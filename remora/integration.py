"""
VLMEvalKit integration. We use an internal queue to coalesce sequential calls
into batches that your ragged/W8A16 pipeline can consume.
"""

import logging
import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional

try:
    from vlmeval.api import CustomLLM
except Exception:  # pragma: no cover - optional dependency
    class CustomLLM:  # type: ignore
        pass

from .engine import GenerationRequest, RemoraEngine

logger = logging.getLogger(__name__)


@dataclass
class _QueuedRequest:
    idx: int
    prompt: str
    vision: Any
    future: Future
    tokenizer_kwargs: Dict[str, Any]
    generate_kwargs: Dict[str, Any]
    vision_key: Any


class VibeCheckModel(CustomLLM):
    """
    Drop-in wrapper for VLMEvalKit. Calls are buffered and flushed into batches
    so your ragged batching path sees meaningful batch sizes even when the
    upstream loop is strictly sequential.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        batch_size: int = 4,
        flush_ms: float = 10.0,
        max_queue: int = 64,
        evaluator: Optional[RemoraEngine] = None,
    ):
        super().__init__()
        self.engine = evaluator or RemoraEngine(model=model, tokenizer=tokenizer)
        self.batch_size = batch_size
        self.flush_ms = flush_ms
        self.max_queue = max_queue
        self._queue: Deque[_QueuedRequest] = deque()
        self._cv = threading.Condition()
        self._counter = 0
        self._closed = False
        self._worker = threading.Thread(target=self._drain_loop, daemon=True)
        self._worker.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # Public API expected by VLMEvalKit ------------------------------------
    def generate(self, prompt: str, image: Any = None, **kwargs) -> str:
        fut = self._enqueue(prompt, image=image, kwargs=kwargs)
        return fut.result()

    def generate_batch(self, prompts: List[str], images: Optional[List[Any]] = None, **kwargs) -> List[str]:
        images = images or [None] * len(prompts)
        futures = [self._enqueue(p, image=im, kwargs=kwargs, flush_now=True) for p, im in zip(prompts, images)]
        return [f.result() for f in futures]

    # Internal worker ------------------------------------------------------
    def _enqueue(self, prompt: str, image: Any, kwargs: Dict[str, Any], flush_now: bool = False) -> Future:
        with self._cv:
            if self._closed:
                raise RuntimeError("VibeCheckModel has been closed and cannot accept new work.")
            while len(self._queue) >= self.max_queue and not self._closed:
                self._cv.wait(timeout=0.001)
            if self._closed:
                raise RuntimeError("VibeCheckModel has been closed and cannot accept new work.")
            payload = dict(kwargs) if kwargs else {}
            fut: Future = Future()
            tokenizer_kwargs = payload.pop("tokenizer_kwargs", {})
            vision_key = payload.pop("vision_key", None)
            item = _QueuedRequest(
                idx=self._counter,
                prompt=prompt,
                vision=image,
                future=fut,
                tokenizer_kwargs=tokenizer_kwargs,
                generate_kwargs=payload,
                vision_key=vision_key,
            )
            self._counter += 1
            self._queue.append(item)
            self._cv.notify_all()
            if flush_now:
                self._cv.notify()
        return fut

    def _drain_loop(self):
        while True:
            batch: List[_QueuedRequest] = []
            with self._cv:
                if not self._queue:
                    self._cv.wait(timeout=self.flush_ms / 1000.0)
                if not self._queue and self._closed:
                    break
                if (
                    not self._closed
                    and self._queue
                    and len(self._queue) < self.batch_size
                ):
                    # Delay slightly to allow sequential callers to accumulate into a batch.
                    self._cv.wait(timeout=self.flush_ms / 1000.0)
                while self._queue and len(batch) < self.batch_size:
                    batch.append(self._queue.popleft())
                self._cv.notify_all()

            if not batch:
                continue

            try:
                gen_reqs = [
                    GenerationRequest(
                        prompt=req.prompt,
                        vision=req.vision,
                        tokenizer_kwargs=req.tokenizer_kwargs,
                        generate_kwargs=req.generate_kwargs,
                        vision_key=req.vision_key,
                    )
                    for req in batch
                ]
                results = self.engine.generate_batch(gen_reqs)
            except Exception as exc:
                for req in batch:
                    if not req.future.cancelled():
                        req.future.set_exception(exc)
                continue

            for i, req in enumerate(batch):
                if req.future.cancelled():
                    continue
                out = results.get(i)
                if out is None:
                    error_msg = f"Missing generation output for batch index {i}."
                    logger.error(error_msg)
                    req.future.set_exception(RuntimeError(error_msg))
                    continue
                
                if "error" in out:
                    logger.error(f"Generation error for request {i}: {out['error']}")
                    req.future.set_exception(RuntimeError(out["error"]))
                    continue

                req.future.set_result(out.get("text", ""))

    def close(self, cancel_pending: bool = False, timeout: float = 0.5):
        self._closed = True
        with self._cv:
            if cancel_pending:
                while self._queue:
                    req = self._queue.popleft()
                    if not req.future.done():
                        req.future.cancel()
            self._cv.notify_all()
        self._worker.join(timeout=timeout)
