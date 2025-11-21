"""
VLMEvalKit integration. We use an internal queue to coalesce sequential calls from
the evaluator into GPU-friendly batches.
"""

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

from .engine import GenerationRequest, TritonEvaluator


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
    so Triton kernels see meaningful batch sizes even when the upstream loop is
    strictly sequential.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        batch_size: int = 4,
        flush_ms: float = 10.0,
        max_queue: int = 64,
        evaluator: Optional[TritonEvaluator] = None,
    ):
        super().__init__()
        self.engine = evaluator or TritonEvaluator(model=model, tokenizer=tokenizer)
        self.batch_size = batch_size
        self.flush_ms = flush_ms
        self.max_queue = max_queue
        self._queue: Deque[_QueuedRequest] = deque()
        self._cv = threading.Condition()
        self._counter = 0
        self._closed = False
        self._worker = threading.Thread(target=self._drain_loop, daemon=True)
        self._worker.start()

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
            while len(self._queue) >= self.max_queue and not self._closed:
                self._cv.wait(timeout=0.001)
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
        while not self._closed:
            batch: List[_QueuedRequest] = []
            with self._cv:
                if not self._queue:
                    self._cv.wait(timeout=self.flush_ms / 1000.0)
                elif len(self._queue) < self.batch_size:
                    # Delay slightly to allow sequential callers to accumulate into a batch.
                    self._cv.wait(timeout=self.flush_ms / 1000.0)
                while self._queue and len(batch) < self.batch_size:
                    batch.append(self._queue.popleft())
                self._cv.notify_all()

            if not batch:
                continue

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
            for i, req in enumerate(batch):
                out = results.get(i, {})
                if not req.future.cancelled():
                    req.future.set_result(out.get("text", ""))

    def close(self):
        self._closed = True
        with self._cv:
            self._cv.notify_all()
        self._worker.join(timeout=0.1)
