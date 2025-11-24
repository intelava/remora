"""
TritonEvaluator coordinates batching, prefix caching, and Triton-backed generation.
"""

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch

from .kernels import w8a16_gemm


def _default_vision_key(image: Any) -> Any:
    # Heuristic: use object id for tensors; otherwise rely on hashable identifier.
    if torch.is_tensor(image):
        return (id(image), tuple(image.shape), image.device.type)
    return image


class PrefixCache:
    def __init__(self, capacity: int = 8):
        self.capacity = capacity
        self._store: OrderedDict[Any, Any] = OrderedDict()

    def get_or_set(self, key: Any, fn: Callable[[], Any]) -> Any:
        if self.capacity == 0:
            return fn()
        if key in self._store:
            value = self._store.pop(key)
            self._store[key] = value
            return value
        value = fn()
        self._store[key] = value
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)
        return value


@dataclass
class GenerationRequest:
    prompt: str
    vision: Any = None
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    generate_kwargs: Dict[str, Any] = field(default_factory=dict)
    vision_key: Any = None


class TritonEvaluator:
    """
    Thin orchestrator that sits between VLMEvalKit and the model. It batches text,
    reuses vision prefixes, and leans on Triton-backed linear layers installed via
    hijack_model.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any = None,
        max_prefix_cache: int = 8,
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        if device is None:
            if hasattr(model, "device"):
                device = model.device
            else:
                try:
                    device = next(model.parameters()).device
                except Exception:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.prefix_cache = PrefixCache(max_prefix_cache)
        self.model.to(self.device)
        if tokenizer is not None and hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = "left"

    # Vision encoding hooks -------------------------------------------------
    def _encode_vision(self, vision: Any) -> Any:
        if vision is None:
            return None

        if hasattr(self.model, "get_vision_tower"):
            vt = self.model.get_vision_tower()
            prefix = vt(vision)
            return prefix.to(self.device) if torch.is_tensor(prefix) else prefix
        if hasattr(self.model, "vision_encoder"):
            prefix = self.model.vision_encoder(vision)
            return prefix.to(self.device) if torch.is_tensor(prefix) else prefix
        # Assume caller already passed encoded features.
        return vision

    def _get_cached_vision_prefix(self, req: GenerationRequest) -> Any:
        vkey = req.vision_key or _default_vision_key(req.vision)
        return self.prefix_cache.get_or_set(vkey, lambda: self._encode_vision(req.vision))

    def _prepare_inputs(self, req: GenerationRequest) -> Dict[str, Any]:
        if self.tokenizer is not None and callable(self.tokenizer):
            if req.vision is not None:
                try:
                    with torch.inference_mode():
                        tokenized = self.tokenizer(
                            req.prompt,
                            images=req.vision,
                            return_tensors="pt",
                            **req.tokenizer_kwargs,
                        )
                    return tokenized.to(self.device)
                except TypeError:
                    # Tokenizer does not accept images keyword; fall back to encoded vision path below.
                    pass
            tokenized = self.tokenizer(req.prompt, return_tensors="pt", **req.tokenizer_kwargs)
            return tokenized.to(self.device)

        # Tokenizer unavailable; rely on caller-provided ids and optional encoded vision.
        text_inputs: Dict[str, Any] = {"input_ids": req.prompt}
        if req.vision is not None:
            vision_prefix = self._get_cached_vision_prefix(req)
            if torch.is_tensor(vision_prefix):
                vision_prefix = vision_prefix.to(self.device)
            text_inputs["vision_hidden_states"] = vision_prefix
        return text_inputs

    # Core batching path ----------------------------------------------------
    def generate_batch(self, requests: List[GenerationRequest]) -> Dict[int, Dict[str, Any]]:
        """
        Executes a batch of requests. Returns a map of index -> dict with text + metadata.
        """
        prepared_inputs: List[Dict[str, Any]] = []
        for req in requests:
            prepared_inputs.append(self._prepare_inputs(req))

        outputs: Dict[int, Dict[str, Any]] = {}
        start = time.time()
        for idx, (req, inputs) in enumerate(zip(requests, prepared_inputs)):
            # Use the underlying HF generate as the high-level loop; the heavy
            # matmuls within have been swapped to TritonBitLinear via hijack_model.
            generate_kwargs = {"max_new_tokens": 64, "use_cache": True}
            generate_kwargs.update(req.generate_kwargs)
            with torch.inference_mode():
                result = self.model.generate(**inputs, **generate_kwargs)
            if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
                text = self.tokenizer.decode(result[0], skip_special_tokens=True)
            else:
                text = result
            outputs[idx] = {
                "text": text,
                "tokens": result,
            }
        elapsed = time.time() - start

        # A basic TPS metric used by run_eval.py.
        total_new_tokens = 0
        for val in outputs.values():
            tokens = val.get("tokens")
            if torch.is_tensor(tokens):
                total_new_tokens += int(tokens.shape[-1])
        outputs["tps"] = total_new_tokens / max(elapsed, 1e-5)
        return outputs

    # Lower-level API for custom callers -----------------------------------
    def triton_linear(self, a: torch.Tensor, w_int8: torch.Tensor, scale: torch.Tensor, bias=None) -> torch.Tensor:
        return w8a16_gemm(a, w_int8, scale, bias)
