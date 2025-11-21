"""
remora: Triton-accelerated evaluation harness for vision-language models.
"""

from .engine import TritonEvaluator
from .integration import VibeCheckModel
from .models import MODEL_PRESETS, load_model_and_tokenizer
from .surgery import TritonBitLinear, hijack_model

__all__ = [
    "TritonEvaluator",
    "VibeCheckModel",
    "TritonBitLinear",
    "hijack_model",
    "MODEL_PRESETS",
    "load_model_and_tokenizer",
    "accelerate",
]


def accelerate(model, tokenizer=None, **kwargs):
    """
    Convenience entrypoint so callers can simply do:

    >>> import vibecheck
    >>> vibecheck.accelerate(model)
    """
    return TritonEvaluator(model=model, tokenizer=tokenizer, **kwargs)
