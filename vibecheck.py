"""
Lightweight facade so users can `import vibecheck; vibecheck.accelerate(model)`.
"""

from remora import (
    MODEL_PRESETS,
    TritonEvaluator,
    VibeCheckModel,
    hijack_model,
    load_model_and_tokenizer,
)


def accelerate(model, tokenizer=None, **kwargs):
    """
    Applies Triton surgery and returns a TritonEvaluator instance.
    """
    hijack_model(model, verbose=kwargs.pop("verbose", True))
    return TritonEvaluator(model=model, tokenizer=tokenizer, **kwargs)


__all__ = [
    "accelerate",
    "TritonEvaluator",
    "VibeCheckModel",
    "hijack_model",
    "MODEL_PRESETS",
    "load_model_and_tokenizer",
]
