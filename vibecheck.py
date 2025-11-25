"""
Lightweight facade so users can `import vibecheck; vibecheck.accelerate(model)`.

For full functionality, use `import remora` directly.
"""

from remora import (
    MODEL_PRESETS,
    JaggedTensor,
    TritonEvaluator,
    TritonVisionProjector,
    VibeCheckModel,
    full_vlm_surgery,
    hijack_model,
    hijack_vision_projector,
    load_model_and_tokenizer,
    pack_sequences,
)


def accelerate(model, tokenizer=None, full_vlm: bool = False, **kwargs):
    """
    Applies Triton surgery and returns a TritonEvaluator instance.

    Args:
        model: PyTorch model to accelerate
        tokenizer: Optional tokenizer
        full_vlm: If True, also replaces vision projector with fused version
    """
    verbose = kwargs.pop("verbose", True)
    if full_vlm:
        full_vlm_surgery(model, verbose=verbose)
    else:
        hijack_model(model, verbose=verbose)
    return TritonEvaluator(model=model, tokenizer=tokenizer, **kwargs)


__all__ = [
    "accelerate",
    "TritonEvaluator",
    "VibeCheckModel",
    "TritonVisionProjector",
    "hijack_model",
    "hijack_vision_projector",
    "full_vlm_surgery",
    "JaggedTensor",
    "pack_sequences",
    "MODEL_PRESETS",
    "load_model_and_tokenizer",
]
