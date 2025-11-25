"""
remora: Triton-accelerated inference for vision-language models.

Key features that go beyond torch.compile:
1. W8A16 Quantization - int8 weights with fp16 activations, 2x memory bandwidth savings
2. Jagged Tensors - variable-length sequences without padding overhead
3. Fused Vision Projector - optimized "modality gap" layer with GELU fusion

Quick start:
    >>> import remora
    >>> # Full VLM optimization (projector + LLM layers)
    >>> remora.accelerate(model, tokenizer, full_vlm=True)

    >>> # Or step-by-step control
    >>> from remora import hijack_vision_projector, hijack_model
    >>> hijack_vision_projector(model)  # Fused projector
    >>> hijack_model(model)  # W8A16 linear layers

For jagged tensor operations:
    >>> from remora import JaggedTensor, pack_sequences
    >>> jagged = pack_sequences([seq1, seq2, seq3])  # No padding!
    >>> out = remora.jagged_w8a16_gemm(jagged, weights, scales)
"""

from .engine import TritonEvaluator
from .integration import VibeCheckModel
from .kernels import (
    JaggedTensor,
    is_triton_available,
    jagged_w8a16_gemm,
    jagged_w8a16_gelu_gemm,
    pack_sequences,
    pad_jagged,
    quantize_weight_per_channel,
    unpack_sequences,
    w8a16_gemm,
    w8a16_gelu_gemm,
)
from .models import MODEL_PRESETS, load_model_and_tokenizer
from .surgery import (
    TritonBitLinear,
    TritonGELULinear,
    TritonVisionProjector,
    find_vision_projector,
    full_vlm_surgery,
    hijack_model,
    hijack_vision_projector,
)

__all__ = [
    # High-level API
    "accelerate",
    "TritonEvaluator",
    "VibeCheckModel",
    # Model surgery
    "hijack_model",
    "hijack_vision_projector",
    "full_vlm_surgery",
    "find_vision_projector",
    # Module replacements
    "TritonBitLinear",
    "TritonGELULinear",
    "TritonVisionProjector",
    # Jagged tensor support
    "JaggedTensor",
    "pack_sequences",
    "unpack_sequences",
    "pad_jagged",
    # Kernels
    "w8a16_gemm",
    "w8a16_gelu_gemm",
    "jagged_w8a16_gemm",
    "jagged_w8a16_gelu_gemm",
    "quantize_weight_per_channel",
    "is_triton_available",
    # Model loading
    "MODEL_PRESETS",
    "load_model_and_tokenizer",
]


def accelerate(model, tokenizer=None, full_vlm: bool = False, **kwargs):
    """
    Applies Triton surgery and returns a TritonEvaluator instance.

    Args:
        model: PyTorch model to accelerate
        tokenizer: Optional tokenizer for the model
        full_vlm: If True, also attempts to replace the vision projector
                  with a fused TritonVisionProjector (recommended for VLMs)
        **kwargs: Additional arguments passed to TritonEvaluator

    Returns:
        TritonEvaluator instance wrapping the accelerated model

    Example:
        >>> import remora
        >>> # Basic acceleration (LLM layers only)
        >>> evaluator = remora.accelerate(model, tokenizer)

        >>> # Full VLM acceleration (projector + LLM layers)
        >>> evaluator = remora.accelerate(model, tokenizer, full_vlm=True)
    """
    verbose = kwargs.pop("verbose", True)

    if full_vlm:
        full_vlm_surgery(model, verbose=verbose)
    else:
        hijack_model(model, verbose=verbose)

    return TritonEvaluator(model=model, tokenizer=tokenizer, **kwargs)
