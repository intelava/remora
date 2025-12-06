"""
remora: scaffold for ragged batching + W8A16 quantization.

The heavy lifting is intentionally left to you. Fill in the TODOs across
`remora/engine.py` and `remora/kernels.py` to:
- Build ragged batches for mixed text/vision inputs
- Run your model with W8A16 quantized matmuls
- Decode outputs back into text
"""

from .engine import GenerationRequest, RemoraEngine
from .integration import VibeCheckModel
from .kernels import (
    JaggedTensor,
    jagged_w8a16_gelu_gemm,
    jagged_w8a16_gemm,
    pack_sequences,
    pad_jagged,
    quantize_weight_per_channel,
    unpack_sequences,
    w8a16_gelu_gemm,
    w8a16_gemm,
)
from .models import MODEL_PRESETS, load_model_and_tokenizer
from .surgery import full_vlm_surgery, hijack_model, hijack_vision_projector

__all__ = [
    "accelerate",
    "GenerationRequest",
    "RemoraEngine",
    "VibeCheckModel",
    "JaggedTensor",
    "pack_sequences",
    "unpack_sequences",
    "pad_jagged",
    "w8a16_gemm",
    "w8a16_gelu_gemm",
    "jagged_w8a16_gemm",
    "jagged_w8a16_gelu_gemm",
    "quantize_weight_per_channel",
    "MODEL_PRESETS",
    "load_model_and_tokenizer",
    "hijack_model",
    "hijack_vision_projector",
    "full_vlm_surgery",
]


import logging

# Configure a null handler so usage without logging configuration doesn't complain
logging.getLogger(__name__).addHandler(logging.NullHandler())


def accelerate(model, tokenizer=None, **kwargs):
    """
    Convenience wrapper that returns a RemoraEngine instance.
    No automatic surgery or kernel swapping is performed here.
    """
    return RemoraEngine(model=model, tokenizer=tokenizer, **kwargs)
