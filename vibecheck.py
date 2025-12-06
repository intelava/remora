"""
Lightweight facade so users can `import vibecheck; vibecheck.accelerate(model)`.
Everything is scaffolded for ragged batching + W8A16 quantization, with TODOs
left for you to fill in.
"""

from remora import (
    MODEL_PRESETS,
    GenerationRequest,
    JaggedTensor,
    RemoraEngine,
    VibeCheckModel,
    accelerate,
    jagged_w8a16_gelu_gemm,
    jagged_w8a16_gemm,
    load_model_and_tokenizer,
    pack_sequences,
    pad_jagged,
    quantize_weight_per_channel,
    unpack_sequences,
    w8a16_gelu_gemm,
    w8a16_gemm,
)


__all__ = [
    "accelerate",
    "RemoraEngine",
    "VibeCheckModel",
    "GenerationRequest",
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
]
