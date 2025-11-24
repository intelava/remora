"""
Preset model loader utilities.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)


@dataclass
class ModelSpec:
    model_id: str
    tokenizer_id: Optional[str]
    description: str
    supports_vision: bool = True
    trust_remote_code: bool = True
    model_loader: Optional[Callable[..., torch.nn.Module]] = None


MODEL_PRESETS: Dict[str, ModelSpec] = {
    "molmo-7b": ModelSpec(
        model_id="allenai/Molmo-7B-D-0924",
        tokenizer_id="allenai/Molmo-7B-D-0924",
        supports_vision=True,
        description="Molmo 7B vision-language model",
    ),
    "smolvlm-base": ModelSpec(
        model_id="HuggingFaceTB/SmolVLM-Base",
        tokenizer_id="HuggingFaceTB/SmolVLM-Base",
        supports_vision=True,
        description="SmolVLM Base vision-language model",
        model_loader=AutoModelForVision2Seq.from_pretrained,
    ),
    "qwen2.5-vl-7b": ModelSpec(
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        tokenizer_id="Qwen/Qwen2.5-VL-7B-Instruct",
        supports_vision=True,
        description="Qwen2.5-VL 7B vision-language model",
    ),
}


def load_model_and_tokenizer(
    preset: str,
    device: Optional[str] = None,
) -> Tuple[torch.nn.Module, object]:
    """
    Loads a preset model + tokenizer pair for convenience.
    """
    if preset not in MODEL_PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {', '.join(MODEL_PRESETS.keys())}")
    spec = MODEL_PRESETS[preset]
    model_fn = spec.model_loader or AutoModelForCausalLM.from_pretrained
    model = model_fn(
        spec.model_id,
        trust_remote_code=spec.trust_remote_code,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            spec.tokenizer_id or spec.model_id,
            trust_remote_code=spec.trust_remote_code,
        )
    except Exception:
        tokenizer = AutoProcessor.from_pretrained(
            spec.tokenizer_id or spec.model_id,
            trust_remote_code=spec.trust_remote_code,
        )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, tokenizer
