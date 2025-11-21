"""
Preset model loader utilities.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


@dataclass
class ModelSpec:
    model_id: str
    tokenizer_id: Optional[str]
    description: str
    supports_vision: bool = True
    trust_remote_code: bool = True


MODEL_PRESETS: Dict[str, ModelSpec] = {
    "molmo-7b": ModelSpec(
        model_id="allenai/Molmo-7B-D-0924",
        tokenizer_id="allenai/Molmo-7B-D-0924",
        supports_vision=True,
        description="Molmo 7B vision-language model",
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
    model = AutoModelForCausalLM.from_pretrained(
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
