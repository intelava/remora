"""
Preset model loader utilities.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    model_id: str
    tokenizer_id: Optional[str]
    description: str
    trust_remote_code: bool = True
    model_loader: Optional[Callable[..., torch.nn.Module]] = None


MODEL_PRESETS: Dict[str, ModelSpec] = {
    "molmo-7b": ModelSpec(
        model_id="allenai/Molmo-7B-D-0924",
        tokenizer_id="allenai/Molmo-7B-D-0924",
        description="Molmo 7B vision-language model",
    ),
    "smolvlm-base": ModelSpec(
        model_id="HuggingFaceTB/SmolVLM-Base",
        tokenizer_id="HuggingFaceTB/SmolVLM-Base",
        description="SmolVLM Base vision-language model",
        model_loader=AutoModelForVision2Seq.from_pretrained,
    ),
    "qwen2.5-vl-7b": ModelSpec(
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        tokenizer_id="Qwen/Qwen2.5-VL-7B-Instruct",
        description="Qwen2.5-VL 7B vision-language model",
    ),
    "phi-3.5-vision": ModelSpec(
        model_id="microsoft/Phi-3.5-vision-instruct",
        tokenizer_id="microsoft/Phi-3.5-vision-instruct",
        description="Phi-3.5 Vision (4.2B) - efficient multimodal model",
    ),
    "qwen2-vl-2b": ModelSpec(
        model_id="Qwen/Qwen2-VL-2B-Instruct",
        tokenizer_id="Qwen/Qwen2-VL-2B-Instruct",
        description="Qwen2-VL 2B vision-language model - compact and efficient",
    ),
    "llava-1.5-7b": ModelSpec(
        model_id="llava-hf/llava-1.5-7b-hf",
        tokenizer_id="llava-hf/llava-1.5-7b-hf",
        description="LLaVA 1.5 7B - popular open-source vision-language model",
    ),
    "internvl2-4b": ModelSpec(
        model_id="OpenGVLab/InternVL2-4B",
        tokenizer_id="OpenGVLab/InternVL2-4B",
        description="InternVL2 4B - strong performance with 4B parameters",
    ),
    "idefics2-8b": ModelSpec(
        model_id="HuggingFaceM4/idefics2-8b",
        tokenizer_id="HuggingFaceM4/idefics2-8b",
        description="Idefics2 8B - HuggingFace's multimodal model",
        model_loader=AutoModelForVision2Seq.from_pretrained,
    ),
    "molmo-7b-o": ModelSpec(
        model_id="allenai/Molmo-7B-O-0924",
        tokenizer_id="allenai/Molmo-7B-O-0924",
        description="Molmo 7B-O open weights vision-language model",
    ),
    "qwen2.5-vl-3b": ModelSpec(
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        tokenizer_id="Qwen/Qwen2.5-VL-3B-Instruct",
        description="Qwen2.5-VL 3B - smaller but capable vision-language model",
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
        error_msg = f"Unknown preset '{preset}'. Available: {', '.join(MODEL_PRESETS.keys())}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    spec = MODEL_PRESETS[preset]
    logger.info(f"Loading preset '{preset}': {spec.description}")

    model_fn = spec.model_loader or AutoModelForCausalLM.from_pretrained
    try:
        model = model_fn(
            spec.model_id,
            trust_remote_code=spec.trust_remote_code,
        )
    except Exception as e:
        logger.error(f"Failed to load model '{spec.model_id}': {e}")
        raise

    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            spec.tokenizer_id or spec.model_id,
            trust_remote_code=spec.trust_remote_code,
        )
    except Exception as e:
        logger.info(f"AutoTokenizer failed for '{spec.tokenizer_id or spec.model_id}', trying AutoProcessor. Error: {e}")
        try:
            tokenizer = AutoProcessor.from_pretrained(
                spec.tokenizer_id or spec.model_id,
                trust_remote_code=spec.trust_remote_code,
            )
        except Exception as e2:
            logger.error(f"Both AutoTokenizer and AutoProcessor failed for '{spec.tokenizer_id or spec.model_id}'. Last error: {e2}")
            # We don't raise here immediately, as some workflows might work without tokenizer? 
            # But usually it's fatal. Let's raise.
            raise RuntimeError(f"Could not load tokenizer or processor for {preset}") from e2

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Moving model to {device}")
    model = model.to(device)
    return model, tokenizer
