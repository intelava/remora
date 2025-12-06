"""
Optional model-surgery hooks.

If you want to swap modules to use your W8A16 quantized path, add that logic
here. The previous accelerator-specific replacements have been removed.
"""

from typing import Any


def hijack_model(model: Any, *args, **kwargs) -> Any:
    """
    TODO: replace this with your own model surgery (e.g., swap nn.Linear for
    a W8A16-aware module). Return the modified model.
    """
    raise NotImplementedError("Implement your model surgery here.")


def hijack_vision_projector(model: Any, *args, **kwargs) -> Any:
    """
    TODO: optionally replace vision projection layers to use your quantized path.
    """
    raise NotImplementedError("Implement optional vision projector surgery here.")


def full_vlm_surgery(model: Any, *args, **kwargs) -> Any:
    """
    TODO: orchestrate any multi-step surgery you need for full VLMs.
    """
    raise NotImplementedError("Implement full VLM surgery here.")
