"""
Model surgery utilities for Triton-accelerated VLM inference.

Provides:
1. TritonBitLinear - W8A16 drop-in replacement for nn.Linear
2. TritonVisionProjector - Fused 2-layer MLP for vision-to-language projection
3. hijack_model - Recursive nn.Linear replacement
4. hijack_vision_projector - Targeted projector replacement for VLMs
"""

from typing import Any, Iterable, Optional, Tuple

import math
import torch
import torch.nn as nn

from .kernels import quantize_weight_per_channel, w8a16_gemm, w8a16_gelu_gemm


class TritonBitLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear using int8 weights and the Triton W8A16 GEMM.
    Dequantizes weights on-the-fly in SRAM for 2x memory bandwidth savings.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, **factory_kwargs)) if bias else None
        )
        self.register_buffer(
            "weight_int8", torch.empty(0, dtype=torch.int8), persistent=False
        )
        self.register_buffer(
            "weight_scale", torch.empty(0, dtype=torch.float32), persistent=False
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        self._requantize()

    def _requantize(self):
        w_int8, scale = quantize_weight_per_channel(self.weight.detach())
        self.weight_int8 = w_int8
        self.weight_scale = scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.weight_int8.numel() == 0:
            self._requantize()
        return w8a16_gemm(input, self.weight_int8, self.weight_scale, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class TritonGELULinear(nn.Module):
    """
    Fused Linear + GELU layer using W8A16 quantization.
    Saves one memory round-trip by computing GELU in the same kernel as the matmul.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, **factory_kwargs)) if bias else None
        )
        self.register_buffer(
            "weight_int8", torch.empty(0, dtype=torch.int8), persistent=False
        )
        self.register_buffer(
            "weight_scale", torch.empty(0, dtype=torch.float32), persistent=False
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        self._requantize()

    def _requantize(self):
        w_int8, scale = quantize_weight_per_channel(self.weight.detach())
        self.weight_int8 = w_int8
        self.weight_scale = scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.weight_int8.numel() == 0:
            self._requantize()
        return w8a16_gelu_gemm(input, self.weight_int8, self.weight_scale, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, activation=GELU"


class TritonVisionProjector(nn.Module):
    """
    Fused two-layer MLP projector for vision-to-language embedding conversion.
    Uses W8A16 quantization with fused GELU for the first layer.

    Architecture: Linear(in_dim → hidden_dim) → GELU → Linear(hidden_dim → out_dim)

    This is the "Modality Gap" fusion that torch.compile struggles with because:
    1. Vision projectors process fixed token counts (e.g., 576 image tokens)
    2. The shapes change dynamically between vision encoder and LLM
    3. Standard compilation breaks on these boundaries

    Memory savings:
    - First layer: W8A16 + fused GELU saves 1 read + 1 write of intermediate
    - Second layer: W8A16 saves memory bandwidth on weight loading
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        factory_kwargs = {"device": device, "dtype": dtype}

        # First layer weights (with fused GELU)
        self.weight1 = nn.Parameter(
            torch.empty((hidden_features, in_features), **factory_kwargs)
        )
        self.bias1 = (
            nn.Parameter(torch.empty(hidden_features, **factory_kwargs))
            if bias
            else None
        )

        # Second layer weights
        self.weight2 = nn.Parameter(
            torch.empty((out_features, hidden_features), **factory_kwargs)
        )
        self.bias2 = (
            nn.Parameter(torch.empty(out_features, **factory_kwargs)) if bias else None
        )

        # Quantized buffers for both layers
        self.register_buffer(
            "weight1_int8", torch.empty(0, dtype=torch.int8), persistent=False
        )
        self.register_buffer(
            "weight1_scale", torch.empty(0, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "weight2_int8", torch.empty(0, dtype=torch.int8), persistent=False
        )
        self.register_buffer(
            "weight2_scale", torch.empty(0, dtype=torch.float32), persistent=False
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        if self.bias1 is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight1)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias1, -bound, bound)
        if self.bias2 is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight2)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias2, -bound, bound)
        self._requantize()

    def _requantize(self):
        self.weight1_int8, self.weight1_scale = quantize_weight_per_channel(
            self.weight1.detach()
        )
        self.weight2_int8, self.weight2_scale = quantize_weight_per_channel(
            self.weight2.detach()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [N, in_features] where N is typically
               the number of vision tokens (e.g., 576 for a 24x24 patch grid)

        Returns:
            Output tensor of shape [N, out_features] ready for the LLM
        """
        if self.weight1_int8.numel() == 0:
            self._requantize()

        # Flatten if needed (handle [B, N, D] input)
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.flatten(0, -2)

        # First layer with fused GELU
        h = w8a16_gelu_gemm(x, self.weight1_int8, self.weight1_scale, self.bias1)

        # Second layer
        out = w8a16_gemm(h, self.weight2_int8, self.weight2_scale, self.bias2)

        # Restore shape if needed
        if len(orig_shape) > 2:
            out = out.view(*orig_shape[:-1], -1)

        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, hidden_features={self.hidden_features}, "
            f"out_features={self.out_features}, bias={self.bias1 is not None}"
        )


# =============================================================================
# Model Surgery Functions
# =============================================================================


def _replace_linear(
    module: nn.Module, prefix: str, names: Iterable[str], verbose: bool
) -> int:
    """Recursively replace nn.Linear with TritonBitLinear."""
    replaced = 0
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            if names and not any(tag in full_name for tag in names):
                continue
            new_layer = TritonBitLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                device=child.weight.device,
                dtype=child.weight.dtype,
            )
            with torch.no_grad():
                new_layer.weight.copy_(child.weight)
                if child.bias is not None:
                    new_layer.bias.copy_(child.bias)
                new_layer._requantize()
            setattr(module, name, new_layer)
            replaced += 1
            if verbose:
                print(f"[remora] swapped {full_name} -> TritonBitLinear")
        else:
            replaced += _replace_linear(child, full_name, names, verbose)
    return replaced


def hijack_model(
    model: Any, include: Optional[Iterable[str]] = None, verbose: bool = True
) -> Any:
    """
    Recursively replaces nn.Linear layers with TritonBitLinear in the provided model.

    Args:
        model: PyTorch model to modify
        include: Optional list of substrings to filter which layers to replace
        verbose: Print replacement messages

    Returns:
        The modified model (in-place modification)
    """
    count = _replace_linear(model, prefix="", names=include or [], verbose=verbose)
    if verbose:
        print(f"[remora] Replaced {count} Linear layers with TritonBitLinear")
    return model


def find_vision_projector(model: nn.Module) -> Optional[Tuple[str, nn.Module]]:
    """
    Attempts to locate the vision-to-language projector in a VLM.

    Common projector names across different VLM architectures:
    - LLaVA: 'mm_projector', 'multi_modal_projector'
    - Qwen-VL: 'visual.merger', 'visual_projector'
    - InternVL: 'mlp1', 'connector'
    - Idefics: 'modality_projection'

    Returns:
        Tuple of (full_name, module) if found, None otherwise
    """
    projector_names = [
        "mm_projector",
        "multi_modal_projector",
        "vision_proj",
        "visual_proj",
        "connector",
        "vision_projector",
        "modality_projection",
        "visual.merger",
        "mlp1",
    ]

    for name, module in model.named_modules():
        module_name = name.split(".")[-1].lower()
        # Check if any projector name matches
        if any(pn.lower() in name.lower() for pn in projector_names):
            return name, module

    return None


def _extract_mlp_structure(
    module: nn.Module,
) -> Optional[Tuple[nn.Linear, nn.Linear, bool]]:
    """
    Extract the two linear layers from an MLP-style projector.

    Returns:
        Tuple of (first_linear, last_linear, has_gelu) or None if not recognized
    """
    linears = []
    has_gelu = False

    for name, child in module.named_modules():
        if isinstance(child, nn.Linear):
            linears.append(child)
        if isinstance(child, nn.GELU) or "gelu" in name.lower():
            has_gelu = True

    if len(linears) >= 2:
        return linears[0], linears[-1], has_gelu

    return None


def hijack_vision_projector(model: nn.Module, verbose: bool = True) -> bool:
    """
    Replaces the vision projector with a TritonVisionProjector if found.

    This targets the "Modality Gap" - the projection layer between vision encoder
    and language model that torch.compile struggles to optimize due to:
    1. Dynamic shape transitions
    2. Fixed vision token counts (e.g., 576) vs variable text lengths
    3. Non-standard activation patterns

    Args:
        model: VLM model to modify
        verbose: Print status messages

    Returns:
        True if replacement was made, False otherwise
    """
    result = find_vision_projector(model)
    if result is None:
        if verbose:
            print("[remora] No vision projector found to replace.")
        return False

    full_name, proj = result

    # Try to extract MLP structure
    mlp_info = _extract_mlp_structure(proj)
    if mlp_info is None:
        if verbose:
            print(
                f"[remora] Projector '{full_name}' structure not recognized (expected 2-layer MLP), skipping."
            )
        return False

    first_linear, last_linear, has_gelu = mlp_info

    # Get dimensions
    in_features = first_linear.in_features
    hidden_features = first_linear.out_features
    out_features = last_linear.out_features

    # Validate hidden dimension matches
    if last_linear.in_features != hidden_features:
        if verbose:
            print(
                f"[remora] Projector '{full_name}' has mismatched hidden dims "
                f"({hidden_features} vs {last_linear.in_features}), skipping."
            )
        return False

    # Create replacement
    new_proj = TritonVisionProjector(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        bias=first_linear.bias is not None,
        device=first_linear.weight.device,
        dtype=first_linear.weight.dtype,
    )

    # Copy weights
    with torch.no_grad():
        new_proj.weight1.copy_(first_linear.weight)
        new_proj.weight2.copy_(last_linear.weight)
        if first_linear.bias is not None and new_proj.bias1 is not None:
            new_proj.bias1.copy_(first_linear.bias)
        if last_linear.bias is not None and new_proj.bias2 is not None:
            new_proj.bias2.copy_(last_linear.bias)
        new_proj._requantize()

    # Replace in model
    parts = full_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_proj)

    if verbose:
        print(
            f"[remora] Replaced '{full_name}' with TritonVisionProjector "
            f"({in_features} → {hidden_features} → {out_features})"
        )

    return True


def full_vlm_surgery(
    model: nn.Module,
    include_linear: Optional[Iterable[str]] = None,
    replace_projector: bool = True,
    verbose: bool = True,
) -> nn.Module:
    """
    Complete VLM optimization: replaces both the vision projector and LLM linear layers.

    This is the recommended entry point for VLM optimization, applying:
    1. Vision projector replacement (if found) with fused GELU
    2. LLM linear layer replacement with W8A16

    Args:
        model: VLM model to optimize
        include_linear: Optional filter for which linear layers to replace
        replace_projector: Whether to attempt projector replacement
        verbose: Print status messages

    Returns:
        The optimized model (modified in-place)
    """
    if replace_projector:
        hijack_vision_projector(model, verbose=verbose)

    hijack_model(model, include=include_linear, verbose=verbose)

    return model
