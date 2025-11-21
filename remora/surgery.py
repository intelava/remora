"""
Model surgery utilities that swap nn.Linear for TritonBitLinear in-place.
"""

from typing import Any, Iterable, Optional

import math
import torch
import torch.nn as nn

from .kernels import quantize_weight_per_channel, w8a16_gemm


class TritonBitLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear using int8 weights and the Triton W8A16 GEMM.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs)) if bias else None
        self.register_buffer("weight_int8", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("weight_scale", torch.empty(0, dtype=torch.float32), persistent=False)
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


def _replace_linear(module: nn.Module, prefix: str, names: Iterable[str], verbose: bool) -> int:
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


def hijack_model(model: Any, include: Optional[Iterable[str]] = None, verbose: bool = True) -> Any:
    """
    Recursively replaces nn.Linear layers with TritonBitLinear in the provided model.
    """
    _replace_linear(model, prefix="", names=include or [], verbose=verbose)
    return model
