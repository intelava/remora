import torch
import torch.nn as nn
from . import kernels

def _quantize_weight(weight: torch.Tensor):
    """
    Performs row-wise absolute max quantization for a weight tensor.
    """
    absmax = weight.abs().max(dim=-1, keepdim=True)[0]
    scales = absmax / 127.0
    scales.clamp_(min=1e-5)
    quantized_weights = (weight / scales).round().to(torch.int8)
    return quantized_weights, scales.to(torch.float16)

class W8A16Linear(nn.Module):
    """
    A Remora-optimized Linear layer that performs weight-only 8-bit quantization
    and uses a high-performance Triton kernel for w8a16 matrix multiplication.
    This layer is a drop-in replacement for nn.Linear.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, linear_layer: nn.Linear = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if linear_layer is None:
            # If no layer is provided, create a placeholder one to get weights
            linear_layer = nn.Linear(in_features, out_features, bias=bias, dtype=torch.float16)

        # Quantize the weights and register them as buffers
        weight = linear_layer.weight.clone().to(torch.float16).cuda()
        int8_weights, scales = _quantize_weight(weight)
        self.register_buffer('int8_weights', int8_weights)
        self.register_buffer('scales', scales)
        
        # Register bias if it exists
        if bias and linear_layer.bias is not None:
            self.bias = linear_layer.bias.clone().to(torch.float16)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass using the custom w8a16 Triton kernel.
        """
        x = x.to(torch.float16)
        
        # Reshape for matrix multiplication if necessary
        original_shape = x.shape
        if len(original_shape) > 2:
            x = x.view(-1, self.in_features)

        # Call the custom Triton kernel launcher
        # The kernel now handles the dequantization scaling
        output = kernels.w8a16_matmul(x, self.int8_weights, self.scales)

        # Add bias if it exists
        if self.bias is not None:
            output += self.bias
        
        # Reshape back to original dimensions if necessary
        if len(original_shape) > 2:
            output = output.view(*original_shape[:-1], self.out_features)
            
        return output

    def __repr__(self):
        return f"Remora_W8A16Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"