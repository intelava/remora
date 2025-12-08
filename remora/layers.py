import torch
import torch.nn as nn
from . import kernels

def _quantize_weight(weight: torch.Tensor):
    """
    Performs row-wise absolute max quantization for a weight tensor.
    This is a common technique for 8-bit weight-only quantization.

    Args:
        weight (torch.Tensor): A fp16 or fp32 tensor of shape (out_features, in_features).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of (int8_weights, scales).
                                           The weights are int8 and scales are fp16.
    """
    # Calculate the absolute maximum value for each row (per-output-channel).
    # This value is used to scale the weights to the [-127, 127] range.
    absmax = weight.abs().max(dim=-1, keepdim=True)[0]
    
    # Calculate the scaling factor. We divide by 127 to map the max value to 127.
    scales = absmax / 127.0
    
    # Clamp scales to avoid division by zero in case of all-zero rows.
    scales.clamp_(min=1e-5)
    
    # Apply the scaling and round to the nearest integer.
    quantized_weights = (weight / scales).round().to(torch.int8)
    
    # Return quantized weights and fp16 scales.
    return quantized_weights, scales.to(torch.float16)

class W8A16Linear(nn.Module):
    """
    An nn.Module that implements a weight-only 8-bit quantized linear layer.
    It uses a high-performance Triton kernel for the w8a16 matrix multiplication
    (fp16 activation, int8 weight).

    This layer demonstrates how to integrate custom kernels for performance-critical
    operations while maintaining the familiar PyTorch API.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, linear_layer: nn.Linear = None):
        """
        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool): If True, adds a learnable bias to the output.
            linear_layer (nn.Linear, optional): A pre-trained nn.Linear layer to quantize.
                                                If None, random weights are created for demonstration.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if linear_layer:
            # If a layer is provided, use its weights and bias
            assert linear_layer.in_features == in_features
            assert linear_layer.out_features == out_features
            weight = linear_layer.weight.clone()
            self.bias = linear_layer.bias.clone().to(torch.float16) if bias and linear_layer.bias is not None else None
        else:
            # Otherwise, initialize with random weights for demonstration purposes
            weight = torch.randn(out_features, in_features, dtype=torch.float32)
            self.bias = nn.Parameter(torch.randn(out_features, dtype=torch.float16)) if bias else None
        
        # Quantize the weights and register them as buffers.
        # Buffers are part of the module's state but are not considered model parameters.
        int8_weights, scales = _quantize_weight(weight.cuda())
        self.register_buffer('int8_weights', int8_weights)
        self.register_buffer('scales', scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass using the custom w8a16 Triton kernel.

        The operation is equivalent to `F.linear(x, dequantized_weight, self.bias)`,
        but is executed much faster.

        Args:
            x (torch.Tensor): Input tensor of shape (..., in_features).

        Returns:
            torch.Tensor: Output tensor of shape (..., out_features).
        """
        # The input activation is expected to be fp16.
        x = x.to(torch.float16).cuda()
        
        # Reshape input tensor to 2D for matrix multiplication: (..., in_features) -> (M, in_features)
        original_shape = x.shape
        x_reshaped = x.view(-1, self.in_features)

        # Call the custom Triton kernel launcher for w8a16 matmul.
        # This computes: output_q = x_reshaped @ int8_weights.T
        output_q = kernels.w8a16_matmul(x_reshaped, self.int8_weights)
        
        # Dequantize the output by multiplying with the scales.
        # output_q has shape (M, out_features)
        # self.scales has shape (out_features, 1) -> .T gives (1, out_features)
        # Broadcasting applies the correct scale to each column of the output.
        output = output_q * self.scales.T

        # Add bias if it exists
        if self.bias is not None:
            output += self.bias
            
        # Reshape the output back to its original batch dimensions
        return output.view(*original_shape[:-1], self.out_features)

    def __repr__(self):
        return f"W8A16Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"
