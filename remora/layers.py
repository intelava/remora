import torch
import triton
import triton.language as tl
import torch.nn as nn

from remora.kernels import _rmsnorm_kernel, _binary_classifier_kernel

class TritonRMSNorm(nn.Module):
    def __init__(self, original_norm):
        super().__init__()
        self.weight = original_norm.weight
        self.eps = original_norm.variance_epsilon if hasattr(original_norm, 'variance_epsilon') else original_norm.eps
    
    def forward(self, x):
        orig_shape = x.shape
        hidden_dim = orig_shape[-1]
        
        x_flat = x.view(-1, hidden_dim)
        y_flat = torch.empty_like(x_flat)
        M, N = x_flat.shape
        
        BLOCK_SIZE = triton.next_power_of_2(N)
        
        num_warps = 4
        if BLOCK_SIZE >= 2048: num_warps = 8
        if BLOCK_SIZE >= 4096: num_warps = 16

        _rmsnorm_kernel[(M,)](
            x_flat, self.weight, y_flat, 
            x_flat.stride(0), x_flat.stride(1), 
            self.weight.stride(0), self.weight.stride(0),
            y_flat.stride(0), y_flat.stride(1),
            N, self.eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
        )
        
        return y_flat.view(orig_shape)


class TritonBinaryHead(torch.nn.Module):
    def __init__(self, original_head, up_id, down_id):
        super().__init__()
        self.weight = original_head.weight
        self.up_id = up_id
        self.down_id = down_id
        self.hidden_dim = self.weight.shape[1]

    def forward(self, hidden_states):
        if hidden_states.dim() == 3:
            x = hidden_states[:, -1, :].contiguous()
        else:
            x = hidden_states.contiguous()
            
        batch_size = x.shape[0]
        output = torch.empty(batch_size, dtype=torch.int32, device=x.device)
        BLOCK_SIZE = triton.next_power_of_2(self.hidden_dim)
        
        _binary_classifier_kernel[(batch_size,)](
            x, self.weight, output,
            x.stride(0), x.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            self.up_id, self.down_id,
            K=self.hidden_dim,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return output.long().unsqueeze(1)
