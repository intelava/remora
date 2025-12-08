import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 4}),
        # Additional configs for H100 might benefit from larger block sizes
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 8}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def w8a16_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Kernel for W8A16 (int8 weight, fp16 activation) matrix multiplication.
    C = A @ B.T, where B is int8 and A is fp16.

    This kernel is optimized for modern GPUs (e.g., Ampere, Hopper) by:
    1. Using vectorized loads to read int8 weights.
    2. Casting the entire block of weights to fp16 at once.
    3. Using tl.dot() to leverage Tensor Cores for the matrix multiplication.
    4. Using grouped ordering to improve L2 cache reuse.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        
        # Load int8 weights and cast to fp16
        b_int8 = tl.load(b_ptrs, mask=b_mask, other=0)
        b = b_int8.to(tl.float16)
        
        # Load fp16 activations
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Perform matrix multiplication
        accumulator += tl.dot(a, b)

    c = accumulator.to(tl.float16)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def w8a16_matmul(a: torch.Tensor, b_quant: torch.Tensor, b_scale: torch.Tensor) -> torch.Tensor:
    """
    Launcher for the w8a16_matmul kernel.

    Args:
        a (torch.Tensor): The fp16 activation tensor of shape (M, K).
        b_quant (torch.Tensor): The int8 quantized weight tensor of shape (N, K).
        b_scale (torch.Tensor): The fp16 scale tensor of shape (N, 1).
    """
    M, K = a.shape
    N, _ = b_quant.shape

    # Output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Transpose weights for coalesced memory access
    b_quant_t = b_quant.T.contiguous()

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    w8a16_matmul_kernel[grid](
        a, b_quant_t, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b_quant_t.stride(0), b_quant_t.stride(1),
        c.stride(0), c.stride(1),
    )
    
    # Dequantize the output
    return c * b_scale.T