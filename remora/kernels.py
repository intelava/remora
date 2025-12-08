import torch
import triton
import triton.language as tl

# This is a PoC and may require a specific Triton version.
# Developed with triton==2.1.0 and torch==2.1.0 on a T4 GPU.

@triton.jit
def w8a16_matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables tell us how to move from one element to the next
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Kernel for computing C = A @ B.
    A is of shape (M, K) and is fp16.
    B is of shape (K, N) and is int8 (transposed from original (N,K)).
    C is of shape (M, N) and is fp16.

    This kernel is designed for efficiency by loading B in large contiguous blocks (vectors).
    B, the weight matrix, is assumed to be transposed and contiguous, enabling
    vectorized reads along the K dimension.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first block of C.
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # -----------------------------------------------------------
    # Create accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # -----------------------------------------------------------
    # Iterate over K in blocks of BLOCK_SIZE_K.
    # We use a standard for loop instead of tl.static_for because the number of iterations
    # is data-dependent (on K).
    a_offs = stride_am * offs_m[:, None]
    b_offs = stride_bn * offs_n[None, :]
    
    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        
        # Load the block of A (activation)
        a_ptrs = a_ptr + a_offs + stride_ak * offs_k[None, :]
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load the block of B (weights)
        b_ptrs = b_ptr + stride_bk * offs_k[:, None] + b_offs
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        
        # --- The Critical Optimization ---
        # 1. Load the int8 weights in a large contiguous block (vectorized load).
        b_int8 = tl.load(b_ptrs, mask=b_mask, other=0)
        
        # 2. Cast the entire vector to fp16 in a single operation.
        # This is much faster than casting scalar values inside the kernel.
        b = b_int8.to(tl.float16)
        
        # 3. Perform the matrix multiplication using the fp16 accumulator.
        accumulator += tl.dot(a, b)

    # Convert accumulator to fp16 and store the result
    c = accumulator.to(tl.float16)
    tl.store(c_ptrs, c, mask=c_mask)

def w8a16_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Launcher for the w8a16_matmul_kernel.

    Args:
        a (torch.Tensor): The fp16 activation tensor of shape (M, K).
        b (torch.Tensor): The int8 weight tensor of shape (N, K).
    """
    # Check inputs
    assert a.is_contiguous(), "Input tensor A must be contiguous"
    assert a.dtype == torch.float16
    assert b.dtype == torch.int8
    
    M, K = a.shape
    N, K_b = b.shape
    assert K == K_b, f"Shapes {a.shape} and {b.shape} are not compatible for matmul"

    # Transpose b to (K, N) and ensure it's contiguous for efficient memory access.
    # This layout is crucial for the kernel's vectorized loads along the K dimension.
    b_transposed = b.T.contiguous()
    assert b_transposed.is_contiguous()

    # Allocates output buffer
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Grid dimensions
    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    # Heuristics for block sizes and other meta-parameters
    # These may need tuning for different GPU architectures
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8
    num_warps = 4
    num_stages = 3

    # Launch kernel
    w8a16_matmul_kernel[grid](
        a, b_transposed, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b_transposed.stride(0), b_transposed.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return c
