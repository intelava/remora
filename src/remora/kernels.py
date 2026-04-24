import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    X_ptr, W_ptr, Out_ptr,
    stride_x_row, stride_x_col,
    stride_w_col,
    stride_out_row, stride_out_col,
    N_COLS, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N_COLS

    x = tl.load(X_ptr + row * stride_x_row + cols * stride_x_col, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols * stride_w_col, mask=mask, other=0.0).to(tl.float32)

    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / N_COLS + eps)
    tl.store(Out_ptr + row * stride_out_row + cols * stride_out_col, x * rstd * w, mask=mask)


@triton.jit
def _rope_kernel(
    Q_ptr, Cos_ptr, Sin_ptr,
    stride_q_row, stride_q_col,
    stride_cos_col,
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < head_dim

    q = tl.load(Q_ptr + row * stride_q_row + offsets * stride_q_col, mask=mask, other=0.0)
    cos = tl.load(Cos_ptr + offsets * stride_cos_col, mask=mask, other=1.0)
    sin = tl.load(Sin_ptr + offsets * stride_cos_col, mask=mask, other=0.0)

    half_dim = head_dim // 2
    swap = (offsets + half_dim) % head_dim
    q_swap = tl.load(Q_ptr + row * stride_q_row + swap * stride_q_col, mask=mask, other=0.0)
    sign = tl.where(offsets < half_dim, -1.0, 1.0)

    tl.store(Q_ptr + row * stride_q_row + offsets * stride_q_col, q * cos + q_swap * sin * sign, mask=mask)


@triton.jit
def _binary_classifier_kernel(
    X_ptr, W_ptr, Out_ptr,
    stride_x_row, stride_x_col,
    stride_w_row, stride_w_col,
    idx_up, idx_down,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    k_offsets = tl.arange(0, BLOCK_SIZE)
    mask = k_offsets < K

    x = tl.load(X_ptr + pid * stride_x_row + k_offsets * stride_x_col, mask=mask, other=0.0)
    w_up = tl.load(W_ptr + idx_up * stride_w_row + k_offsets * stride_w_col, mask=mask, other=0.0)
    w_down = tl.load(W_ptr + idx_down * stride_w_row + k_offsets * stride_w_col, mask=mask, other=0.0)

    logit_up = tl.sum(x * w_up)
    logit_down = tl.sum(x * w_down)
    tl.store(Out_ptr + pid, tl.where(logit_up > logit_down, idx_up, idx_down))


@triton.jit
def _swiglu_kernel(
    Gate_ptr, Up_ptr, Out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused SiLU(gate) * up in a single kernel pass.

    Replaces three PyTorch elementwise kernels (silu, mul, store) with one,
    cutting DRAM round-trips from 5 tensor passes to 3 (load gate, load up,
    store out).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    gate = tl.load(Gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    up   = tl.load(Up_ptr   + offs, mask=mask, other=0.0).to(tl.float32)

    out = gate * tl.sigmoid(gate) * up          # SiLU(gate) * up
    tl.store(Out_ptr + offs, out.to(tl.float16), mask=mask)


@triton.jit
def _fused_add_rmsnorm_kernel(
    X_ptr, Residual_ptr, W_ptr, Out_ptr, Res_out_ptr,
    stride_row, stride_col,
    stride_w,
    N_COLS, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Add residual in-place, then apply RMSNorm — two ops, one DRAM round-trip.

    Saves one full read+write of the hidden state compared to separate
    residual-add and RMSNorm kernels (common bottleneck in decoder blocks).
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N_COLS

    x   = tl.load(X_ptr        + row * stride_row + cols * stride_col, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(Residual_ptr  + row * stride_row + cols * stride_col, mask=mask, other=0.0).to(tl.float32)
    w   = tl.load(W_ptr         + cols * stride_w,                      mask=mask, other=1.0).to(tl.float32)

    x_added = x + res                                       # residual add
    rstd = tl.rsqrt(tl.sum(x_added * x_added, axis=0) / N_COLS + eps)
    normed = x_added * rstd * w

    tl.store(Res_out_ptr + row * stride_row + cols * stride_col, x_added.to(tl.float16), mask=mask)
    tl.store(Out_ptr     + row * stride_row + cols * stride_col, normed.to(tl.float16),  mask=mask)


def triton_rope(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    n_rows, dim = q.shape
    BLOCK_SIZE = triton.next_power_of_2(dim)
    _rope_kernel[(n_rows,)](
        q, cos, sin,
        q.stride(0), q.stride(1),
        cos.stride(1),
        head_dim=dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return q
