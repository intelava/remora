import torch
import triton
import triton.language as tl
import torch.nn as nn

@triton.jit
def _rmsnorm_kernel(X_ptr, W_ptr, Out_ptr, stride_x_row, stride_x_col, stride_w_row, stride_w_col, stride_out_row, stride_out_col, N_COLS, eps, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N_COLS
    x = tl.load(X_ptr + row_idx * stride_x_row + cols * stride_x_col, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols * stride_w_col, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt((tl.sum(x * x, axis=0) / N_COLS) + eps)
    tl.store(Out_ptr + row_idx * stride_out_row + cols * stride_out_col, x * rstd * w, mask=mask)


@triton.jit
def _rope_kernel(
    Q_ptr, Cos_ptr, Sin_ptr,
    stride_q_row, stride_q_col,
    stride_cos_row, stride_cos_col,
    seq_len, head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row_idx = pid 
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < head_dim
    q_ptr = Q_ptr + row_idx * stride_q_row
    q = tl.load(q_ptr + offsets * stride_q_col, mask=mask, other=0.0)
    cos = tl.load(Cos_ptr + offsets * stride_cos_col, mask=mask, other=1.0)
    sin = tl.load(Sin_ptr + offsets * stride_cos_col, mask=mask, other=0.0)
    half_dim = head_dim // 2
    swap_offsets = (offsets + half_dim) % head_dim
    q_swapped = tl.load(q_ptr + swap_offsets * stride_q_col, mask=mask, other=0.0)
    sign = tl.where(offsets < half_dim, -1.0, 1.0)
    q_out = (q * cos) + (q_swapped * sin * sign)
    tl.store(q_ptr + offsets * stride_q_col, q_out, mask=mask)

@triton.jit
def _binary_classifier_kernel(
    X_ptr, W_ptr, Out_ptr,
    stride_x_row, stride_x_col,
    stride_w_row, stride_w_col,
    idx_up, idx_down,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row_start_ptr = X_ptr + pid * stride_x_row
    k_offsets = tl.arange(0, BLOCK_SIZE)
    mask = k_offsets < K
    x = tl.load(row_start_ptr + k_offsets * stride_x_col, mask=mask, other=0.0)
    w_up = tl.load(W_ptr + idx_up * stride_w_row + k_offsets * stride_w_col, mask=mask, other=0.0)
    w_down = tl.load(W_ptr + idx_down * stride_w_row + k_offsets * stride_w_col, mask=mask, other=0.0)
    logit_up = tl.sum(x * w_up)
    logit_down = tl.sum(x * w_down)
    result = tl.where(logit_up > logit_down, idx_up, idx_down)
    tl.store(Out_ptr + pid, result)

def triton_rope(q, cos, sin):
    n_rows, dim = q.shape
    BLOCK_SIZE = triton.next_power_of_2(dim)
    _rope_kernel[(n_rows,)](
        q, cos, sin,
        q.stride(0), q.stride(1),
        cos.stride(0), cos.stride(1),
        1, dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return q
