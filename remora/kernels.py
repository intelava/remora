"""
Triton kernels for mixed-precision GEMM.

The main target is a W8A16 GEMM (int8 weights, fp16 activations) that supports
flattened (ragged) activation layouts commonly found in KV-cache paging or
packed batches.
"""

from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional dependency for CPU dev environments
    triton = None
    tl = None


def is_triton_available() -> bool:
    return triton is not None and tl is not None


if is_triton_available():

    @triton.autotune(
        configs=[
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "SPLIT_K": 1},
                num_warps=8,
                num_stages=2,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1},
                num_warps=8,
                num_stages=2,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": 1},
                num_warps=8,
                num_stages=2,
            ),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def _w8a16_matmul(
        a_ptr,
        b_ptr,
        c_ptr,
        scales_ptr,
        bias_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_scale,
        apply_bias: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            k_inds = k + offs_k
            a = tl.load(
                a_ptr + offs_m[:, None] * stride_am + k_inds[None, :] * stride_ak,
                mask=(offs_m[:, None] < M) & (k_inds[None, :] < K),
                other=0.0,
            ).to(tl.float16)

            b = tl.load(
                b_ptr + k_inds[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                mask=(k_inds[:, None] < K) & (offs_n[None, :] < N),
                other=0,
            ).to(tl.float16)

            scale = tl.load(
                scales_ptr + offs_n * stride_scale,
                mask=offs_n < N,
                other=1.0,
            ).to(tl.float16)

            # Convert int8 weights to fp16 on the fly using scale.
            b = b * scale[None, :]
            acc += tl.dot(a, b, out_dtype=tl.float32)

        if apply_bias:
            bias = tl.load(
                bias_ptr + offs_n,
                mask=offs_n < N,
                other=0.0,
            )
            acc += bias[None, :]

        c = acc.to(tl.float16)
        tl.store(
            c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
            c,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


def w8a16_gemm(
    a: torch.Tensor,
    w_int8: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Performs (A @ W) using int8 weights and fp16 activations.

    Args:
        a: Activation tensor of shape [M, K] in fp16/bf16.
        w_int8: Quantized weights of shape [N, K] or [K, N] in int8.
        w_scale: Per-output-channel scale of shape [N].
        bias: Optional bias of shape [N].
    """
    if a.dim() != 2:
        a = a.flatten(0, -2)
    if w_int8.dim() != 2:
        raise ValueError("Expected 2D weights for w8a16_gemm.")

    if not torch.is_floating_point(a):
        a = a.float()

    # Ensure weights are laid out as [K, N] to match kernel expectation.
    if w_int8.shape[0] == a.shape[1]:
        w_int8_t = w_int8
    else:
        w_int8_t = w_int8.t().contiguous()

    M, K = a.shape
    K_w, N = w_int8_t.shape
    if K_w != K:
        raise ValueError(f"K mismatch: activation K={K}, weight K={K_w}")

    w_int8_t = w_int8_t.to(a.device)
    w_scale = w_scale.to(a.device)
    if bias is not None:
        bias = bias.to(a.device)

    if is_triton_available() and a.is_cuda and w_int8_t.is_cuda:
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)

        grid = (
            triton.cdiv(M, 64),
            triton.cdiv(N, 64),
        )
        _w8a16_matmul[grid](
            a,
            w_int8_t,
            c,
            w_scale,
            bias if bias is not None else torch.empty(1, device=a.device),
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            w_int8_t.stride(0),
            w_int8_t.stride(1),
            c.stride(0),
            c.stride(1),
            w_scale.stride(0),
            bias is not None,
        )
        return c

    # Fallback: dequantize and use a standard matmul for CPU/dev environments.
    dequant_w = (w_int8_t.float() * w_scale[None, :]).to(a.dtype)
    out = a @ dequant_w
    if bias is not None:
        out += bias
    return out


def quantize_weight_per_channel(
    weight: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes weights to int8 with per-output-channel scales.
    """
    # weight: [out_features, in_features]
    if weight.dim() != 2:
        raise ValueError("Expected 2D weight tensor.")
    max_val = weight.abs().amax(dim=1) + eps
    scales = (max_val / 127.0).to(torch.float32)
    w_int8 = torch.round(weight / scales[:, None]).clamp(-127, 127).to(torch.int8)
    return w_int8, scales
