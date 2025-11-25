"""
Triton kernels for mixed-precision GEMM with support for:

1. W8A16 GEMM - int8 weights, fp16 activations with on-the-fly dequantization
2. Jagged/Ragged tensors - variable-length sequences without padding overhead
3. Fused GELU+Linear - saves memory bandwidth by fusing activation with matmul

These kernels target the gaps that torch.compile cannot easily optimize:
- Mixed-precision integer math (W8A16)
- Dynamic shapes without recompilation (jagged tensors)
- Custom fusion patterns (GELU+Linear)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional dependency for CPU dev environments
    triton = None
    tl = None


def is_triton_available() -> bool:
    return triton is not None and tl is not None


# =============================================================================
# Jagged Tensor Support
# =============================================================================


@dataclass
class JaggedTensor:
    """
    Represents variable-length sequences packed into a single contiguous tensor.
    Eliminates padding overhead for mixed image+text token batches.

    Attributes:
        data: Flattened tensor of shape [total_tokens, hidden_dim]
        cu_seqlens: Cumulative sequence lengths of shape [batch_size + 1]
                    cu_seqlens[i] is the start index of sequence i
                    cu_seqlens[batch_size] == total_tokens

    Example:
        3 sequences of lengths [5, 3, 7] -> cu_seqlens = [0, 5, 8, 15]
        Sequence 0: data[0:5], Sequence 1: data[5:8], Sequence 2: data[8:15]
    """

    data: torch.Tensor
    cu_seqlens: torch.Tensor

    @property
    def batch_size(self) -> int:
        return len(self.cu_seqlens) - 1

    @property
    def total_tokens(self) -> int:
        return self.data.shape[0]

    @property
    def hidden_dim(self) -> int:
        return self.data.shape[-1]

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    def get_sequence(self, idx: int) -> torch.Tensor:
        """Extract a single sequence by batch index."""
        start = int(self.cu_seqlens[idx].item())
        end = int(self.cu_seqlens[idx + 1].item())
        return self.data[start:end]

    def sequence_lengths(self) -> torch.Tensor:
        """Returns tensor of individual sequence lengths."""
        return self.cu_seqlens[1:] - self.cu_seqlens[:-1]


def pack_sequences(sequences: List[torch.Tensor]) -> JaggedTensor:
    """
    Pack a list of variable-length sequences into a JaggedTensor.

    Args:
        sequences: List of tensors, each of shape [seq_len_i, hidden_dim]

    Returns:
        JaggedTensor with packed data and cumulative sequence lengths

    Example:
        >>> seqs = [torch.randn(5, 64), torch.randn(3, 64), torch.randn(7, 64)]
        >>> jagged = pack_sequences(seqs)
        >>> jagged.total_tokens
        15
        >>> jagged.cu_seqlens
        tensor([0, 5, 8, 15])
    """
    if not sequences:
        raise ValueError("Cannot pack empty sequence list")

    device = sequences[0].device
    hidden_dim = sequences[0].shape[-1]

    # Validate all sequences have same hidden dim
    for i, seq in enumerate(sequences):
        if seq.shape[-1] != hidden_dim:
            raise ValueError(
                f"Sequence {i} has hidden_dim {seq.shape[-1]}, expected {hidden_dim}"
            )

    lengths = [seq.shape[0] for seq in sequences]
    cu_seqlens = torch.zeros(len(sequences) + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(
        torch.tensor(lengths, dtype=torch.int32, device=device), dim=0
    )

    data = torch.cat(sequences, dim=0)
    return JaggedTensor(data=data, cu_seqlens=cu_seqlens)


def unpack_sequences(jagged: JaggedTensor) -> List[torch.Tensor]:
    """
    Unpack a JaggedTensor into a list of variable-length sequences.

    Args:
        jagged: JaggedTensor to unpack

    Returns:
        List of tensors, one per sequence
    """
    return [jagged.get_sequence(i) for i in range(jagged.batch_size)]


def pad_jagged(
    jagged: JaggedTensor, max_len: Optional[int] = None, pad_value: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert JaggedTensor to padded format (for compatibility with non-jagged ops).

    Args:
        jagged: JaggedTensor to pad
        max_len: Maximum sequence length (defaults to longest sequence)
        pad_value: Value to use for padding

    Returns:
        Tuple of (padded_tensor [B, max_len, hidden], attention_mask [B, max_len])
    """
    lengths = jagged.sequence_lengths()
    if max_len is None:
        max_len = int(lengths.max().item())

    B = jagged.batch_size
    H = jagged.hidden_dim

    padded = torch.full(
        (B, max_len, H), pad_value, dtype=jagged.dtype, device=jagged.device
    )
    mask = torch.zeros(B, max_len, dtype=torch.bool, device=jagged.device)

    for i in range(B):
        seq_len = int(lengths[i].item())
        padded[i, :seq_len] = jagged.get_sequence(i)
        mask[i, :seq_len] = True

    return padded, mask


# =============================================================================
# Triton Kernels
# =============================================================================

if is_triton_available():

    @triton.jit
    def _gelu_approx(x):
        """
        Fast GELU approximation: x * sigmoid(1.702 * x)
        Matches PyTorch's approximate='tanh' GELU closely.
        """
        return x * tl.sigmoid(1.702 * x)

    # -------------------------------------------------------------------------
    # W8A16 GEMM Kernel (no activation)
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # W8A16 GEMM + GELU Fused Kernel
    # -------------------------------------------------------------------------

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
    def _w8a16_gelu_matmul(
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
        """W8A16 matmul with fused GELU activation on the output."""
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

            b = b * scale[None, :]
            acc += tl.dot(a, b, out_dtype=tl.float32)

        if apply_bias:
            bias = tl.load(
                bias_ptr + offs_n,
                mask=offs_n < N,
                other=0.0,
            )
            acc += bias[None, :]

        # Fused GELU activation
        acc = _gelu_approx(acc)

        c = acc.to(tl.float16)
        tl.store(
            c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
            c,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


# =============================================================================
# Python API
# =============================================================================


def w8a16_gemm(
    a: torch.Tensor,
    w_int8: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Performs (A @ W) using int8 weights and fp16 activations.
    Dequantizes weights on-the-fly in SRAM for 2x memory bandwidth savings.

    Args:
        a: Activation tensor of shape [..., K] in fp16/bf16.
        w_int8: Quantized weights of shape [N, K] or [K, N] in int8.
        w_scale: Per-output-channel scale of shape [N].
        bias: Optional bias of shape [N].

    Returns:
        Output tensor of shape [..., N] in fp16 (same batch dims as input).
    """
    # Save original shape to restore later
    orig_shape = a.shape
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
        # Restore original batch dimensions
        if len(orig_shape) > 2:
            c = c.view(*orig_shape[:-1], N)
        return c

    # Fallback: dequantize and use a standard matmul for CPU/dev environments.
    dequant_w = (w_int8_t.float() * w_scale[None, :]).to(a.dtype)
    out = a @ dequant_w
    if bias is not None:
        out += bias
    # Restore original batch dimensions
    if len(orig_shape) > 2:
        out = out.view(*orig_shape[:-1], N)
    return out


def w8a16_gelu_gemm(
    a: torch.Tensor,
    w_int8: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Performs GELU(A @ W + bias) using int8 weights and fp16 activations.
    Fuses the linear layer with GELU activation to save one memory round-trip.

    This is particularly valuable for MLP layers where the activation would
    otherwise require writing to and reading from global memory.

    Args:
        a: Activation tensor of shape [..., K] in fp16/bf16.
        w_int8: Quantized weights of shape [N, K] or [K, N] in int8.
        w_scale: Per-output-channel scale of shape [N].
        bias: Optional bias of shape [N].

    Returns:
        Output tensor of shape [..., N] with GELU applied, in fp16.
    """
    # Save original shape to restore later
    orig_shape = a.shape
    if a.dim() != 2:
        a = a.flatten(0, -2)
    if w_int8.dim() != 2:
        raise ValueError("Expected 2D weights for w8a16_gelu_gemm.")

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
        _w8a16_gelu_matmul[grid](
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
        # Restore original batch dimensions
        if len(orig_shape) > 2:
            c = c.view(*orig_shape[:-1], N)
        return c

    # Fallback: dequantize and use standard ops for CPU/dev environments.
    import torch.nn.functional as F

    dequant_w = (w_int8_t.float() * w_scale[None, :]).to(a.dtype)
    out = a @ dequant_w
    if bias is not None:
        out = out + bias
    out = F.gelu(out, approximate="tanh")
    # Restore original batch dimensions
    if len(orig_shape) > 2:
        out = out.view(*orig_shape[:-1], N)
    return out


def jagged_w8a16_gemm(
    jagged: JaggedTensor,
    w_int8: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> JaggedTensor:
    """
    Applies W8A16 GEMM to a JaggedTensor, preserving the jagged structure.

    Since linear layers are token-independent, we simply apply the GEMM to the
    flattened data tensor and return a new JaggedTensor with the same cu_seqlens.

    Args:
        jagged: Input JaggedTensor with data of shape [total_tokens, in_features]
        w_int8: Quantized weights of shape [out_features, in_features] in int8.
        w_scale: Per-output-channel scale of shape [out_features].
        bias: Optional bias of shape [out_features].

    Returns:
        JaggedTensor with data of shape [total_tokens, out_features]
    """
    out_data = w8a16_gemm(jagged.data, w_int8, w_scale, bias)
    return JaggedTensor(data=out_data, cu_seqlens=jagged.cu_seqlens)


def jagged_w8a16_gelu_gemm(
    jagged: JaggedTensor,
    w_int8: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> JaggedTensor:
    """
    Applies fused W8A16 GELU GEMM to a JaggedTensor, preserving the jagged structure.

    Args:
        jagged: Input JaggedTensor with data of shape [total_tokens, in_features]
        w_int8: Quantized weights of shape [out_features, in_features] in int8.
        w_scale: Per-output-channel scale of shape [out_features].
        bias: Optional bias of shape [out_features].

    Returns:
        JaggedTensor with data of shape [total_tokens, out_features], GELU applied.
    """
    out_data = w8a16_gelu_gemm(jagged.data, w_int8, w_scale, bias)
    return JaggedTensor(data=out_data, cu_seqlens=jagged.cu_seqlens)


def quantize_weight_per_channel(
    weight: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes weights to int8 with per-output-channel scales.

    Args:
        weight: FP16/FP32 weight tensor of shape [out_features, in_features]
        eps: Small constant to avoid division by zero

    Returns:
        Tuple of (w_int8, scales) where:
        - w_int8: int8 weights of shape [out_features, in_features]
        - scales: fp32 per-channel scales of shape [out_features]
    """
    # weight: [out_features, in_features]
    if weight.dim() != 2:
        raise ValueError("Expected 2D weight tensor.")
    max_val = weight.abs().amax(dim=1) + eps
    scales = (max_val / 127.0).to(torch.float32)
    w_int8 = torch.round(weight / scales[:, None]).clamp(-127, 127).to(torch.int8)
    return w_int8, scales
