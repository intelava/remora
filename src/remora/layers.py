from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

from .kernels import (
    _binary_classifier_kernel,
    _fused_add_rmsnorm_kernel,
    _rmsnorm_kernel,
    _swiglu_kernel,
)


class W8A16Linear(nn.Module):
    """INT8 weight-only quantized linear layer (W8A16).

    Weights are stored as int8 with per-row symmetric scale factors.
    Halves weight memory vs fp16; torch.compile fuses the dequantize + matmul.
    """

    weight_int8: torch.Tensor
    scale: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        linear_layer: Optional[nn.Linear] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if linear_layer is not None:
            w = linear_layer.weight.data.float()
            scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0
            w_int8 = (w / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
            self.register_buffer("weight_int8", w_int8)
            self.register_buffer("scale", scale.half())
            if bias and linear_layer.bias is not None:
                self.bias = nn.Parameter(linear_layer.bias.data.half())
            else:
                self.bias = None
        else:
            self.register_buffer("weight_int8", torch.zeros(out_features, in_features, dtype=torch.int8))
            self.register_buffer("scale", torch.ones(out_features, dtype=torch.float16))
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_fp16 = self.weight_int8.half() * self.scale.unsqueeze(1)
        return F.linear(x.half(), w_fp16, self.bias).to(x.dtype)


class TritonRMSNorm(nn.Module):
    """Fused RMS normalization via a custom Triton kernel."""

    def __init__(self, original_norm: nn.Module) -> None:
        super().__init__()
        self.weight = original_norm.weight
        self.eps: float = getattr(original_norm, "variance_epsilon", None) or getattr(original_norm, "eps", 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        hidden_dim = orig_shape[-1]
        x_flat = x.reshape(-1, hidden_dim).contiguous()
        out_flat = torch.empty_like(x_flat)
        M = x_flat.shape[0]
        BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
        num_warps = 4
        if BLOCK_SIZE >= 2048:
            num_warps = 8
        if BLOCK_SIZE >= 4096:
            num_warps = 16
        _rmsnorm_kernel[(M,)](
            x_flat, self.weight, out_flat,
            x_flat.stride(0), x_flat.stride(1),
            self.weight.stride(0),
            out_flat.stride(0), out_flat.stride(1),
            hidden_dim, self.eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return out_flat.reshape(orig_shape)


class FusedSwiGLUMLP(nn.Module):
    """Drop-in for Qwen2MLP that fuses gate+up into a single GEMM.

    Original path: gate_proj(x) → GEMM₁; up_proj(x) → GEMM₂; silu(gate)*up.
    Fused path:    cat([gate_w, up_w]) @ x → one GEMM, split, silu*up.

    Saves one GEMM launch and one full read of hidden_states per layer (×24).
    Uses PyTorch's F.silu for the elementwise step — it's already optimized and
    faster than a hand-rolled Triton kernel at this operand size.
    """

    def __init__(self, original_mlp: nn.Module) -> None:
        super().__init__()
        gate_w = original_mlp.gate_proj.weight.data
        up_w   = original_mlp.up_proj.weight.data
        self.gate_dim = gate_w.shape[0]
        self.register_buffer("fused_weight", torch.cat([gate_w, up_w], dim=0))
        if original_mlp.gate_proj.bias is not None:
            self.fused_bias = nn.Parameter(
                torch.cat([original_mlp.gate_proj.bias.data,
                           original_mlp.up_proj.bias.data], dim=0)
            )
        else:
            self.fused_bias = None
        self.down_proj = original_mlp.down_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = F.linear(x, self.fused_weight, self.fused_bias)
        gate, up = gate_up[..., :self.gate_dim], gate_up[..., self.gate_dim:]
        return self.down_proj(F.silu(gate) * up)


class TritonSwiGLUMLP(nn.Module):
    """Elementwise-fused SwiGLU (kept for reference — slower than FusedSwiGLUMLP).

    Fuses SiLU(gate)*up into one Triton kernel but still issues two separate GEMMs.
    Profiling showed the elementwise kernel is not the bottleneck; the extra GEMM
    and hidden-state re-read are.  Use FusedSwiGLUMLP instead.
    """

    def __init__(self, original_mlp: nn.Module) -> None:
        super().__init__()
        self.gate_proj = original_mlp.gate_proj
        self.up_proj   = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up   = self.up_proj(x)
        out = torch.empty_like(gate)
        n = gate.numel()
        BLOCK = min(triton.next_power_of_2(n), 4096)
        _swiglu_kernel[((n + BLOCK - 1) // BLOCK,)](
            gate.contiguous(), up.contiguous(), out,
            n, BLOCK_SIZE=BLOCK,
        )
        return self.down_proj(out)


class FusedQKVAttention(nn.Module):
    """Attention wrapper that fuses Q+K+V projections into a single GEMM.

    Original: q_proj(x), k_proj(x), v_proj(x) — three GEMMs, three reads of x.
    Fused:    cat([q_w, k_w, v_w]) @ x — one GEMM, one read of x, then split.

    Wraps the original attention module; only the projection step is changed.
    All other logic (RoPE, SDPA, output projection) runs unchanged.
    """

    def __init__(self, original_attn: nn.Module) -> None:
        super().__init__()
        self._attn = original_attn

        q_w = original_attn.q_proj.weight.data
        k_w = original_attn.k_proj.weight.data
        v_w = original_attn.v_proj.weight.data
        self._q_dim = q_w.shape[0]
        self._k_dim = k_w.shape[0]
        self._v_dim = v_w.shape[0]
        self.register_buffer("_qkv_weight", torch.cat([q_w, k_w, v_w], dim=0))

        has_bias = original_attn.q_proj.bias is not None
        if has_bias:
            self._qkv_bias = nn.Parameter(torch.cat([
                original_attn.q_proj.bias.data,
                original_attn.k_proj.bias.data,
                original_attn.v_proj.bias.data,
            ], dim=0))
        else:
            self._qkv_bias = None

        # Monkey-patch the three projection methods so the wrapped forward uses ours.
        self._patch_projections()

    def _patch_projections(self) -> None:
        qkv_weight = self._qkv_weight
        qkv_bias   = self._qkv_bias
        q_dim, k_dim, v_dim = self._q_dim, self._k_dim, self._v_dim

        # Thread-safe per-instance cache stored on the module itself.
        _cache: dict = {}

        class _QProj(nn.Module):
            def forward(self, x):
                out = F.linear(x, qkv_weight, qkv_bias)
                _cache["k"] = out[..., q_dim : q_dim + k_dim]
                _cache["v"] = out[..., q_dim + k_dim :]
                return out[..., :q_dim]

        class _KProj(nn.Module):
            def forward(self, x):
                return _cache["k"]

        class _VProj(nn.Module):
            def forward(self, x):
                return _cache["v"]

        self._attn.q_proj = _QProj()
        self._attn.k_proj = _KProj()
        self._attn.v_proj = _VProj()

    def forward(self, *args, **kwargs):
        return self._attn(*args, **kwargs)

    # Proxy attribute access so the rest of the model can still inspect
    # self_attn.num_heads etc. without touching _attn directly.
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._attn, name)


class TritonFusedAddRMSNorm(nn.Module):
    """Fused residual-add + RMSNorm — one DRAM round-trip instead of two.

    Decoder blocks do: hidden = hidden + residual; hidden = RMSNorm(hidden).
    Replacing both with this module halves memory traffic for the hidden state
    across all 24 decoder layers.
    """

    def __init__(self, original_norm: nn.Module) -> None:
        super().__init__()
        self.weight = original_norm.weight
        self.eps: float = getattr(original_norm, "variance_epsilon", None) or getattr(original_norm, "eps", 1e-6)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        D = orig_shape[-1]
        x_flat   = x.reshape(-1, D).contiguous()
        res_flat = residual.reshape(-1, D).contiguous()
        out_flat     = torch.empty_like(x_flat)
        res_out_flat = torch.empty_like(x_flat)
        M = x_flat.shape[0]
        BLOCK = triton.next_power_of_2(D)
        num_warps = 4 if BLOCK < 2048 else (8 if BLOCK < 4096 else 16)
        _fused_add_rmsnorm_kernel[(M,)](
            x_flat, res_flat, self.weight, out_flat, res_out_flat,
            x_flat.stride(0), x_flat.stride(1),
            self.weight.stride(0),
            D, self.eps,
            BLOCK_SIZE=BLOCK,
            num_warps=num_warps,
        )
        return out_flat.reshape(orig_shape), res_out_flat.reshape(orig_shape)


class TritonBinaryHead(nn.Module):
    """Triton-accelerated binary classifier head.

    Instead of computing the full vocabulary logits, only computes dot
    products for two target token IDs (e.g. 'up' and 'down').
    """

    def __init__(self, original_head: nn.Module, up_id: int, down_id: int) -> None:
        super().__init__()
        self.weight = original_head.weight
        self.up_id = up_id
        self.down_id = down_id
        self.hidden_dim = self.weight.shape[1]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states[:, -1, :].contiguous() if hidden_states.dim() == 3 else hidden_states.contiguous()
        batch_size = x.shape[0]
        out = torch.empty(batch_size, dtype=torch.int32, device=x.device)
        BLOCK_SIZE = triton.next_power_of_2(self.hidden_dim)
        _binary_classifier_kernel[(batch_size,)](
            x, self.weight, out,
            x.stride(0), x.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            self.up_id, self.down_id,
            K=self.hidden_dim,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out.long().unsqueeze(1)
