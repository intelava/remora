from __future__ import annotations

import torch
import torch.nn as nn

from .layers import (
    FusedQKVAttention,
    FusedSwiGLUMLP,
    TritonBinaryHead,
    TritonFusedAddRMSNorm,
    TritonRMSNorm,
)


def replace_rmsnorm(module: nn.Module) -> int:
    """Recursively swap every RMSNorm for a fused Triton kernel. Returns count."""
    count = 0
    for name, child in module.named_children():
        if "RMSNorm" in child.__class__.__name__:
            setattr(module, name, TritonRMSNorm(child))
            count += 1
        else:
            count += replace_rmsnorm(child)
    return count


def replace_mlp(module: nn.Module) -> int:
    """Fuse gate+up projections into a single GEMM per MLP layer. Returns count."""
    count = 0
    for name, child in module.named_children():
        if (hasattr(child, "gate_proj") and hasattr(child, "up_proj") and hasattr(child, "down_proj")):
            setattr(module, name, FusedSwiGLUMLP(child))
            count += 1
        else:
            count += replace_mlp(child)
    return count


def replace_qkv(module: nn.Module) -> int:
    """Fuse Q+K+V projections into a single GEMM per attention layer. Returns count."""
    count = 0
    for name, child in module.named_children():
        if (hasattr(child, "q_proj") and hasattr(child, "k_proj") and hasattr(child, "v_proj")
                and isinstance(child.q_proj, nn.Linear)):
            setattr(module, name, FusedQKVAttention(child))
            count += 1
        else:
            count += replace_qkv(child)
    return count


def optimize_model(model: nn.Module, processor) -> nn.Module:
    """Apply all Remora optimizations to a LLaVA-style VLM.

    Fusion stack (ordered by profiled impact):
      1. FusedSwiGLUMLP   — gate+up into one GEMM (was two). Halves hidden-state
                            reads and cuts one GEMM launch per decoder layer ×24.
                            Uses PyTorch F.silu for the elementwise step (faster
                            than a hand-rolled Triton kernel at this operand size).
      2. FusedQKVAttention — Q+K+V into one GEMM (was three). Cuts 2 extra reads
                             of hidden_states per attention layer ×24.
      3. TritonRMSNorm    — fused RMS norm across 24+1 decoder norm layers.
      4. TritonBinaryHead — 2 logits instead of 152k vocab.

    Not applied:
      - torch.compile: blocked by transformers 5.x output_capturing wrapper.
      - Custom flash attention: SDPA already active (pytorch_flash backend).
      - cuBLAS GEMMs (52.6%): already near-peak on A100 with Ampere tensor cores.
    """
    n_mlp = replace_mlp(model.model.language_model)
    print(f"  Fused {n_mlp} MLP gate+up projections (FusedSwiGLUMLP)")

    n_qkv = replace_qkv(model.model.language_model)
    print(f"  Fused {n_qkv} QKV projections (FusedQKVAttention)")

    n_norm = replace_rmsnorm(model)
    print(f"  Replaced {n_norm} RMSNorm layers with TritonRMSNorm")

    up_id   = processor.tokenizer.encode("up")[-1]
    down_id = processor.tokenizer.encode("down")[-1]
    model.lm_head = TritonBinaryHead(model.lm_head, up_id, down_id)
    print(f"  Replaced lm_head with TritonBinaryHead (up={up_id}, down={down_id})")

    return model
