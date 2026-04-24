"""Benchmark: LLaVA-OneVision inference — baseline vs. Remora optimizations.

Runs the same forward pass (image + text → action token) under progressively
more aggressive optimizations and prints a comparison table.

Usage:
    python benchmarks/bench_vlm.py
    python benchmarks/bench_vlm.py --warmup 10 --frames 100
    python benchmarks/bench_vlm.py --model-id llava-hf/llava-onevision-qwen2-0.5b-si-hf
"""

from __future__ import annotations

import argparse
import statistics
from typing import Callable, List

import numpy as np
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

TARGET_MS = 1000.0 / 30.0  # 33.3 ms → 30 FPS
MODEL_ID = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _cuda_latencies(fn: Callable, warmup: int, frames: int) -> List[float]:
    """Return per-call GPU latencies in milliseconds (warmup calls excluded)."""
    latencies: List[float] = []
    for i in range(warmup + frames):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        if i >= warmup:
            latencies.append(start.elapsed_time(end))
    return latencies


def _report(label: str, latencies: List[float]) -> None:
    mean = statistics.mean(latencies)
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    fps = 1000.0 / mean
    status = "PASS" if mean <= TARGET_MS else "FAIL"
    bar = "✓" if status == "PASS" else "✗"
    print(f"  {bar} mean {mean:6.1f} ms | p95 {p95:6.1f} ms | {fps:5.1f} FPS | [{status}]  ← {label}")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _make_prompt(processor) -> str:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "You are a Pong agent. The ball is moving towards you. "
                    "Should you move the paddle up or down? Reply with one word."
                )},
                {"type": "image"},
            ],
        }
    ]
    return processor.apply_chat_template(conversation, add_generation_prompt=True)


def _make_inputs(processor, prompt: str, frame: np.ndarray, device: str):
    img = torch.tensor(frame, dtype=torch.float16, device=device) / 255.0
    return processor(images=img, text=prompt, return_tensors="pt").to(device, torch.float16)


def _attach_hook(model):
    """Attach a forward hook to capture the last hidden state before lm_head."""
    container: dict = {}

    def hook_fn(module, inp, out):
        container["payload"] = out

    handle = model.model.language_model.norm.register_forward_hook(hook_fn)
    return container, handle


# ---------------------------------------------------------------------------
# Benchmark tiers
# ---------------------------------------------------------------------------

def bench_baseline(model, processor, prompt: str, frame: np.ndarray, warmup: int, frames: int) -> List[float]:
    """model.generate — exactly what you'd write without Remora."""
    device = next(model.parameters()).device.type
    inputs = _make_inputs(processor, prompt, frame, device)

    def run():
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=1, do_sample=False)

    return _cuda_latencies(run, warmup, frames)


def bench_forward_hook(model, processor, prompt: str, frame: np.ndarray, warmup: int, frames: int) -> List[float]:
    """Direct forward pass + hook-captured hidden state + lm_head.

    Skips the generate() overhead; baseline for the optimized path.
    """
    device = next(model.parameters()).device.type
    inputs = _make_inputs(processor, prompt, frame, device)
    container, handle = _attach_hook(model)

    def run():
        with torch.no_grad():
            model(**inputs)
            model.lm_head(container["payload"])

    latencies = _cuda_latencies(run, warmup, frames)
    handle.remove()
    return latencies


def bench_remora(model, processor, prompt: str, frame: np.ndarray, warmup: int, frames: int) -> List[float]:
    """Remora-optimized: binary head + Triton RMSNorm + compiled vision tower."""
    device = next(model.parameters()).device.type
    inputs = _make_inputs(processor, prompt, frame, device)
    container, handle = _attach_hook(model)

    def run():
        with torch.no_grad():
            model(**inputs)
            model.lm_head(container["payload"])

    latencies = _cuda_latencies(run, warmup, frames)
    handle.remove()
    return latencies


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Remora VLM latency benchmark")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--warmup", type=int, default=8, help="warmup frames (excluded from stats)")
    parser.add_argument("--frames", type=int, default=50, help="benchmark frames")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  Remora VLM Benchmark")
    print(f"  Model  : {args.model_id}")
    print(f"  Device : {args.device}")
    print(f"  Target : {TARGET_MS:.1f} ms/frame  ({1000/TARGET_MS:.0f} FPS)")
    print(f"{'=' * 60}\n")

    synthetic_frame = np.zeros((210, 160, 3), dtype=np.uint8)

    # ------------------------------------------------------------------ load
    print("Loading model (baseline copy)...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    baseline_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.float16, device_map=args.device,
    ).eval()

    prompt = _make_prompt(processor)

    # --------------------------------------------------------- tier 0: generate
    print("[0] model.generate  (standard HuggingFace path)")
    t0 = bench_baseline(baseline_model, processor, prompt, synthetic_frame, args.warmup, args.frames)
    _report("baseline — model.generate(max_new_tokens=1)", t0)

    # ------------------------------------------- tier 1: forward + full lm_head
    print("\n[1] forward + full lm_head  (no generate overhead)")
    t1 = bench_forward_hook(baseline_model, processor, prompt, synthetic_frame, args.warmup, args.frames)
    _report("forward + full vocab lm_head", t1)

    del baseline_model
    torch.cuda.empty_cache()

    # ------------------------------------------------- load + optimize
    print("\nLoading model (Remora-optimized copy)...")
    from remora.optimizer import optimize_model

    opt_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.float16, device_map=args.device,
    ).eval()
    opt_model = optimize_model(opt_model, processor)

    # -------------------------------------------- tier 2: remora (first pass triggers compile)
    print("\n[2] Remora  (FusedSwiGLUMLP + FusedQKVAttention + TritonRMSNorm + TritonBinaryHead)")
    t2 = bench_remora(opt_model, processor, prompt, synthetic_frame, warmup=args.warmup, frames=args.frames)
    _report("Remora — fused QKV + fused gate/up + RMSNorm + BinaryHead", t2)

    del opt_model
    torch.cuda.empty_cache()

    # ------------------------------------------------------------ summary
    mean0 = statistics.mean(t0)
    mean2 = statistics.mean(t2)
    speedup = mean0 / mean2

    print(f"\n{'─' * 60}")
    print(f"  Speedup  : {speedup:.2f}×  (baseline → full Remora)")
    print(f"  Baseline : {mean0:.1f} ms  ({1000/mean0:.1f} FPS)")
    print(f"  Remora   : {mean2:.1f} ms  ({1000/mean2:.1f} FPS)")
    target_met = mean2 <= TARGET_MS
    print(f"  30 FPS   : {'✓ ACHIEVED' if target_met else '✗ NOT YET — try fp8 quant or flash-attn'}")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    main()
