"""
VLMEvalKit bridging script. Loads a preset model, applies Triton surgery, wraps it
with VibeCheckModel, and calls VLMEvalKit on chosen datasets (e.g., MME, TextVQA).

Example:
python vlmeval_bench.py --preset molmo-7b --datasets MME,TextVQA --batch-size 4
"""

import argparse
import inspect
import os
import sys
from typing import List

import torch

from remora.integration import VibeCheckModel
from remora.models import MODEL_PRESETS, load_model_and_tokenizer
from remora.surgery import hijack_model


def _run_vlmeval(vibe_model: VibeCheckModel, datasets: List[str], output_dir: str, **kwargs):
    """
    Best-effort wrapper that calls VLMEvalKit's run_eval if available. This keeps
    all configuration in Python so users can tweak without shelling out.
    """
    try:
        from vlmeval.api import run_eval  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "vlmeval is required for this script. Install via `pip install vlmeval`."
        ) from exc

    sig = inspect.signature(run_eval)
    params = sig.parameters
    call_kwargs = {}
    if "model" in params:
        call_kwargs["model"] = vibe_model
    elif "llm" in params:
        call_kwargs["llm"] = vibe_model
    if "datasets" in params:
        call_kwargs["datasets"] = datasets
    if "save_dir" in params:
        call_kwargs["save_dir"] = output_dir
    if "out_dir" in params:
        call_kwargs["out_dir"] = output_dir
    if "num_workers" in params and "num_workers" in kwargs:
        call_kwargs["num_workers"] = kwargs["num_workers"]
    if "root" in params and "root" in kwargs:
        call_kwargs["root"] = kwargs["root"]

    return run_eval(**call_kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VLMEvalKit benchmarks with remora.")
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(MODEL_PRESETS.keys()),
        default="molmo-7b",
        help="Preset model alias to load.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="MME",
        help="Comma-separated dataset names (e.g., MME,TextVQA,GQA).",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for VibeCheck queue.")
    parser.add_argument("--flush-ms", type=float, default=10.0, help="Flush interval for the batching queue.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Result directory for VLMEvalKit.")
    parser.add_argument("--num-workers", type=int, default=1, help="VLMEvalKit worker threads.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[remora] Loading preset '{args.preset}' on {device}")
    model, tokenizer = load_model_and_tokenizer(args.preset, device=device)

    print("[remora] Applying Triton surgery...")
    hijack_model(model)

    vibe = VibeCheckModel(
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        flush_ms=args.flush_ms,
    )

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[remora] Launching VLMEvalKit for datasets: {datasets}")
    _run_vlmeval(vibe, datasets, args.output_dir, num_workers=args.num_workers)


if __name__ == "__main__":
    main()
