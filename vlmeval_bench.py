"""
VLMEvalKit bridging script. Loads a preset model, wraps it with VibeCheckModel,
and calls VLMEvalKit on chosen datasets (e.g., MME, TextVQA).

Fill in the TODOs inside remora before running this so the ragged/W8A16 path is functional.

Example:
python vlmeval_bench.py --preset molmo-7b --datasets MME,TextVQA --batch-size 4
"""

import argparse
import os
import sys
import subprocess
from typing import List

import torch

from remora.integration import VibeCheckModel
from remora.models import MODEL_PRESETS, load_model_and_tokenizer


def _run_vlmeval(vibe_model: VibeCheckModel, datasets: List[str], output_dir: str, **kwargs):
    """
    Best-effort wrapper that calls VLMEvalKit's run_eval if available. This keeps
    all configuration in Python so users can tweak without shelling out.
    """
    run_eval = None
    for path in ("vlmeval.api", "vlmeval.evaluate", "vlmeval"):
        try:
            module = __import__(path, fromlist=["run_eval"])
            run_eval = getattr(module, "run_eval", None)
            if run_eval is not None:
                break
        except Exception as exc:  # pragma: no cover - defensive
            import_error = exc
            continue

    if run_eval is None:
        # CLI fallback: call `python -m vlmeval.evaluate ...`
        cmd = [
            sys.executable,
            "-m",
            "vlmeval.evaluate",
            "--model",
            "vibecheck",
            "--datasets",
            ",".join(datasets),
            "--save_dir",
            output_dir,
        ]
        if "num_workers" in kwargs:
            cmd.extend(["--num_workers", str(kwargs["num_workers"])])
        print(f"[remora] Falling back to VLMEvalKit CLI: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return None

    return run_eval(
        model=vibe_model,
        datasets=datasets,
        save_dir=output_dir,
        num_workers=kwargs.get("num_workers", 1),
    )


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
        "--list-presets",
        action="store_true",
        help="List available presets and exit.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="MME",
        help="Comma-separated dataset names (e.g., MME,TextVQA,GQA).",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for VibeCheck queue.")
    parser.add_argument("--flush-ms", type=float, default=10.0, help="Flush interval for the batching queue.")
    parser.add_argument("--max-queue", type=int, default=64, help="Maximum buffered requests before throttling.")
    parser.add_argument(
        "--device",
        type=str,
        help="Device override (e.g., cpu, cuda, cuda:0). Defaults to CUDA if available.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs", help="Result directory for VLMEvalKit.")
    parser.add_argument("--num-workers", type=int, default=1, help="VLMEvalKit worker threads.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_presets:
        print("Available presets:")
        for name, spec in MODEL_PRESETS.items():
            print(f"- {name}: {spec.description}")
        return

    if args.device:
        try:
            torch.device(args.device)
        except Exception as exc:
            raise SystemExit(f"Invalid --device '{args.device}': {exc}") from exc
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[remora] Loading preset '{args.preset}' on {device}")
    model, tokenizer = load_model_and_tokenizer(args.preset, device=device)

    vibe = VibeCheckModel(
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        flush_ms=args.flush_ms,
        max_queue=args.max_queue,
    )

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if not datasets:
        raise SystemExit("No datasets provided; supply at least one via --datasets.")
    print(
        f"[remora] Queue config: batch_size={args.batch_size}, flush_ms={args.flush_ms}, max_queue={args.max_queue}"
    )
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[remora] Launching VLMEvalKit for datasets: {datasets}")
    _run_vlmeval(vibe, datasets, args.output_dir, num_workers=args.num_workers)


if __name__ == "__main__":
    main()
