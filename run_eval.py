"""
Simple benchmark harness that contrasts stock PyTorch vs. your ragged/W8A16 path.
Fill in the TODOs inside remora before running this.

Usage:
python run_eval.py --model-path /path/to/model --prompt "Describe the image" --image /path/to/image.png
"""

import argparse
import time
from typing import Optional

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional
    Image = None

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForVision2Seq
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit("transformers required to run the benchmark") from exc

from remora.integration import VibeCheckModel
from remora.models import MODEL_PRESETS, load_model_and_tokenizer


def _token_length(tokenizer, text: str) -> int:
    if hasattr(tokenizer, "encode"):
        return len(tokenizer.encode(text))
    if callable(tokenizer):
        out = tokenizer(text, return_tensors="pt")
        if "input_ids" in out:
            return out["input_ids"].shape[-1]
    return len(text.split())


def measure_tps(tokenizer, generate_fn, prompt: str, image=None, steps: int = 2) -> float:
    total_tokens = 0
    prompt_tokens = _token_length(tokenizer, prompt)
    start = time.time()
    for _ in range(steps):
        text = generate_fn(prompt, image)
        total_tokens += max(_token_length(tokenizer, text) - prompt_tokens, 1)
    return total_tokens / max(time.time() - start, 1e-5)


def _load_image(path: Optional[str]):
    if not path:
        return None
    if Image is None:
        raise SystemExit("Pillow is required for --image-path usage. Install via `pip install pillow`.")
    return Image.open(path).convert("RGB")


def _build_inputs(tokenizer, prompt: str, image=None, device="cpu"):
    if image is not None:
        try:
            return tokenizer(prompt, images=image, return_tensors="pt").to(device)
        except TypeError:
            # Fallback if tokenizer doesn't accept images
            return tokenizer(prompt, return_tensors="pt").to(device)
    return tokenizer(prompt, return_tensors="pt").to(device)


def run(args: argparse.Namespace):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.preset:
        if args.preset not in MODEL_PRESETS:
            raise SystemExit(f"Unknown preset '{args.preset}'. Options: {', '.join(MODEL_PRESETS)}")
        print(f"Loading preset '{args.preset}' ({MODEL_PRESETS[args.preset].description}) on {device}")
        model, tokenizer = load_model_and_tokenizer(args.preset, device=device)
    else:
        print(f"Loading model from {args.model_path} on {device}")
        model = AutoModelForVision2Seq.from_pretrained(args.model_path, trust_remote_code=True).to(device)#change it to a modular Auto
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    image = _load_image(args.image_path)

    print("Measuring stock PyTorch generate()...")

    def stock_generate(prompt, image=None):
        inputs = _build_inputs(tokenizer, prompt, image=image, device=model.device)
        result = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        if hasattr(tokenizer, "decode"):
            return tokenizer.decode(result[0], skip_special_tokens=True)
        return str(result)

    stock_tps = measure_tps(tokenizer, stock_generate, args.prompt, image=image)
    print(f"Stock TPS: {stock_tps:.2f}")

    print("Evaluating the VibeCheck path (requires your ragged/W8A16 implementation)...")
    vibe = VibeCheckModel(model=model, tokenizer=tokenizer, batch_size=args.batch_size)

    def vibe_generate(prompt, image=None):
        return vibe.generate(prompt, image=image, max_new_tokens=args.max_new_tokens)

    tuned_tps = measure_tps(tokenizer, vibe_generate, args.prompt, image=image)
    print(f"Vibe TPS: {tuned_tps:.2f}")

    improvement = (tuned_tps - stock_tps) / max(stock_tps, 1e-9) * 100
    print(f"Speedup: {improvement:.1f}%")
    vibe.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick TPS comparison for remora.")
    parser.add_argument("--model-path", type=str, help="Explicit HF path; ignored if --preset is set.")
    parser.add_argument("--preset", type=str, choices=list(MODEL_PRESETS.keys()), help="Preset model to load.")
    parser.add_argument("--prompt", type=str, default="Describe the image.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for VibeCheck queue flushing.")
    parser.add_argument("--image-path", type=str, help="Optional image file to include in the prompt.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
