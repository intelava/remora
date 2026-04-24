"""Benchmark: padded batching vs. ragged (jagged) packing.

Usage:
    python benchmarks/bench_ragged.py
    python benchmarks/bench_ragged.py --batch-size 32 --max-len 512
    python benchmarks/bench_ragged.py --model-id HuggingFaceTB/SmolLM2-135M
"""

import argparse
import logging
import random
import time
from typing import List, Optional, Tuple

import torch

from remora import pack_token_ids, pad_jagged_token_ids

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _make_sequences(
    count: int,
    min_len: int,
    max_len: int,
    device: torch.device,
    tokenizer=None,
) -> List[torch.Tensor]:
    seqs: List[torch.Tensor] = []
    for _ in range(count):
        if tokenizer is None:
            length = random.randint(min_len, max_len)
            seqs.append(torch.randint(5, 250, (length,), device=device, dtype=torch.long))
        else:
            target = random.randint(min_len, max_len)
            words: List[str] = []
            tokens: List[int] = []
            while len(tokens) < target:
                words.append(f"word{random.randint(0, 999)}")
                tokens = tokenizer.encode(" ".join(words), add_special_tokens=False)
            seqs.append(torch.tensor(tokens[:target], device=device, dtype=torch.long))
    return seqs


def _time(fn, n: int) -> Tuple[float, float]:
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times), min(times)


def benchmark(
    count: int,
    min_len: int,
    max_len: int,
    iterations: int,
    device: torch.device,
    model=None,
    tokenizer=None,
    max_new_tokens: int = 8,
) -> None:
    pad_id = int(getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None) or 0) if tokenizer else 0
    seqs = _make_sequences(count, min_len, max_len, device, tokenizer=tokenizer)

    actual_tokens = sum(len(s) for s in seqs)
    padded_tokens = count * max_len
    savings = (1 - actual_tokens / padded_tokens) * 100
    log.info(
        "Batch %d seqs | max_len=%d | %d padded vs %d jagged (%.1f%% saved)",
        count, max_len, padded_tokens, actual_tokens, savings,
    )

    def run_padded():
        out = torch.full((count, max_len), 0, device=device, dtype=torch.long)
        attn = torch.zeros_like(out, dtype=torch.bool)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = s
            attn[i, : len(s)] = True
        return out, attn

    def run_ragged():
        return pack_token_ids(seqs)

    padded_avg, padded_best = _time(run_padded, iterations)
    ragged_avg, ragged_best = _time(run_ragged, iterations)

    log.info("  Padded  : avg %.3f ms  best %.3f ms", padded_avg * 1e3, padded_best * 1e3)
    log.info("  Ragged  : avg %.3f ms  best %.3f ms", ragged_avg * 1e3, ragged_best * 1e3)
    log.info("  Speedup : %.2fx  (pack-only vs padded)", padded_avg / ragged_avg if ragged_avg else 0)

    if model is None or tokenizer is None:
        return

    def gen_padded():
        with torch.inference_mode():
            ids, attn = run_padded()
            return model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=max_new_tokens, do_sample=False)

    def gen_ragged():
        with torch.inference_mode():
            jagged = pack_token_ids(seqs)
            ids, attn = pad_jagged_token_ids(jagged, pad_token_id=pad_id)
            return model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=max_new_tokens, do_sample=False)

    gp_avg, gp_best = _time(gen_padded, iterations)
    gr_avg, gr_best = _time(gen_ragged, iterations)
    log.info("  Model generate (padded) : avg %.3f ms  best %.3f ms", gp_avg * 1e3, gp_best * 1e3)
    log.info("  Model generate (ragged) : avg %.3f ms  best %.3f ms", gr_avg * 1e3, gr_best * 1e3)
    log.info("  Model speedup           : %.2fx", gp_avg / gr_avg if gr_avg else 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark padded vs ragged batching")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-len", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model-id", type=str, default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--no-model", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = tokenizer = None

    if not args.no_model:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            log.info("Loading %s ...", args.model_id)
            tokenizer = AutoTokenizer.from_pretrained(args.model_id)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            model = AutoModelForCausalLM.from_pretrained(args.model_id).to(device).eval()
        except ImportError:
            log.warning("transformers not installed — skipping model benchmark")

    benchmark(args.batch_size, args.min_len, args.max_len, args.iterations, device,
              model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
