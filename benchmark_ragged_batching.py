import argparse
import logging
import time
from typing import List, Tuple

import torch

from remora import pad_jagged_token_ids, pack_token_ids

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
logger = logging.getLogger(__name__)


def make_token_sequences(
    count: int,
    min_len: int,
    max_len: int,
    device: torch.device,
    tokenizer=None,
) -> List[torch.Tensor]:
    import random

    if tokenizer is None:
        seqs: List[torch.Tensor] = []
        for _ in range(count):
            length = random.randint(min_len, max_len)
            seq = torch.randint(low=5, high=250, size=(length,), device=device, dtype=torch.long)
            seqs.append(seq)
        return seqs

    seqs = []
    for _ in range(count):
        target_len = random.randint(min_len, max_len)
        words: List[str] = []
        tokens: List[int] = []
        while len(tokens) < target_len:
            words.append(f"word{random.randint(0, 999)}")
            tokens = tokenizer.encode(" ".join(words), add_special_tokens=False)
        tokens = tokens[:target_len]
        seqs.append(torch.tensor(tokens, device=device, dtype=torch.long))
    return seqs


def time_it(fn, iterations: int) -> Tuple[float, float]:
    times: List[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times.append(time.perf_counter() - start)
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
    pad_token_id = 0
    if tokenizer is not None:
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    seqs = make_token_sequences(count, min_len, max_len, device, tokenizer=tokenizer)
    padded_tokens = count * max_len
    actual_tokens = sum(len(s) for s in seqs)
    savings = (1 - actual_tokens / padded_tokens) * 100
    logger.info(
        "Batch of %d sequences (max_len=%d): %d tokens padded vs %d jagged (%.1f%% saved)",
        count,
        max_len,
        padded_tokens,
        actual_tokens,
        savings,
    )

    def padded():
        padded = torch.full((count, max_len), 0, device=device, dtype=torch.long)
        for i, seq in enumerate(seqs):
            padded[i, : len(seq)] = seq
        attn = torch.zeros_like(padded, dtype=torch.bool)
        for i, seq in enumerate(seqs):
            attn[i, : len(seq)] = True
        return padded, attn

    def ragged_pack_only():
        return pack_token_ids(seqs)

    padded_avg, padded_best = time_it(padded, iterations)
    pack_only_avg, pack_only_best = time_it(ragged_pack_only, iterations)

    logger.info("Padded: avg %.3f ms, best %.3f ms", padded_avg * 1000, padded_best * 1000)
    logger.info("Ragged pack-only: avg %.3f ms, best %.3f ms", pack_only_avg * 1000, pack_only_best * 1000)
    if pack_only_avg > 0:
        logger.info("Pack-only vs padded speedup: %.2fx", padded_avg / pack_only_avg)

    if model is None or tokenizer is None:
        return

    def gen_padded():
        with torch.inference_mode():
            input_ids, attn = padded()
            return model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

    def gen_ragged():
        with torch.inference_mode():
            jagged = pack_token_ids(seqs)
            input_ids, attn = pad_jagged_token_ids(jagged, pad_token_id=pad_token_id)
            return model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

    gen_padded_avg, gen_padded_best = time_it(gen_padded, iterations)
    gen_ragged_avg, gen_ragged_best = time_it(gen_ragged, iterations)
    logger.info("Model generate (padded): avg %.3f ms, best %.3f ms", gen_padded_avg * 1000, gen_padded_best * 1000)
    logger.info("Model generate (ragged pack->pad): avg %.3f ms, best %.3f ms", gen_ragged_avg * 1000, gen_ragged_best * 1000)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark padded vs ragged packing.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-len", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model-id", type=str, default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--no-model", action="store_true")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = None
    tokenizer = None
    if not args.no_model:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            logger.warning("transformers not installed; skipping model benchmark")
        else:
            logger.info("Loading model %s ...", args.model_id)
            tokenizer = AutoTokenizer.from_pretrained(args.model_id)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            model = AutoModelForCausalLM.from_pretrained(args.model_id)
            model.to(device)
            model.eval()

    benchmark(
        args.batch_size,
        args.min_len,
        args.max_len,
        args.iterations,
        device,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
