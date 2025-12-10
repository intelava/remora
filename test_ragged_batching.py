import logging
import sys
from dataclasses import dataclass
from typing import List

import torch

from remora import GenerationRequest, RemoraEngine

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)


class ToyTokenizer:
    pad_token_id = 0

    def encode(self, text: str, **_: object) -> List[int]:
        return [len(word) for word in text.split()]

    def decode(self, tokens, skip_special_tokens: bool = True) -> str:
        ints = [int(t) for t in tokens]
        return " ".join(f"tok{val}" for val in ints)


class EchoModel(torch.nn.Module):

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        logger.debug("EchoModel.generate called with input_ids shape %s", getattr(input_ids, "shape", None))
        return input_ids


def main() -> bool:
    tokenizer = ToyTokenizer()
    model = EchoModel()
    engine = RemoraEngine(model=model, tokenizer=tokenizer)

    requests = [
        GenerationRequest(prompt="short example"),
        GenerationRequest(prompt="this request is noticeably longer than the first one"),
        GenerationRequest(prompt="mid size"),
    ]

    batch = engine.build_ragged_batch(requests)
    jagged = batch["input_ids_jagged"]
    print("\nRagged packing:")
    print(f"  total tokens: {jagged.total_tokens}")
    print(f"  batch size:   {jagged.batch_size}")
    print(f"  cu_seqlens:   {jagged.cu_seqlens.tolist()}")

    outputs = engine.generate_batch(requests)
    print("\nDecoded outputs (EchoModel returns the inputs):")
    for idx, result in outputs.items():
        print(f"  [{idx}] tokens: {result['tokens'].tolist()}")
        print(f"      text:   {result['text']}")

    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)








