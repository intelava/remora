"""Tests for RemoraEngine (text generation) with a toy model."""

from typing import List

import torch

from remora import GenerationRequest, RemoraEngine


class ToyTokenizer:
    pad_token_id = 0

    def encode(self, text: str, **_) -> List[int]:
        return [len(w) for w in text.split()]

    def decode(self, tokens, skip_special_tokens: bool = True) -> str:
        return " ".join(f"tok{int(t)}" for t in tokens)


class EchoModel(torch.nn.Module):
    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        return input_ids


def _make_engine() -> RemoraEngine:
    return RemoraEngine(model=EchoModel(), tokenizer=ToyTokenizer())


def test_build_ragged_batch_sizes():
    engine = _make_engine()
    requests = [
        GenerationRequest(prompt="short example"),
        GenerationRequest(prompt="this request is noticeably longer than the first one"),
        GenerationRequest(prompt="mid size"),
    ]
    batch = engine.build_ragged_batch(requests)
    jagged = batch["input_ids_jagged"]
    assert jagged.batch_size == 3
    assert jagged.total_tokens == sum(len(r.prompt.split()) for r in requests)


def test_build_ragged_batch_cu_seqlens():
    engine = _make_engine()
    requests = [GenerationRequest(prompt="a b"), GenerationRequest(prompt="c d e")]
    batch = engine.build_ragged_batch(requests)
    assert batch["input_ids_jagged"].cu_seqlens.tolist() == [0, 2, 5]


def test_generate_batch_returns_all_indices():
    engine = _make_engine()
    requests = [GenerationRequest(prompt="hello world"), GenerationRequest(prompt="foo")]
    outputs = engine.generate_batch(requests)
    assert set(outputs.keys()) == {0, 1}
    for v in outputs.values():
        assert "tokens" in v
        assert "text" in v


def test_generate_batch_echo_model():
    engine = _make_engine()
    requests = [GenerationRequest(prompt="one two three")]
    outputs = engine.generate_batch(requests)
    tokens = outputs[0]["tokens"].tolist()
    assert tokens == [3, 3, 5]


def test_generation_request_defaults():
    req = GenerationRequest(prompt="test")
    assert req.max_new_tokens == 32
