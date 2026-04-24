"""Tests for ragged/jagged sequence packing utilities."""

import pytest
import torch

from remora import (
    JaggedTensor,
    JaggedTokenIds,
    pack_sequences,
    pack_token_ids,
    pad_jagged,
    pad_jagged_token_ids,
    unpack_sequences,
    unpack_token_ids,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# pack / unpack round-trips
# ---------------------------------------------------------------------------

def test_pack_unpack_token_ids():
    seq1 = torch.tensor([101, 2023, 2003, 1037, 3231, 1012], device=DEVICE)
    seq2 = torch.tensor([101, 2023, 2003, 1037, 3231, 1012, 2003, 1037, 3231, 1012], device=DEVICE)
    seq3 = torch.tensor([101, 2023], device=DEVICE)

    jagged = pack_token_ids([seq1, seq2, seq3])

    assert jagged.batch_size == 3
    assert jagged.total_tokens == len(seq1) + len(seq2) + len(seq3)
    assert jagged.cu_seqlens.tolist() == [0, 6, 16, 18]

    unpacked = unpack_token_ids(jagged)
    for orig, restored in zip([seq1, seq2, seq3], unpacked):
        assert torch.equal(orig, restored)


def test_pack_token_ids_efficiency():
    seq1 = torch.tensor([1, 2, 3], device=DEVICE)
    seq2 = torch.tensor([4, 5, 6, 7, 8, 9, 10], device=DEVICE)
    seq3 = torch.tensor([11], device=DEVICE)

    jagged = pack_token_ids([seq1, seq2, seq3])
    max_len = max(len(seq1), len(seq2), len(seq3))
    padded_size = jagged.batch_size * max_len

    assert jagged.total_tokens < padded_size


def test_pad_jagged_token_ids_shape():
    seq1 = torch.tensor([1, 2, 3], device=DEVICE)
    seq2 = torch.tensor([4, 5, 6, 7, 8], device=DEVICE)
    jagged = pack_token_ids([seq1, seq2])

    padded, mask = pad_jagged_token_ids(jagged, pad_token_id=0)

    assert padded.shape == (2, 5)
    assert mask.shape == (2, 5)
    assert mask[0, :3].all()
    assert not mask[0, 3:].any()
    assert mask[1, :5].all()


def test_pad_jagged_token_ids_values():
    seq = torch.tensor([10, 20, 30], device=DEVICE)
    jagged = pack_token_ids([seq])
    padded, _ = pad_jagged_token_ids(jagged, max_len=5, pad_token_id=99)
    assert padded[0, :3].tolist() == [10, 20, 30]
    assert padded[0, 3:].tolist() == [99, 99]


# ---------------------------------------------------------------------------
# pack_sequences (hidden-state tensors)
# ---------------------------------------------------------------------------

def test_pack_unpack_sequences():
    h = 16
    s1 = torch.randn(3, h, device=DEVICE)
    s2 = torch.randn(7, h, device=DEVICE)
    s3 = torch.randn(2, h, device=DEVICE)

    jagged = pack_sequences([s1, s2, s3])
    assert jagged.batch_size == 3
    assert jagged.total_tokens == 12
    assert jagged.hidden_dim == h

    restored = unpack_sequences(jagged)
    for orig, r in zip([s1, s2, s3], restored):
        assert torch.allclose(orig, r)


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------

def test_pack_empty_raises():
    with pytest.raises(ValueError):
        pack_token_ids([])


def test_pack_scalar_token_id():
    scalar = torch.tensor(42, device=DEVICE)
    jagged = pack_token_ids([scalar])
    assert jagged.total_tokens == 1


def test_pack_2d_token_ids_raises():
    bad = torch.tensor([[1, 2], [3, 4]], device=DEVICE)
    with pytest.raises(ValueError):
        pack_token_ids([bad])
