from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class JaggedTensor:
    """Variable-length sequences packed contiguously with cumulative length offsets."""

    data: torch.Tensor
    cu_seqlens: torch.Tensor

    @property
    def batch_size(self) -> int:
        return len(self.cu_seqlens) - 1

    @property
    def total_tokens(self) -> int:
        return self.data.shape[0]

    @property
    def hidden_dim(self) -> int:
        return self.data.shape[-1]

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    def get_sequence(self, idx: int) -> torch.Tensor:
        start = int(self.cu_seqlens[idx].item())
        end = int(self.cu_seqlens[idx + 1].item())
        return self.data[start:end]

    def sequence_lengths(self) -> torch.Tensor:
        return self.cu_seqlens[1:] - self.cu_seqlens[:-1]


@dataclass
class JaggedTokenIds:
    """Variable-length token ID sequences packed contiguously."""

    data: torch.Tensor
    cu_seqlens: torch.Tensor

    @property
    def batch_size(self) -> int:
        return len(self.cu_seqlens) - 1

    @property
    def total_tokens(self) -> int:
        return self.data.shape[0]

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    def get_sequence(self, idx: int) -> torch.Tensor:
        start = int(self.cu_seqlens[idx].item())
        end = int(self.cu_seqlens[idx + 1].item())
        return self.data[start:end]

    def sequence_lengths(self) -> torch.Tensor:
        return self.cu_seqlens[1:] - self.cu_seqlens[:-1]


def pack_sequences(sequences: List[torch.Tensor]) -> JaggedTensor:
    if not sequences:
        raise ValueError("Cannot pack an empty sequence list")
    device = sequences[0].device
    hidden_dim = sequences[0].shape[-1]
    for i, seq in enumerate(sequences):
        if seq.shape[-1] != hidden_dim:
            raise ValueError(f"Sequence {i} has hidden_dim {seq.shape[-1]}, expected {hidden_dim}")
    lengths = [seq.shape[0] for seq in sequences]
    cu_seqlens = torch.zeros(len(sequences) + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(torch.tensor(lengths, dtype=torch.int32, device=device), dim=0)
    return JaggedTensor(data=torch.cat(sequences, dim=0), cu_seqlens=cu_seqlens)


def unpack_sequences(jagged: JaggedTensor) -> List[torch.Tensor]:
    return [jagged.get_sequence(i) for i in range(jagged.batch_size)]


def pad_jagged(
    jagged: JaggedTensor,
    max_len: Optional[int] = None,
    pad_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = jagged.sequence_lengths()
    if max_len is None:
        max_len = int(lengths.max().item())
    padded = torch.full(
        (jagged.batch_size, max_len, jagged.hidden_dim),
        pad_value,
        dtype=jagged.dtype,
        device=jagged.device,
    )
    mask = torch.zeros(jagged.batch_size, max_len, dtype=torch.bool, device=jagged.device)
    for i in range(jagged.batch_size):
        seq_len = int(lengths[i].item())
        padded[i, :seq_len] = jagged.get_sequence(i)
        mask[i, :seq_len] = True
    return padded, mask


def pack_token_ids(token_ids_list: List[torch.Tensor]) -> JaggedTokenIds:
    if not token_ids_list:
        raise ValueError("Cannot pack an empty token ID list")
    device = token_ids_list[0].device
    dtype = token_ids_list[0].dtype
    normalized: List[torch.Tensor] = []
    for i, tok_ids in enumerate(token_ids_list):
        if not isinstance(tok_ids, torch.Tensor):
            tok_ids = torch.tensor(tok_ids, dtype=dtype, device=device)
        if tok_ids.ndim == 0:
            tok_ids = tok_ids.unsqueeze(0)
        elif tok_ids.ndim > 1:
            raise ValueError(f"Token IDs {i} has shape {tok_ids.shape}, expected 1D")
        normalized.append(tok_ids)
    lengths = [s.shape[0] for s in normalized]
    cu_seqlens = torch.zeros(len(normalized) + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(torch.tensor(lengths, dtype=torch.int32, device=device), dim=0)
    return JaggedTokenIds(data=torch.cat(normalized, dim=0), cu_seqlens=cu_seqlens)


def unpack_token_ids(jagged: JaggedTokenIds) -> List[torch.Tensor]:
    return [jagged.get_sequence(i) for i in range(jagged.batch_size)]


def pad_jagged_token_ids(
    jagged: JaggedTokenIds,
    max_len: Optional[int] = None,
    pad_token_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = jagged.sequence_lengths()
    if max_len is None:
        max_len = int(lengths.max().item())
    padded = torch.full(
        (jagged.batch_size, max_len),
        pad_token_id,
        dtype=jagged.dtype,
        device=jagged.device,
    )
    mask = torch.zeros(jagged.batch_size, max_len, dtype=torch.bool, device=jagged.device)
    for i in range(jagged.batch_size):
        seq_len = int(lengths[i].item())
        padded[i, :seq_len] = jagged.get_sequence(i)
        mask[i, :seq_len] = True
    return padded, mask
