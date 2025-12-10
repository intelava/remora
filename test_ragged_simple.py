import sys

import torch

from remora import pack_token_ids, pad_jagged_token_ids, unpack_token_ids


def main() -> bool:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    seq1 = torch.tensor([101, 2023, 2003, 1037, 3231, 1012], device=device)
    seq2 = torch.tensor([101, 2023, 2003, 1037, 3231, 1012, 2003, 1037, 3231, 1012], device=device)
    seq3 = torch.tensor([101, 2023], device=device)

    print("\nPacking jagged batch...")
    jagged = pack_token_ids([seq1, seq2, seq3])
    print(f"  total tokens: {jagged.total_tokens}")
    print(f"  cumulative lengths: {jagged.cu_seqlens.tolist()}")

    max_len = max(len(seq1), len(seq2), len(seq3))
    padded_size = jagged.batch_size * max_len
    savings = (1 - jagged.total_tokens / padded_size) * 100
    print(f"  would have padded to {padded_size} tokens ({savings:.1f}% saved)")

    unpacked = unpack_token_ids(jagged)
    assert all(torch.equal(orig, up) for orig, up in zip([seq1, seq2, seq3], unpacked))
    print("  unpacked sequences match")

    padded, mask = pad_jagged_token_ids(jagged, pad_token_id=0)
    print(f"\nConverted to padded form: shape={padded.shape}, mask shape={mask.shape}")
    print(f"  first row: {padded[0].tolist()}")
    print(f"  mask: {mask[0].int().tolist()}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)








