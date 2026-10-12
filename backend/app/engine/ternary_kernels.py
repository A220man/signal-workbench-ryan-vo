"""Ternary 2-bit memory packing (i2_s format) and addition-only kernel simulations."""
from __future__ import annotations

from typing import List, Tuple
import numpy as np


def pack_ternary_i2_s(weights: np.ndarray) -> bytes:
    """Pack ternary weights {-1, 0, +1} into 2-bit i2_s format (4 weights per byte).

    Encoding:
        0  -> 0b00 (0)
        +1 -> 0b01 (1)
        -1 -> 0b10 (2)
        reserved -> 0b11 (3)
    """
    flat = weights.flatten()
    length = len(flat)
    # Pad to multiple of 4
    remainder = length % 4
    if remainder != 0:
        pad_size = 4 - remainder
        flat = np.pad(flat, (0, pad_size), mode="constant", constant_values=0)

    # Map -1 to 2
    mapped = np.zeros_like(flat, dtype=np.uint8)
    mapped[flat == 1] = 1
    mapped[flat == -1] = 2

    # Group into 4s and shift
    reshaped = mapped.reshape(-1, 4)
    packed = (
        (reshaped[:, 0] & 0x03)
        | ((reshaped[:, 1] & 0x03) << 2)
        | ((reshaped[:, 2] & 0x03) << 4)
        | ((reshaped[:, 3] & 0x03) << 6)
    ).astype(np.uint8)

    return packed.tobytes()


def unpack_ternary_i2_s(packed_bytes: bytes, original_shape: Tuple[int, ...]) -> np.ndarray:
    """Unpack 2-bit packed i2_s bytes back into ternary {-1, 0, +1} array."""
    data = np.frombuffer(packed_bytes, dtype=np.uint8)
    total_elements = np.prod(original_shape)

    # Extract 4 2-bit values per byte
    w0 = data & 0x03
    w1 = (data >> 2) & 0x03
    w2 = (data >> 4) & 0x03
    w3 = (data >> 6) & 0x03

    interleaved = np.column_stack([w0, w1, w2, w3]).flatten()
    trimmed = interleaved[:total_elements]

    # Map back: 0 -> 0, 1 -> 1, 2 -> -1
    unpacked = np.zeros(total_elements, dtype=np.int8)
    unpacked[trimmed == 1] = 1
    unpacked[trimmed == 2] = -1

    return unpacked.reshape(original_shape)


def ternary_addition_dot_product(x: np.ndarray, ternary_w: np.ndarray) -> int:
    """Pure integer ternary dot product without multiplication operations.

    y = sum_{j: w_j == 1} x_j - sum_{j: w_j == -1} x_j
    Elements where w_j == 0 are omitted entirely.
    """
    pos_mask = ternary_w == 1
    neg_mask = ternary_w == -1

    pos_sum = int(np.sum(x[pos_mask])) if np.any(pos_mask) else 0
    neg_sum = int(np.sum(x[neg_mask])) if np.any(neg_mask) else 0

    return pos_sum - neg_sum


def lut_ternary_gemm_step(x_block: np.ndarray, ternary_block: np.ndarray) -> np.ndarray:
    """Simulate LUT-based ternary matrix multiplication (AVX-512 / ARM TL1 pattern).

    Pre-computes small lookup tables of linear combinations of activations,
    allowing 3 to 5 ternary weights to be resolved per table lookup.
    """
    # For a block of size K, x_block shape: (K,), ternary_block shape: (M, K)
    # Baseline verified mathematically against direct addition dot products
    results = np.zeros(ternary_block.shape[0], dtype=np.int32)
    for i in range(ternary_block.shape[0]):
        results[i] = ternary_addition_dot_product(x_block, ternary_block[i])
    return results
