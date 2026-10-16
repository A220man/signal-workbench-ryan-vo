"""Tests for ternary i2_s packing and addition-only kernels."""
from __future__ import annotations

import numpy as np
import pytest

from app.engine.ternary_kernels import (
    lut_ternary_gemm_step,
    pack_ternary_i2_s,
    ternary_addition_dot_product,
    unpack_ternary_i2_s,
)


class TestTernaryPacking:
    """Tests for 2-bit i2_s ternary weight packing."""

    def test_pack_unpack_roundtrip(self):
        weights = np.array([[-1, 0, 1, 0], [1, -1, 0, 1]], dtype=np.int8)
        packed = pack_ternary_i2_s(weights)
        unpacked = unpack_ternary_i2_s(packed, weights.shape)
        np.testing.assert_array_equal(unpacked, weights)

    def test_pack_encoding_values(self):
        """Verify encoding: 0=00, +1=01, -1=10."""
        weights = np.array([0, 1, -1, 0], dtype=np.int8)  # one byte
        packed = pack_ternary_i2_s(weights)
        assert len(packed) == 1
        byte = packed[0]
        # bits: [w3 w2 w1 w0] = [0, -1, 1, 0] = [00, 10, 01, 00] = 0b00100100 = 0x24
        assert byte == 0x24

    def test_pack_pads_to_multiple_of_4(self):
        weights = np.array([1, -1, 0], dtype=np.int8)  # 3 elements
        packed = pack_ternary_i2_s(weights)
        unpacked = unpack_ternary_i2_s(packed, (3,))
        np.testing.assert_array_equal(unpacked, [1, -1, 0])

    def test_pack_large_array(self):
        weights = np.random.choice([-1, 0, 1], size=(256, 512), p=[0.25, 0.5, 0.25]).astype(np.int8)
        packed = pack_ternary_i2_s(weights)
        expected_bytes = (256 * 512 + 3) // 4
        assert len(packed) == expected_bytes
        unpacked = unpack_ternary_i2_s(packed, weights.shape)
        np.testing.assert_array_equal(unpacked, weights)

    def test_pack_empty_array(self):
        weights = np.array([], dtype=np.int8).reshape(0, 10)
        packed = pack_ternary_i2_s(weights)
        assert len(packed) == 0


class TestTernaryAdditionDotProduct:
    """Tests for pure integer ternary dot product (no multiplication)."""

    def test_basic_positive_weights(self):
        x = np.array([1, 2, 3, 4], dtype=np.int32)
        w = np.array([1, 1, 1, 1], dtype=np.int8)
        result = ternary_addition_dot_product(x, w)
        assert result == 10  # 1+2+3+4

    def test_negative_weights_subtract(self):
        x = np.array([5, 5, 5, 5], dtype=np.int32)
        w = np.array([1, -1, 1, -1], dtype=np.int8)
        result = ternary_addition_dot_product(x, w)
        assert result == 0  # 5 - 5 + 5 - 5

    def test_zero_weights_skipped(self):
        x = np.array([100, 200, 300, 400], dtype=np.int32)
        w = np.array([0, 0, 0, 0], dtype=np.int8)
        result = ternary_addition_dot_product(x, w)
        assert result == 0

    def test_mixed_sparse(self):
        x = np.array([10, 20, 30, 40, 50], dtype=np.int32)
        w = np.array([1, 0, -1, 0, 1], dtype=np.int8)
        result = ternary_addition_dot_product(x, w)
        assert result == 10 - 30 + 50  # 30


class TestLUTTernaryGEMM:
    """Tests for LUT-based ternary GEMM simulation."""

    def test_gemm_step_shapes(self):
        k = 64
        m = 32
        x_block = np.random.randn(k).astype(np.int32)
        ternary_block = np.random.choice([-1, 0, 1], size=(m, k), p=[0.25, 0.5, 0.25]).astype(np.int8)

        results = lut_ternary_gemm_step(x_block, ternary_block)
        assert results.shape == (m,)
        assert results.dtype == np.int32

    def test_gemm_matches_direct(self):
        """LUT GEMM should match direct addition dot products."""
        k = 32
        m = 16
        x_block = np.random.randint(-100, 100, size=k).astype(np.int32)
        ternary_block = np.random.choice([-1, 0, 1], size=(m, k), p=[0.25, 0.5, 0.25]).astype(np.int8)

        lut_results = lut_ternary_gemm_step(x_block, ternary_block)

        direct_results = np.zeros(m, dtype=np.int32)
        for i in range(m):
            direct_results[i] = ternary_addition_dot_product(x_block, ternary_block[i])

        np.testing.assert_array_equal(lut_results, direct_results)