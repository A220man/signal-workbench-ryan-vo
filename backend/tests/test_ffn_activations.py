"""Tests for FFN activation functions (ReLU² vs SiLU) - core Issue #588 validation."""
from __future__ import annotations

import numpy as np
import pytest

from app.engine.ffn_activations import (
    compare_activation_energy,
    ffn_sublayer,
    relu_squared,
    silu,
)


class TestReLUSquared:
    """Tests for squared ReLU activation: relu(x)^2."""

    def test_positive_input(self):
        x = np.array([1.0, 2.0, 3.0])
        out = relu_squared(x)
        expected = np.array([1.0, 4.0, 9.0])
        np.testing.assert_allclose(out, expected)

    def test_negative_input(self):
        x = np.array([-1.0, -2.0, -3.0])
        out = relu_squared(x)
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(out, expected)

    def test_mixed_input(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        out = relu_squared(x)
        expected = np.array([0.0, 0.0, 0.0, 1.0, 4.0])
        np.testing.assert_allclose(out, expected)

    def test_zero_gradient_at_origin(self):
        """ReLU² has zero gradient at x=0, unlike SiLU."""
        x = np.array([0.0])
        out = relu_squared(x)
        assert out[0] == 0.0

    def test_preserves_non_negativity(self):
        """ReLU² output is always >= 0."""
        x = np.random.randn(1000)
        out = relu_squared(x)
        assert np.all(out >= 0.0)


class TestSiLU:
    """Tests for SiLU (Swish) activation: x * sigmoid(x)."""

    def test_positive_input(self):
        x = np.array([1.0, 2.0])
        out = silu(x)
        # SiLU(1) = 1 * sigmoid(1) ≈ 0.731
        # SiLU(2) = 2 * sigmoid(2) ≈ 1.762
        assert out[0] > 0 and out[0] < 1.0
        assert out[1] > 1.0 and out[1] < 2.0

    def test_negative_input(self):
        x = np.array([-1.0, -2.0])
        out = silu(x)
        # SiLU allows negative outputs (unlike ReLU²)
        assert out[0] < 0.0
        assert out[1] < 0.0
    def test_large_negative_clipping(self):
        """SiLU should handle large negative inputs without underflow."""
        x = np.array([-100.0])
        out = silu(x)
        # sigmoid(-20) ≈ 2e-9, so -100 * 2e-9 ≈ -2e-7, very close to 0
        assert abs(out[0]) < 1e-6
    def test_large_positive(self):
        """SiLU(x) ≈ x for large positive x."""
        x = np.array([50.0])
        out = silu(x)
        assert np.isclose(out[0], 50.0, rtol=1e-3)


class TestFFNSublayer:
    """Tests for BitNet FFN sublayer with both activations."""

    def test_relu_sqr_shape(self):
        batch, dim, hidden = 4, 64, 256
        x = np.random.randn(batch, dim).astype(np.float32)
        w_gate = np.random.randn(hidden, dim).astype(np.float32)
        w_up = np.random.randn(hidden, dim).astype(np.float32)
        w_down = np.random.randn(dim, hidden).astype(np.float32)

        out = ffn_sublayer(x, w_gate, w_up, w_down, activation="relu_sqr")
        assert out.shape == (batch, dim)

    def test_silu_shape(self):
        batch, dim, hidden = 2, 32, 128
        x = np.random.randn(batch, dim).astype(np.float32)
        w_gate = np.random.randn(hidden, dim).astype(np.float32)
        w_up = np.random.randn(hidden, dim).astype(np.float32)
        w_down = np.random.randn(dim, hidden).astype(np.float32)

        out = ffn_sublayer(x, w_gate, w_up, w_down, activation="silu")
        assert out.shape == (batch, dim)

    def test_invalid_activation_raises(self):
        x = np.random.randn(2, 16)
        w_gate = w_up = w_down = np.random.randn(32, 16)
        with pytest.raises(ValueError, match="Unsupported activation"):
            ffn_sublayer(x, w_gate, w_up, w_down, activation="gelu")

    def test_deterministic_given_same_inputs(self):
        """Same inputs should produce identical outputs."""
        x = np.ones((1, 16))
        w_gate = w_up = np.ones((32, 16))
        w_down = np.ones((16, 32))

        out1 = ffn_sublayer(x, w_gate, w_up, w_down, activation="relu_sqr")
        out2 = ffn_sublayer(x, w_gate, w_up, w_down, activation="relu_sqr")
        np.testing.assert_allclose(out1, out2)


class TestActivationEnergyComparison:
    """Tests for diagnostic comparison (Issue #588 root cause analysis)."""

    def test_energy_ratio_computed(self):
        x = np.random.randn(1000).astype(np.float32)
        result = compare_activation_energy(x)
        assert "relu_sqr_l2_norm" in result
        assert "silu_l2_norm" in result
        assert "energy_ratio_silu_to_relu_sqr" in result
        assert result["energy_ratio_silu_to_relu_sqr"] >= 0.0

    def test_sparsity_difference(self):
        """ReLU² should have higher sparsity (more zeros) than SiLU."""
        x = np.random.randn(10000).astype(np.float32)
        result = compare_activation_energy(x)
        # ReLU² zeros out all negative inputs; SiLU only near-zero
        assert result["relu_sqr_sparsity"] >= result["silu_sparsity"]

    def test_relu_sqr_non_negative_mean(self):
        x = np.random.randn(1000)
        result = compare_activation_energy(x)
        assert result["relu_sqr_mean"] >= 0.0