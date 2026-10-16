"""Tests for BitNet ternary quantization engine."""
from __future__ import annotations

import numpy as np
import pytest

from app.engine.bitnet_quant import (
    compute_quantization_stats,
    quantize_activations_absmax,
    quantize_weights_absmean,
    ternary_linear_forward,
)


class TestWeightQuantization:
    """Tests for absmean ternary weight quantization."""

    def test_quantize_zero_weights(self):
        weights = np.zeros((64, 64), dtype=np.float32)
        ternary, gamma = quantize_weights_absmean(weights)
        assert ternary.shape == weights.shape
        assert np.all(ternary == 0)
        assert gamma == 1.0  # fallback for zero weights

    def test_quantize_all_positive(self):
        weights = np.ones((32, 32), dtype=np.float32) * 0.5
        ternary, gamma = quantize_weights_absmean(weights)
        assert np.all(ternary == 1)
        assert gamma == 0.5

    def test_quantize_mixed_signs(self):
        weights = np.array([[-2.0, -0.5, 0.0], [0.5, 1.5, 3.0]], dtype=np.float32)
        ternary, gamma = quantize_weights_absmean(weights)
        assert ternary.shape == weights.shape
        assert set(np.unique(ternary)).issubset({-1, 0, 1})
        assert gamma > 0

    def test_quantize_preserves_shape(self):
        for shape in [(10, 10), (128, 256), (1, 1000), (512, 1)]:
            weights = np.random.randn(*shape).astype(np.float32)
            ternary, _ = quantize_weights_absmean(weights)
            assert ternary.shape == shape


class TestActivationQuantization:
    """Tests for dynamic absmax activation quantization."""

    def test_quantize_zero_activations(self):
        acts = np.zeros((8, 256), dtype=np.float32)
        quantized, alpha = quantize_activations_absmax(acts)
        assert quantized.shape == acts.shape
        assert np.all(quantized == 0)
        assert alpha == 1.0

    def test_quantize_8bit_range(self):
        acts = np.array([[-127.0, 0.0, 127.0]], dtype=np.float32)
        quantized, alpha = quantize_activations_absmax(acts, num_bits=8)
        assert quantized.shape == acts.shape
        assert np.max(np.abs(quantized)) <= 127

    def test_quantize_clips_overflow(self):
        acts = np.array([[1000.0]], dtype=np.float32)
        quantized, _ = quantize_activations_absmax(acts, num_bits=8)
        assert np.max(np.abs(quantized)) <= 127


class TestTernaryLinearForward:
    """Tests for ternary linear layer forward pass."""

    def test_forward_shapes(self):
        x = np.random.randn(4, 64).astype(np.float32)
        w = np.random.choice([-1, 0, 1], size=(128, 64), p=[0.25, 0.5, 0.25]).astype(np.int8)
        gamma = 0.5
        out = ternary_linear_forward(x, w, gamma)
        assert out.shape == (4, 128)

    def test_forward_with_activation_quant(self):
        x = np.random.randn(2, 32).astype(np.float32)
        w = np.random.choice([-1, 0, 1], size=(16, 32), p=[0.25, 0.5, 0.25]).astype(np.int8)
        gamma = 0.3
        out = ternary_linear_forward(x, w, gamma, quantize_activations=True)
        assert out.shape == (2, 16)

    def test_forward_zero_input(self):
        x = np.zeros((1, 16), dtype=np.float32)
        w = np.ones((8, 16), dtype=np.int8)
        gamma = 1.0
        out = ternary_linear_forward(x, w, gamma)
        assert np.allclose(out, 0.0)


class TestQuantizationStats:
    """Tests for quantization statistics computation."""

    def test_stats_keys_present(self):
        orig = np.random.randn(64, 64).astype(np.float32)
        ternary, gamma = quantize_weights_absmean(orig)
        stats = compute_quantization_stats(orig, ternary, gamma)
        expected_keys = {
            "sparsity",
            "mse",
            "weight_distribution_neg",
            "weight_distribution_zero",
            "weight_distribution_pos",
            "gamma",
        }
        assert set(stats.keys()) == expected_keys

    def test_sparsity_calculation(self):
        orig = np.zeros((100, 100), dtype=np.float32)
        ternary, gamma = quantize_weights_absmean(orig)
        stats = compute_quantization_stats(orig, ternary, gamma)
        assert stats["sparsity"] == 1.0
        assert stats["weight_distribution_zero"] == 1.0

    def test_gamma_recorded(self):
        orig = np.ones((10, 10), dtype=np.float32) * 2.5
        ternary, gamma = quantize_weights_absmean(orig)
        stats = compute_quantization_stats(orig, ternary, gamma)
        assert stats["gamma"] == gamma