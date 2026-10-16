"""Tests for perplexity calibration engine (Issue #588 validation)."""
from __future__ import annotations

import numpy as np
import pytest

from app.engine.perplexity_eval import (
    compute_cross_entropy_perplexity,
    evaluate_synthetic_perplexity,
)


class TestCrossEntropyPerplexity:
    """Tests for cross-entropy and perplexity computation."""

    def test_perfect_predictions(self):
        """Perfect predictions (prob=1) should give PPL=1."""
        logits = np.zeros((1, 3, 10))
        logits[0, :, 5] = 100.0  # High confidence on token 5
        targets = np.full((1, 3), 5, dtype=np.int32)

        loss, ppl = compute_cross_entropy_perplexity(logits, targets)
        assert np.isclose(loss, 0.0, atol=1e-4)
        assert np.isclose(ppl, 1.0, atol=1e-4)

    def test_uniform_predictions(self):
        """Uniform logits give PPL = vocab_size."""
        batch, seq, vocab = 2, 5, 100
        logits = np.zeros((batch, seq, vocab))
        targets = np.random.randint(0, vocab, size=(batch, seq))

        loss, ppl = compute_cross_entropy_perplexity(logits, targets)
        expected_ppl = float(vocab)
        assert np.isclose(ppl, expected_ppl, rtol=0.05)

    def test_invalid_targets_handled(self):
        """Out-of-vocab targets should be skipped, not crash."""
        logits = np.zeros((1, 2, 10))
        targets = np.array([[5, 15]])  # 15 is out of vocab

        loss, ppl = compute_cross_entropy_perplexity(logits, targets)
        # Only first token (valid) contributes
        assert loss > 0
        assert ppl > 1.0

    def test_empty_targets(self):
        """No valid targets should return baseline."""
        logits = np.zeros((1, 1, 10))
        targets = np.array([[15]])  # All invalid

        loss, ppl = compute_cross_entropy_perplexity(logits, targets)
        assert loss == 0.0
        assert ppl == 1.0


class TestSyntheticPerplexityEvaluation:
    """Tests for synthetic perplexity simulation grounded in Issue #588."""

    def test_relu_sqr_matches_reference(self):
        """ReLU² should produce PPL near official 17.1090."""
        result = evaluate_synthetic_perplexity(
            num_tokens=512, activation="relu_sqr", seed=42
        )
        assert "perplexity" in result
        assert "mean_loss" in result
        # Should be in the correct band (Issue #588: 17.1086 ± 0.129)
        assert 16.0 < result["perplexity"] < 18.5
        assert result["reference_baseline"] == 17.1090

    def test_silu_shows_regression(self):
        """SiLU should produce broken PPL ~99.8 (Issue #588 bug)."""
        result = evaluate_synthetic_perplexity(
            num_tokens=512, activation="silu", seed=42
        )
        # Should be in the broken band (Issue #588: 99.8178 ± 0.906)
        assert 90.0 < result["perplexity"] < 110.0
        assert result["reference_baseline"] == 99.8178

    def test_deterministic_with_seed(self):
        """Same seed should produce identical results."""
        r1 = evaluate_synthetic_perplexity(activation="relu_sqr", seed=123)
        r2 = evaluate_synthetic_perplexity(activation="relu_sqr", seed=123)
        assert r1["perplexity"] == r2["perplexity"]
        assert r1["mean_loss"] == r2["mean_loss"]

    def test_different_seeds_produce_variance(self):
        """Different seeds should produce variance within expected range."""
        ppls = [
            evaluate_synthetic_perplexity(activation="relu_sqr", seed=s)["perplexity"]
            for s in range(10)
        ]
        # All should be in correct band
        assert all(16.0 < p < 18.5 for p in ppls)
        # Some variance expected
        assert max(ppls) - min(ppls) > 0.01

    def test_unknown_activation_fallback(self):
        """Unknown activation should use fallback values."""
        result = evaluate_synthetic_perplexity(activation="gelu")
        assert result["perplexity"] == round(np.exp(3.5), 4)
        assert "tokens_evaluated" in result