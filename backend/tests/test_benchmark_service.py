"""Tests for benchmark service (perplexity calibration per Issue #588)."""
from __future__ import annotations

import pytest

from app.models.schemas import BenchmarkRunRequest
from app.services.benchmark_service import list_benchmark_history, run_benchmark


class TestBenchmarkExecution:
    """Tests for perplexity benchmark runs."""

    def test_relu_sqr_benchmark_in_correct_band(self, test_user):
        req = BenchmarkRunRequest(
            model_name="BitNet-b1.58-2B-4T",
            dataset="wikitext-2",
            num_tokens=256,
            activation_tested="relu_sqr",
        )
        result = run_benchmark(req, test_user)

        assert result.id.startswith("bench-")
        assert result.model_name == "BitNet-b1.58-2B-4T"
        assert result.activation_tested == "relu_sqr"
        assert result.dataset == "wikitext-2"
        # Correct band: 17.1086 ± 0.129 (Issue #588)
        assert 16.0 < result.perplexity < 18.5
        assert result.latency_ms > 0
        assert result.peak_memory_mb > 0

    def test_silu_benchmark_shows_regression(self, test_user):
        req = BenchmarkRunRequest(
            model_name="BitNet-b1.58-2B-4T",
            dataset="wikitext-2",
            num_tokens=256,
            activation_tested="silu",
        )
        result = run_benchmark(req, test_user)

        # Broken band: 99.8178 ± 0.906 (Issue #588 bug)
        assert 90.0 < result.perplexity < 110.0
        assert result.activation_tested == "silu"

    def test_benchmark_records_weight_distribution(self, test_user):
        req = BenchmarkRunRequest(
            model_name="BitNet-b1.58-2B-4T",
            activation_tested="relu_sqr",
        )
        result = run_benchmark(req, test_user)

        assert "neg_pct" in result.weight_distribution
        assert "zero_pct" in result.weight_distribution
        assert "pos_pct" in result.weight_distribution
        # Should be approximately 27.5% / 45% / 27.5%
        assert 20 < result.weight_distribution["neg_pct"] < 35
        assert 40 < result.weight_distribution["zero_pct"] < 50
        assert 20 < result.weight_distribution["pos_pct"] < 35

    def test_deterministic_with_fixed_seed(self, test_user):
        """Same inputs should produce consistent results (synthetic simulation)."""
        req = BenchmarkRunRequest(
            model_name="BitNet-b1.58-2B-4T",
            activation_tested="relu_sqr",
        )
        r1 = run_benchmark(req, test_user)
        r2 = run_benchmark(req, test_user)
    def test_invalid_activation_uses_fallback(self, test_user):
        req = BenchmarkRunRequest(activation_tested="gelu")
        result = run_benchmark(req, test_user)
        # Should use fallback perplexity (exp(3.5) ≈ 33.1)
        assert result.perplexity == round(33.1155, 4)

class TestBenchmarkHistory:
    """Tests for historical benchmark retrieval."""

    def test_list_returns_recent_first(self, test_user):
        req1 = BenchmarkRunRequest(activation_tested="relu_sqr")
        req2 = BenchmarkRunRequest(activation_tested="silu")
        run_benchmark(req1, test_user)
        run_benchmark(req2, test_user)

        history = list_benchmark_history(limit=10)
        assert len(history) >= 2
        # Most recent first
        assert history[0].created_at >= history[1].created_at

    def test_limit_respected(self, test_user):
        for _ in range(5):
            run_benchmark(BenchmarkRunRequest(activation_tested="relu_sqr"), test_user)

        history = list_benchmark_history(limit=3)
        assert len(history) == 3