"""Tests for hardware memory planner and throughput estimation."""
from __future__ import annotations

import pytest

from app.engine.memory_planner import BITNET_SPECS, calculate_memory_and_throughput


class TestBitNetSpecs:
    """Tests for known model specifications."""

    def test_all_scales_defined(self):
        expected = {"0.6B", "2B", "7B", "70B", "405B"}
        assert set(BITNET_SPECS.keys()) == expected

    def test_specs_have_required_fields(self):
        for scale, spec in BITNET_SPECS.items():
            assert "params" in spec
            assert "layers" in spec
            assert "heads" in spec
            assert "kv_heads" in spec
            assert "hidden_dim" in spec
            assert spec["params"] > 0
            assert spec["layers"] > 0


class TestMemoryPlanning:
    """Tests for memory footprint calculation."""

    def test_2b_fits_in_16gb(self):
        """2B model should fit comfortably in 16GB with context 2048."""
        result = calculate_memory_and_throughput(
            model_scale="2B",
            context_length=2048,
            device_memory_gb=16.0,
            memory_bandwidth_gbps=200.0,
        )
        assert result.fits is True
        assert result.total_memory_required_gb < 16.0
        assert result.ternary_weight_gb > 0
        assert result.kv_cache_gb > 0

    def test_70b_exceeds_16gb(self):
        """70B model should exceed 16GB."""
        result = calculate_memory_and_throughput(
            model_scale="70B",
            context_length=2048,
            device_memory_gb=16.0,
            memory_bandwidth_gbps=200.0,
        )
        assert result.fits is False

    def test_405b_requires_high_memory(self):
        """405B model requires substantial memory."""
        result = calculate_memory_and_throughput(
            model_scale="405B",
            context_length=2048,
            device_memory_gb=128.0,
            memory_bandwidth_gbps=400.0,
        )
        # 405B ternary weights ~101 GB + KV cache + overhead > 128 GB
        assert result.ternary_weight_gb > 90.0

    def test_kv_cache_scales_with_context(self):
        """KV cache should grow linearly with context length."""
        r1 = calculate_memory_and_throughput("2B", 1024, 32.0, 200.0)
        r2 = calculate_memory_and_throughput("2B", 4096, 32.0, 200.0)
        r3 = calculate_memory_and_throughput("2B", 8192, 32.0, 200.0)

        # KV cache should roughly double each step
        assert r2.kv_cache_gb > r1.kv_cache_gb * 1.8
        assert r3.kv_cache_gb > r2.kv_cache_gb * 1.8

    def test_throughput_increases_with_bandwidth(self):
        """Estimated tokens/s should scale with memory bandwidth."""
        r1 = calculate_memory_and_throughput("2B", 2048, 32.0, 100.0)
        r2 = calculate_memory_and_throughput("2B", 2048, 32.0, 400.0)
        assert r2.estimated_tps > r1.estimated_tps * 3.5

    def test_unknown_scale_defaults_to_2b(self):
        result = calculate_memory_and_throughput("999B", 2048, 32.0, 200.0)
        assert result.model_scale == "2B"

    def test_reduction_percentage_computed(self):
        """FP16 to ternary reduction should be ~87.5% (2 bits vs 16 bits)."""
        result = calculate_memory_and_throughput("2B", 2048, 32.0, 200.0)
        # 2/16 = 0.125, so reduction = 87.5%
        assert 85.0 < result.reduction_pct < 90.0


class TestSizingResponse:
    """Tests for response structure."""

    def test_all_fields_present(self):
        result = calculate_memory_and_throughput("2B", 2048, 16.0, 200.0)
        assert hasattr(result, "model_scale")
        assert hasattr(result, "fp16_weight_gb")
        assert hasattr(result, "ternary_weight_gb")
        assert hasattr(result, "kv_cache_gb")
        assert hasattr(result, "total_memory_required_gb")
        assert hasattr(result, "reduction_pct")
        assert hasattr(result, "fits")
        assert hasattr(result, "estimated_tps")
        assert hasattr(result, "notes")
        assert isinstance(result.notes, list)
        assert len(result.notes) > 0