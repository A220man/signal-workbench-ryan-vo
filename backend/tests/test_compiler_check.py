"""Tests for compiler and CPU feature diagnostics (Issues 180, 334, 185, 480)."""
from __future__ import annotations

import platform
from unittest.mock import patch

import pytest

from app.engine.compiler_check import (
    detect_cpu_features,
    detect_toolchain_versions,
    run_compiler_diagnostics,
)


class TestCPUFeatureDetection:
    """Tests for SIMD instruction detection."""

    def test_returns_dict_with_known_keys(self):
        features = detect_cpu_features()
        assert isinstance(features, dict)
        expected_keys = {"avx2", "avx512f", "avx512_vnni", "neon", "sse4_2"}
        assert set(features.keys()) == expected_keys
        assert all(isinstance(v, bool) for v in features.values())

    def test_arm_sets_neon(self):
        with patch("platform.machine", return_value="aarch64"):
            features = detect_cpu_features()
            assert features["neon"] is True

    def test_x86_does_not_set_neon_by_default(self):
        with patch("platform.machine", return_value="x86_64"):
            features = detect_cpu_features()
            # neon only set for arm/aarch64
            assert features["neon"] is False


class TestToolchainDetection:
    """Tests for clang and cmake version detection."""

    @patch("shutil.which")
    def test_detects_clang_when_available(self, mock_which):
        mock_which.side_effect = lambda x: "/usr/bin/clang" if x == "clang" else None
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = "clang version 18.1.8\n"
            result = detect_toolchain_versions()
            assert result["clang"] is not None
            assert "18" in result["clang"]

    @patch("shutil.which")
    def test_detects_clang_versioned_binary(self, mock_which):
        mock_which.side_effect = lambda x: "/usr/bin/clang-18" if x == "clang-18" else None
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = "clang version 18.1.8\n"
            result = detect_toolchain_versions()
            assert result["clang"] is not None

    @patch("shutil.which")
    def test_detects_cmake(self, mock_which):
        mock_which.side_effect = lambda x: "/usr/bin/cmake" if x == "cmake" else None
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = "cmake version 3.28.3\n"
            result = detect_toolchain_versions()
            assert result["cmake"] is not None
            assert "3.28" in result["cmake"]

    @patch("shutil.which", return_value=None)
    def test_returns_none_when_missing(self, mock_which):
        result = detect_toolchain_versions()
        assert result["clang"] is None
        assert result["cmake"] is None


class TestCompilerDiagnostics:
    """Tests for full diagnostic suite."""

    def test_returns_complete_response(self):
        result = run_compiler_diagnostics()
        assert hasattr(result, "cpu_vendor")
        assert hasattr(result, "cpu_features")
        assert hasattr(result, "clang_version")
        assert hasattr(result, "cmake_version")
        assert hasattr(result, "known_issues_addressed")
    def test_known_issues_include_references(self):
        result = run_compiler_diagnostics()
        issue_refs = [str(issue) for issue in result.known_issues_addressed]
        # Should reference the key upstream build issues
        assert any("180" in ref for ref in issue_refs)  # Build flags
        assert any("334" in ref for ref in issue_refs)  # Missing header
        assert any("480" in ref for ref in issue_refs)  # Clang 20 const-correctness