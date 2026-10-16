"""Integration tests for FastAPI endpoints (full HTTP boundary)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    """Tests for /api/health."""

    def test_health_returns_ok(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "1.1.0"
        assert "environment" in data
        assert "demo_mode" in data


class TestModelEndpoints:
    """Tests for /api/models CRUD."""

    def test_load_model_creates_session(self, client):
        payload = {
            "model_id": "api-test-1",
            "model_path": "/models/test",
            "quantization": "i2_s",
            "context_length": 2048,
            "activation_function": "relu_sqr",
        }
        response = client.post("/api/models", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == "api-test-1"
        assert data["status"] == "loaded_in_ram"

    def test_load_model_with_silu_activation(self, client):
        payload = {
            "model_id": "api-silu",
            "model_path": "/models/silu",
            "quantization": "tl1",
            "context_length": 2048,
            "activation_function": "silu",
        }
        response = client.post("/api/models", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["activation_function"] == "silu"

    def test_list_models_returns_registered(self, client):
        client.post("/api/models", json={"model_id": "list-1", "model_path": "/p1", "quantization": "i2_s", "context_length": 2048, "activation_function": "relu_sqr"})
        client.post("/api/models", json={"model_id": "list-2", "model_path": "/p2", "quantization": "i2_s", "context_length": 2048, "activation_function": "relu_sqr"})

        response = client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 2
        ids = {m["id"] for m in data}
        assert "list-1" in ids
        assert "list-2" in ids

    def test_get_model_by_id(self, client):
        client.post("/api/models", json={"model_id": "get-1", "model_path": "/p", "quantization": "i2_s", "context_length": 2048, "activation_function": "relu_sqr"})
        response = client.get("/api/models/get-1")
        assert response.status_code == 200
        assert response.json()["id"] == "get-1"

    def test_get_nonexistent_model_404(self, client):
        response = client.get("/api/models/does-not-exist")
        assert response.status_code == 404

    def test_unload_model_evicts_session(self, client):
        client.post("/api/models", json={"model_id": "unload-1", "model_path": "/p", "quantization": "i2_s", "context_length": 2048, "activation_function": "relu_sqr"})
        response = client.delete("/api/models/unload-1")
        assert response.status_code == 204

        # Model record persists but session is gone
        response = client.get("/api/models/unload-1")
        assert response.status_code == 200
        assert response.json()["status"] != "loaded_in_ram"


class TestInferenceEndpoint:
    """Tests for /api/inference against warm sessions."""

    def test_inference_requires_loaded_model(self, client):
        payload = {"model_id": "not-loaded", "prompt": "Hello", "max_tokens": 10}
        response = client.post("/api/inference", json=payload)
        assert response.status_code == 404

    def test_inference_works_with_resident_model(self, client):
        client.post("/api/models", json={"model_id": "inf-api", "model_path": "/p", "quantization": "i2_s", "context_length": 2048, "activation_function": "relu_sqr"})

        payload = {"model_id": "inf-api", "prompt": "Test prompt", "max_tokens": 20}
        response = client.post("/api/inference", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "inf-api"
        assert data["prompt"] == "Test prompt"
        assert len(data["completion"]) > 0
        assert data["completion_tokens"] > 0
        assert data["latency_ms"] > 0

    def test_inference_activation_override(self, client):
        client.post("/api/models", json={"model_id": "ovr-api", "model_path": "/p", "quantization": "i2_s", "context_length": 2048, "activation_function": "relu_sqr"})

        payload = {"model_id": "ovr-api", "prompt": "Test", "max_tokens": 10, "activation_override": "silu"}
        response = client.post("/api/inference", json=payload)
    def test_inference_rejects_invalid_max_tokens(self, client):
        client.post("/api/models", json={"model_id": "val-api", "model_path": "/p", "quantization": "i2_s", "context_length": 2048, "activation_function": "relu_sqr"})
        payload = {"model_id": "val-api", "prompt": "x", "max_tokens": 5000}
        response = client.post("/api/inference", json=payload)
        # Pydantic validation returns 422 for invalid input
        assert response.status_code == 422
    """Tests for /api/benchmarks."""

    def test_create_benchmark_relu_sqr(self, client):
        payload = {
            "model_name": "BitNet-b1.58-2B-4T",
            "dataset": "wikitext-2",
            "num_tokens": 128,
            "activation_tested": "relu_sqr",
        }
        response = client.post("/api/benchmarks", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["activation_tested"] == "relu_sqr"
        assert 16.0 < data["perplexity"] < 18.5

    def test_create_benchmark_silu(self, client):
        payload = {"activation_tested": "silu"}
        response = client.post("/api/benchmarks", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["activation_tested"] == "silu"
        assert 90.0 < data["perplexity"] < 110.0

    def test_list_benchmarks(self, client):
        client.post("/api/benchmarks", json={"activation_tested": "relu_sqr"})
        client.post("/api/benchmarks", json={"activation_tested": "silu"})

        response = client.get("/api/benchmarks")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 2


class TestSizingEndpoint:
    """Tests for /api/sizing."""

    def test_sizing_2b_fits_16gb(self, client):
        payload = {
            "model_scale": "2B",
            "context_length": 2048,
            "device_memory_gb": 16.0,
            "memory_bandwidth_gbps": 200.0,
        }
        response = client.post("/api/sizing", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["fits"] is True
        assert data["model_scale"] == "2B"

    def test_sizing_70b_exceeds_16gb(self, client):
        payload = {
            "model_scale": "70B",
            "context_length": 2048,
            "device_memory_gb": 16.0,
            "memory_bandwidth_gbps": 200.0,
        }
        response = client.post("/api/sizing", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["fits"] is False


class TestCompilerDiagnosticsEndpoint:
    """Tests for /api/compiler."""

    def test_compiler_diagnostics_returns_structure(self, client):
        response = client.get("/api/compiler")
        assert response.status_code == 200
        data = response.json()
        assert "cpu_vendor" in data
        assert "cpu_features" in data
        assert "clang_version" in data
        assert "cmake_version" in data
        assert "known_issues_addressed" in data
        assert isinstance(data["known_issues_addressed"], list)


class TestAdvisoryEndpoint:
    """Tests for /api/advisory (opt-in, server-side LLM)."""

    def test_advisory_requires_llm_key(self, client):
        # LLM_API_KEY is set in conftest, so this should work
        payload = {"topic": "Why does ReLU² beat SiLU for BitNet?", "context": {}}
        response = client.post("/api/advisory", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "disclaimer" in data
        assert "ADVISORY ONLY" in data["disclaimer"]
        assert data["topic"] == payload["topic"]


class TestAuditEndpoint:
    """Tests for /api/audit (admin only)."""

    def test_audit_endpoint_exists(self, client):
        response = client.get("/api/audit")
        # In demo mode, admin check passes
        assert response.status_code == 200
        assert isinstance(response.json(), list)