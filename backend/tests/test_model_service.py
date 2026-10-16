"""Tests for model registry and in-memory inference service (Issue #206)."""
from __future__ import annotations

import pytest

from app.models.schemas import InferenceRequest, ModelLoadRequest
from app.services.model_service import (
    InMemoryModelSession,
    list_models,
    register_and_load_model,
    run_inference,
)


class TestModelRegistration:
    """Tests for loading models into persistent storage and RAM."""

    def test_register_model_creates_session(self, test_user):
        req = ModelLoadRequest(
            model_id="bitnet-2b-test",
            model_path="/models/bitnet-2b",
            quantization="i2_s",
            context_length=2048,
            activation_function="relu_sqr",
        )
        model_info = register_and_load_model(req, test_user)

        assert model_info.id == "bitnet-2b-test"
        assert model_info.name == "bitnet-2b-test"
        assert model_info.architecture == "BitNet-b1.58-2B-4T"
        assert model_info.parameter_count == 2_000_000_000
        assert model_info.quantization_format == "i2_s"
        assert model_info.activation_function == "relu_sqr"
        assert model_info.status == "loaded_in_ram"
        assert model_info.memory_bytes > 0

    def test_duplicate_model_id_updates_session(self, test_user):
        req1 = ModelLoadRequest(model_id="dup-test", model_path="/v1", quantization="i2_s", context_length=2048, activation_function="relu_sqr")
        req2 = ModelLoadRequest(model_id="dup-test", model_path="/v2", quantization="tl1", context_length=2048, activation_function="silu")

        m1 = register_and_load_model(req1, test_user)
        m2 = register_and_load_model(req2, test_user)

        assert m2.quantization_format == "tl1"
        assert m2.activation_function == "silu"
        # Only one session should exist
        models = list_models()
        assert len([m for m in models if m.id == "dup-test"]) == 1

    def test_list_models_returns_registered(self, test_user):
        register_and_load_model(ModelLoadRequest(model_id="m1", model_path="/p1", quantization="i2_s", context_length=2048, activation_function="relu_sqr"), test_user)
        register_and_load_model(ModelLoadRequest(model_id="m2", model_path="/p2", quantization="tl1", context_length=2048, activation_function="silu"), test_user)

        models = list_models()
        assert len(models) == 2
        ids = {m.id for m in models}
        assert ids == {"m1", "m2"}

    def test_model_persists_after_registration(self, test_user):
        req = ModelLoadRequest(model_id="persist-test", model_path="/persist", quantization="i2_s", context_length=2048, activation_function="relu_sqr")
        register_and_load_model(req, test_user)

        models = list_models()
        found = next(m for m in models if m.id == "persist-test")
        assert found.name == "persist-test"
        assert found.updated_at is not None


class TestInMemorySessions:
    """Tests for warm in-memory model residency."""

    def test_session_tracks_memory(self):
        session = InMemoryModelSession(
            model_id="mem-test",
            model_path="/test",
            quantization="i2_s",
            context_length=2048,
            activation_function="relu_sqr",
        )
        assert session.resident_memory_bytes > 1000

    def test_session_activation_function_stored(self):
        session = InMemoryModelSession(
            model_id="act-test",
            model_path="/test",
            quantization="i2_s",
            context_length=2048,
            activation_function="silu",
        )
        assert session.activation_function == "silu"


class TestInference:
    """Tests for inference against warm sessions."""

    def test_inference_requires_resident_model(self, test_user):
        req = InferenceRequest(model_id="nonexistent", prompt="Hello", max_tokens=10)
        with pytest.raises(KeyError, match="not loaded in memory"):
            run_inference(req, test_user)

    def test_inference_returns_response(self, test_user):
        # Register and load a model first
        register_and_load_model(
            ModelLoadRequest(model_id="inf-test", model_path="/test", quantization="i2_s", context_length=2048, activation_function="relu_sqr"),
            test_user,
        )

        req = InferenceRequest(
            model_id="inf-test",
            prompt="Test prompt",
            max_tokens=50,
            temperature=0.7,
            activation_override=None,
        )
        response = run_inference(req, test_user)

        assert response.id.startswith("inf-")
        assert response.model_id == "inf-test"
        assert response.prompt == "Test prompt"
        assert len(response.completion) > 0
        assert response.latency_ms > 0
        assert response.completion_tokens > 0
        assert response.activation_used == "relu_sqr"

    def test_inference_activation_override(self, test_user):
        register_and_load_model(
            ModelLoadRequest(model_id="ovr-test", model_path="/test", quantization="i2_s", context_length=2048, activation_function="relu_sqr"),
            test_user,
        )

        req = InferenceRequest(
            model_id="ovr-test",
            prompt="Test",
            max_tokens=10,
            activation_override="silu",
        )
        response = run_inference(req, test_user)
    def test_inference_validates_max_tokens(self, test_user):
        register_and_load_model(ModelLoadRequest(model_id="val-test", model_path="/t", quantization="i2_s", context_length=2048, activation_function="relu_sqr"), test_user)

        # Pydantic validates max_tokens <= 2048 at schema level
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            InferenceRequest(model_id="val-test", prompt="x", max_tokens=5000)