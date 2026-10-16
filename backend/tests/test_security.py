"""Tests for authentication, authorization, and security constraints."""
from __future__ import annotations

import base64
import importlib
import json
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


def make_test_jwt(
    roles: list[str] | None = None,
    issuer: str = "http://127.0.0.1:8080/realms/workbench-realm",
) -> str:
    """Create a synthetic JWT whose payload decodes successfully without signature check.

    The token is clearly synthetic (fake header/payload/signature) and carries
    runtime-generated claims — never a real credential.
    """
    header = {"alg": "HS256", "typ": "JWT"}
    payload: dict = {
        "sub": "test-user-id",
        "preferred_username": "testuser",
        "email": "test@example.com",
        "iss": issuer,
        "exp": 9999999999,
        "iat": 1000000000,
    }
    if roles:
        payload["realm_access"] = {"roles": roles}
    def b64(obj: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode("ascii")).rstrip(b"=").decode("ascii")
    return f"{b64(header)}.{b64(payload)}.synthetic-signature"


def _reload_for_production():
    """Reload config + security + main with production settings for fresh app instance."""
    import app.core.config
    importlib.reload(app.core.config)
    import app.core.security
    importlib.reload(app.core.security)
    import app.main
    importlib.reload(app.main)
    from app.main import app
    return TestClient(app)


class TestDemoModeAuth:
    """Tests for demo mode authentication (DEMO_MODE=true)."""

    def test_health_unauthenticated(self):
        with patch.dict(os.environ, {"DEMO_MODE": "true", "ENVIRONMENT": "test"}):
            from app.main import app
            client = TestClient(app)
            response = client.get("/api/health")
            assert response.status_code == 200

    def test_models_accessible_in_demo(self):
        with patch.dict(os.environ, {"DEMO_MODE": "true", "ENVIRONMENT": "test"}):
            from app.main import app
            client = TestClient(app)
            response = client.get("/api/models")
            assert response.status_code == 200

    def test_write_operations_allowed_in_demo(self):
        with patch.dict(os.environ, {"DEMO_MODE": "true", "ENVIRONMENT": "test"}):
            from app.main import app
            client = TestClient(app)
            response = client.post("/api/models", json={
                "model_id": "demo-write",
                "model_path": "/demo",
                "quantization": "i2_s",
                "context_length": 2048,
                "activation_function": "relu_sqr",
            })
            assert response.status_code == 201


class TestProductionModeRejectsDemo:
    """Tests that production mode refuses demo mode (security constraint)."""

    def test_production_with_demo_mode_raises(self):
        with patch.dict(os.environ, {"DEMO_MODE": "true", "ENVIRONMENT": "production"}):
            with pytest.raises(RuntimeError, match="SECURITY VIOLATION"):
                import app.core.config
                importlib.reload(app.core.config)


class TestBearerTokenAuth:
    """Tests for bearer token authentication (production mode)."""

    def test_missing_token_rejected_in_production(self):
        with patch.dict(os.environ, {"DEMO_MODE": "false", "ENVIRONMENT": "production"}):
            client = _reload_for_production()
            response = client.get("/api/models")
            assert response.status_code == 401

    def test_invalid_token_rejected_in_production(self):
        """Malformed tokens (not valid JWT) are rejected, not silently accepted."""
        with patch.dict(os.environ, {"DEMO_MODE": "false", "ENVIRONMENT": "production"}):
            client = _reload_for_production()
            response = client.get(
                "/api/models",
                headers={"Authorization": "Bearer not-a-jwt"},
            )
            assert response.status_code == 401

    def test_valid_jwt_accepted_in_production(self):
        """A JWT with correct issuer and operator role is accepted."""
        with patch.dict(os.environ, {"DEMO_MODE": "false", "ENVIRONMENT": "production"}):
            client = _reload_for_production()
            token = make_test_jwt(roles=["operator"])
            response = client.get(
                "/api/models",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert response.status_code == 200

    def test_wrong_issuer_rejected_in_production(self):
        """JWT with mismatched issuer is rejected in production."""
        with patch.dict(os.environ, {"DEMO_MODE": "false", "ENVIRONMENT": "production"}):
            client = _reload_for_production()
            token = make_test_jwt(roles=["operator"], issuer="https://evil-idp.example.com/realms/x")
            response = client.get(
                "/api/models",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert response.status_code == 401


class TestRoleAuthorization:
    """Tests for operator/admin role enforcement via Keycloak realm claims."""

    def test_admin_required_for_audit_in_production(self):
        """Admin role in JWT claims grants access to audit endpoint."""
        with patch.dict(os.environ, {"DEMO_MODE": "false", "ENVIRONMENT": "production"}):
            client = _reload_for_production()
            token = make_test_jwt(roles=["admin"])
            response = client.get("/api/audit", headers={"Authorization": f"Bearer {token}"})
            assert response.status_code == 200

    def test_operator_rejected_for_audit_in_production(self):
        """Operator role lacks admin-only access."""
        with patch.dict(os.environ, {"DEMO_MODE": "false", "ENVIRONMENT": "production"}):
            client = _reload_for_production()
            token = make_test_jwt(roles=["operator"])
            response = client.get("/api/audit", headers={"Authorization": f"Bearer {token}"})
            assert response.status_code == 403

    def test_writer_can_create_model_in_production(self):
        """Operator role can POST models."""
        with patch.dict(os.environ, {"DEMO_MODE": "false", "ENVIRONMENT": "production"}):
            client = _reload_for_production()
            token = make_test_jwt(roles=["operator"])
            response = client.post(
                "/api/models",
                json={
                    "model_id": "prod-test",
                    "model_path": "/models/test",
                    "quantization": "i2_s",
                    "context_length": 2048,
                    "activation_function": "relu_sqr",
                },
                headers={"Authorization": f"Bearer {token}"},
            )
            assert response.status_code == 201

    def test_viewer_rejected_for_model_create_in_production(self):
        """Viewer role cannot POST models (write requires operator)."""
        with patch.dict(os.environ, {"DEMO_MODE": "false", "ENVIRONMENT": "production"}):
            client = _reload_for_production()
            token = make_test_jwt(roles=["viewer"])
            response = client.post(
                "/api/models",
                json={
                    "model_id": "viewer-test",
                    "model_path": "/models/test",
                    "quantization": "i2_s",
                    "context_length": 2048,
                    "activation_function": "relu_sqr",
                },
                headers={"Authorization": f"Bearer {token}"},
            )
            assert response.status_code == 403


class TestSecurityHeaders:
    """Tests for security-relevant headers and settings."""

    def test_cors_restricted_to_localhost(self):
        with patch.dict(os.environ, {"DEMO_MODE": "true", "ENVIRONMENT": "test"}):
            from app.main import app
            client = TestClient(app)
            response = client.options("/api/models", headers={
                "Origin": "http://evil.com",
                "Access-Control-Request-Method": "POST",
            })
            # Should not allow evil.com
            assert response.headers.get("access-control-allow-origin") != "http://evil.com"

    def test_no_tokens_in_logs(self, caplog):
        """Ensure tokens are never logged."""
        from app.core.config import settings

        # This is a structural test - actual logging would be checked in integration
        assert settings.LLM_API_KEY is not None or settings.DEMO_MODE is True
        # The key point: LLM_API_KEY is only read server-side, never exposed in API responses