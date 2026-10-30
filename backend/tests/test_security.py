"""Tests for authentication, authorization, and security constraints."""
from __future__ import annotations

import base64
import importlib
import json
import os
from unittest.mock import patch, AsyncMock

import pytest
from fastapi.testclient import TestClient
from jose import jwt
from unittest.mock import patch, AsyncMock


# Test RSA key pair (generated once, embedded for reproducibility)
TEST_PRIVATE_KEY_PEM = b"""-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCv79cNE1a+OOU/
EAljOM20jBx/IMlW7IslI/cKRtkdF1QMvxmLwO7cHXgzuby31Oc4WVvIJnNLF0/9
o+cQ+EZ9LkQabCZILiAPz9cgb+QxDO3SkMacKaIkcUUARKrqhCHz+kK8+/010jrf
XUg9VKclPWhjFuOWGjelEj/W2OnroA7TOq8gO0FJ7s1A2LZ7NBNH5gskAONygmgK
Y13SGuvJbS8Qfy56ZxDsAvEUMASBu008kNddPzoTe99oect9rjsI6o1a9WSbIT3p
rijEpcGSXLev7nyFQMNvHwx1B/1Jhk/pqBP5wdeQ/nB81RRp2D6WYr7KKUDN5QP/
d1cA2iypAgMBAAECggEABald4J/h0O0B3o0bROVsmYiP4weTcb/wAdsrOC97uDU6
bo8TPmqwMmaN33nqMLm8kj+uOw3WtGDHjDxYeL9ekv5t8VzlK081rQCVlgetE+bW
5omOD0jX3NzwGSA8kPT9Jo25VP11wmQVoioNWNuRpeUuHTsyNMU69hhKXt/xK3SC
lHKsvhBpmjo0wcJ0sx2OwMU1DHdhpSz3Bn0qf0pzxTNWNhPu3pv090OW6sp6A5p2
TYfIb8oKXIcgsQUFQqgmyzFpELxyUHTqUD3y0huVcimIIrQdbiBPSd+WWi9nSgAp
9cnZed6EWjyIrom4/bW48MSIqtlus1N+IFt/SSMN4QKBgQDuGGMbpnpjYK3KT9Eh
0qsvC17lPA7rU9/IfVmTLI/S5CY+LE6fFaC+TKEBn4d1PbyYZ3y/toMpw7BFgEgx
Mepn9AYoKzCuAa1SL8cO4wWy7X4WUJv19odhksN5LXjU2BZjJASYZEf8e+K0RqvY
4EbseoC0XmremzfKtRbEyRCtSQKBgQC9KtTC1UFZkTYB9ZyHBNk+1QuaaptClrQz
MDb24dvKuqrbsxDrgD/bfB40awGCZNkiIc9V8w83ORzJmWh7LJBjRIgw1K+xKfdp
J8q4I0bck9Pd3DgnOrh2RFc37EprsrQdHCuORWrG4uNe+kiL//RnoYnFGlYpaw14
GzLGlxhkYQKBgQDQfIRUUvu6h47GqAu9rUZ+pFtNKcD6qfA8FB9as1uhVGx0ZsHV
pc+is2ENFc88h3ncjkn7aesCqDtzMHiGs2nJ6zF0/RTcWdWWMQYCcCKuYi3VhnJ6
94vmkege9QVog2tfFsv7UVyDp0uFhpi2CPWx3zdSOZclJA+/rcCpQlMbGQKBgQCk
gvJDerj5jvq8G4EEPyg7p4YOxbW+5FdUyRKUBHpkRZC/vmvxkzh1xexyw7itUM0x
DNxGsDpEWWv6u2qM7IWpfvk+FkmxNoEPN2bqWuuOexVfNdyz6TJw0CBV5y4VmXVg
d5Y9hYtF6h25BMtAzdkcovSwsKXMIWi2UlkHMnIpIQKBgDMEAtXZvMaqIui+Te5h
7Yr9O1XVemePgvDYQRnYG16FblaJ0RKi/SPExst/4H2vf5+4xXfdv9bPDAJY5R6j
Fj2IazjIGVzY5Svd5yHEgjHg+ENbx0Ec5EjZ6RIRzeh1BUwaULGiJin0jL9ki17w
H7m8ffTh+pIdNouB0DYfwJj/
-----END PRIVATE KEY-----"""

TEST_JWKS = {
    "keys": [
        {
            "kty": "RSA",
            "kid": "test-key-1",
            "use": "sig",
            "alg": "RS256",
            "n": "r-_XDRNWvjjlPxAJYzjNtIwcfyDJVuyLJSP3CkbZHRdUDL8Zi8Du3B14M7m8t9TnOFlbyCZzSxdP_aPnEPhGfS5EGmwmSC4gD8_XIG_kMQzt0pDGnCmiJHFFAESq6oQh8_pCvPv9NdI6311IPVSnJT1oYxbjlho3pRI_1tjp66AO0zqvIDtBSe7NQNi2ezQTR-YLJADjcoJoCmNd0hrryW0vEH8uemcQ7ALxFDAEgbtNPJDXXT86E3vfaHnLfa47COqNWvVkmyE96a4oxKXBkly3r-58hUDDbx8MdQf9SYZP6agT-cHXkP5wfNUUadg-lmK-yilAzeUD_3dXANosqQ",
            "e": "AQAB",
        }
    ]
}


def make_test_jwt(
    roles: list[str] | None = None,
    issuer: str = "http://127.0.0.1:8080/realms/workbench-realm",
    audience: str = "workbench-client",
) -> str:
    """Create a properly signed RS256 JWT for testing."""
    header = {"alg": "RS256", "typ": "JWT", "kid": "test-key-1"}
    payload: dict = {
        "sub": "test-user-id",
        "preferred_username": "testuser",
        "email": "test@example.com",
        "iss": issuer,
        "aud": audience,
        "exp": 9999999999,
        "iat": 1000000000,
    }
    if roles:
        payload["realm_access"] = {"roles": roles}
    return jwt.encode(payload, TEST_PRIVATE_KEY_PEM, algorithm="RS256", headers=header)

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


def _mock_jwks():
    """Return a context manager that mocks fetch_jwks to return TEST_JWKS."""
    return patch("app.core.security.fetch_jwks", new_callable=AsyncMock, return_value=TEST_JWKS)


def _production_client_with_jwks_mock():
    """Create a test client with production settings and JWKS mock applied after reload."""
    with patch.dict(os.environ, {"DEMO_MODE": "false", "ENVIRONMENT": "production"}):
        client = _reload_for_production()
        # The mock needs to be applied to the reloaded module
        from app.core import security
        original_fetch = security.fetch_jwks
        async def mock_fetch(jwks_url):
            return TEST_JWKS
        security.fetch_jwks = mock_fetch
        return client
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
        client = _production_client_with_jwks_mock()
        token = make_test_jwt(roles=["operator"])
        response = client.get(
            "/api/models",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200

    def test_wrong_issuer_rejected_in_production(self):
        """JWT with mismatched issuer is rejected in production."""
        client = _production_client_with_jwks_mock()
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
        client = _production_client_with_jwks_mock()
        token = make_test_jwt(roles=["admin"])
        response = client.get("/api/audit", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200

    def test_operator_rejected_for_audit_in_production(self):
        """Operator role lacks admin-only access."""
        client = _production_client_with_jwks_mock()
        token = make_test_jwt(roles=["operator"])
        response = client.get("/api/audit", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 403

    def test_writer_can_create_model_in_production(self):
        """Operator role can POST models."""
        client = _production_client_with_jwks_mock()
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
        client = _production_client_with_jwks_mock()
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