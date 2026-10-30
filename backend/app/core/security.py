"""Authentication and authorization utilities for OIDC/PKCE and role checks."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import time
from typing import Dict, List, Optional
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import httpx
from jose import jwt
from jose.exceptions import JOSEError, ExpiredSignatureError
from app.core.config import settings

security_bearer = HTTPBearer(auto_error=False)

# In-memory test token store for local protocol tests and mock OIDC verification
_MOCK_OIDC_KEYS: Dict[str, dict] = {}

# JWKS cache with TTL
_JWKS_CACHE: Dict[str, tuple[Dict, float]] = {}
_JWKS_TTL_SECONDS = 3600  # 1 hour


class UserContext:
    """Authenticated user context with assigned roles."""

    def __init__(self, user_id: str, username: str, email: str, roles: List[str]):
        self.user_id = user_id
        self.username = username
        self.email = email
        self.roles = roles

    def has_role(self, role: str) -> bool:
        if "admin" in self.roles:
            return True
        return role in self.roles

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "roles": self.roles,
        }


def generate_pkce_pair() -> Dict[str, str]:
    """Generate PKCE code_verifier and code_challenge (S256)."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return {
        "code_verifier": verifier,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }


def verify_pkce_challenge(verifier: str, challenge: str) -> bool:
    """Verify that a code_verifier hashes to the stored code_challenge."""
    if not verifier or not challenge:
        return False
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    computed = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return computed == challenge


def decode_jwt_unverified(token: str) -> dict:
    """Parse JWT claims safely without logging token content."""
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid JWT format")
    payload_b64 = parts[1]
    # Pad base64 if necessary
    padding = len(payload_b64) % 4
    if padding:
        payload_b64 += "=" * (4 - padding)
    decoded_bytes = base64.urlsafe_b64decode(payload_b64.encode("ascii"))
    return json.loads(decoded_bytes.decode("utf-8"))


async def fetch_jwks(jwks_url: str) -> Dict:
    """Fetch JWKS from the OIDC provider with caching."""
    global _JWKS_CACHE
    now = time.time()

    # Check cache
    if jwks_url in _JWKS_CACHE:
        cached_jwks, cached_at = _JWKS_CACHE[jwks_url]
        if now - cached_at < _JWKS_TTL_SECONDS:
            return cached_jwks

    # Fetch fresh JWKS
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(jwks_url)
            resp.raise_for_status()
            jwks = resp.json()
    except Exception as e:
        # If we have stale cache, use it; otherwise raise
        if jwks_url in _JWKS_CACHE:
            return _JWKS_CACHE[jwks_url][0]
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Unable to fetch JWKS from OIDC provider: {e}",
        )

    _JWKS_CACHE[jwks_url] = (jwks, now)
    return jwks


async def validate_jwt_signature(token: str) -> dict:
    """Validate JWT signature using Keycloak JWKS.

    Returns the verified claims if valid, raises HTTPException otherwise.
    """
    # Extract kid from token header
    try:
        header = jwt.get_unverified_header(token)
    except JOSEError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid JWT header: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    kid = header.get("kid")
    if not kid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT header missing 'kid'",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get JWKS URL from discovery endpoint
    jwks_url = f"{settings.OIDC_ISSUER}/protocol/openid-connect/certs"

    jwks = await fetch_jwks(jwks_url)

    # Find matching key
    key = None
    for k in jwks.get("keys", []):
        if k.get("kid") == kid:
            key = k
            break

    if not key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"No matching key found for kid: {kid}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Validate token with the public key
    try:
        claims = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            audience=settings.OIDC_AUDIENCE,
            issuer=settings.OIDC_ISSUER,
            options={"verify_aud": True, "verify_iss": True, "verify_exp": True},
        )
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except JOSEError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"JWT validation failed: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return claims


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security_bearer),
) -> UserContext:
    """Extract and validate the current user from the Authorization bearer token."""
    # When in demo mode and no credentials provided, return local operator user
    if credentials is None:
        if settings.DEMO_MODE:
            return UserContext(
                user_id="demo-user-1",
                username="demo-operator",
                email="operator@local.demo",
                roles=["admin", "operator", "viewer"],
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials.strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Empty Bearer authentication token",
        )

    # In demo mode, accept synthetic demo tokens for test automation
    if settings.DEMO_MODE and token.startswith("demo-token-"):
        role_type = token.replace("demo-token-", "")
        if role_type in ("admin", "operator", "viewer"):
            return UserContext(
                user_id=f"demo-{role_type}",
                username=f"demo-{role_type}",
                email=f"{role_type}@local.demo",
                roles=[role_type, "viewer"] if role_type != "viewer" else ["viewer"],
            )

    # Production mode: validate JWT signature via JWKS
    claims = await validate_jwt_signature(token)

    sub = claims.get("sub", "unknown")
    username = claims.get("preferred_username", claims.get("name", sub))
    email = claims.get("email", f"{username}@domain.local")

    # Extract realm or resource roles from Keycloak payload
    roles: List[str] = []
    realm_access = claims.get("realm_access", {})
    if isinstance(realm_access, dict):
        roles.extend(realm_access.get("roles", []))
    resource_access = claims.get("resource_access", {}).get(settings.OIDC_CLIENT_ID, {})
    if isinstance(resource_access, dict):
        roles.extend(resource_access.get("roles", []))

    if not roles:
        roles = ["viewer"]

    return UserContext(user_id=sub, username=username, email=email, roles=roles)


def require_role(required_role: str):
    """Dependency factory checking if current user possesses the required role."""

    def _role_checker(user: UserContext = Depends(get_current_user)) -> UserContext:
        if not user.has_role(required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' required for this action",
            )
        return user

    return _role_checker
