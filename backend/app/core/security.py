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
from app.core.config import settings

security_bearer = HTTPBearer(auto_error=False)

# In-memory test token store for local protocol tests and mock OIDC verification
_MOCK_OIDC_KEYS: Dict[str, dict] = {}


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
            headers={"WWW-Authenticate": "Bearer"},
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

    try:
        claims = decode_jwt_unverified(token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid JWT bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Validate standard OIDC claims
    exp = claims.get("exp")
    if exp and exp < time.time():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )

    iss = claims.get("iss")
    if settings.OIDC_ISSUER and iss and iss != settings.OIDC_ISSUER:
        if not settings.DEMO_MODE:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Token issuer mismatch",
                headers={"WWW-Authenticate": "Bearer"},
            )

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
