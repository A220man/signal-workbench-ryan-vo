"""Application configuration with environment parsing and validation."""
from __future__ import annotations

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime configuration for Signal Workbench."""
    APP_NAME: str = "Signal Workbench"
    VERSION: str = "1.1.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")

    # Persistent Storage
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./workbench.db")

    # Authentication & Security
    DEMO_MODE: bool = os.getenv("DEMO_MODE", "true").lower() in ("true", "1", "yes")
    OIDC_ISSUER: str = os.getenv("OIDC_ISSUER", "http://127.0.0.1:8080/realms/workbench-realm")
    OIDC_CLIENT_ID: str = os.getenv("OIDC_CLIENT_ID", "workbench-client")
    OIDC_CLIENT_SECRET: Optional[str] = os.getenv("OIDC_CLIENT_SECRET", None)
    OIDC_AUDIENCE: str = os.getenv("OIDC_AUDIENCE", "workbench-client")
    OIDC_DISCOVERY_URL: str = os.getenv(
        "OIDC_DISCOVERY_URL",
        "http://127.0.0.1:8080/realms/workbench-realm/.well-known/openid-configuration"
    )

    # Advisory LLM Configuration (opt-in, server-side only)
    LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY", None)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai-compatible")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "meta-llama/llama-3.1-8b-instruct")
    LLM_BASE_URL: Optional[str] = os.getenv("LLM_BASE_URL", None)

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

# Enforce security constraint: Demo mode must be rejected in production
if settings.ENVIRONMENT.lower() == "production" and settings.DEMO_MODE:
    raise RuntimeError(
        "SECURITY VIOLATION: DEMO_MODE cannot be enabled in production environment. "
        "Explicit OIDC authentication must be configured."
    )
