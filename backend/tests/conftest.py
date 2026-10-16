"""Pytest configuration and fixtures for Signal Workbench backend tests."""
from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

# Ensure backend is on path
BACKEND_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

# Test environment: force demo mode, in-memory DB, test settings
os.environ["ENVIRONMENT"] = "test"
os.environ["DEMO_MODE"] = "true"
# Use file-based database for test persistence across TestClient instances
test_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
test_db.close()
os.environ["DATABASE_URL"] = f"sqlite:///{test_db.name}"
os.environ["LLM_API_KEY"] = "test-key-placeholder"  # Synthetic test credential


@pytest.fixture(autouse=True)
def init_database() -> Generator[None, None, None]:
    """Initialize database before each test."""
    from app.core.database import init_db, get_connection
    init_db()
    # Clear all tables for test isolation
    with get_connection() as conn:
        conn.execute("DELETE FROM models")
        conn.execute("DELETE FROM inference_runs")
        conn.execute("DELETE FROM benchmark_runs")
        conn.execute("DELETE FROM audit_logs")
        conn.commit()
    yield


@pytest.fixture(autouse=True)
def reset_services() -> Generator[None, None, None]:
    """Reset global service state between tests."""
    from app.services.model_service import _ACTIVE_SESSIONS

    _ACTIVE_SESSIONS.clear()
    yield
    _ACTIVE_SESSIONS.clear()


@pytest.fixture
def test_user() -> str:
    """Test user identifier."""
    return "test-user-123"


@pytest.fixture
def sample_ternary_weights() -> "np.ndarray":
    """Sample ternary weight matrix for testing."""
    import numpy as np

    rng = np.random.default_rng(42)
    # 128x256 weight matrix with ~45% zeros, 27.5% -1, 27.5% +1
    weights = rng.choice([-1, 0, 1], size=(128, 256), p=[0.275, 0.45, 0.275])
    return weights.astype(np.int8)