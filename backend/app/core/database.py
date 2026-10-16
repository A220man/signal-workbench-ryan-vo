"""Persistent SQLite storage with parameterized SQL, schema migrations, and audit logging."""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from typing import Any, Dict, List, Optional
from app.core.config import settings


def get_db_path() -> str:
    """Extract filesystem path from sqlite URI or return default."""
    url = settings.DATABASE_URL
    if url.startswith("sqlite:///"):
        return url.replace("sqlite:///", "")
    if url.startswith("sqlite://"):
        # Handle sqlite:///file:... URIs
        return url.replace("sqlite://", "")
    return "workbench.db"


def get_connection() -> sqlite3.Connection:
    """Create a connection to SQLite with row factories and WAL mode enabled."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    if db_path != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def init_db() -> None:
    """Initialize database tables with schema migrations."""
    with get_connection() as conn:
        cursor = conn.cursor()

        # Models table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                architecture TEXT NOT NULL,
                parameter_count INTEGER NOT NULL,
                quantization_format TEXT NOT NULL,
                activation_function TEXT NOT NULL,
                status TEXT NOT NULL,
                memory_bytes INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)

        # Inference executions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inference_runs (
                id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                completion TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                latency_ms REAL NOT NULL,
                tokens_per_sec REAL NOT NULL,
                quant_type TEXT NOT NULL,
                activation TEXT NOT NULL,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (model_id) REFERENCES models (id) ON DELETE CASCADE
            );
        """)

        # Benchmark runs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                dataset TEXT NOT NULL,
                tokens_evaluated INTEGER NOT NULL,
                activation_tested TEXT NOT NULL,
                measured_perplexity REAL NOT NULL,
                mean_loss REAL NOT NULL,
                latency_ms REAL NOT NULL,
                memory_peak_mb REAL NOT NULL,
                details_json TEXT NOT NULL,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
        """)

        # Audit logs for mutations and operational history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                action TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                resource_id TEXT NOT NULL,
                status TEXT NOT NULL,
                details TEXT NOT NULL
            );
        """)

        conn.commit()


def log_audit_event(
    action: str,
    resource_type: str,
    resource_id: str,
    user_id: str,
    status: str,
    details: Dict[str, Any],
) -> None:
    """Record an immutable audit event in SQLite."""
    now_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO audit_logs (timestamp, user_id, action, resource_type, resource_id, status, details)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_utc,
                user_id,
                action,
                resource_type,
                resource_id,
                status,
                json.dumps(details),
            ),
        )
        conn.commit()


def query_audit_logs(
    limit: int = 50,
    offset: int = 0,
    action: Optional[str] = None,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve paginated audit records."""
    query = "SELECT id, timestamp, user_id, action, resource_type, resource_id, status, details FROM audit_logs"
    params: List[Any] = []
    conditions: List[str] = []

    if action:
        conditions.append("action = ?")
        params.append(action)
    if user_id:
        conditions.append("user_id = ?")
        params.append(user_id)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        results = []
        for r in rows:
            results.append({
                "id": r["id"],
                "timestamp": r["timestamp"],
                "user_id": r["user_id"],
                "action": r["action"],
                "resource_type": r["resource_type"],
                "resource_id": r["resource_id"],
                "status": r["status"],
                "details": json.loads(r["details"]) if r["details"] else {},
            })
        return results
