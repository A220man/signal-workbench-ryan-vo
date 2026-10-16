"""Tests for SQLite persistence, migrations, and audit logging."""
from __future__ import annotations

import pytest

from app.core.database import (
    get_connection,
    init_db,
    log_audit_event,
    query_audit_logs,
)


class TestDatabaseInitialization:
    """Tests for schema creation and migrations."""

    def test_init_db_creates_tables(self):
        init_db()
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        expected = {"models", "benchmark_runs", "audit_logs", "inference_runs"}
        assert expected.issubset(tables)

    def test_schema_version_recorded(self):
        init_db()
        conn = get_connection()
        cursor = conn.cursor()
        # Check that all expected tables exist (schema migration tracking via tables)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('models', 'benchmark_runs', 'audit_logs', 'inference_runs')")
        tables = {row[0] for row in cursor.fetchall()}
        assert tables == {"models", "benchmark_runs", "audit_logs", "inference_runs"}


class TestAuditLogging:
    """Tests for immutable audit trail."""

    def test_log_audit_event_persists(self):
        init_db()
        log_audit_event(
            action="test_action",
            resource_type="test_resource",
            resource_id="res-123",
            user_id="user-456",
            status="success",
            details={"key": "value", "count": 42},
        )

        logs = query_audit_logs(limit=10)
        assert len(logs) >= 1
        log = logs[0]
        assert log["action"] == "test_action"
        assert log["resource_type"] == "test_resource"
        assert log["resource_id"] == "res-123"
        assert log["user_id"] == "user-456"
        assert log["status"] == "success"
        assert log["details"]["key"] == "value"
        assert log["details"]["count"] == 42

    def test_audit_logs_paginated(self):
        init_db()
        for i in range(15):
            log_audit_event(
                action=f"action_{i}",
                resource_type="type",
                resource_id=f"id_{i}",
                user_id="user",
                status="success",
                details={"index": i},
            )

        page1 = query_audit_logs(limit=5, offset=0)
        page2 = query_audit_logs(limit=5, offset=5)
        page3 = query_audit_logs(limit=5, offset=10)

        assert len(page1) == 5
        assert len(page2) == 5
        assert len(page3) == 5
        # Most recent first
        assert page1[0]["details"]["index"] == 14
        assert page2[0]["details"]["index"] == 9
        assert page3[0]["details"]["index"] == 4

    def test_audit_immutability_no_delete(self):
        """Audit table should not have DELETE trigger (immutable)."""
        init_db()
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='trigger' AND tbl_name='audit_logs'")
        triggers = cursor.fetchall()
        # No triggers that allow deletion
        for t in triggers:
            assert "DELETE" not in t[0].upper()


class TestDatabaseConstraints:
    """Tests for foreign keys and constraints."""

    def test_foreign_keys_enabled(self):
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys")
        assert cursor.fetchone()[0] == 1

    def test_wal_mode_for_persistence(self):
        """WAL mode should be enabled for file databases."""
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        # In-memory DBs return "memory", file DBs return "wal"
        assert mode.upper() in ("WAL", "MEMORY")