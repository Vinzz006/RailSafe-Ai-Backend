"""Unit tests for Firestore to Realtime Database migration utility."""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from src.config import SystemConfig
from src.migrate_firestore_to_rtdb import (
    run_migration,
    transform_firestore_doc_to_rtdb,
)
from tests.test_realtime_database_repository import MockRTDBRef


def test_transform_firestore_doc_to_rtdb() -> None:
    """Verify document transformation preserves all fields and generates epoch milliseconds."""
    firestore_data = {
        "event_uid": "EVT-TEST-99",
        "timestamp": "2026-10-11T02:00:00Z",
        "created_at": "2026-10-11T02:00:00Z",
        "camera_id": "CAM-PLATFORM-03",
        "track_id": 7,
        "object_class": "suitcase",
        "event_duration": 22.5,
        "reason": "Unattended bag detected",
        "snapshot_key": "snap_CAM03_EVT99.jpg",
        "snapshot_url": "/api/alerts/EVT-TEST-99/snapshot",
        "clip_key": "clip_CAM03_EVT99.mp4",
        "clip_url": "/api/alerts/EVT-TEST-99/clip",
        "review_status": "verified",
        "reviewed_by": "officer-uid-01",
        "reviewed_at": "2026-10-11T02:10:00Z",
        "cleared_timestamp": None,
    }

    rtdb_rec = transform_firestore_doc_to_rtdb("EVT-TEST-99", firestore_data)

    assert rtdb_rec["id"] == "EVT-TEST-99"
    assert rtdb_rec["event_uid"] == "EVT-TEST-99"
    assert rtdb_rec["status"] == "active"
    assert rtdb_rec["review_status"] == "verified"
    assert rtdb_rec["reviewed_by"] == "officer-uid-01"
    assert rtdb_rec["snapshot_key"] == "snap_CAM03_EVT99.jpg"
    assert rtdb_rec["clip_key"] == "clip_CAM03_EVT99.mp4"
    assert isinstance(rtdb_rec["created_at_ms"], int)
    assert isinstance(rtdb_rec["timestamp_ms"], int)
    assert rtdb_rec["created_at_ms"] > 1700000000000


def test_transform_cleared_alert() -> None:
    """Verify alert with cleared_timestamp receives status='cleared'."""
    firestore_data = {
        "event_uid": "EVT-CLEARED-01",
        "cleared_timestamp": "2026-10-11T02:30:00Z",
    }
    rtdb_rec = transform_firestore_doc_to_rtdb("EVT-CLEARED-01", firestore_data)
    assert rtdb_rec["status"] == "cleared"
    assert rtdb_rec["cleared_timestamp"] == "2026-10-11T02:30:00Z"


def test_migration_dry_run_and_idempotency(monkeypatch) -> None:
    """Verify migration dry run does not write to RTDB and live migration skips existing records."""
    tree: dict = {}
    mock_rtdb_root = MockRTDBRef(tree)

    # Mock Firestore doc
    mock_doc = MagicMock()
    mock_doc.id = "EVT-MIGRATE-01"
    mock_doc.to_dict.return_value = {
        "event_uid": "EVT-MIGRATE-01",
        "timestamp": "2026-10-11T01:00:00Z",
        "camera_id": "CAM-01",
        "track_id": 1,
        "object_class": "backpack",
        "event_duration": 15.0,
        "reason": "Dry run migration test",
        "review_status": "unreviewed",
    }

    mock_firestore_db = MagicMock()
    mock_coll = MagicMock()
    mock_coll.stream.return_value = [mock_doc]
    mock_firestore_db.collection.return_value = mock_coll

    monkeypatch.setattr("src.migrate_firestore_to_rtdb.get_firestore_db", lambda *args, **kwargs: mock_firestore_db)
    monkeypatch.setattr("src.migrate_firestore_to_rtdb.get_realtime_db", lambda *args, **kwargs: mock_rtdb_root)

    cfg = SystemConfig()

    # 1. Test Dry Run: should not write to RTDB tree
    summary_dry = run_migration(cfg, dry_run=True, overwrite=False)
    assert summary_dry["total_scanned"] == 1
    assert summary_dry["migrated"] == 1
    assert summary_dry["failed"] == 0
    assert "railguard" not in tree or "alerts" not in tree.get("railguard", {})

    # 2. Test Live Migration: should write to RTDB tree
    summary_live = run_migration(cfg, dry_run=False, overwrite=False)
    assert summary_live["total_scanned"] == 1
    assert summary_live["migrated"] == 1
    assert tree["railguard"]["alerts"]["EVT-MIGRATE-01"]["event_uid"] == "EVT-MIGRATE-01"
    assert tree["railguard"]["event_uid_index"]["EVT-MIGRATE-01"] == "EVT-MIGRATE-01"

    # 3. Test Re-run without overwrite: should skip existing record
    summary_rerun = run_migration(cfg, dry_run=False, overwrite=False)
    assert summary_rerun["total_scanned"] == 1
    assert summary_rerun["skipped"] == 1
    assert summary_rerun["migrated"] == 0
