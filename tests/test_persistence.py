"""Tests verifying persistent storage integrity across simulated application restarts."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from src.config import load_config
from src.db import AlertDatabaseManager, AlertRecord


def test_sqlite_and_evidence_persistence_across_restart() -> None:
    """Verify that alerts, review status, and evidence files persist across application restarts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        persist_dir = Path(tmpdir) / "var_data"
        persist_dir.mkdir(parents=True, exist_ok=True)
        db_path = persist_dir / "security_alerts.db"
        alerts_dir = persist_dir / "alerts"
        clips_dir = persist_dir / "clips"
        alerts_dir.mkdir(parents=True, exist_ok=True)
        clips_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------------------
        # Phase 1: First Application Lifecycle
        # -------------------------------------------------------------
        db1 = AlertDatabaseManager(str(db_path))

        # Create dummy evidence files
        snap1 = alerts_dir / "snapshot_CAM1_trk1_20261011.jpg"
        snap1.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)

        clip1 = clips_dir / "clip_CAM1_trk1_20261011.mp4"
        clip1.write_bytes(b"\x00\x00\x00 ftypmp42" + b"\x00" * 40)

        alert_id = db1.insert_alert(
            timestamp="2026-10-11T01:00:00Z",
            camera_id="CAM-PLATFORM-01",
            track_id=10,
            object_class="backpack",
            event_duration=19.2,
            reason="Unattended backpack condition persisted > 15s",
            snapshot_path=str(snap1),
            clip_path=str(clip1),
            review_status="unreviewed",
            event_uid="UID-PERSIST-TEST-001",
        )
        assert alert_id == 1

        # Operator reviews alert in Phase 1
        db1.update_review_status(alert_id, "verified")

        # Simulate shutdown by deleting db1 handle
        del db1

        # -------------------------------------------------------------
        # Phase 2: Simulated Service Restart on Persistent Disk
        # -------------------------------------------------------------
        db2 = AlertDatabaseManager(str(db_path))

        record = db2.get_alert_by_id(alert_id)
        assert record is not None
        assert record.id == 1
        assert record.event_uid == "UID-PERSIST-TEST-001"
        assert record.camera_id == "CAM-PLATFORM-01"
        assert record.track_id == 10
        assert record.review_status == "verified"
        assert record.event_duration == 19.2

        # Check evidence files are still accessible at configured paths
        assert Path(record.snapshot_path).is_file()
        assert Path(record.snapshot_path).read_bytes()[:4] == b"\xff\xd8\xff\xe0"

        assert record.clip_path is not None
        assert Path(record.clip_path).is_file()
        assert b"ftyp" in Path(record.clip_path).read_bytes()

        # Check running stats reflect persisted state
        rate, total, false_alarms = db2.get_false_alarm_rate()
        assert total == 1
        assert false_alarms == 0
        assert rate == 0.0


def test_data_dir_env_var_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that RAILGUARD_DATA_DIR standardizes subpaths under persistent storage mount."""
    with tempfile.TemporaryDirectory() as tmpdir:
        simulated_render_disk = str(Path(tmpdir) / "var_data")
        monkeypatch.setenv("RAILGUARD_DATA_DIR", simulated_render_disk)

        cfg = load_config()
        assert cfg.storage.data_dir == simulated_render_disk
        assert Path(cfg.storage.db_path).resolve().parent == Path(simulated_render_disk).resolve()
        assert Path(cfg.storage.alerts_dir).resolve().parent == Path(simulated_render_disk).resolve()
        assert Path(cfg.storage.clips_dir).resolve().parent == Path(simulated_render_disk).resolve()
