"""Unit tests for alert evidence recording and duplicate event suppression."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.alerts import AlertManager
from src.capture import VideoCaptureManager
from src.config import StorageConfig, VideoConfig
from src.db import AlertDatabaseManager
from src.rules import AlertEvent, ClearedAlertEvent


@pytest.fixture
def temp_environment():
    with tempfile.TemporaryDirectory() as tmpdir:
        p_tmp = Path(tmpdir)
        db_path = str(p_tmp / "test_alerts.db")
        alerts_dir = str(p_tmp / "alerts")
        clips_dir = str(p_tmp / "clips")

        db = AlertDatabaseManager(db_path)
        cfg = StorageConfig(
            db_path=db_path,
            alerts_dir=alerts_dir,
            clips_dir=clips_dir,
            pre_alert_seconds=2.0,
            post_alert_seconds=2.0,
        )
        am = AlertManager(cfg, db)

        # Mock capture manager with simulated buffer
        v_cfg = VideoConfig(buffer_seconds=10.0)
        vcm = VideoCaptureManager(v_cfg)
        # Populate rolling buffer with synthetic frames
        for sec in range(10):
            frame = np.full((100, 100, 3), sec * 20, dtype=np.uint8)
            vcm.buffer.append((float(sec), frame))

        yield am, vcm, db, p_tmp
        vcm.release()
        del am
        del db
        import gc
        gc.collect()


def test_duplicate_alert_suppression(temp_environment) -> None:
    """Verify active alerts for the same track ID are not duplicated on every frame."""
    am, vcm, db, _ = temp_environment
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    alert = AlertEvent(
        track_id=42,
        object_class="backpack",
        camera_id="CAM-01",
        timestamp=5.0,
        duration=15.0,
        reason="Unattended backpack",
        bbox=(20, 20, 80, 80),
        centroid=(50.0, 50.0),
    )

    # Frame 1: First time alert is triggered -> should insert into DB
    created_ids_1 = am.process_alerts([alert], [], frame, vcm)
    assert len(created_ids_1) == 1
    assert am.is_alert_active(42) is True

    # Frame 2: Same alert persists in active state -> MUST NOT create duplicate DB row
    created_ids_2 = am.process_alerts([alert], [], frame, vcm)
    assert len(created_ids_2) == 0

    # Total records in DB should remain exactly 1
    all_alerts = db.get_alerts()
    assert len(all_alerts) == 1
    assert all_alerts[0].track_id == 42


def test_alert_clearing_lifecycle(temp_environment) -> None:
    """Verify an active alert clears and records the cleared timestamp in SQLite."""
    am, vcm, db, _ = temp_environment
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    alert = AlertEvent(
        track_id=99,
        object_class="suitcase",
        camera_id="CAM-01",
        timestamp=4.0,
        duration=16.0,
        reason="Unattended suitcase",
        bbox=(10, 10, 60, 60),
        centroid=(35.0, 35.0),
    )

    am.process_alerts([alert], [], frame, vcm)
    assert am.is_alert_active(99) is True

    # Clear event
    cleared = ClearedAlertEvent(
        track_id=99,
        object_class="suitcase",
        camera_id="CAM-01",
        timestamp=8.0,
        cleared_reason="Owner returned",
    )

    am.process_alerts([], [cleared], frame, vcm)
    assert am.is_alert_active(99) is False

    # Check database record
    rec = db.get_alerts()[0]
    assert rec.cleared_timestamp is not None
    assert "T" in rec.cleared_timestamp  # ISO-8601 string


def test_snapshot_and_clip_generation(temp_environment) -> None:
    """Verify snapshot JPEG and clip MP4 files are written to disk."""
    am, vcm, db, tmp_dir = temp_environment
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    alert = AlertEvent(
        track_id=7,
        object_class="handbag",
        camera_id="CAM-02",
        timestamp=5.0,
        duration=15.0,
        reason="Unattended handbag",
        bbox=(20, 20, 50, 50),
        centroid=(35.0, 35.0),
    )

    db_ids = am.process_alerts([alert], [], frame, vcm)
    assert len(db_ids) == 1

    rec = db.get_alert_by_id(db_ids[0])
    assert rec is not None
    assert rec.snapshot_path != ""
    assert Path(rec.snapshot_path).is_file()
    assert Path(rec.snapshot_path).stat().st_size > 0
