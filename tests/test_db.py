"""Unit tests for SQLite database operations, review actions, and false-alarm rate calculations."""

import gc
import tempfile
from pathlib import Path

import pytest

from src.db import AlertDatabaseManager


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test_db.db")
        db = AlertDatabaseManager(db_path)
        yield db
        del db
        gc.collect()


def test_insert_and_get_alert(temp_db: AlertDatabaseManager) -> None:
    alert_id = temp_db.insert_alert(
        timestamp="2026-10-10T12:00:00",
        camera_id="CAM-PLATFORM-01",
        track_id=101,
        object_class="backpack",
        event_duration=15.5,
        reason="Unattended backpack for 15.5s",
        snapshot_path="/path/to/snap.jpg",
        clip_path="/path/to/clip.mp4",
        review_status="unreviewed",
    )
    assert alert_id == 1

    alert = temp_db.get_alert_by_id(1)
    assert alert is not None
    assert alert.track_id == 101
    assert alert.object_class == "backpack"
    assert alert.review_status == "unreviewed"
    assert alert.event_duration == 15.5
    assert alert.cleared_timestamp is None


def test_update_cleared_timestamp(temp_db: AlertDatabaseManager) -> None:
    temp_db.insert_alert(
        timestamp="2026-10-10T12:00:00",
        camera_id="CAM-PLATFORM-01",
        track_id=202,
        object_class="suitcase",
        event_duration=20.0,
        reason="Left suitcase",
        snapshot_path="",
    )

    rows_affected = temp_db.update_cleared_timestamp(
        track_id=202,
        camera_id="CAM-PLATFORM-01",
        cleared_timestamp="2026-10-10T12:00:30",
    )
    assert rows_affected == 1

    updated = temp_db.get_alert_by_id(1)
    assert updated is not None
    assert updated.cleared_timestamp == "2026-10-10T12:00:30"


def test_update_review_status(temp_db: AlertDatabaseManager) -> None:
    aid = temp_db.insert_alert(
        timestamp="2026-10-10T12:00:00",
        camera_id="CAM-01",
        track_id=303,
        object_class="handbag",
        event_duration=16.0,
        reason="Unattended handbag",
        snapshot_path="",
    )

    # Valid updates
    assert temp_db.update_review_status(aid, "verified") is True
    assert temp_db.get_alert_by_id(aid).review_status == "verified"

    assert temp_db.update_review_status(aid, "false_alarm") is True
    assert temp_db.get_alert_by_id(aid).review_status == "false_alarm"

    # Invalid status raises ValueError
    with pytest.raises(ValueError, match="Invalid review_status"):
        temp_db.update_review_status(aid, "unknown_status")


def test_false_alarm_rate_calculation(temp_db: AlertDatabaseManager) -> None:
    # 0 reviewed alerts initially
    rate, total_reviewed, fa_count = temp_db.get_false_alarm_rate()
    assert rate is None
    assert total_reviewed == 0
    assert fa_count == 0

    # Insert 4 alerts
    a1 = temp_db.insert_alert("2026-10-10T10:00:00", "C1", 1, "bag", 15.0, "reason", "")
    a2 = temp_db.insert_alert("2026-10-10T10:05:00", "C1", 2, "bag", 16.0, "reason", "")
    a3 = temp_db.insert_alert("2026-10-10T10:10:00", "C1", 3, "bag", 17.0, "reason", "")
    a4 = temp_db.insert_alert("2026-10-10T10:15:00", "C1", 4, "bag", 18.0, "reason", "")

    # Review 1 verified, 1 false alarm, 2 remain unreviewed
    temp_db.update_review_status(a1, "verified")
    temp_db.update_review_status(a2, "false_alarm")

    rate, total_reviewed, fa_count = temp_db.get_false_alarm_rate()
    assert rate == 0.5  # 1 false alarm out of 2 reviewed
    assert total_reviewed == 2
    assert fa_count == 1

    # Review another as false alarm (now 1 verified, 2 false alarms)
    temp_db.update_review_status(a3, "false_alarm")
    rate2, total_reviewed2, fa_count2 = temp_db.get_false_alarm_rate()
    assert round(rate2, 4) == round(2.0 / 3.0, 4)
    assert total_reviewed2 == 3
    assert fa_count2 == 2
