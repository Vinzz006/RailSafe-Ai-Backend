"""Unit tests for RealtimeDatabaseAlertRepository operations and query semantics."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import pytest

from src.realtime_database_repository import (
    RealtimeDatabaseAlertRepository,
    iso_to_epoch_ms,
    sanitize_rtdb_key,
)


class MockRTDBQuery:
    """Simulates query builder returned by order_by_child / order_by_key."""

    def __init__(self, data: Dict[str, Any], order_by_child: Optional[str] = None, order_by_key: bool = False) -> None:
        self.data = dict(data)
        self.child_key = order_by_child
        self.is_key_order = order_by_key
        self.filter_val: Any = None
        self.has_filter = False
        self.limit_val: Optional[int] = None
        self.limit_first = False

    def equal_to(self, val: Any) -> "MockRTDBQuery":
        self.filter_val = val
        self.has_filter = True
        return self

    def limit_to_last(self, limit: int) -> "MockRTDBQuery":
        self.limit_val = limit
        self.limit_first = False
        return self

    def limit_to_first(self, limit: int) -> "MockRTDBQuery":
        self.limit_val = limit
        self.limit_first = True
        return self

    def get(self) -> Optional[Dict[str, Any]]:
        items = list(self.data.items())

        if self.has_filter and self.child_key:
            items = [
                (k, v)
                for k, v in items
                if isinstance(v, dict) and v.get(self.child_key) == self.filter_val
            ]

        if self.child_key:
            items.sort(key=lambda kv: kv[1].get(self.child_key) if isinstance(kv[1], dict) else 0)
        elif self.is_key_order:
            items.sort(key=lambda kv: kv[0])

        if self.limit_val is not None:
            if self.limit_first:
                items = items[: self.limit_val]
            else:
                items = items[-self.limit_val :]

        if not items:
            return None
        return dict(items)


class MockRTDBRef:
    """In-memory tree node simulating Firebase Realtime Database reference."""

    def __init__(self, tree: Dict[str, Any], path: str = "") -> None:
        self.tree = tree
        self.path = path.strip("/")

    def _resolve(self) -> Tuple[Optional[Dict[str, Any]], str]:
        parts = [p for p in self.path.split("/") if p]
        curr = self.tree
        for p in parts[:-1]:
            if p not in curr or not isinstance(curr[p], dict):
                curr[p] = {}
            curr = curr[p]
        last = parts[-1] if parts else ""
        return curr, last

    def child(self, child_path: str) -> "MockRTDBRef":
        child_clean = child_path.strip("/")
        new_path = f"{self.path}/{child_clean}" if self.path else child_clean
        return MockRTDBRef(self.tree, new_path)

    def get(self) -> Any:
        if not self.path:
            return self.tree
        parts = [p for p in self.path.split("/") if p]
        curr = self.tree
        for p in parts:
            if not isinstance(curr, dict) or p not in curr:
                return None
            curr = curr[p]
        return curr

    def set(self, val: Any) -> None:
        parts = [p for p in self.path.split("/") if p]
        if not parts:
            self.tree.clear()
            if isinstance(val, dict):
                self.tree.update(val)
            return
        curr = self.tree
        for p in parts[:-1]:
            if p not in curr or not isinstance(curr[p], dict):
                curr[p] = {}
            curr = curr[p]
        curr[parts[-1]] = val

    def update(self, val: Dict[str, Any]) -> None:
        existing = self.get()
        if not isinstance(existing, dict):
            existing = {}
        existing.update(val)
        self.set(existing)

    def transaction(self, update_fn: Any) -> Any:
        current = self.get()
        new_val = update_fn(current)
        if new_val is not None:
            self.set(new_val)
        return new_val

    def order_by_child(self, child_name: str) -> MockRTDBQuery:
        data = self.get() or {}
        return MockRTDBQuery(data if isinstance(data, dict) else {}, order_by_child=child_name)

    def order_by_key(self) -> MockRTDBQuery:
        data = self.get() or {}
        return MockRTDBQuery(data if isinstance(data, dict) else {}, order_by_key=True)


@pytest.fixture
def rtdb_repo() -> Tuple[RealtimeDatabaseAlertRepository, Dict[str, Any]]:
    tree: Dict[str, Any] = {}
    mock_root = MockRTDBRef(tree)
    repo = RealtimeDatabaseAlertRepository(base_path="railguard", db_root=mock_root)
    return repo, tree


def test_sanitize_rtdb_key() -> None:
    """Verify keys with RTDB disallowed characters are sanitized properly."""
    assert sanitize_rtdb_key("normal_key-123") == "normal_key-123"
    assert sanitize_rtdb_key("alert.with.dots") == "alert_with_dots"
    assert sanitize_rtdb_key("alert$key#test") == "alert_key_test"
    assert sanitize_rtdb_key("alert[0]/segment") == "alert_0__segment"
    assert sanitize_rtdb_key("") == "unknown"


def test_iso_to_epoch_ms() -> None:
    """Verify ISO timestamps correctly convert to integer epoch milliseconds."""
    ms = iso_to_epoch_ms("2026-10-11T00:00:00+00:00")
    assert isinstance(ms, int)
    assert ms > 1700000000000
    # Invalid fallback
    fallback_ms = iso_to_epoch_ms("not-a-date")
    assert isinstance(fallback_ms, int)
    assert fallback_ms > 0


def test_check_readiness(rtdb_repo) -> None:
    """Verify readiness check succeeds and handles empty database."""
    repo, _ = rtdb_repo
    assert repo.check_readiness() is True


def test_create_alert_and_idempotency(rtdb_repo) -> None:
    """Verify idempotent alert creation using event_uid."""
    repo, tree = rtdb_repo

    alert_data = {
        "event_uid": "EVT-RTDB-01",
        "timestamp": "2026-10-11T01:00:00Z",
        "camera_id": "CAM-01",
        "track_id": 10,
        "object_class": "backpack",
        "event_duration": 18.0,
        "reason": "Unattended bag",
    }

    # 1. First creation
    doc_id, created = repo.create_or_update_alert(alert_data)
    assert doc_id == "EVT-RTDB-01"
    assert created is True

    # Check alert was stored in RTDB alerts node
    alert = repo.get_alert("EVT-RTDB-01")
    assert alert is not None
    assert alert["event_uid"] == "EVT-RTDB-01"
    assert alert["status"] == "active"
    assert alert["review_status"] == "unreviewed"
    assert alert["created_at_ms"] > 0
    assert alert["timestamp_ms"] > 0

    # 2. Duplicate insertion with same event_uid
    doc_id2, created2 = repo.create_or_update_alert(alert_data)
    assert doc_id2 == "EVT-RTDB-01"
    assert created2 is False

    # Check alerts count remains 1
    assert len(tree["railguard"]["alerts"]) == 1
    # Check event_uid_index was written
    assert tree["railguard"]["event_uid_index"]["EVT-RTDB-01"] == "EVT-RTDB-01"


def test_list_alerts_and_filters(rtdb_repo) -> None:
    """Verify listing alerts, status filtering, and sorting."""
    repo, _ = rtdb_repo

    # Insert 3 alerts with distinct timestamps
    for i, st in enumerate(["unreviewed", "verified", "false_alarm"]):
        repo.create_or_update_alert({
            "event_uid": f"EVT-LIST-{i}",
            "timestamp": f"2026-10-11T0{i+1}:00:00Z",
            "camera_id": "CAM-01",
            "track_id": i,
            "object_class": "backpack",
            "event_duration": 15.0,
            "reason": f"Alert {i}",
            "review_status": st,
        })

    # Unfiltered listing
    all_alerts = repo.list_alerts(limit=50)
    assert len(all_alerts) == 3
    # Ordered descending by created_at_ms
    assert all_alerts[0]["event_uid"] == "EVT-LIST-2"

    # Filter unreviewed_only
    unreviewed = repo.list_alerts(limit=50, unreviewed_only=True)
    assert len(unreviewed) == 1
    assert unreviewed[0]["event_uid"] == "EVT-LIST-0"

    # Filter status="verified"
    verified = repo.list_alerts(limit=50, review_status="verified")
    assert len(verified) == 1
    assert verified[0]["event_uid"] == "EVT-LIST-1"

    # Filter status="false_alarm"
    false_alarm = repo.list_alerts(limit=50, review_status="false_alarm")
    assert len(false_alarm) == 1
    assert false_alarm[0]["event_uid"] == "EVT-LIST-2"


def test_update_review_status_and_audit(rtdb_repo) -> None:
    """Verify review status updates record reviewer UID and reviewed_at timestamp."""
    repo, _ = rtdb_repo

    repo.create_or_update_alert({
        "event_uid": "EVT-REV-01",
        "timestamp": "2026-10-11T01:00:00Z",
        "camera_id": "CAM-01",
        "track_id": 5,
        "object_class": "luggage",
        "event_duration": 12.0,
        "reason": "Test review",
    })

    # Update to verified
    ok = repo.update_review_status(
        alert_id="EVT-REV-01",
        review_status="verified",
        reviewed_by="officer-uid-42",
    )
    assert ok is True

    updated = repo.get_alert("EVT-REV-01")
    assert updated is not None
    assert updated["review_status"] == "verified"
    assert updated["reviewed_by"] == "officer-uid-42"
    assert updated["reviewed_at"] is not None

    # Invalid status raises ValueError
    with pytest.raises(ValueError):
        repo.update_review_status("EVT-REV-01", "invalid_status")

    # Non-existent alert returns False
    assert repo.update_review_status("NON-EXISTENT", "verified") is False


def test_update_cleared_timestamp(rtdb_repo) -> None:
    """Verify luggage retrieval updates active alerts for matching camera and track."""
    repo, _ = rtdb_repo

    repo.create_or_update_alert({
        "event_uid": "EVT-CLR-01",
        "timestamp": "2026-10-11T01:00:00Z",
        "camera_id": "CAM-PLATFORM-02",
        "track_id": 99,
        "object_class": "suitcase",
        "event_duration": 20.0,
        "reason": "Clear test",
    })

    # Clear matching alert
    cleared_time = "2026-10-11T01:05:00Z"
    count = repo.update_cleared_timestamp(
        camera_id="CAM-PLATFORM-02",
        track_id=99,
        cleared_timestamp=cleared_time,
    )
    assert count == 1

    cleared_alert = repo.get_alert("EVT-CLR-01")
    assert cleared_alert is not None
    assert cleared_alert["cleared_timestamp"] == cleared_time
    assert cleared_alert["status"] == "cleared"

    # Calling again should update 0 alerts since it is already cleared
    count2 = repo.update_cleared_timestamp(
        camera_id="CAM-PLATFORM-02",
        track_id=99,
        cleared_timestamp="2026-10-11T01:10:00Z",
    )
    assert count2 == 0


def test_false_alarm_rate_calculation(rtdb_repo) -> None:
    """Verify false alarm rate computes correctly from verified and false_alarm reviews."""
    repo, _ = rtdb_repo

    # Zero reviewed
    rate, total, false_alarms = repo.get_false_alarm_rate()
    assert rate is None
    assert total == 0
    assert false_alarms == 0

    # 1 verified, 1 false alarm -> 50%
    repo.create_or_update_alert({
        "event_uid": "EVT-ST-1",
        "timestamp": "2026-10-11T01:00:00Z",
        "camera_id": "CAM-01",
        "track_id": 1,
        "object_class": "bag",
        "event_duration": 10.0,
        "reason": "stat 1",
        "review_status": "verified",
    })
    repo.create_or_update_alert({
        "event_uid": "EVT-ST-2",
        "timestamp": "2026-10-11T01:05:00Z",
        "camera_id": "CAM-01",
        "track_id": 2,
        "object_class": "bag",
        "event_duration": 10.0,
        "reason": "stat 2",
        "review_status": "false_alarm",
    })

    rate, total, false_alarms = repo.get_false_alarm_rate()
    assert rate == 0.5
    assert total == 2
    assert false_alarms == 1


def test_sanitization_collision_disambiguation(rtdb_repo) -> None:
    """Verify that two distinct event_uids that sanitize to the same base key do not overwrite each other."""
    repo, tree = rtdb_repo

    # Both 'EVT/001' and 'EVT.001' sanitize to 'EVT_001'
    alert1 = {
        "event_uid": "EVT/001",
        "timestamp": "2026-10-11T01:00:00Z",
        "camera_id": "CAM-01",
        "track_id": 1,
        "object_class": "bag",
        "event_duration": 5.0,
        "reason": "First alert",
    }
    alert2 = {
        "event_uid": "EVT.001",
        "timestamp": "2026-10-11T01:01:00Z",
        "camera_id": "CAM-01",
        "track_id": 2,
        "object_class": "suitcase",
        "event_duration": 8.0,
        "reason": "Collision alert with distinct event_uid",
    }

    id1, created1 = repo.create_or_update_alert(alert1)
    assert created1 is True
    assert id1 == "EVT_001"

    id2, created2 = repo.create_or_update_alert(alert2)
    assert created2 is True
    assert id2 != id1
    assert id2.startswith("EVT_001_")

    # Verify both records coexist in the database with their respective event_uids
    doc1 = repo.get_alert(id1)
    doc2 = repo.get_alert(id2)
    assert doc1 is not None and doc1["event_uid"] == "EVT/001"
    assert doc2 is not None and doc2["event_uid"] == "EVT.001"

    # Idempotent re-ingestion of both
    id1_retry, created1_retry = repo.create_or_update_alert(alert1)
    assert id1_retry == id1
    assert created1_retry is False

    id2_retry, created2_retry = repo.create_or_update_alert(alert2)
    assert id2_retry == id2
    assert created2_retry is False


def test_list_alerts_handles_malformed_records(rtdb_repo) -> None:
    """Verify list_alerts does not crash when RTDB contains non-dict or malformed records."""
    repo, tree = rtdb_repo

    # Add a valid alert
    repo.create_or_update_alert({
        "event_uid": "EVT-VALID-01",
        "timestamp": "2026-10-11T02:00:00Z",
        "camera_id": "CAM-01",
        "track_id": 1,
        "object_class": "bag",
        "event_duration": 5.0,
        "reason": "Valid alert",
    })

    # Inject legacy malformed records directly into tree
    alerts_node = tree["railguard"]["alerts"]
    alerts_node["MALFORMED_STRING_MS"] = {
        "id": "MALFORMED_STRING_MS",
        "event_uid": "EVT-BAD-1",
        "created_at_ms": "not-an-int",
        "created_at": "invalid-date",
        "camera_id": "CAM-01",
        "track_id": 2,
        "review_status": "unreviewed",
    }

    alerts = repo.list_alerts(limit=10)
    assert len(alerts) >= 1
    # Verify the valid alert is present
    uids = [a["event_uid"] for a in alerts]
    assert "EVT-VALID-01" in uids
