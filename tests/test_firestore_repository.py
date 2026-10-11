"""Unit tests for FirestoreAlertRepository operations and error handling."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.firestore_repository import FirestoreAlertRepository


class MockDocSnapshot:
    def __init__(self, doc_id: str, data: dict, exists: bool = True, reference: any = None) -> None:
        self.id = doc_id
        self._data = data
        self.exists = exists
        self.reference = reference

    def to_dict(self) -> dict:
        return dict(self._data)


class MockDocReference:
    def __init__(self, doc_id: str, storage_dict: dict) -> None:
        self.id = doc_id
        self.storage = storage_dict

    def get(self) -> MockDocSnapshot:
        if self.id in self.storage:
            return MockDocSnapshot(self.id, self.storage[self.id], exists=True, reference=self)
        return MockDocSnapshot(self.id, {}, exists=False, reference=self)

    def set(self, data: dict) -> None:
        self.storage[self.id] = dict(data)

    def update(self, data: dict) -> None:
        if self.id in self.storage:
            self.storage[self.id].update(data)


class MockCollectionReference:
    def __init__(self, storage_dict: dict) -> None:
        self.storage = storage_dict

    def document(self, doc_id: str) -> MockDocReference:
        return MockDocReference(doc_id, self.storage)

    def where(self, field: str, op: str, value: any) -> "MockCollectionReference":
        filtered = {}
        for k, v in self.storage.items():
            if op == "==" and v.get(field) == value:
                filtered[k] = v
        return self.__class__(filtered)

    def order_by(self, field: str, direction: str = "ASCENDING") -> "MockCollectionReference":
        return self

    def limit(self, count: int) -> "MockCollectionReference":
        return self

    def stream(self):
        for k, v in self.storage.items():
            yield MockDocSnapshot(k, v, exists=True, reference=self.document(k))


class MockFirestoreClient:
    def __init__(self) -> None:
        self.storage: dict = {}

    def collection(self, name: str) -> MockCollectionReference:
        return MockCollectionReference(self.storage)


def test_repository_create_and_idempotency() -> None:
    """Verify document creation and duplicate suppression on event_uid."""
    client = MockFirestoreClient()
    repo = FirestoreAlertRepository(collection_name="test_alerts", db_client=client)

    alert_data = {
        "event_uid": "EVT-FS-01",
        "timestamp": "2026-10-11T01:00:00Z",
        "camera_id": "CAM-01",
        "track_id": 10,
        "object_class": "backpack",
        "event_duration": 18.0,
        "reason": "Test Firestore Ingestion",
    }

    # First write -> Created
    doc_id, created = repo.create_or_update_alert(alert_data)
    assert doc_id == "EVT-FS-01"
    assert created is True

    # Duplicate write -> Suppressed
    doc_id2, created2 = repo.create_or_update_alert(alert_data)
    assert doc_id2 == "EVT-FS-01"
    assert created2 is False
    assert len(client.storage) == 1


def test_repository_get_and_review_lifecycle() -> None:
    """Verify alert retrieval and operator review status updates."""
    client = MockFirestoreClient()
    repo = FirestoreAlertRepository(collection_name="test_alerts", db_client=client)

    repo.create_or_update_alert({
        "event_uid": "EVT-REV-99",
        "timestamp": "2026-10-11T01:00:00Z",
        "camera_id": "CAM-01",
        "track_id": 99,
        "object_class": "suitcase",
        "event_duration": 25.0,
        "reason": "Review test",
    })

    # Retrieve
    alert = repo.get_alert("EVT-REV-99")
    assert alert is not None
    assert alert["review_status"] == "unreviewed"

    # Review as verified
    updated = repo.update_review_status("EVT-REV-99", "verified", reviewed_by="officer_123")
    assert updated is True

    reviewed_doc = repo.get_alert("EVT-REV-99")
    assert reviewed_doc["review_status"] == "verified"
    assert reviewed_doc["reviewed_by"] == "officer_123"

    # Invalid status raises ValueError
    with pytest.raises(ValueError, match="Invalid review_status"):
        repo.update_review_status("EVT-REV-99", "invalid_status")


def test_repository_cleared_timestamp_and_false_alarm_rate() -> None:
    """Verify cleared timestamp update and false alarm rate calculation."""
    client = MockFirestoreClient()
    repo = FirestoreAlertRepository(collection_name="test_alerts", db_client=client)

    repo.create_or_update_alert({
        "event_uid": "EVT-A1",
        "camera_id": "CAM-1",
        "track_id": 5,
        "review_status": "unreviewed",
        "cleared_timestamp": None,
    })
    repo.create_or_update_alert({
        "event_uid": "EVT-A2",
        "camera_id": "CAM-1",
        "track_id": 6,
        "review_status": "unreviewed",
        "cleared_timestamp": None,
    })

    # Mark A1 cleared
    rows = repo.update_cleared_timestamp("CAM-1", 5, "2026-10-11T01:05:00Z")
    assert rows == 1
    assert repo.get_alert("EVT-A1")["cleared_timestamp"] == "2026-10-11T01:05:00Z"

    # Review A1 as verified, A2 as false_alarm
    repo.update_review_status("EVT-A1", "verified")
    repo.update_review_status("EVT-A2", "false_alarm")

    rate, total, false_alarms = repo.get_false_alarm_rate()
    assert total == 2
    assert false_alarms == 1
    assert rate == 0.5


def test_list_alerts_composite_index_fallback() -> None:
    """Verify list_alerts catches composite index errors, logs warning, and sorts in memory."""
    storage = {
        "EVT-1": {"event_uid": "EVT-1", "review_status": "unreviewed", "created_at": "2026-10-11T01:00:00Z"},
        "EVT-2": {"event_uid": "EVT-2", "review_status": "unreviewed", "created_at": "2026-10-11T01:10:00Z"},
    }

    class IndexFailingQuery(MockCollectionReference):
        def stream(self):
            # Simulate Google Cloud FailedPrecondition on indexed stream
            raise Exception("FAILED_PRECONDITION: The query requires an index: https://console.firebase.google.com/...")

    class FallbackCollection(MockCollectionReference):
        def order_by(self, field: str, direction: str = "ASCENDING"):
            return IndexFailingQuery(self.storage)

    class FallbackClient:
        def collection(self, name: str):
            return FallbackCollection(storage)

    repo = FirestoreAlertRepository(collection_name="test_alerts", db_client=FallbackClient())
    results = repo.list_alerts(limit=50, unreviewed_only=True)
    assert len(results) == 2
    # Verify sorted in descending order by created_at in memory fallback
    assert results[0]["event_uid"] == "EVT-2"
    assert results[1]["event_uid"] == "EVT-1"

