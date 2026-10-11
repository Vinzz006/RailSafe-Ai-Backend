"""Unit and integration tests for RailGuard Vision FastAPI Cloud Backend with Firestore & Firebase Auth."""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import pytest
from fastapi.testclient import TestClient

import app.api as api_module
from app.api import app, get_repository
from src.config import FirebaseConfig, SystemConfig


class MockFirestoreRepository:
    """Deterministic in-memory Firestore repository for isolated unit testing."""

    def __init__(self) -> None:
        self.documents: Dict[str, Dict[str, Any]] = {}

    def check_readiness(self) -> bool:
        return True

    def create_or_update_alert(self, alert_data: Dict[str, Any]) -> Tuple[str, bool]:
        event_uid = alert_data.get("event_uid")
        if not event_uid:
            raise ValueError("event_uid required")

        if event_uid in self.documents:
            return event_uid, False

        doc = dict(alert_data)
        doc["id"] = event_uid
        self.documents[event_uid] = doc
        return event_uid, True

    def get_alert(self, alert_id: str) -> Optional[Dict[str, Any]]:
        return self.documents.get(alert_id)

    def list_alerts(
        self,
        limit: int = 50,
        review_status: Optional[str] = None,
        unreviewed_only: bool = False,
    ) -> List[Dict[str, Any]]:
        results = list(self.documents.values())
        if unreviewed_only:
            results = [r for r in results if r.get("review_status") == "unreviewed"]
        elif review_status:
            results = [r for r in results if r.get("review_status") == review_status]
        return results[:limit]

    def update_review_status(
        self,
        alert_id: str,
        review_status: str,
        reviewed_by: Optional[str] = None,
    ) -> bool:
        if alert_id not in self.documents:
            return False
        if review_status not in ("verified", "false_alarm", "unreviewed"):
            raise ValueError("Invalid review_status")
        self.documents[alert_id]["review_status"] = review_status
        if reviewed_by:
            self.documents[alert_id]["reviewed_by"] = reviewed_by
        return True

    def update_cleared_timestamp(
        self,
        camera_id: str,
        track_id: int,
        cleared_timestamp: str,
    ) -> int:
        count = 0
        for doc in self.documents.values():
            if (
                doc.get("camera_id") == camera_id
                and doc.get("track_id") == track_id
                and not doc.get("cleared_timestamp")
            ):
                doc["cleared_timestamp"] = cleared_timestamp
                count += 1
        return count

    def get_false_alarm_rate(self) -> Tuple[Optional[float], int, int]:
        total = 0
        false_alarms = 0
        for doc in self.documents.values():
            st = doc.get("review_status")
            if st in ("verified", "false_alarm"):
                total += 1
                if st == "false_alarm":
                    false_alarms += 1
        if total == 0:
            return None, 0, 0
        return false_alarms / total, total, false_alarms


@pytest.fixture
def test_env() -> Generator[dict, None, None]:
    """Provide isolated test environment with mock Firestore repository and test credentials."""
    mock_repo = MockFirestoreRepository()

    # Configure API module
    api_module.config.firebase.allow_dev_auth_bypass = True
    api_module.config.security.agent_token = "test-agent-token"
    api_module.config.security.api_token = "test-operator-token"

    # Dependency override for Firestore repository
    app.dependency_overrides[get_repository] = lambda: mock_repo

    client = TestClient(app)

    yield {
        "client": client,
        "repo": mock_repo,
        "agent_token": "test-agent-token",
        "operator_token": "test-operator-token",
    }

    app.dependency_overrides.clear()


def test_health_check(test_env: dict) -> None:
    """Verify GET /health executes without requiring /var/data or database connectivity."""
    client: TestClient = test_env["client"]
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["persistence"] == "Firebase Realtime Database"
    assert "FOR HUMAN REVIEW ONLY" in data["disclaimer"]
    # Ensure no internal tokens or secrets are leaked
    body_str = res.text
    assert "test-agent-token" not in body_str
    assert "test-operator-token" not in body_str
    assert "/var/data" not in body_str


def test_root_info_endpoint(test_env: dict) -> None:
    """Verify GET / returns public service discovery information."""
    client: TestClient = test_env["client"]
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["service"] == "RailGuard Vision Cloud Backend"
    assert data["status"] == "online"
    assert data["endpoints"]["health"] == "/health"
    assert data["endpoints"]["readiness"] == "/ready"


def test_readiness_check_healthy_and_unhealthy(test_env: dict) -> None:
    """Verify GET /ready correctly handles healthy and unhealthy Realtime Database dependencies."""
    client: TestClient = test_env["client"]
    repo: MockFirestoreRepository = test_env["repo"]

    # 1. Healthy state -> 200 OK without requiring authentication
    res = client.get("/ready")
    assert res.status_code == 200
    assert res.json()["status"] == "ready"
    assert res.json()["rtdb_connected"] is True
    assert res.json()["firestore_connected"] is True

    # 2. Unhealthy state (simulated Realtime Database disconnection) -> 503 Service Unavailable
    repo.check_readiness = lambda: False
    res_down = client.get("/ready")
    assert res_down.status_code == 503
    assert "unavailable" in res_down.json()["detail"].lower()

    # Restore healthy
    repo.check_readiness = lambda: True


def test_authentication_enforcement(test_env: dict) -> None:
    """Verify rejection of unauthenticated or invalid token requests across operator endpoints."""
    client: TestClient = test_env["client"]

    # 1. No token on operator endpoint -> 401
    res = client.get("/api/alerts")
    assert res.status_code == 401
    assert "Missing authentication token" in res.json()["detail"]

    # 2. Invalid token on operator endpoint -> 401
    res_bad = client.get("/api/alerts", headers={"Authorization": "Bearer bad-token"})
    assert res_bad.status_code == 401

    # 3. Valid operator token (via test bypass/verifier) -> 200
    res_ok = client.get(
        "/api/alerts",
        headers={"Authorization": f"Bearer {test_env['operator_token']}"},
    )
    assert res_ok.status_code == 200
    assert isinstance(res_ok.json(), list)


def test_ingestion_agent_token_enforcement(test_env: dict) -> None:
    """Verify that ingestion strictly requires valid RAILGUARD_AGENT_TOKEN."""
    client: TestClient = test_env["client"]
    correct_token = test_env["agent_token"]

    fake_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 30
    form_data = {
        "event_uid": "EVT-AUTH-TEST-01",
        "timestamp": "2026-10-11T01:00:00Z",
        "camera_id": "CAM-01",
        "track_id": "1",
        "object_class": "luggage",
        "event_duration": "10.0",
        "reason": "Auth test",
    }
    files = {"snapshot": ("snap.jpg", io.BytesIO(fake_jpeg), "image/jpeg")}

    # Case 1: Ingestion with NO token -> 401
    res_no_token = client.post("/api/ingest/alerts", data=form_data, files=files)
    assert res_no_token.status_code == 401
    assert "Invalid camera ingestion agent token" in res_no_token.json()["detail"]

    # Case 2: Ingestion with WRONG token -> 401
    files_wrong = {"snapshot": ("snap.jpg", io.BytesIO(fake_jpeg), "image/jpeg")}
    res_wrong_token = client.post(
        "/api/ingest/alerts",
        data=form_data,
        files=files_wrong,
        headers={"Authorization": "Bearer wrong-agent-token"},
    )
    assert res_wrong_token.status_code == 401

    # Case 3: Ingestion with operator token (instead of agent token) -> 401 (Separation of concerns)
    files_op = {"snapshot": ("snap.jpg", io.BytesIO(fake_jpeg), "image/jpeg")}
    res_op_token = client.post(
        "/api/ingest/alerts",
        data=form_data,
        files=files_op,
        headers={"Authorization": f"Bearer {test_env['operator_token']}"},
    )
    assert res_op_token.status_code == 401

    # Case 4: Ingestion with CORRECT agent token -> 201
    files_ok = {"snapshot": ("snap.jpg", io.BytesIO(fake_jpeg), "image/jpeg")}
    res_ok = client.post(
        "/api/ingest/alerts",
        data=form_data,
        files=files_ok,
        headers={"Authorization": f"Bearer {correct_token}"},
    )
    assert res_ok.status_code == 201



def test_ingest_alert_valid_and_idempotent(test_env: dict) -> None:
    """Verify laptop agent ingestion, Firestore persistence, and idempotency."""
    client: TestClient = test_env["client"]
    agent_token = test_env["agent_token"]
    repo: MockFirestoreRepository = test_env["repo"]

    fake_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb" + b"\x00" * 40
    form_data = {
        "event_uid": "EVT-CAM01-TRK42-20261011T010000Z",
        "timestamp": "2026-10-11T01:00:00Z",
        "camera_id": "CAM-PLATFORM-01",
        "track_id": "42",
        "object_class": "backpack",
        "event_duration": "18.5",
        "reason": "Stationary unattended luggage detected for > 15s",
    }
    files = {
        "snapshot": ("snap.jpg", io.BytesIO(fake_jpeg), "image/jpeg"),
    }
    headers = {"Authorization": f"Bearer {agent_token}"}

    # Step 1: Initial Ingest
    res = client.post("/api/ingest/alerts", data=form_data, files=files, headers=headers)
    assert res.status_code == 201
    data = res.json()
    assert data["status"] == "created"
    assert data["alert_id"] == "EVT-CAM01-TRK42-20261011T010000Z"

    # Verify document exists in Firestore repository
    doc = repo.get_alert("EVT-CAM01-TRK42-20261011T010000Z")
    assert doc is not None
    assert doc["camera_id"] == "CAM-PLATFORM-01"
    assert doc["track_id"] == 42
    assert doc["object_class"] == "backpack"
    assert doc["review_status"] == "unreviewed"

    # Step 2: Idempotent Retry (Same event_uid re-submitted)
    files2 = {"snapshot": ("snap.jpg", io.BytesIO(fake_jpeg), "image/jpeg")}
    res_retry = client.post("/api/ingest/alerts", data=form_data, files=files2, headers=headers)
    assert res_retry.status_code == 200
    retry_data = res_retry.json()
    assert retry_data["status"] == "already_exists"
    assert retry_data["alert_id"] == "EVT-CAM01-TRK42-20261011T010000Z"

    # Verify no duplicate document was created
    assert len(repo.documents) == 1


def test_ingest_rejection_invalid_fields(test_env: dict) -> None:
    """Verify rejection of invalid fields or non-image snapshot file."""
    client: TestClient = test_env["client"]
    agent_token = test_env["agent_token"]
    headers = {"Authorization": f"Bearer {agent_token}"}

    bad_files = {"snapshot": ("script.sh", io.BytesIO(b"echo hack"), "text/plain")}
    bad_data = {
        "event_uid": "EVT-BAD",
        "timestamp": "2026-10-11T01:00:00Z",
        "camera_id": "CAM-01",
        "track_id": "1",
        "object_class": "backpack",
        "event_duration": "5.0",
        "reason": "Test",
    }
    res = client.post("/api/ingest/alerts", data=bad_data, files=bad_files, headers=headers)
    assert res.status_code == 400
    assert "must be a valid image" in res.json()["detail"]


def test_ingest_clear_event(test_env: dict) -> None:
    """Verify POST /api/ingest/alerts/clear records cleared timestamp."""
    client: TestClient = test_env["client"]
    agent_token = test_env["agent_token"]
    repo: MockFirestoreRepository = test_env["repo"]

    # Seed active alert in repo
    repo.create_or_update_alert({
        "event_uid": "EVT-CLR-01",
        "timestamp": "2026-10-11T01:00:00Z",
        "camera_id": "CAM-CLR-01",
        "track_id": 99,
        "object_class": "suitcase",
        "event_duration": 22.0,
        "reason": "Clear test",
        "review_status": "unreviewed",
        "cleared_timestamp": None,
    })

    clear_payload = {
        "camera_id": "CAM-CLR-01",
        "track_id": 99,
        "cleared_timestamp": "2026-10-11T01:05:00Z",
    }
    res = client.post(
        "/api/ingest/alerts/clear",
        json=clear_payload,
        headers={"Authorization": f"Bearer {agent_token}"},
    )
    assert res.status_code == 200
    assert res.json()["rows_updated"] == 1

    updated = repo.get_alert("EVT-CLR-01")
    assert updated is not None
    assert updated["cleared_timestamp"] == "2026-10-11T01:05:00Z"


def test_operator_review_and_stats(test_env: dict) -> None:
    """Verify operator review updates and false alarm rate calculation."""
    client: TestClient = test_env["client"]
    operator_token = test_env["operator_token"]
    repo: MockFirestoreRepository = test_env["repo"]
    headers = {"Authorization": f"Bearer {operator_token}"}

    # Seed two alerts
    repo.create_or_update_alert({
        "event_uid": "EVT-REV-1",
        "timestamp": "2026-10-11T01:00:00Z",
        "camera_id": "CAM-1",
        "track_id": 1,
        "object_class": "backpack",
        "event_duration": 16.0,
        "reason": "Test 1",
        "review_status": "unreviewed",
    })
    repo.create_or_update_alert({
        "event_uid": "EVT-REV-2",
        "timestamp": "2026-10-11T01:01:00Z",
        "camera_id": "CAM-1",
        "track_id": 2,
        "object_class": "handbag",
        "event_duration": 17.0,
        "reason": "Test 2",
        "review_status": "unreviewed",
    })

    # Review 1: verified
    res1 = client.patch(
        "/api/alerts/EVT-REV-1/review",
        json={"review_status": "verified"},
        headers=headers,
    )
    assert res1.status_code == 200
    assert res1.json()["review_status"] == "verified"
    assert res1.json()["reviewed_by"] is not None

    # Review 2: false_alarm
    res2 = client.patch(
        "/api/alerts/EVT-REV-2/review",
        json={"review_status": "false_alarm"},
        headers=headers,
    )
    assert res2.status_code == 200
    assert res2.json()["review_status"] == "false_alarm"

    # Check stats endpoint
    stats_res = client.get("/api/alerts/stats/false-alarm-rate", headers=headers)
    assert stats_res.status_code == 200
    stats = stats_res.json()
    assert stats["total_reviewed"] == 2
    assert stats["false_alarms"] == 1
    assert stats["false_alarm_rate"] == 0.5


def test_evidence_delivery_and_path_traversal_protection(test_env: dict) -> None:
    """Verify evidence streaming and path traversal rejection."""
    client: TestClient = test_env["client"]
    operator_token = test_env["operator_token"]
    repo: MockFirestoreRepository = test_env["repo"]
    headers = {"Authorization": f"Bearer {operator_token}"}

    # Create legitimate snapshot inside safe container directory
    snap_file = api_module.CONTAINER_SNAPSHOTS_DIR / "legit_snap.jpg"
    snap_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 30)

    repo.create_or_update_alert({
        "event_uid": "EVT-SNAP-01",
        "timestamp": "2026-10-11T01:00:00Z",
        "camera_id": "CAM-1",
        "track_id": 50,
        "object_class": "backpack",
        "event_duration": 15.0,
        "reason": "Snap test",
        "snapshot_key": "legit_snap.jpg",
    })

    # Valid fetch
    res = client.get("/api/alerts/EVT-SNAP-01/snapshot", headers=headers)
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"

    # Traversal attack attempt: pointing to file outside container evidence dir
    outside_file = Path(tempfile.gettempdir()) / "outside_secret.txt"
    outside_file.write_text("SENSITIVE")

    repo.create_or_update_alert({
        "event_uid": "EVT-EVIL-01",
        "timestamp": "2026-10-11T01:00:00Z",
        "camera_id": "CAM-1",
        "track_id": 51,
        "object_class": "backpack",
        "event_duration": 15.0,
        "reason": "Evil test",
        "snapshot_key": "../outside_secret.txt",
    })

    res_evil = client.get("/api/alerts/EVT-EVIL-01/snapshot", headers=headers)
    assert res_evil.status_code in (403, 404)


def test_cors_preflight_options(test_env: dict) -> None:
    """Verify OPTIONS preflight from http://localhost:54313 succeeds with CORS headers."""
    client: TestClient = test_env["client"]
    res = client.options(
        "/api/alerts",
        headers={
            "Origin": "http://localhost:54313",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "http://localhost:54313"
    assert "GET" in res.headers.get("access-control-allow-methods", "")


def test_cors_authorized_get_request(test_env: dict) -> None:
    """Verify authorized GET request from permitted origin includes CORS headers."""
    client: TestClient = test_env["client"]
    operator_token = test_env["operator_token"]
    res = client.get(
        "/api/alerts?limit=50&unreviewed_only=true",
        headers={
            "Origin": "http://localhost:54313",
            "Authorization": f"Bearer {operator_token}",
        },
    )
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "http://localhost:54313"


def test_cors_headers_on_401_and_403_responses(test_env: dict) -> None:
    """Verify 401 and 403 responses include CORS headers for permitted origin."""
    client: TestClient = test_env["client"]

    # Missing token -> 401
    res_401 = client.get(
        "/api/alerts",
        headers={"Origin": "http://localhost:54313"},
    )
    assert res_401.status_code == 401
    assert res_401.headers.get("access-control-allow-origin") == "http://localhost:54313"

    # Ingestion endpoint with invalid agent token -> 401
    res_ingest_401 = client.post(
        "/api/ingest/alerts",
        json={"event_uid": "EVT-TEST", "camera_id": "CAM-1", "track_id": 1, "object_class": "bag", "event_duration": 15.0, "reason": "test"},
        headers={"Origin": "http://localhost:54313", "X-Agent-Token": "wrong-agent-token"},
    )
    assert res_ingest_401.status_code == 401
    assert res_ingest_401.headers.get("access-control-allow-origin") == "http://localhost:54313"


def test_cors_headers_on_500_response(test_env: dict) -> None:
    """Verify controlled HTTP 500 includes CORS headers and does not mask errors as empty lists."""
    client: TestClient = test_env["client"]
    operator_token = test_env["operator_token"]
    repo: MockFirestoreRepository = test_env["repo"]

    # Force repository to raise unexpected database error
    def failing_list_alerts(**kwargs):
        raise RuntimeError("Unexpected Firestore connection failure")

    repo.list_alerts = failing_list_alerts

    res = client.get(
        "/api/alerts?limit=50",
        headers={
            "Origin": "http://localhost:54313",
            "Authorization": f"Bearer {operator_token}",
        },
    )
    # Must NOT mask exception as an empty alert list
    assert res.status_code == 500
    assert res.headers.get("access-control-allow-origin") == "http://localhost:54313"
    assert "Failed to retrieve alerts" in res.json()["detail"]


def test_cors_unapproved_origin_rejected(test_env: dict) -> None:
    """Verify unapproved origin does NOT receive CORS access headers."""
    client: TestClient = test_env["client"]
    operator_token = test_env["operator_token"]
    unapproved_origin = "http://unapproved-hacker-site.com"

    res = client.get(
        "/api/alerts",
        headers={
            "Origin": unapproved_origin,
            "Authorization": f"Bearer {operator_token}",
        },
    )
    assert res.headers.get("access-control-allow-origin") != unapproved_origin


def test_alerts_listing_query_parameters(test_env: dict) -> None:
    """Verify alert listing handles limit=50, unreviewed_only, and status filters correctly."""
    client: TestClient = test_env["client"]
    operator_token = test_env["operator_token"]
    repo: MockFirestoreRepository = test_env["repo"]
    headers = {"Authorization": f"Bearer {operator_token}"}

    # Populate 3 alerts with different review statuses
    for i, st in enumerate(["unreviewed", "verified", "false_alarm"]):
        repo.create_or_update_alert({
            "event_uid": f"EVT-PARAM-{i}",
            "timestamp": "2026-10-11T01:00:00Z",
            "camera_id": "CAM-PARAM",
            "track_id": i,
            "object_class": "luggage",
            "event_duration": 15.0 + i,
            "reason": f"Param test {i}",
            "review_status": st,
        })

    # Test limit=50
    res_limit = client.get("/api/alerts?limit=50", headers=headers)
    assert res_limit.status_code == 200
    assert len(res_limit.json()) == 3

    # Test unreviewed_only=true
    res_unreviewed = client.get("/api/alerts?limit=50&unreviewed_only=true", headers=headers)
    assert res_unreviewed.status_code == 200
    assert len(res_unreviewed.json()) == 1
    assert res_unreviewed.json()[0]["review_status"] == "unreviewed"

    # Test status=verified
    res_verified = client.get("/api/alerts?limit=50&status=verified", headers=headers)
    assert res_verified.status_code == 200
    assert len(res_verified.json()) == 1
    assert res_verified.json()[0]["review_status"] == "verified"

    # Test status=all
    res_all = client.get("/api/alerts?limit=50&status=all", headers=headers)
    assert res_all.status_code == 200
    assert len(res_all.json()) == 3

    # Test invalid status returns 400 Bad Request
    res_invalid = client.get("/api/alerts?limit=50&status=bad_status", headers=headers)
    assert res_invalid.status_code == 400

