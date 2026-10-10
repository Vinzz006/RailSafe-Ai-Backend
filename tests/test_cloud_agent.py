"""Unit tests for the laptop edge camera agent and asynchronous cloud forwarder."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent import CloudForwarder, IngestTask
from src.db import AlertRecord


def test_cloud_forwarder_queue_and_upload() -> None:
    """Verify that CloudForwarder enqueues alerts and sends authenticated HTTP requests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        snap_file = Path(tmpdir) / "test_snap.jpg"
        snap_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 10)

        record = AlertRecord(
            id=1,
            timestamp="2026-10-11T01:00:00Z",
            camera_id="CAM-AGENT-01",
            track_id=77,
            object_class="backpack",
            event_duration=16.0,
            reason="Unattended bag test",
            snapshot_path=str(snap_file),
            clip_path=None,
            review_status="unreviewed",
        )

        forwarder = CloudForwarder(
            cloud_url="https://mock-render-api.railguard.local",
            agent_token="test-secret-agent-token",
            max_queue_size=10,
        )

        with patch("httpx.Client.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 201
            mock_post.return_value = mock_resp

            success = forwarder._upload_alert(record, "UID-TEST-77")
            assert success is True
            assert mock_post.called

            # Check that authorization header and multipart files were sent
            call_kwargs = mock_post.call_args[1]
            assert "headers" in call_kwargs
            assert call_kwargs["headers"]["Authorization"] == "Bearer test-secret-agent-token"
            assert "data" in call_kwargs
            assert call_kwargs["data"]["event_uid"] == "UID-TEST-77"
            assert "files" in call_kwargs
            assert "snapshot" in call_kwargs["files"]


def test_cloud_forwarder_clear_upload() -> None:
    """Verify clear notification upload request."""
    forwarder = CloudForwarder(
        cloud_url="https://mock-render-api.railguard.local",
        agent_token="test-agent-token",
    )

    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        success = forwarder._upload_clear("CAM-01", 12, "2026-10-11T01:05:00Z")
        assert success is True
        assert mock_post.called
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["camera_id"] == "CAM-01"
        assert call_kwargs["json"]["track_id"] == 12


def test_cloud_forwarder_queue_overflow_protection() -> None:
    """Verify that bounded queue prevents unbounded memory growth when offline."""
    forwarder = CloudForwarder(
        cloud_url="https://offline.local",
        agent_token="dummy",
        max_queue_size=2,  # Tiny queue to test overflow
    )

    record = AlertRecord(
        id=1,
        timestamp="2026-10-11T01:00:00Z",
        camera_id="CAM-1",
        track_id=1,
        object_class="backpack",
        event_duration=10.0,
        reason="Test",
        snapshot_path="dummy.jpg",
        clip_path=None,
        review_status="unreviewed",
    )

    # First two should succeed
    assert forwarder.enqueue_alert(record, "UID-1") is True
    assert forwarder.enqueue_alert(record, "UID-2") is True
    # Third should be dropped safely rather than crashing or growing infinitely
    assert forwarder.enqueue_alert(record, "UID-3") is False


def test_upload_rejection_auth_failure_not_marked_success() -> None:
    """Verify that HTTP 401 rejection is not treated as success and aborts non-retryable loops."""
    with tempfile.TemporaryDirectory() as tmpdir:
        snap_file = Path(tmpdir) / "test_snap.jpg"
        snap_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 10)

        record = AlertRecord(
            id=1,
            timestamp="2026-10-11T01:00:00Z",
            camera_id="CAM-01",
            track_id=10,
            object_class="suitcase",
            event_duration=15.0,
            reason="Auth failure test",
            snapshot_path=str(snap_file),
            clip_path=None,
            review_status="unreviewed",
        )

        forwarder = CloudForwarder(
            cloud_url="https://mock-render-api.railguard.local",
            agent_token="wrong-token",
        )

        task = IngestTask(task_type="alert", alert_record=record, event_uid="UID-AUTH-FAIL")
        with patch("httpx.Client.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 401
            mock_resp.text = "Unauthorized: Invalid camera ingestion agent token."
            mock_post.return_value = mock_resp

            success = forwarder._upload_alert(record, "UID-AUTH-FAIL", task=task)
            assert success is False
            assert task.non_retryable is True  # Must not retry infinitely on auth failure


def test_upload_aborted_when_agent_token_missing() -> None:
    """Verify that an empty or missing token aborts upload cleanly without sending requests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        snap_file = Path(tmpdir) / "test_snap.jpg"
        snap_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 10)

        record = AlertRecord(
            id=1,
            timestamp="2026-10-11T01:00:00Z",
            camera_id="CAM-01",
            track_id=10,
            object_class="suitcase",
            event_duration=15.0,
            reason="Missing token test",
            snapshot_path=str(snap_file),
            clip_path=None,
            review_status="unreviewed",
        )

        forwarder = CloudForwarder(
            cloud_url="https://mock-render-api.railguard.local",
            agent_token="",
        )

        task = IngestTask(task_type="alert", alert_record=record, event_uid="UID-NO-TOKEN")
        with patch("httpx.Client.post") as mock_post:
            success = forwarder._upload_alert(record, "UID-NO-TOKEN", task=task)
            assert success is False
            assert task.non_retryable is True
            assert not mock_post.called  # Should abort before network call


def test_agent_precedence_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CLI argument > Environment variable > Default precedence rules."""
    from src.agent import LocalDetectionAgent

    # 1. Environment variable set
    monkeypatch.setenv("RAILGUARD_CLOUD_URL", "https://env-url.onrender.com")
    monkeypatch.setenv("RAILGUARD_AGENT_TOKEN", "env-secret-token")

    # When no explicit args passed, env var wins
    agent1 = LocalDetectionAgent()
    assert agent1.cloud_url == "https://env-url.onrender.com"
    assert agent1.agent_token == "env-secret-token"

    # When explicit CLI parameters passed, CLI parameter overrides env var
    agent2 = LocalDetectionAgent(
        cloud_url="https://cli-override.onrender.com",
        agent_token="cli-override-token",
    )
    assert agent2.cloud_url == "https://cli-override.onrender.com"
    assert agent2.agent_token == "cli-override-token"


def test_run_agent_synthetic_ingest_test() -> None:
    """Verify synthetic ingest test runner produces valid requests and validates auth."""
    from run_agent import run_synthetic_ingest_test

    # 1. Missing token -> Fails without network request
    assert run_synthetic_ingest_test("https://test.local", agent_token=None) is False

    # 2. Valid token with mock 201 response -> Succeeds
    with patch("httpx.Client.post") as mock_post:
        resp_alert = MagicMock(status_code=201, text='{"status":"created"}')
        resp_clear = MagicMock(status_code=200, text='{"status":"success"}')
        mock_post.side_effect = [resp_alert, resp_clear]

        res = run_synthetic_ingest_test("https://test.local", agent_token="valid-secret")
        assert res is True
        assert mock_post.call_count == 2
        # Verify Authorization header attached to both calls
        first_call_headers = mock_post.call_args_list[0][1]["headers"]
        second_call_headers = mock_post.call_args_list[1][1]["headers"]
        assert first_call_headers["Authorization"] == "Bearer valid-secret"
        assert second_call_headers["Authorization"] == "Bearer valid-secret"

