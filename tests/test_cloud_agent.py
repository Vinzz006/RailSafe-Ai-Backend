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
