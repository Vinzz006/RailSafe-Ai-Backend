"""Unit tests for Firebase Admin SDK service module (initialization, auth, evidence storage)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock
import pytest

from src.config import FirebaseConfig
from src.firebase_service import (
    download_evidence_blob,
    get_firebase_app,
    get_realtime_db,
    reset_firebase_app,
    upload_evidence_blob,
    verify_firebase_id_token,
)


def test_verify_empty_token_raises_value_error() -> None:
    """Verify that empty, None, or whitespace tokens are rejected."""
    with pytest.raises(ValueError, match="empty or missing"):
        verify_firebase_id_token("")

    with pytest.raises(ValueError, match="empty or missing"):
        verify_firebase_id_token("   ")


def test_dev_bypass_token_allowed_in_dev_rejected_in_prod(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify dev bypass works in non-production when enabled, but is strictly rejected in production."""
    # 1. In development with allow_dev_bypass=True
    monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)

    claims = verify_firebase_id_token("DEMO_OFFICER_TOKEN", allow_dev_bypass=True)
    assert claims["role"] == "operator"
    assert claims["uid"] == "dev-officer-uid"

    # 2. In production (RENDER=true), bypass must be rejected even if allow_dev_bypass=True
    monkeypatch.setenv("RENDER", "true")
    with pytest.raises(ValueError):
        verify_firebase_id_token("DEMO_OFFICER_TOKEN", allow_dev_bypass=True)

    # 3. In production with ENVIRONMENT=production
    monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "production")
    with pytest.raises(ValueError):
        verify_firebase_id_token("dev-test-token", allow_dev_bypass=True)


def test_evidence_blob_path_traversal_rejection() -> None:
    """Verify upload and download reject paths containing directory traversal or leading separators."""
    # Relative path traversal
    assert upload_evidence_blob("../malicious.jpg", b"fake", config=FirebaseConfig()) is False
    assert download_evidence_blob("../malicious.jpg", config=FirebaseConfig()) is None

    # Absolute path attempts
    assert upload_evidence_blob("/etc/passwd", b"fake", config=FirebaseConfig()) is False
    assert download_evidence_blob("/var/data/alert.jpg", config=FirebaseConfig()) is None
    assert upload_evidence_blob("\\windows\\win.ini", b"fake", config=FirebaseConfig()) is False


def test_evidence_blob_upload_and_download_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify successful upload and download using a mock Firebase Storage bucket."""
    mock_blob = MagicMock()
    mock_blob.exists.return_value = True
    mock_blob.download_as_bytes.return_value = b"\xff\xd8\xff\xe0JPEG_DATA"

    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob

    monkeypatch.setattr("src.firebase_service.get_storage_bucket", lambda *args, **kwargs: mock_bucket)

    # Test upload
    uploaded = upload_evidence_blob("alerts/snap_CAM01_EVT01.jpg", b"\xff\xd8\xff\xe0JPEG_DATA")
    assert uploaded is True
    mock_bucket.blob.assert_called_with("alerts/snap_CAM01_EVT01.jpg")
    mock_blob.upload_from_string.assert_called_once()

    # Test download
    downloaded = download_evidence_blob("alerts/snap_CAM01_EVT01.jpg")
    assert downloaded == b"\xff\xd8\xff\xe0JPEG_DATA"
    mock_blob.download_as_bytes.assert_called_once()


def test_evidence_download_missing_blob(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify download returns None when the blob does not exist in the bucket."""
    mock_blob = MagicMock()
    mock_blob.exists.return_value = False

    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob

    monkeypatch.setattr("src.firebase_service.get_storage_bucket", lambda *args, **kwargs: mock_bucket)

    assert download_evidence_blob("alerts/non_existent.jpg") is None


def test_firebase_app_invalid_sa_json() -> None:
    """Verify invalid JSON in FIREBASE_SERVICE_ACCOUNT_JSON raises ValueError."""
    reset_firebase_app()
    bad_config = FirebaseConfig(service_account_json="not-a-valid-json")
    with pytest.raises(ValueError, match="Invalid FIREBASE_SERVICE_ACCOUNT_JSON"):
        get_firebase_app(bad_config)
    reset_firebase_app()


def test_firebase_app_missing_sa_path() -> None:
    """Verify missing file path raises FileNotFoundError."""
    reset_firebase_app()
    bad_config = FirebaseConfig(service_account_path="non_existent_credentials_file.json")
    with pytest.raises(FileNotFoundError):
        get_firebase_app(bad_config)
    reset_firebase_app()
