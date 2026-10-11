"""Unit tests for RailGuard Vision configuration loader and validation."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.config import SystemConfig, load_config


def test_load_default_config() -> None:
    """Test loading configuration with default YAML or fallbacks."""
    cfg = load_config()
    assert isinstance(cfg, SystemConfig)
    assert cfg.app.name == "RailGuard Vision"
    assert "FOR HUMAN REVIEW ONLY" in cfg.app.disclaimer
    assert cfg.video.target_width > 0
    assert cfg.video.target_height > 0
    assert cfg.rules.stationary_pixels == 18.0
    assert cfg.rules.proximity_pixels == 140.0
    assert cfg.rules.warning_seconds < cfg.rules.unattended_seconds


def test_custom_yaml_loading() -> None:
    """Test loading from a custom temporary YAML configuration."""
    custom_data = {
        "video": {
            "source": "data/videos/custom_test.mp4",
            "camera_id": "CAM-TEST-99",
            "target_width": 1280,
            "target_height": 720,
        },
        "rules": {
            "stationary_pixels": 25.0,
            "warning_seconds": 3.0,
            "unattended_seconds": 10.0,
        },
    }

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(custom_data, tmp)
        tmp_path = tmp.name

    try:
        cfg = load_config(tmp_path)
        assert cfg.video.source == "data/videos/custom_test.mp4"
        assert cfg.video.camera_id == "CAM-TEST-99"
        assert cfg.video.target_width == 1280
        assert cfg.rules.stationary_pixels == 25.0
        assert cfg.rules.warning_seconds == 3.0
        assert cfg.rules.unattended_seconds == 10.0
    finally:
        os.remove(tmp_path)


def test_validation_errors() -> None:
    """Test that invalid config parameters raise ValueError."""
    # Invalid resolution
    cfg = SystemConfig()
    cfg.video.target_width = -100
    with pytest.raises(ValueError, match="Invalid video dimensions"):
        cfg.validate()

    # Invalid warning vs unattended relation
    cfg2 = SystemConfig()
    cfg2.rules.warning_seconds = 20.0
    cfg2.rules.unattended_seconds = 10.0
    with pytest.raises(ValueError, match="warning_seconds.*must be < unattended_seconds"):
        cfg2.validate()

    # Invalid confidence threshold
    cfg3 = SystemConfig()
    cfg3.model.confidence_threshold = 1.5
    with pytest.raises(ValueError, match="confidence_threshold"):
        cfg3.validate()


def test_env_var_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test environment variable overrides for sensitive or deployment parameters."""
    monkeypatch.setenv("RAILGUARD_VIDEO_SOURCE", "rtsp://secure-camera.local/feed")
    monkeypatch.setenv("RAILGUARD_CAMERA_ID", "CAM-ENV-01")
    monkeypatch.setenv("RAILGUARD_MODEL_WEIGHTS", "yolov8m.pt")
    monkeypatch.setenv("RAILGUARD_DB_PATH", "custom_alerts.db")
    monkeypatch.setenv("RTDB_ALERTS_PATH", "custom_railguard/custom_alerts")
    monkeypatch.setenv("RTDB_EVENT_UID_INDEX_PATH", "custom_railguard/custom_index")

    cfg = load_config()
    assert cfg.video.source == "rtsp://secure-camera.local/feed"
    assert cfg.video.camera_id == "CAM-ENV-01"
    assert cfg.model.weights == "yolov8m.pt"
    assert cfg.storage.db_path == "custom_alerts.db"
    assert cfg.firebase.rtdb_alerts_path == "custom_railguard/custom_alerts"
    assert cfg.firebase.rtdb_event_uid_index_path == "custom_railguard/custom_index"


def test_cors_origin_parsing_and_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test CORS origin parsing, trailing slash trimming, and production isolation."""
    # 1. In production (RENDER=true), do NOT automatically append localhost
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://railsafe-ai.web.app/ , https://operator.railsafe.in")

    cfg_prod = load_config()
    assert "https://railsafe-ai.web.app" in cfg_prod.security.cors_allowed_origins
    assert "https://operator.railsafe.in" in cfg_prod.security.cors_allowed_origins
    assert "http://localhost:54313" not in cfg_prod.security.cors_allowed_origins

    # 2. In local dev (no RENDER), default origins are provided when empty
    monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "")

    cfg_dev = load_config()
    assert "http://localhost:54313" in cfg_dev.security.cors_allowed_origins


def test_production_auth_bypass_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure development auth bypass is strictly disabled when RENDER or ENVIRONMENT=production is set."""
    # Attempt to enable bypass on Render
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setenv("ALLOW_DEV_AUTH_BYPASS", "true")

    cfg = load_config()
    assert cfg.firebase.allow_dev_auth_bypass is False

    # In local development, bypass can be enabled
    monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.setenv("ALLOW_DEV_AUTH_BYPASS", "true")

    cfg_dev = load_config()
    assert cfg_dev.firebase.allow_dev_auth_bypass is True
