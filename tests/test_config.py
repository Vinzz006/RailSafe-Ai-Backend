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

    cfg = load_config()
    assert cfg.video.source == "rtsp://secure-camera.local/feed"
    assert cfg.video.camera_id == "CAM-ENV-01"
    assert cfg.model.weights == "yolov8m.pt"
    assert cfg.storage.db_path == "custom_alerts.db"
