"""Configuration loader and schema definitions for RailGuard Vision."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class AppConfig:
    name: str = "RailGuard Vision"
    version: str = "1.0.0"
    institution: str = "Indian Railways Surveillance & Security Assistance Prototype"
    disclaimer: str = (
        "FOR HUMAN REVIEW ONLY — Does not detect explosives, chemical substances, or narcotics."
    )


@dataclass
class VideoConfig:
    source: Union[int, str] = 0
    camera_id: str = "CAM-PLATFORM-01"
    target_width: int = 640
    target_height: int = 480
    frame_skip: int = 1
    buffer_seconds: float = 10.0


@dataclass
class ModelConfig:
    weights: str = "yolov8n.pt"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    device: str = "cpu"
    input_size: int = 640


@dataclass
class RulesConfig:
    stationary_pixels: float = 18.0
    stationary_window_seconds: float = 5.0
    proximity_pixels: float = 140.0
    warning_seconds: float = 4.0
    unattended_seconds: float = 15.0
    track_drop_grace_seconds: float = 2.5
    owner_left_grace_seconds: float = 8.0
    loiter_seconds: float = 90.0
    restricted_zone_enabled: bool = False
    restricted_zone_polygon: List[List[int]] = field(
        default_factory=lambda: [[20, 20], [300, 20], [300, 200], [20, 200]]
    )
    bag_classes: List[str] = field(
        default_factory=lambda: ["backpack", "handbag", "suitcase"]
    )
    person_class: str = "person"


@dataclass
class StorageConfig:
    db_path: str = "data/security_alerts.db"
    alerts_dir: str = "data/alerts"
    clips_dir: str = "data/clips"
    pre_alert_seconds: float = 5.0
    post_alert_seconds: float = 5.0
    data_dir: Optional[str] = None


@dataclass
class SecurityConfig:
    api_token: str = "dev-railguard-token-change-in-production"
    agent_token: str = "dev-agent-token-change-in-production"
    max_upload_bytes: int = 30 * 1024 * 1024  # 30 MB
    rate_limit_per_minute: int = 120
    cors_allowed_origins: List[str] = field(
        default_factory=lambda: [
            "http://localhost:54313",
            "http://127.0.0.1:54313",
            "http://localhost:3000",
            "http://localhost:8000",
        ]
    )


@dataclass
class FirebaseConfig:
    project_id: str = "railsafe-ai-92d9f"
    database_url: str = (
        "https://railsafe-ai-92d9f-default-rtdb.asia-southeast1.firebasedatabase.app/"
    )
    rtdb_alerts_path: str = "railguard/alerts"
    rtdb_event_uid_index_path: str = "railguard/event_uid_index"
    service_account_json: Optional[str] = None
    service_account_path: Optional[str] = None
    alerts_collection: str = "railguard_alerts"
    storage_bucket: Optional[str] = "railsafe-ai-92d9f.firebasestorage.app"
    allow_dev_auth_bypass: bool = False


@dataclass
class EvaluationConfig:
    ground_truth_path: str = "data/ground_truth/ground_truth_template.csv"
    time_tolerance_seconds: float = 4.0


@dataclass
class SystemConfig:
    app: AppConfig = field(default_factory=AppConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    rules: RulesConfig = field(default_factory=RulesConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    firebase: FirebaseConfig = field(default_factory=FirebaseConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        """Validate config parameters for physical and numerical sanity."""
        if self.video.target_width <= 0 or self.video.target_height <= 0:
            raise ValueError(f"Invalid video dimensions: {self.video.target_width}x{self.video.target_height}")
        if self.video.frame_skip < 1:
            raise ValueError(f"frame_skip must be >= 1, got {self.video.frame_skip}")
        if not (0.0 < self.model.confidence_threshold <= 1.0):
            raise ValueError(f"confidence_threshold must be in (0, 1], got {self.model.confidence_threshold}")
        if self.rules.stationary_pixels <= 0:
            raise ValueError("stationary_pixels must be > 0")
        if self.rules.proximity_pixels <= 0:
            raise ValueError("proximity_pixels must be > 0")
        if self.rules.warning_seconds >= self.rules.unattended_seconds:
            raise ValueError(
                f"warning_seconds ({self.rules.warning_seconds}) must be < unattended_seconds ({self.rules.unattended_seconds})"
            )


def find_default_config_path() -> Optional[Path]:
    """Locate config.yaml across expected relative paths."""
    env_config = os.getenv("RAILGUARD_CONFIG")
    if env_config:
        p = Path(env_config)
        if p.is_file():
            return p.resolve()

    candidates = [
        Path("configs/config.yaml"),
        Path("RailGuard_Vision/configs/config.yaml"),
        Path(__file__).resolve().parent.parent / "configs" / "config.yaml",
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


def load_config(config_path: Optional[Union[str, Path]] = None) -> SystemConfig:
    """Load system configuration from YAML file with environment variable overrides."""
    resolved_path: Optional[Path] = None
    if config_path is not None:
        p = Path(config_path)
        if p.is_file():
            resolved_path = p.resolve()
    else:
        resolved_path = find_default_config_path()

    raw: Dict[str, Any] = {}
    if resolved_path and resolved_path.is_file():
        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
                if isinstance(content, dict):
                    raw = content
        except Exception:
            raw = {}

    app_data = raw.get("app", {})
    video_data = raw.get("video", {})
    model_data = raw.get("model", {})
    rules_data = raw.get("rules", {})
    storage_data = raw.get("storage", {})
    security_data = raw.get("security", {})
    eval_data = raw.get("evaluation", {})

    # Environment variable overrides
    env_data_dir = os.getenv("RAILGUARD_DATA_DIR")
    if env_data_dir:
        storage_data["data_dir"] = env_data_dir
        # Standardize default subpaths under persistent data_dir unless specific env vars set
        if not os.getenv("RAILGUARD_DB_PATH"):
            storage_data["db_path"] = str(Path(env_data_dir) / "security_alerts.db")
        if not os.getenv("RAILGUARD_ALERTS_DIR"):
            storage_data["alerts_dir"] = str(Path(env_data_dir) / "alerts")
        if not os.getenv("RAILGUARD_CLIPS_DIR"):
            storage_data["clips_dir"] = str(Path(env_data_dir) / "clips")

    env_source = os.getenv("RAILGUARD_VIDEO_SOURCE")
    if env_source is not None:
        if env_source.isdigit():
            video_data["source"] = int(env_source)
        else:
            video_data["source"] = env_source

    env_cam_id = os.getenv("RAILGUARD_CAMERA_ID")
    if env_cam_id:
        video_data["camera_id"] = env_cam_id

    env_weights = os.getenv("RAILGUARD_MODEL_WEIGHTS")
    if env_weights:
        model_data["weights"] = env_weights

    env_db = os.getenv("RAILGUARD_DB_PATH")
    if env_db:
        storage_data["db_path"] = env_db

    env_alerts_dir = os.getenv("RAILGUARD_ALERTS_DIR")
    if env_alerts_dir:
        storage_data["alerts_dir"] = env_alerts_dir

    env_clips_dir = os.getenv("RAILGUARD_CLIPS_DIR")
    if env_clips_dir:
        storage_data["clips_dir"] = env_clips_dir

    env_api_token = os.getenv("RAILGUARD_API_TOKEN")
    if env_api_token:
        security_data["api_token"] = env_api_token

    env_agent_token = os.getenv("RAILGUARD_AGENT_TOKEN")
    if env_agent_token:
        security_data["agent_token"] = env_agent_token

    env_cors = os.getenv("CORS_ALLOWED_ORIGINS")
    if env_cors:
        origins = []
        for item in env_cors.split(","):
            cleaned = item.strip().rstrip("/")
            if cleaned and cleaned not in origins:
                origins.append(cleaned)
        # Ensure development origin is included unless explicitly restricted
        if "http://localhost:54313" not in origins:
            origins.append("http://localhost:54313")
        security_data["cors_allowed_origins"] = origins

    # Firebase environment variable overrides
    firebase_data = raw.get("firebase", {})
    env_fb_project = os.getenv("FIREBASE_PROJECT_ID")
    if env_fb_project:
        firebase_data["project_id"] = env_fb_project

    env_fb_db_url = os.getenv("FIREBASE_DATABASE_URL")
    if env_fb_db_url:
        firebase_data["database_url"] = env_fb_db_url.strip()

    env_fb_sa_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    if env_fb_sa_json:
        firebase_data["service_account_json"] = env_fb_sa_json

    env_fb_sa_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
    if env_fb_sa_path:
        firebase_data["service_account_path"] = env_fb_sa_path

    env_fb_collection = os.getenv("FIRESTORE_ALERTS_COLLECTION")
    if env_fb_collection:
        firebase_data["alerts_collection"] = env_fb_collection

    env_fb_bucket = os.getenv("FIREBASE_STORAGE_BUCKET")
    if env_fb_bucket:
        firebase_data["storage_bucket"] = env_fb_bucket

    env_fb_bypass = os.getenv("ALLOW_DEV_AUTH_BYPASS")
    if env_fb_bypass is not None:
        firebase_data["allow_dev_auth_bypass"] = env_fb_bypass.lower() in ("true", "1", "yes")

    # Video source type conversion
    src_val = video_data.get("source", 0)
    if isinstance(src_val, str) and src_val.isdigit():
        video_data["source"] = int(src_val)

    cfg = SystemConfig(
        app=AppConfig(**app_data),
        video=VideoConfig(**video_data),
        model=ModelConfig(**model_data),
        rules=RulesConfig(**rules_data),
        storage=StorageConfig(**storage_data),
        security=SecurityConfig(**security_data),
        firebase=FirebaseConfig(**firebase_data),
        evaluation=EvaluationConfig(**eval_data),
    )
    cfg.validate()
    return cfg
