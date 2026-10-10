"""Integrated video processing pipeline connecting capture, detection, tracking, rules, and alerts."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.alerts import AlertManager
from src.capture import VideoCaptureManager
from src.config import SystemConfig
from src.db import AlertDatabaseManager
from src.detector import YOLODetector
from src.rules import AlertEvent, ClearedAlertEvent, UnattendedBagRulesEngine
from src.tracker import TrackManager, TrackState


@dataclass
class PipelineFrameResult:
    """Encapsulates all processing results for a single processed frame."""

    annotated_frame: np.ndarray
    raw_frame: np.ndarray
    timestamp: float
    fps: float
    frame_index: int
    tracks: List[TrackState] = field(default_factory=list)
    active_alerts: List[AlertEvent] = field(default_factory=list)
    cleared_alerts: List[ClearedAlertEvent] = field(default_factory=list)
    new_db_alert_ids: List[int] = field(default_factory=list)
    is_eof: bool = False
    error_message: Optional[str] = None


class RailGuardPipeline:
    """Master video processing pipeline for RailGuard Vision."""

    def __init__(self, config: SystemConfig) -> None:
        self.config = config
        self.capture = VideoCaptureManager(config.video)
        self.detector = YOLODetector(config.model)
        self.tracker = TrackManager(
            track_drop_grace_seconds=config.rules.track_drop_grace_seconds
        )
        self.rules = UnattendedBagRulesEngine(
            config=config.rules,
            camera_id=config.video.camera_id,
        )
        self.db = AlertDatabaseManager(config.storage.db_path)
        self.alerts = AlertManager(config.storage, self.db)

        # Performance monitoring
        self._prev_time: float = time.time()
        self.current_fps: float = 0.0

    def start(self) -> bool:
        """Initialize detector and open video source."""
        success = self.capture.open()
        if not success:
            return False
        self.detector.load_model()
        self.rules.reset()
        self._prev_time = time.time()
        return True

    def step(self) -> PipelineFrameResult:
        """Process one frame through the entire screening pipeline."""
        ret, frame, timestamp = self.capture.read()
        if not ret or frame is None:
            return PipelineFrameResult(
                annotated_frame=np.zeros((480, 640, 3), dtype=np.uint8),
                raw_frame=np.zeros((480, 640, 3), dtype=np.uint8),
                timestamp=timestamp,
                fps=self.current_fps,
                frame_index=self.capture.frame_idx,
                is_eof=self.capture.is_eof,
                error_message=self.capture.error_message,
            )

        # Calculate instantaneous FPS
        now = time.time()
        dt = now - self._prev_time
        self._prev_time = now
        if dt > 0:
            instant_fps = 1.0 / dt
            self.current_fps = 0.85 * self.current_fps + 0.15 * instant_fps if self.current_fps > 0 else instant_fps

        # 1. Detection & ByteTrack association
        detections = self.detector.track(frame)

        # 2. Update track history and owner association
        active_tracks = self.tracker.update_tracks(
            detections=detections,
            timestamp=timestamp,
            stationary_threshold=self.config.rules.stationary_pixels,
            proximity_threshold=self.config.rules.proximity_pixels,
        )

        # 3. Evaluate unattended bag conditions
        active_alerts, cleared_alerts = self.rules.evaluate(
            tracks=active_tracks,
            current_time=timestamp,
        )

        # 4. Handle alert persistence and evidence recording
        new_db_ids = self.alerts.process_alerts(
            active_alerts=active_alerts,
            cleared_alerts=cleared_alerts,
            current_frame=frame,
            capture_manager=self.capture,
        )

        # 5. Render overlays and safety telemetry
        annotated = self.tracker.draw_annotations(
            frame=frame,
            tracks=active_tracks,
            proximity_threshold=self.config.rules.proximity_pixels,
        )

        # Draw mandatory top HUD header banner
        hud_h = 36
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], hud_h), (25, 25, 25), -1)
        hud_text = (
            f"RAILGUARD VISION | {self.config.video.camera_id} | FPS: {self.current_fps:.1f} | "
            f"Tracks: {len(active_tracks)} | FOR HUMAN REVIEW ONLY"
        )
        cv2.putText(
            annotated,
            hud_text,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # If active alerts exist, draw high-visibility alert banner at bottom
        if active_alerts:
            alert_bar_y = annotated.shape[0] - 28
            cv2.rectangle(
                annotated,
                (0, alert_bar_y),
                (annotated.shape[1], annotated.shape[0]),
                (0, 0, 200),
                -1,
            )
            alert_text = (
                f"⚠️ ATTENTION: {len(active_alerts)} UNATTENDED OBJECT ALERT(S) ACTIVE — "
                "FOR HUMAN REVIEW ONLY. VERIFY PHYSICAL CONTEXT."
            )
            cv2.putText(
                annotated,
                alert_text,
                (10, annotated.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return PipelineFrameResult(
            annotated_frame=annotated,
            raw_frame=frame,
            timestamp=timestamp,
            fps=self.current_fps,
            frame_index=self.capture.frame_idx,
            tracks=active_tracks,
            active_alerts=active_alerts,
            cleared_alerts=cleared_alerts,
            new_db_alert_ids=new_db_ids,
            is_eof=False,
            error_message=None,
        )

    def stop(self) -> None:
        """Release video capture device and flush buffers."""
        self.capture.release()
