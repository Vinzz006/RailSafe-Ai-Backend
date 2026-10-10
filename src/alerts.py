"""Alert evidence collection (snapshots, video clips) and SQLite integration for RailGuard Vision."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from src.capture import VideoCaptureManager
from src.config import StorageConfig
from src.db import AlertDatabaseManager
from src.rules import AlertEvent, ClearedAlertEvent


class AlertManager:
    """Orchestrates evidence capture (snapshots & clips) and prevents duplicate database events."""

    def __init__(
        self,
        config: StorageConfig,
        db_manager: AlertDatabaseManager,
    ) -> None:
        self.config = config
        self.db = db_manager

        self.alerts_dir = Path(config.alerts_dir).resolve()
        self.clips_dir = Path(config.clips_dir).resolve()
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)

        # Tracks currently in active ALERT state to prevent duplicate DB writes
        self._active_alert_ids: Set[int] = set()

    def is_alert_active(self, track_id: int) -> bool:
        return track_id in self._active_alert_ids

    def save_snapshot(
        self, frame: np.ndarray, camera_id: str, track_id: int, timestamp: float
    ) -> Optional[str]:
        """Save high-resolution evidence snapshot to disk."""
        try:
            ts_str = datetime.datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{camera_id}_trk{track_id}_{ts_str}.jpg"
            filepath = self.alerts_dir / filename
            success = cv2.imwrite(str(filepath), frame)
            if success:
                return str(filepath)
            return None
        except Exception as e:
            print(f"[ALERT_MANAGER] Snapshot save failure: {e}")
            return None

    def save_clip(
        self,
        capture_manager: VideoCaptureManager,
        camera_id: str,
        track_id: int,
        alert_timestamp: float,
    ) -> Optional[str]:
        """Synthesize and save an MP4 evidence clip using the rolling frame buffer."""
        start_time = alert_timestamp - self.config.pre_alert_seconds
        end_time = alert_timestamp + self.config.post_alert_seconds
        buffered = capture_manager.get_buffered_frames(start_time, end_time)

        if not buffered:
            return None

        try:
            ts_str = datetime.datetime.fromtimestamp(alert_timestamp).strftime("%Y%m%d_%H%M%S")
            filename = f"clip_{camera_id}_trk{track_id}_{ts_str}.mp4"
            filepath = self.clips_dir / filename

            h, w = buffered[0][1].shape[:2]
            fps = capture_manager.fps if capture_manager.fps > 0 else 25.0

            # Use MP4V codec for standard cross-platform MP4 playback
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(filepath), fourcc, fps, (w, h))

            if not writer.isOpened():
                # Fallback to alternative codec
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                filepath = self.clips_dir / f"clip_{camera_id}_trk{track_id}_{ts_str}.avi"
                writer = cv2.VideoWriter(str(filepath), fourcc, fps, (w, h))

            if not writer.isOpened():
                print(f"[ALERT_MANAGER] VideoWriter could not open for {filepath}")
                return None

            frames_written = 0
            for _, frame in buffered:
                writer.write(frame)
                frames_written += 1

            writer.release()

            if frames_written > 0 and filepath.is_file() and filepath.stat().st_size > 0:
                return str(filepath)
            return None

        except Exception as e:
            print(f"[ALERT_MANAGER] Clip synthesis error: {e}")
            return None

    def process_alerts(
        self,
        active_alerts: List[AlertEvent],
        cleared_alerts: List[ClearedAlertEvent],
        current_frame: np.ndarray,
        capture_manager: VideoCaptureManager,
    ) -> List[int]:
        """Process active and cleared alert events for the current frame.
        
        Returns:
            List of database IDs for newly created alerts in this step.
        """
        created_db_ids: List[int] = []

        # 1. Handle cleared alerts
        for clr in cleared_alerts:
            if clr.track_id in self._active_alert_ids:
                self._active_alert_ids.remove(clr.track_id)
                iso_cleared = datetime.datetime.fromtimestamp(clr.timestamp).isoformat()
                self.db.update_cleared_timestamp(
                    track_id=clr.track_id,
                    camera_id=clr.camera_id,
                    cleared_timestamp=iso_cleared,
                )

        # 2. Handle active alerts (deduplicating against already active tracks)
        for alert in active_alerts:
            tid = alert.track_id
            if tid in self._active_alert_ids:
                # Already active: do not write duplicate database record every frame
                continue

            # New active alert: save snapshot, clip, and insert into DB
            self._active_alert_ids.add(tid)
            snapshot_path = self.save_snapshot(
                current_frame, alert.camera_id, tid, alert.timestamp
            ) or ""
            clip_path = self.save_clip(
                capture_manager, alert.camera_id, tid, alert.timestamp
            )

            iso_ts = datetime.datetime.fromtimestamp(alert.timestamp).isoformat()
            db_id = self.db.insert_alert(
                timestamp=iso_ts,
                camera_id=alert.camera_id,
                track_id=tid,
                object_class=alert.object_class,
                event_duration=alert.duration,
                reason=alert.reason,
                snapshot_path=snapshot_path,
                clip_path=clip_path,
                review_status="unreviewed",
                cleared_timestamp=None,
            )
            created_db_ids.append(db_id)

        return created_db_ids
