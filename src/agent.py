"""Laptop Camera Edge Agent for RailGuard Vision.

Runs the local video detection pipeline (YOLOv8 + ByteTrack + Rules Engine)
on the station laptop and forwards alerts and evidence asynchronously to the
cloud-hosted FastAPI backend on Render over authenticated HTTPS.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import httpx

from src.config import SystemConfig, load_config
from src.db import AlertDatabaseManager, AlertRecord
from src.pipeline import PipelineFrameResult, RailGuardPipeline

logger = logging.getLogger("railguard_agent")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [LOCAL_AGENT] %(message)s",
)


@dataclass
class IngestTask:
    """Represents an alert upload task to be pushed to the cloud backend."""

    task_type: str  # "alert" or "clear"
    alert_record: Optional[AlertRecord] = None
    camera_id: str = ""
    track_id: int = 0
    cleared_timestamp: str = ""
    event_uid: str = ""
    retry_count: int = 0
    non_retryable: bool = False


class CloudForwarder:
    """Asynchronous, thread-safe background uploader with bounded retry logic."""

    def __init__(
        self,
        cloud_url: str,
        agent_token: str,
        max_queue_size: int = 100,
        max_retries: int = 5,
        timeout_seconds: float = 15.0,
    ) -> None:
        self.cloud_url = cloud_url.rstrip("/")
        self.agent_token = agent_token
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

        self._queue: queue.Queue[IngestTask] = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the background uploader worker thread."""
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="CloudForwarderWorker",
        )
        self._worker_thread.start()
        logger.info("Cloud forwarding worker started. Target: %s", self.cloud_url)

    def stop(self, timeout: float = 3.0) -> None:
        """Signal worker to stop and wait for completion."""
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
        logger.info("Cloud forwarding worker stopped.")

    def enqueue_alert(self, alert: AlertRecord, event_uid: str) -> bool:
        """Enqueue an alert upload task without blocking the video capture loop."""
        task = IngestTask(
            task_type="alert",
            alert_record=alert,
            camera_id=alert.camera_id,
            track_id=alert.track_id,
            event_uid=event_uid,
        )
        try:
            self._queue.put_nowait(task)
            logger.info("Enqueued alert %s for cloud upload (Q-size: %d)", event_uid, self._queue.qsize())
            return True
        except queue.Full:
            logger.warning("Upload queue full (max=%d). Dropping upload for %s to prevent memory leaks.", self._queue.maxsize, event_uid)
            return False

    def enqueue_clear(self, camera_id: str, track_id: int, cleared_timestamp: str) -> bool:
        """Enqueue an alert clear task."""
        task = IngestTask(
            task_type="clear",
            camera_id=camera_id,
            track_id=track_id,
            cleared_timestamp=cleared_timestamp,
        )
        try:
            self._queue.put_nowait(task)
            return True
        except queue.Full:
            logger.warning("Upload queue full. Dropping clear event for cam=%s trk=%d", camera_id, track_id)
            return False

    def _worker_loop(self) -> None:
        """Main upload loop processing tasks with bounded exponential backoff."""
        while not self._stop_event.is_set():
            try:
                task = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            success = False
            try:
                if task.task_type == "alert" and task.alert_record:
                    success = self._upload_alert(task.alert_record, task.event_uid, task=task)
                elif task.task_type == "clear":
                    success = self._upload_clear(task.camera_id, task.track_id, task.cleared_timestamp, task=task)
            except Exception as e:
                logger.warning("Error processing cloud upload task: %s", e)
                success = False

            if not success and not task.non_retryable and task.retry_count < self.max_retries:
                task.retry_count += 1
                backoff = min(2 ** task.retry_count, 30)
                logger.info("Task failed. Retrying in %ds (attempt %d/%d)", backoff, task.retry_count, self.max_retries)
                time.sleep(backoff)
                try:
                    self._queue.put_nowait(task)
                except queue.Full:
                    logger.warning("Queue full during retry requeue; dropping task %s", task.event_uid)
            elif not success:
                if task.non_retryable:
                    logger.error("Task %s rejected with non-retryable error (e.g. invalid credentials). Abandoning.", task.event_uid)
                else:
                    logger.error("Task %s exceeded max retries (%d). Abandoning upload.", task.event_uid, self.max_retries)

            self._queue.task_done()

    def _upload_alert(
        self,
        alert: AlertRecord,
        event_uid: str,
        task: Optional[IngestTask] = None,
    ) -> bool:
        """Execute authenticated multipart HTTP upload to cloud backend.
        
        Returns:
            bool: True on success (HTTP 200/201), False otherwise.
        """
        if not self.agent_token or not self.agent_token.strip():
            logger.error("Upload aborted for %s: Missing agent token. Ingestion requires RAILGUARD_AGENT_TOKEN.", event_uid)
            if task:
                task.non_retryable = True
            return False

        snap_path = Path(alert.snapshot_path)
        if not snap_path.is_file():
            logger.error("Snapshot file %s not found on disk. Cannot upload alert.", snap_path)
            if task:
                task.non_retryable = True
            return False

        files: Dict[str, Any] = {}
        snap_file = open(snap_path, "rb")
        files["snapshot"] = (snap_path.name, snap_file, "image/jpeg")

        clip_file = None
        if alert.clip_path and Path(alert.clip_path).is_file():
            clip_file = open(alert.clip_path, "rb")
            ext = Path(alert.clip_path).suffix.lower()
            mime = "video/mp4" if ext == ".mp4" else "video/x-msvideo"
            files["clip"] = (Path(alert.clip_path).name, clip_file, mime)

        data = {
            "event_uid": event_uid,
            "timestamp": alert.timestamp,
            "camera_id": alert.camera_id,
            "track_id": str(alert.track_id),
            "object_class": alert.object_class,
            "event_duration": str(alert.event_duration),
            "reason": alert.reason,
        }

        headers = {
            "Authorization": f"Bearer {self.agent_token.strip()}",
        }

        url = f"{self.cloud_url}/api/ingest/alerts"
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                res = client.post(url, data=data, files=files, headers=headers)
                if res.status_code in (200, 201):
                    logger.info("Successfully uploaded alert %s to cloud (HTTP %d)", event_uid, res.status_code)
                    return True
                elif res.status_code in (401, 403):
                    logger.error(
                        "Cloud backend rejected alert %s (HTTP %d %s). Token invalid or unauthorized.",
                        event_uid,
                        res.status_code,
                        res.text.strip(),
                    )
                    if task:
                        task.non_retryable = True
                    return False
                else:
                    logger.warning("Cloud upload rejected (HTTP %d): %s", res.status_code, res.text)
                    return False
        except httpx.RequestError as e:
            logger.warning("Network failure connecting to cloud (%s): %s", url, e)
            return False
        finally:
            snap_file.close()
            if clip_file:
                clip_file.close()

    def _upload_clear(
        self,
        camera_id: str,
        track_id: int,
        cleared_timestamp: str,
        task: Optional[IngestTask] = None,
    ) -> bool:
        """Execute clear notification upload.
        
        Returns:
            bool: True on success (HTTP 200), False otherwise.
        """
        if not self.agent_token or not self.agent_token.strip():
            logger.error("Clear upload aborted: Missing agent token. Ingestion requires RAILGUARD_AGENT_TOKEN.")
            if task:
                task.non_retryable = True
            return False

        payload = {
            "camera_id": camera_id,
            "track_id": track_id,
            "cleared_timestamp": cleared_timestamp,
        }
        headers = {
            "Authorization": f"Bearer {self.agent_token.strip()}",
            "Content-Type": "application/json",
        }
        url = f"{self.cloud_url}/api/ingest/alerts/clear"
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                res = client.post(url, json=payload, headers=headers)
                if res.status_code == 200:
                    logger.info("Successfully updated cleared state for cam=%s trk=%d in cloud", camera_id, track_id)
                    return True
                elif res.status_code in (401, 403):
                    logger.error(
                        "Cloud backend rejected clear event (HTTP %d %s). Token invalid or unauthorized.",
                        res.status_code,
                        res.text.strip(),
                    )
                    if task:
                        task.non_retryable = True
                    return False
                else:
                    logger.warning("Cloud clear rejected (HTTP %d): %s", res.status_code, res.text)
                    return False
        except httpx.RequestError as e:
            logger.warning("Network failure connecting to cloud clear endpoint: %s", e)
            return False



class LocalDetectionAgent:
    """Main local agent running the vision pipeline and driving cloud forwarding."""

    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        cloud_url: Optional[str] = None,
        agent_token: Optional[str] = None,
        enable_cloud: bool = True,
    ) -> None:
        self.config = config or load_config()
        # Precedence: Explicit parameter > RAILGUARD_CLOUD_URL env > default URL
        self.cloud_url = (
            cloud_url
            or os.getenv("RAILGUARD_CLOUD_URL")
            or "https://railsafe-ai-backend.onrender.com"
        ).rstrip("/")

        # Precedence: Explicit parameter > RAILGUARD_AGENT_TOKEN env > None
        resolved_token = agent_token if agent_token is not None else os.getenv("RAILGUARD_AGENT_TOKEN")
        self.agent_token = resolved_token.strip() if resolved_token else None
        self.enable_cloud = enable_cloud

        self.pipeline = RailGuardPipeline(self.config)
        self.forwarder = (
            CloudForwarder(
                cloud_url=self.cloud_url,
                agent_token=self.agent_token or "",
            )
            if self.enable_cloud
            else None
        )
        self.db = self.pipeline.db
        self._running = False

    def start(self) -> bool:
        """Initialize pipeline and forwarder."""
        success = self.pipeline.start()
        if not success:
            logger.error("Failed to initialize local video pipeline.")
            return False

        if self.forwarder is not None:
            if not self.agent_token:
                logger.warning(
                    "[LOCAL_AGENT] Cloud forwarder active without agent token. "
                    "Uploads will fail. Pass --token or set RAILGUARD_AGENT_TOKEN."
                )
            self.forwarder.start()
        else:
            logger.info("[LOCAL_AGENT] Cloud forwarding disabled. Running in offline/local-only mode.")

        self._running = True
        logger.info("LocalDetectionAgent started on source: %s", self.config.video.source)
        return True

    def step(self) -> PipelineFrameResult:
        """Step the pipeline and forward newly created or cleared alerts."""
        result = self.pipeline.step()

        # Handle newly raised alerts
        if self.forwarder is not None:
            for db_id in result.new_db_alert_ids:
                record = self.db.get_alert_by_id(db_id)
                if record:
                    event_uid = record.event_uid or f"EVT_{record.camera_id}_T{record.track_id}_{int(time.time())}"
                    self.forwarder.enqueue_alert(record, event_uid)

            # Handle cleared alerts
            for clr in result.cleared_alerts:
                iso_cleared = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(clr.timestamp))
                self.forwarder.enqueue_clear(clr.camera_id, clr.track_id, iso_cleared)

        return result

    def stop(self) -> None:
        """Gracefully release camera and stop background forwarder."""
        self._running = False
        self.pipeline.stop()
        if self.forwarder is not None:
            self.forwarder.stop()
        logger.info("LocalDetectionAgent shut down cleanly.")

