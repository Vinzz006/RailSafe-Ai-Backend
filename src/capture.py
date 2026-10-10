"""Video capture engine with rolling buffer support for RailGuard Vision."""

from __future__ import annotations

import collections
import time
from typing import Deque, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.config import VideoConfig


class VideoCaptureManager:
    """Manages video frame capture from webcams, recorded files, and RTSP streams."""

    def __init__(self, config: VideoConfig) -> None:
        self.config = config
        self.source = config.source
        self.target_width = config.target_width
        self.target_height = config.target_height
        self.frame_skip = config.frame_skip
        self.buffer_seconds = config.buffer_seconds

        self.cap: Optional[cv2.VideoCapture] = None
        self.fps: float = 25.0
        self.frame_idx: int = 0
        self.is_eof: bool = False
        self.error_message: Optional[str] = None

        # Rolling buffer storing (timestamp, frame_copy)
        self.buffer: Deque[Tuple[float, np.ndarray]] = collections.deque()
        self._max_buffer_frames = int(max(30, self.fps * self.buffer_seconds))

    @property
    def source_type(self) -> str:
        if isinstance(self.source, int):
            return "webcam"
        s = str(self.source).lower()
        if s.startswith("rtsp://") or s.startswith("rtsps://"):
            return "rtsp"
        return "file"

    @property
    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def open(self) -> bool:
        """Open the configured video source."""
        self.release()
        self.frame_idx = 0
        self.is_eof = False
        self.error_message = None

        try:
            if isinstance(self.source, int):
                # For Windows webcam capture, DirectShow backend (CAP_DSHOW) or default
                self.cap = cv2.VideoCapture(self.source)
            else:
                self.cap = cv2.VideoCapture(str(self.source))

            if not self.cap.isOpened():
                self.error_message = (
                    f"Unable to open video source: '{self.source}' ({self.source_type}). "
                    "Ensure device is connected, file exists, or network stream is reachable."
                )
                return False

            detected_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if detected_fps and detected_fps > 0 and not np.isnan(detected_fps):
                self.fps = float(detected_fps)
            else:
                self.fps = 25.0

            self._max_buffer_frames = int(max(30, self.fps * self.buffer_seconds))
            return True

        except Exception as e:
            self.error_message = f"Exception opening video source '{self.source}': {e}"
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Read the next frame, applying frame skipping and resizing.
        
        Returns:
            (success, frame, timestamp_seconds)
        """
        if not self.is_opened:
            return False, None, 0.0

        for _ in range(self.frame_skip):
            ret, raw_frame = self.cap.read()
            self.frame_idx += 1
            if not ret:
                self.is_eof = True
                return False, None, 0.0

        if raw_frame is None or raw_frame.size == 0:
            return False, None, 0.0

        # Current frame timestamp
        if self.source_type == "file":
            # For recorded video, use video timeline milliseconds
            msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamp = (msec / 1000.0) if msec >= 0 else (self.frame_idx / self.fps)
        else:
            timestamp = time.time()

        # Resize if target resolution specified
        if (
            self.target_width > 0
            and self.target_height > 0
            and (raw_frame.shape[1] != self.target_width or raw_frame.shape[0] != self.target_height)
        ):
            processed_frame = cv2.resize(
                raw_frame,
                (self.target_width, self.target_height),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            processed_frame = raw_frame

        # Append frame copy to rolling buffer for pre-alert context
        self.buffer.append((timestamp, processed_frame.copy()))
        # Evict old frames beyond buffer_seconds
        min_keep_time = timestamp - self.buffer_seconds
        while len(self.buffer) > 1 and self.buffer[0][0] < min_keep_time:
            self.buffer.popleft()

        return True, processed_frame, timestamp

    def get_buffered_frames(
        self, start_time: Optional[float] = None, end_time: Optional[float] = None
    ) -> List[Tuple[float, np.ndarray]]:
        """Retrieve frames from rolling buffer within the requested timestamp interval."""
        if not self.buffer:
            return []

        result: List[Tuple[float, np.ndarray]] = []
        for ts, frame in list(self.buffer):
            if start_time is not None and ts < start_time:
                continue
            if end_time is not None and ts > end_time:
                continue
            result.append((ts, frame))
        return result

    def release(self) -> None:
        """Release video capture device and flush buffer."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.buffer.clear()
