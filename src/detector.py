"""Object detection engine using Ultralytics YOLOv8 for RailGuard Vision."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.config import ModelConfig


# COCO standard class mapping for security screening targets
TARGET_COCO_CLASSES: Dict[int, str] = {
    0: "person",
    24: "backpack",
    26: "handbag",
    28: "suitcase",
}

BAG_CLASS_NAMES = {"backpack", "handbag", "suitcase"}


@dataclass
class Detection:
    """Represents a single detected object in a video frame."""

    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coordinates
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None

    @property
    def centroid(self) -> Tuple[float, float]:
        """Calculates center point (cx, cy) of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def width(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return max(0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def is_bag(self) -> bool:
        return self.class_name in BAG_CLASS_NAMES

    @property
    def is_person(self) -> bool:
        return self.class_name == "person"


class YOLODetector:
    """Ultralytics YOLOv8 detection and tracking wrapper."""

    def __init__(self, config: ModelConfig, model_instance: Optional[Any] = None) -> None:
        self.config = config
        self.weights = config.weights
        self.conf = config.confidence_threshold
        self.iou = config.iou_threshold
        self.device = config.device
        self.input_size = config.input_size

        self.model = model_instance
        self._target_class_ids = list(TARGET_COCO_CLASSES.keys())

    def _resolve_weights_path(self) -> str:
        """Find local weights file if available in workspace, else use model identifier."""
        candidates = [
            Path(self.weights),
            Path("yolov8n.pt"),
            Path("../yolov8n.pt"),
            Path(__file__).resolve().parent.parent.parent / "yolov8n.pt",
            Path(__file__).resolve().parent.parent / "models" / "yolov8n.pt",
        ]
        for c in candidates:
            if c.is_file():
                return str(c.resolve())
        return self.weights

    def load_model(self) -> bool:
        """Initialize the YOLO model."""
        if self.model is not None:
            return True

        try:
            from ultralytics import YOLO

            weights_path = self._resolve_weights_path()
            self.model = YOLO(weights_path)
            return True
        except Exception as e:
            raise RuntimeError(
                f"Failed to load YOLO model with weights '{self.weights}': {e}. "
                "Ensure ultralytics is installed and model weights are accessible."
            )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Perform object detection on a single frame."""
        if self.model is None:
            self.load_model()

        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            classes=self._target_class_ids,
            device=self.device,
            imgsz=self.input_size,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        res = results[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(boxes)):
            cls_id = int(clss[i])
            cls_name = TARGET_COCO_CLASSES.get(cls_id, "unknown")
            box = (int(xyxy[i][0]), int(xyxy[i][1]), int(xyxy[i][2]), int(xyxy[i][3]))
            detections.append(
                Detection(
                    bbox=box,
                    confidence=float(confs[i]),
                    class_id=cls_id,
                    class_name=cls_name,
                    track_id=None,
                )
            )

        return detections

    def track(self, frame: np.ndarray, persist: bool = True) -> List[Detection]:
        """Perform tracking using Ultralytics built-in ByteTrack."""
        if self.model is None:
            self.load_model()

        try:
            results = self.model.track(
                frame,
                persist=persist,
                tracker="bytetrack.yaml",
                conf=self.conf,
                iou=self.iou,
                classes=self._target_class_ids,
                device=self.device,
                imgsz=self.input_size,
                verbose=False,
            )
        except Exception:
            # Fallback to plain detect if tracker throws an exception
            return self.detect(frame)

        detections: List[Detection] = []
        if not results:
            return detections

        res = results[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        # Track IDs may be None if tracking failed to associate
        track_ids = (
            boxes.id.int().cpu().tolist()
            if (hasattr(boxes, "id") and boxes.id is not None)
            else [None] * len(boxes)
        )

        for i in range(len(boxes)):
            cls_id = int(clss[i])
            cls_name = TARGET_COCO_CLASSES.get(cls_id, "unknown")
            box = (int(xyxy[i][0]), int(xyxy[i][1]), int(xyxy[i][2]), int(xyxy[i][3]))
            tid = int(track_ids[i]) if track_ids[i] is not None else None
            detections.append(
                Detection(
                    bbox=box,
                    confidence=float(confs[i]),
                    class_id=cls_id,
                    class_name=cls_name,
                    track_id=tid,
                )
            )

        return detections
