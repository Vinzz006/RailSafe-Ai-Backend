"""Object tracking state management and trajectory analysis for RailGuard Vision."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.detector import Detection


def euclidean_distance(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two 2D points."""
    return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])


@dataclass
class TrackState:
    """Maintains spatial-temporal history and contextual state for a tracked object."""

    track_id: int
    object_class: str
    bbox: Tuple[int, int, int, int]
    first_seen_timestamp: float
    last_seen_timestamp: float
    last_movement_timestamp: float
    state: str = "NORMAL"  # "NORMAL", "WARNING", "ALERT"
    centroid_history: List[Tuple[float, Tuple[float, float]]] = field(default_factory=list)
    associated_person_track_id: Optional[int] = None
    last_associated_person_seen_timestamp: Optional[float] = None
    is_briefly_lost: bool = False
    consecutive_lost_frames: int = 0
    loitering_alert_raised: bool = False

    @property
    def centroid(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def is_bag(self) -> bool:
        return self.object_class in {"backpack", "handbag", "suitcase"}

    @property
    def is_person(self) -> bool:
        return self.object_class == "person"

    def get_displacement_over_window(
        self, current_time: float, window_seconds: float
    ) -> float:
        """Calculate maximum Euclidean displacement from earliest position in the time window."""
        if not self.centroid_history:
            return 0.0

        cutoff = current_time - window_seconds
        points_in_window = [pt for ts, pt in self.centroid_history if ts >= cutoff]
        if not points_in_window:
            points_in_window = [self.centroid_history[-1][1]]

        ref_pt = points_in_window[0]
        max_dist = 0.0
        for pt in points_in_window:
            d = euclidean_distance(ref_pt, pt)
            if d > max_dist:
                max_dist = d
        return max_dist

    def update_position(
        self, bbox: Tuple[int, int, int, int], timestamp: float, stationary_threshold: float
    ) -> None:
        """Update bounding box, centroid history, and movement timestamp."""
        new_centroid = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        old_centroid = self.centroid

        dist_moved = euclidean_distance(old_centroid, new_centroid)
        if dist_moved > stationary_threshold:
            self.last_movement_timestamp = timestamp

        self.bbox = bbox
        self.last_seen_timestamp = timestamp
        self.is_briefly_lost = False
        self.consecutive_lost_frames = 0

        self.centroid_history.append((timestamp, new_centroid))
        # Keep maximum 120 seconds of centroid history to prevent unbounded memory growth
        cutoff = timestamp - 120.0
        while len(self.centroid_history) > 1 and self.centroid_history[0][0] < cutoff:
            self.centroid_history.pop(0)


class TrackManager:
    """Maintains active tracks, performs owner heuristic associations, and cleans expired tracks."""

    def __init__(self, track_drop_grace_seconds: float = 2.5) -> None:
        self.tracks: Dict[int, TrackState] = {}
        self.track_drop_grace_seconds = track_drop_grace_seconds
        self._next_synthetic_id = 9000

    def update_tracks(
        self,
        detections: List[Detection],
        timestamp: float,
        stationary_threshold: float = 18.0,
        proximity_threshold: float = 140.0,
    ) -> List[TrackState]:
        """Update track states from new detections.
        
        Assigns synthetic IDs if detector did not provide ByteTrack IDs.
        Establishes stable initial owner associations for bags.
        """
        seen_track_ids = set()

        # Separate persons and bags for association calculations
        person_detections: List[Detection] = []
        bag_detections: List[Detection] = []

        for det in detections:
            if det.track_id is None:
                det.track_id = self._next_synthetic_id
                self._next_synthetic_id += 1

            seen_track_ids.add(det.track_id)
            if det.is_person:
                person_detections.append(det)
            elif det.is_bag:
                bag_detections.append(det)

        # 1. Update or create person tracks
        for p_det in person_detections:
            tid = p_det.track_id
            assert tid is not None
            if tid in self.tracks:
                self.tracks[tid].update_position(p_det.bbox, timestamp, stationary_threshold)
            else:
                self.tracks[tid] = TrackState(
                    track_id=tid,
                    object_class=p_det.class_name,
                    bbox=p_det.bbox,
                    first_seen_timestamp=timestamp,
                    last_seen_timestamp=timestamp,
                    last_movement_timestamp=timestamp,
                    centroid_history=[(timestamp, p_det.centroid)],
                )

        # 2. Update or create bag tracks with initial owner association
        for b_det in bag_detections:
            tid = b_det.track_id
            assert tid is not None
            if tid in self.tracks:
                bag_track = self.tracks[tid]
                bag_track.update_position(b_det.bbox, timestamp, stationary_threshold)
            else:
                # Newly discovered bag - attempt initial person association
                associated_person_id: Optional[int] = None
                closest_dist = float("inf")
                bag_cx, bag_cy = b_det.centroid

                for p_det in person_detections:
                    d = euclidean_distance((bag_cx, bag_cy), p_det.centroid)
                    if d <= proximity_threshold and d < closest_dist:
                        closest_dist = d
                        associated_person_id = p_det.track_id

                bag_track = TrackState(
                    track_id=tid,
                    object_class=b_det.class_name,
                    bbox=b_det.bbox,
                    first_seen_timestamp=timestamp,
                    last_seen_timestamp=timestamp,
                    last_movement_timestamp=timestamp,
                    centroid_history=[(timestamp, b_det.centroid)],
                    associated_person_track_id=associated_person_id,
                    last_associated_person_seen_timestamp=timestamp if associated_person_id else None,
                )
                self.tracks[tid] = bag_track

        # 3. Handle unseen tracks (dropout grace window)
        to_delete: List[int] = []
        for tid, track in self.tracks.items():
            if tid not in seen_track_ids:
                track.consecutive_lost_frames += 1
                track.is_briefly_lost = True
                elapsed_since_seen = timestamp - track.last_seen_timestamp
                if elapsed_since_seen > self.track_drop_grace_seconds:
                    to_delete.append(tid)

        for tid in to_delete:
            del self.tracks[tid]

        # 4. Update owner visibility timestamps for associated bags
        for track in self.tracks.values():
            if track.is_bag and track.associated_person_track_id is not None:
                owner_id = track.associated_person_track_id
                if owner_id in self.tracks and not self.tracks[owner_id].is_briefly_lost:
                    track.last_associated_person_seen_timestamp = timestamp

        return list(self.tracks.values())

    def draw_annotations(
        self,
        frame: np.ndarray,
        tracks: List[TrackState],
        proximity_threshold: float = 140.0,
        show_trajectories: bool = True,
    ) -> np.ndarray:
        """Draw bounding boxes, IDs, labels, states, and association vectors onto the frame."""
        annotated = frame.copy()

        # Color palette for states
        COLOR_NORMAL = (0, 220, 0)      # Green
        COLOR_WARNING = (0, 165, 255)   # Orange
        COLOR_ALERT = (0, 0, 240)       # Red
        COLOR_PERSON = (200, 200, 50)   # Cyan / Light Blue
        COLOR_LOST = (128, 128, 128)    # Grey

        # 1. Draw association vectors between bags and associated persons
        for t in tracks:
            if t.is_bag and t.associated_person_track_id is not None:
                owner = self.tracks.get(t.associated_person_track_id)
                if owner and not owner.is_briefly_lost:
                    bx, by = int(t.centroid[0]), int(t.centroid[1])
                    ox, oy = int(owner.centroid[0]), int(owner.centroid[1])
                    dist = euclidean_distance(t.centroid, owner.centroid)
                    line_color = (0, 255, 0) if dist <= proximity_threshold else (0, 140, 255)
                    cv2.line(annotated, (bx, by), (ox, oy), line_color, 1, cv2.LINE_AA)

        # 2. Draw tracks
        for t in tracks:
            x1, y1, x2, y2 = t.bbox
            if t.is_briefly_lost:
                color = COLOR_LOST
            elif t.is_person:
                color = COLOR_PERSON
            elif t.state == "ALERT":
                color = COLOR_ALERT
            elif t.state == "WARNING":
                color = COLOR_WARNING
            else:
                color = COLOR_NORMAL

            thickness = 3 if t.state in ("ALERT", "WARNING") else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Centroid trail
            if show_trajectories and len(t.centroid_history) > 1:
                pts = [
                    (int(pt[0]), int(pt[1]))
                    for _, pt in t.centroid_history[-15:]
                ]
                for i in range(1, len(pts)):
                    cv2.line(annotated, pts[i - 1], pts[i], color, 1, cv2.LINE_AA)

            # Label banner
            if t.is_bag:
                owner_str = f"Owner:#{t.associated_person_track_id}" if t.associated_person_track_id else "NoOwner"
                label = f"#{t.track_id} {t.object_class.upper()} [{t.state}] {owner_str}"
            else:
                label = f"#{t.track_id} {t.object_class.upper()}"

            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(
                annotated,
                (x1, max(0, y1 - th - baseline - 4)),
                (x1 + tw + 6, max(th + baseline + 4, y1)),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 3, max(th + 2, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated
