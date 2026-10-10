"""Deterministic spatial-temporal rules engine for RailGuard Vision."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from src.config import RulesConfig
from src.tracker import TrackState, euclidean_distance


def point_in_polygon(point: Tuple[float, float], polygon: List[List[int]]) -> bool:
    """Determine whether a 2D point is inside a polygon using OpenCV."""
    if len(polygon) < 3:
        return False
    pts = np.array(polygon, dtype=np.int32)
    res = cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False)
    return res >= 0


@dataclass
class AlertEvent:
    """Represents an active unattended bag alert requiring human review."""

    track_id: int
    object_class: str
    camera_id: str
    timestamp: float
    duration: float
    reason: str
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    associated_person_id: Optional[int] = None
    in_restricted_zone: bool = False
    disclaimer: str = (
        "FOR HUMAN REVIEW ONLY — This system does not detect explosives or narcotics. "
        "Verify context before taking action."
    )


@dataclass
class ClearedAlertEvent:
    """Represents a previously active alert that has now cleared (e.g. bag picked up or owner returned)."""

    track_id: int
    object_class: str
    camera_id: str
    timestamp: float
    cleared_reason: str


class UnattendedBagRulesEngine:
    """Deterministic spatial-temporal state machine for unattended baggage screening."""

    def __init__(self, config: RulesConfig, camera_id: str = "CAM-PLATFORM-01") -> None:
        self.config = config
        self.camera_id = camera_id

        # State tracking per bag track_id
        # Maps track_id -> timestamp when bag first became stationary and unaccompanied
        self._unaccompanied_since: Dict[int, float] = {}
        # Maps track_id -> timestamp when active alert was first raised
        self._active_alert_since: Dict[int, float] = {}
        # Tracks currently in active ALERT state
        self._active_alert_ids: Set[int] = set()

    def reset(self) -> None:
        """Reset internal rule engine state."""
        self._unaccompanied_since.clear()
        self._active_alert_since.clear()
        self._active_alert_ids.clear()

    def is_bag_stationary(self, track: TrackState, current_time: float) -> bool:
        """A bag is stationary when centroid displacement <= stationary_pixels over stationary_window_seconds."""
        if not track.is_bag:
            return False

        # If bag was observed for less than stationary window, check displacement from inception
        elapsed = current_time - track.first_seen_timestamp
        window = min(elapsed, self.config.stationary_window_seconds)
        if window <= 0.2:
            return False  # Need at least minimal observation time

        displacement = track.get_displacement_over_window(current_time, window)
        return displacement <= self.config.stationary_pixels

    def find_nearest_person(
        self, bag_track: TrackState, all_tracks: List[TrackState]
    ) -> Tuple[Optional[TrackState], float]:
        """Find the nearest visible person track and compute distance to bag."""
        nearest_person: Optional[TrackState] = None
        min_dist = float("inf")

        for t in all_tracks:
            if t.is_person and not t.is_briefly_lost:
                d = euclidean_distance(bag_track.centroid, t.centroid)
                if d < min_dist:
                    min_dist = d
                    nearest_person = t

        return nearest_person, min_dist

    def evaluate(
        self, tracks: List[TrackState], current_time: float
    ) -> Tuple[List[AlertEvent], List[ClearedAlertEvent]]:
        """Evaluate spatial-temporal conditions across all active tracks.
        
        Returns:
            (newly_triggered_or_active_alerts, cleared_alerts)
        """
        active_alerts: List[AlertEvent] = []
        cleared_alerts: List[ClearedAlertEvent] = []
        current_frame_alert_ids: Set[int] = set()

        # Check person tracks for loitering cues (noisy contextual cue, not malicious intent)
        for t in tracks:
            if t.is_person and not t.is_briefly_lost:
                time_in_view = current_time - t.first_seen_timestamp
                if (
                    time_in_view >= self.config.loiter_seconds
                    and not t.loitering_alert_raised
                ):
                    disp = t.get_displacement_over_window(current_time, 30.0)
                    if disp <= (self.config.stationary_pixels * 3.0):
                        t.loitering_alert_raised = True
                        # Marked as contextual note in track, not an unattended baggage alert

        # Evaluate bag tracks
        for bag in tracks:
            if not bag.is_bag:
                continue

            tid = bag.track_id
            is_stationary = self.is_bag_stationary(bag, current_time)

            # Check restricted zone polygon if enabled
            in_restricted_zone = False
            if self.config.restricted_zone_enabled:
                in_restricted_zone = point_in_polygon(
                    bag.centroid, self.config.restricted_zone_polygon
                )

            # Check proximity to nearest person
            nearest_person, nearest_dist = self.find_nearest_person(bag, tracks)
            is_accompanied = nearest_dist <= self.config.proximity_pixels

            # Also check if associated owner specifically is visible and nearby
            owner_present = False
            if bag.associated_person_track_id is not None:
                owner = next((t for t in tracks if t.track_id == bag.associated_person_track_id), None)
                if owner and not owner.is_briefly_lost:
                    owner_dist = euclidean_distance(bag.centroid, owner.centroid)
                    if owner_dist <= self.config.proximity_pixels:
                        owner_present = True

            # If someone is near the bag (either owner or bystander sitting near it)
            if owner_present or is_accompanied:
                # Bag is accompanied -> state returns to NORMAL
                if tid in self._unaccompanied_since:
                    del self._unaccompanied_since[tid]

                if tid in self._active_alert_ids:
                    cleared_reason = (
                        "Owner returned to bag"
                        if owner_present
                        else f"Bystander within proximity perimeter ({nearest_dist:.1f}px)"
                    )
                    cleared_alerts.append(
                        ClearedAlertEvent(
                            track_id=tid,
                            object_class=bag.object_class,
                            camera_id=self.camera_id,
                            timestamp=current_time,
                            cleared_reason=cleared_reason,
                        )
                    )
                    self._active_alert_ids.remove(tid)
                    if tid in self._active_alert_since:
                        del self._active_alert_since[tid]

                bag.state = "NORMAL"
                continue

            # Bag is NOT accompanied. Is it stationary?
            if not is_stationary:
                # Bag is moving (being carried or displaced) -> state is NORMAL
                if tid in self._unaccompanied_since:
                    del self._unaccompanied_since[tid]

                if tid in self._active_alert_ids:
                    cleared_alerts.append(
                        ClearedAlertEvent(
                            track_id=tid,
                            object_class=bag.object_class,
                            camera_id=self.camera_id,
                            timestamp=current_time,
                            cleared_reason="Bag moved or picked up",
                        )
                    )
                    self._active_alert_ids.remove(tid)
                    if tid in self._active_alert_since:
                        del self._active_alert_since[tid]

                bag.state = "NORMAL"
                continue

            # Bag is BOTH stationary AND unaccompanied
            if tid not in self._unaccompanied_since:
                self._unaccompanied_since[tid] = current_time

            unaccompanied_duration = current_time - self._unaccompanied_since[tid]

            # Restricted zone accelerates alert threshold
            alert_threshold = (
                self.config.warning_seconds
                if in_restricted_zone
                else self.config.unattended_seconds
            )

            # State transition: NORMAL -> WARNING -> ALERT
            if unaccompanied_duration >= alert_threshold:
                bag.state = "ALERT"
                current_frame_alert_ids.add(tid)

                if tid not in self._active_alert_since:
                    self._active_alert_since[tid] = current_time
                    self._active_alert_ids.add(tid)

                reason_suffix = " in RESTRICTED ZONE" if in_restricted_zone else ""
                reason = (
                    f"Unattended {bag.object_class} stationary for {unaccompanied_duration:.1f}s "
                    f"without nearby passenger (nearest person {nearest_dist:.1f}px away){reason_suffix}"
                )

                alert_event = AlertEvent(
                    track_id=tid,
                    object_class=bag.object_class,
                    camera_id=self.camera_id,
                    timestamp=current_time,
                    duration=unaccompanied_duration,
                    reason=reason,
                    bbox=bag.bbox,
                    centroid=bag.centroid,
                    associated_person_id=bag.associated_person_track_id,
                    in_restricted_zone=in_restricted_zone,
                )
                active_alerts.append(alert_event)

            elif unaccompanied_duration >= self.config.warning_seconds:
                bag.state = "WARNING"
            else:
                bag.state = "NORMAL"

        # Check for dropped tracks that previously had active alerts
        for tid in list(self._active_alert_ids):
            if tid not in current_frame_alert_ids:
                # Track was lost or removed
                cleared_alerts.append(
                    ClearedAlertEvent(
                        track_id=tid,
                        object_class="bag",
                        camera_id=self.camera_id,
                        timestamp=current_time,
                        cleared_reason="Track left camera field of view or removed",
                    )
                )
                self._active_alert_ids.remove(tid)
                if tid in self._unaccompanied_since:
                    del self._unaccompanied_since[tid]
                if tid in self._active_alert_since:
                    del self._active_alert_since[tid]

        return active_alerts, cleared_alerts
