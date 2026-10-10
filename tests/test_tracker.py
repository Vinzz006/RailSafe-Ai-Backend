"""Unit tests for track state management and spatial association heuristics."""

import numpy as np
import pytest

from src.detector import Detection
from src.tracker import TrackManager, TrackState, euclidean_distance


def test_euclidean_distance() -> None:
    assert euclidean_distance((0, 0), (3, 4)) == 5.0
    assert euclidean_distance((100, 100), (100, 100)) == 0.0


def test_displacement_calculation() -> None:
    track = TrackState(
        track_id=1,
        object_class="backpack",
        bbox=(80, 80, 120, 120),
        first_seen_timestamp=0.0,
        last_seen_timestamp=10.0,
        last_movement_timestamp=0.0,
        centroid_history=[
            (0.0, (100.0, 100.0)),
            (5.0, (105.0, 100.0)),
            (8.0, (110.0, 100.0)),
            (10.0, (112.0, 100.0)),
        ],
    )

    # Over 5 second window (from t=5.0 to 10.0): max displacement from (105, 100) to (112, 100) is 7.0
    disp_5s = track.get_displacement_over_window(10.0, 5.0)
    assert round(disp_5s, 1) == 7.0

    # Over 10 second window (from t=0.0 to 10.0): displacement from (100, 100) to (112, 100) is 12.0
    disp_10s = track.get_displacement_over_window(10.0, 10.0)
    assert round(disp_10s, 1) == 12.0


def test_initial_owner_association_heuristic() -> None:
    """Verify that a newly introduced bag associates with the nearest person within proximity."""
    tm = TrackManager(track_drop_grace_seconds=2.5)

    # Frame 1: Person A at (100, 100), Person B at (300, 300), Bag at (120, 100) -> distance to A is 20px
    pA = Detection(bbox=(80, 80, 120, 120), confidence=0.9, class_id=0, class_name="person", track_id=10)
    pB = Detection(bbox=(280, 280, 320, 320), confidence=0.9, class_id=0, class_name="person", track_id=20)
    bag = Detection(bbox=(100, 80, 140, 120), confidence=0.85, class_id=24, class_name="backpack", track_id=50)

    tracks = tm.update_tracks([pA, pB, bag], timestamp=1.0, proximity_threshold=140.0)
    bag_track = next(t for t in tracks if t.track_id == 50)
    assert bag_track.associated_person_track_id == 10

    # Frame 2: Person C walks very close to the bag (at (110, 100)), but Person A is still owner
    # The initial association MUST remain stable!
    pC = Detection(bbox=(90, 80, 130, 120), confidence=0.9, class_id=0, class_name="person", track_id=30)
    tracks2 = tm.update_tracks([pA, pC, bag], timestamp=2.0, proximity_threshold=140.0)
    bag_track2 = next(t for t in tracks2 if t.track_id == 50)
    assert bag_track2.associated_person_track_id == 10  # Association remained stable!


def test_track_drop_grace_period_and_pruning() -> None:
    """Verify that briefly missing tracks are held in grace period before removal."""
    tm = TrackManager(track_drop_grace_seconds=2.5)

    p1 = Detection(bbox=(50, 50, 90, 90), confidence=0.9, class_id=0, class_name="person", track_id=1)
    tm.update_tracks([p1], timestamp=1.0)
    assert 1 in tm.tracks
    assert tm.tracks[1].is_briefly_lost is False

    # Frame at t=2.0s: p1 is missing (dropout)
    tracks_after_loss = tm.update_tracks([], timestamp=2.0)
    assert 1 in tm.tracks
    assert tm.tracks[1].is_briefly_lost is True  # In grace period!

    # Frame at t=4.0s: p1 has been missing for 3.0s (> 2.5s grace)
    tracks_pruned = tm.update_tracks([], timestamp=4.0)
    assert 1 not in tm.tracks  # Successfully pruned


def test_draw_annotations_smoke() -> None:
    """Verify annotation rendering operates without crashing or modifying input dimensions."""
    tm = TrackManager()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    t1 = TrackState(
        track_id=1,
        object_class="suitcase",
        bbox=(50, 50, 100, 100),
        first_seen_timestamp=0.0,
        last_seen_timestamp=1.0,
        last_movement_timestamp=0.0,
        state="WARNING",
    )
    t2 = TrackState(
        track_id=2,
        object_class="person",
        bbox=(200, 200, 250, 250),
        first_seen_timestamp=0.0,
        last_seen_timestamp=1.0,
        last_movement_timestamp=0.0,
        state="NORMAL",
    )

    annotated = tm.draw_annotations(frame, [t1, t2])
    assert annotated.shape == frame.shape
    assert annotated.dtype == np.uint8
