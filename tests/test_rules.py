"""Unit tests for the deterministic spatial-temporal rules engine."""

from typing import List

import pytest

from src.config import RulesConfig
from src.rules import UnattendedBagRulesEngine
from src.tracker import TrackState


def create_track(
    track_id: int,
    cls_name: str,
    cx: float,
    cy: float,
    first_seen: float,
    last_seen: float,
    associated_owner: int | None = None,
) -> TrackState:
    """Helper to create deterministic synthetic TrackState instances."""
    w, h = 40, 40
    bbox = (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))
    return TrackState(
        track_id=track_id,
        object_class=cls_name,
        bbox=bbox,
        first_seen_timestamp=first_seen,
        last_seen_timestamp=last_seen,
        last_movement_timestamp=first_seen,
        centroid_history=[(first_seen, (cx, cy)), (last_seen, (cx, cy))],
        associated_person_track_id=associated_owner,
    )


def test_stationary_detection() -> None:
    """Verify stationary detection calculates displacement over time window."""
    cfg = RulesConfig(stationary_pixels=18.0, stationary_window_seconds=5.0)
    engine = UnattendedBagRulesEngine(cfg)

    # Stationary bag: moved only 5 pixels in 6 seconds
    bag = create_track(1, "backpack", 100.0, 100.0, 0.0, 6.0)
    bag.centroid_history = [(0.0, (100.0, 100.0)), (3.0, (102.0, 101.0)), (6.0, (103.0, 102.0))]
    assert engine.is_bag_stationary(bag, 6.0) is True

    # Moving bag: displaced by 80 pixels
    moving_bag = create_track(2, "backpack", 100.0, 100.0, 0.0, 6.0)
    moving_bag.centroid_history = [(0.0, (100.0, 100.0)), (3.0, (140.0, 100.0)), (6.0, (180.0, 100.0))]
    assert engine.is_bag_stationary(moving_bag, 6.0) is False


def test_nearby_person_suppression() -> None:
    """Verify that a stationary bag with a nearby person remains in NORMAL state."""
    cfg = RulesConfig(proximity_pixels=140.0, unattended_seconds=15.0)
    engine = UnattendedBagRulesEngine(cfg)

    # Bag at (100, 100), person sitting at (140, 100) -> distance = 40px (< 140px)
    bag = create_track(1, "backpack", 100.0, 100.0, 0.0, 20.0)
    person = create_track(2, "person", 140.0, 100.0, 0.0, 20.0)

    alerts, cleared = engine.evaluate([bag, person], 20.0)
    assert bag.state == "NORMAL"
    assert len(alerts) == 0
    assert len(cleared) == 0


def test_owner_leaves_triggers_warning_then_alert() -> None:
    """Verify state transitions: NORMAL -> WARNING (4s) -> ALERT (15s) when owner walks away."""
    cfg = RulesConfig(
        proximity_pixels=140.0,
        warning_seconds=4.0,
        unattended_seconds=15.0,
    )
    engine = UnattendedBagRulesEngine(cfg)

    bag = create_track(1, "suitcase", 100.0, 100.0, 0.0, 0.0, associated_owner=2)
    # Owner at (500, 500) -> distance ~ 565px (> 140px)
    distant_owner = create_track(2, "person", 500.0, 500.0, 0.0, 0.0)

    # At t=1.0s: observation starting, below warning threshold
    bag.last_seen_timestamp = 1.0
    distant_owner.last_seen_timestamp = 1.0
    alerts, _ = engine.evaluate([bag, distant_owner], 1.0)
    assert bag.state == "NORMAL"
    assert len(alerts) == 0

    # At t=5.0s: exceeds warning_seconds (4s), state should be WARNING
    bag.last_seen_timestamp = 5.0
    distant_owner.last_seen_timestamp = 5.0
    alerts, _ = engine.evaluate([bag, distant_owner], 5.0)
    assert bag.state == "WARNING"
    assert len(alerts) == 0

    # At t=16.0s: exceeds unattended_seconds (15s), state transitions to ALERT
    bag.last_seen_timestamp = 16.0
    distant_owner.last_seen_timestamp = 16.0
    alerts, _ = engine.evaluate([bag, distant_owner], 16.0)
    assert bag.state == "ALERT"
    assert len(alerts) == 1
    assert alerts[0].track_id == 1
    assert "FOR HUMAN REVIEW ONLY" in alerts[0].disclaimer
    assert alerts[0].duration >= 15.0


def test_owner_returns_clears_active_alert() -> None:
    """Verify that an active alert clears when owner returns to bag."""
    cfg = RulesConfig(proximity_pixels=140.0, unattended_seconds=15.0)
    engine = UnattendedBagRulesEngine(cfg)

    bag = create_track(1, "backpack", 100.0, 100.0, 0.0, 16.0, associated_owner=2)
    distant_owner = create_track(2, "person", 500.0, 500.0, 0.0, 16.0)

    # Initial ALERT condition
    engine.evaluate([bag, distant_owner], 1.0)
    alerts, _ = engine.evaluate([bag, distant_owner], 16.0)
    assert len(alerts) == 1
    assert bag.state == "ALERT"

    # Owner returns to (120, 100) -> distance = 20px
    returned_owner = create_track(2, "person", 120.0, 100.0, 0.0, 17.0)
    bag.last_seen_timestamp = 17.0
    alerts, cleared = engine.evaluate([bag, returned_owner], 17.0)

    assert bag.state == "NORMAL"
    assert len(cleared) == 1
    assert cleared[0].track_id == 1
    assert "Owner returned" in cleared[0].cleared_reason


def test_bag_moved_clears_active_alert() -> None:
    """Verify that an active alert clears when the bag is moved or picked up."""
    cfg = RulesConfig(
        stationary_pixels=18.0,
        stationary_window_seconds=5.0,
        unattended_seconds=15.0,
    )
    engine = UnattendedBagRulesEngine(cfg)

    bag = create_track(1, "suitcase", 100.0, 100.0, 0.0, 16.0)
    engine.evaluate([bag], 1.0)
    alerts, _ = engine.evaluate([bag], 16.0)
    assert len(alerts) == 1

    # Bag picked up and moved to (300, 300) -> displacement = 282px
    bag.update_position((280, 280, 320, 320), 17.0, stationary_threshold=18.0)
    alerts, cleared = engine.evaluate([bag], 17.0)

    assert bag.state == "NORMAL"
    assert len(cleared) == 1
    assert "Bag moved or picked up" in cleared[0].cleared_reason


def test_restricted_zone_accelerates_alert() -> None:
    """Verify restricted zone triggers alerts at warning_seconds instead of waiting 15s."""
    cfg = RulesConfig(
        restricted_zone_enabled=True,
        restricted_zone_polygon=[[0, 0], [200, 0], [200, 200], [0, 200]],
        warning_seconds=3.0,
        unattended_seconds=15.0,
    )
    engine = UnattendedBagRulesEngine(cfg)

    # Bag inside restricted polygon at (50, 50)
    bag = create_track(1, "backpack", 50.0, 50.0, 0.0, 3.5)
    engine.evaluate([bag], 0.5)
    alerts, _ = engine.evaluate([bag], 3.5)

    assert bag.state == "ALERT"
    assert len(alerts) == 1
    assert alerts[0].in_restricted_zone is True
    assert "RESTRICTED ZONE" in alerts[0].reason


def test_loitering_contextual_cue() -> None:
    """Verify loitering cue tags passenger track after 90 seconds without triggering false bag alert."""
    cfg = RulesConfig(loiter_seconds=90.0, stationary_pixels=18.0)
    engine = UnattendedBagRulesEngine(cfg)

    person = create_track(10, "person", 200.0, 200.0, 0.0, 95.0)
    alerts, _ = engine.evaluate([person], 95.0)

    assert person.loitering_alert_raised is True
    assert len(alerts) == 0  # Loitering alone MUST NOT raise unattended baggage alert!
