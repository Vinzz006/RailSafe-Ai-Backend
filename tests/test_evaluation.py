"""Unit tests for the evaluation harness metric calculations and matching policy."""

import pytest

from evaluate import GroundTruthEvent, evaluate_alerts
from src.db import AlertRecord


def test_evaluation_metrics_precision_recall_delay() -> None:
    # 2 positive ground truth scenarios, 1 negative scenario
    gt_events = [
        GroundTruthEvent(
            scenario_id="SCENARIO_01",
            camera_id="CAM-01",
            video_file="test.mp4",
            event_type="unattended_bag",
            start_second=10.0,
            expected_alert_second=25.0,
            end_second=40.0,
            notes="Left suitcase",
        ),
        GroundTruthEvent(
            scenario_id="SCENARIO_02",
            camera_id="CAM-01",
            video_file="test.mp4",
            event_type="unattended_bag",
            start_second=50.0,
            expected_alert_second=65.0,
            end_second=80.0,
            notes="Left backpack",
        ),
        GroundTruthEvent(
            scenario_id="SCENARIO_03_NEG",
            camera_id="CAM-01",
            video_file="test.mp4",
            event_type="accompanied_bag",
            start_second=90.0,
            expected_alert_second=None,
            end_second=120.0,
            notes="Passenger sitting near bag",
        ),
    ]

    # Alerts logged:
    # Alert 1: matches SCENARIO_01 at 27.0s (delay = +2.0s) -> TP
    # Alert 2: spurious alert at 100.0s during negative scenario -> FP
    # (SCENARIO_02 has no alert -> FN)
    alerts = [
        AlertRecord(
            id=1,
            timestamp="27.0",
            camera_id="CAM-01",
            track_id=10,
            object_class="suitcase",
            event_duration=15.0,
            reason="Unattended suitcase",
            snapshot_path="",
            clip_path=None,
            review_status="unreviewed",
        ),
        AlertRecord(
            id=2,
            timestamp="100.0",
            camera_id="CAM-01",
            track_id=20,
            object_class="backpack",
            event_duration=15.0,
            reason="Spurious alert",
            snapshot_path="",
            clip_path=None,
            review_status="unreviewed",
        ),
    ]

    report = evaluate_alerts(
        ground_truth=gt_events,
        alerts=alerts,
        time_tolerance_seconds=4.0,
        total_video_duration_seconds=120.0,
    )

    assert report.total_ground_truth_events == 3
    assert report.positive_ground_truth_count == 2
    assert report.negative_ground_truth_count == 1
    assert report.true_positives == 1
    assert report.false_positives == 1
    assert report.false_negatives == 1
    assert report.true_negatives == 1

    # Precision = 1 / (1 + 1) = 0.5
    assert report.precision == 0.5
    # Recall = 1 / (1 + 1) = 0.5
    assert report.recall == 0.5
    # F1 = 0.5
    assert report.f1_score == 0.5

    # Detection delay: 27.0 - 25.0 = 2.0s
    assert report.mean_detection_delay_seconds == 2.0


def test_double_counting_prevention() -> None:
    """Verify an individual alert cannot be matched to multiple ground-truth events."""
    gt_events = [
        GroundTruthEvent(
            scenario_id="SCENARIO_A",
            camera_id="CAM-01",
            video_file="test.mp4",
            event_type="unattended_bag",
            start_second=10.0,
            expected_alert_second=25.0,
            end_second=30.0,
            notes="Event A",
        ),
        GroundTruthEvent(
            scenario_id="SCENARIO_B",
            camera_id="CAM-01",
            video_file="test.mp4",
            event_type="unattended_bag",
            start_second=12.0,
            expected_alert_second=26.0,
            end_second=32.0,
            notes="Event B",
        ),
    ]

    # Only 1 single alert logged at 25.5s
    alerts = [
        AlertRecord(
            id=1,
            timestamp="25.5",
            camera_id="CAM-01",
            track_id=1,
            object_class="bag",
            event_duration=15.0,
            reason="test",
            snapshot_path="",
            clip_path=None,
            review_status="unreviewed",
        )
    ]

    report = evaluate_alerts(gt_events, alerts, time_tolerance_seconds=3.0)
    # Alert must only match ONE scenario; the second scenario remains a False Negative
    assert report.true_positives == 1
    assert report.false_positives == 0
    assert report.false_negatives == 1
