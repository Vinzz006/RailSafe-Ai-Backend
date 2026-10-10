"""Evaluation framework for RailGuard Vision comparing ground-truth annotations with logged alerts."""

from __future__ import annotations

import argparse
import csv
import datetime
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.db import AlertDatabaseManager, AlertRecord


@dataclass
class GroundTruthEvent:
    scenario_id: str
    camera_id: str
    video_file: str
    event_type: str  # 'unattended_bag' (positive) or 'accompanied_bag' / 'normal' (negative)
    start_second: float
    expected_alert_second: Optional[float]
    end_second: float
    notes: str


@dataclass
class EvaluationReport:
    total_ground_truth_events: int
    positive_ground_truth_count: int
    negative_ground_truth_count: int
    total_alerts_logged: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1_score: float
    mean_detection_delay_seconds: Optional[float]
    delays: List[float]
    false_alarms_per_hour: float
    evaluation_duration_hours: float
    matched_pairs: List[Dict[str, Any]]
    unmatched_alerts: List[Dict[str, Any]]
    unmatched_ground_truth: List[Dict[str, Any]]


def parse_ground_truth_csv(filepath: Path) -> List[GroundTruthEvent]:
    """Parse annotated ground truth CSV scenarios."""
    if not filepath.is_file():
        raise FileNotFoundError(f"Ground truth CSV not found: {filepath}")

    events: List[GroundTruthEvent] = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s_alert = row.get("expected_alert_second", "").strip()
            exp_sec = float(s_alert) if s_alert and s_alert != "None" else None

            events.append(
                GroundTruthEvent(
                    scenario_id=row.get("scenario_id", "").strip(),
                    camera_id=row.get("camera_id", "").strip(),
                    video_file=row.get("video_file", "").strip(),
                    event_type=row.get("event_type", "unattended_bag").strip(),
                    start_second=float(row.get("start_second", 0.0)),
                    expected_alert_second=exp_sec,
                    end_second=float(row.get("end_second", 0.0)),
                    notes=row.get("notes", "").strip(),
                )
            )
    return events


def parse_alert_seconds(alert: AlertRecord, reference_start: Optional[datetime.datetime] = None) -> float:
    """Extract timeline seconds from alert record."""
    raw_ts = alert.timestamp
    # If already a float/second string
    try:
        return float(raw_ts)
    except ValueError:
        pass

    # Try ISO timestamp parsing
    try:
        dt = datetime.datetime.fromisoformat(raw_ts)
        if reference_start:
            return (dt - reference_start).total_seconds()
        # Fall back to day seconds for relative evaluation
        return dt.hour * 3600.0 + dt.minute * 60.0 + dt.second + dt.microsecond / 1e6
    except Exception:
        return 0.0


def evaluate_alerts(
    ground_truth: List[GroundTruthEvent],
    alerts: List[AlertRecord],
    time_tolerance_seconds: float = 4.0,
    total_video_duration_seconds: Optional[float] = None,
) -> EvaluationReport:
    """Evaluate detected alerts against ground truth annotations.
    
    Matching Policy:
    1. A positive ground truth event matches an alert if:
       - camera_id matches
       - alert time falls within [expected_alert_second - tolerance, end_second + tolerance]
    2. One-to-one matching: Once an alert matches a ground-truth event, neither can be matched again.
    3. Prevents double-counting.
    """
    matched_alert_ids = set()
    matched_gt_ids = set()
    matched_pairs: List[Dict[str, Any]] = []
    delays: List[float] = []

    positives = [gt for gt in ground_truth if gt.event_type == "unattended_bag" and gt.expected_alert_second is not None]
    negatives = [gt for gt in ground_truth if gt.event_type != "unattended_bag"]

    # Match positive scenarios
    for gt in positives:
        best_alert: Optional[AlertRecord] = None
        best_delta = float("inf")

        for alert in alerts:
            if alert.id in matched_alert_ids:
                continue
            if gt.camera_id and alert.camera_id != gt.camera_id:
                continue

            alert_sec = parse_alert_seconds(alert)
            assert gt.expected_alert_second is not None

            # Window: [expected_alert_second - tolerance, end_second + tolerance]
            window_min = gt.expected_alert_second - time_tolerance_seconds
            window_max = gt.end_second + time_tolerance_seconds

            if window_min <= alert_sec <= window_max:
                delta = abs(alert_sec - gt.expected_alert_second)
                if delta < best_delta:
                    best_delta = delta
                    best_alert = alert

        if best_alert is not None:
            matched_alert_ids.add(best_alert.id)
            matched_gt_ids.add(gt.scenario_id)
            alert_sec = parse_alert_seconds(best_alert)
            delay = alert_sec - (gt.expected_alert_second or 0.0)
            delays.append(delay)
            matched_pairs.append({
                "scenario_id": gt.scenario_id,
                "alert_id": best_alert.id,
                "track_id": best_alert.track_id,
                "expected_second": gt.expected_alert_second,
                "actual_second": alert_sec,
                "detection_delay_seconds": round(delay, 2),
            })

    tp = len(matched_pairs)
    fn = len(positives) - tp
    fp = len(alerts) - len(matched_alert_ids)
    tn = len(negatives)  # Negative scenarios without false alarms

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    mean_delay = (sum(delays) / len(delays)) if delays else None

    # Calculate total duration in hours for false alarms per hour
    if total_video_duration_seconds and total_video_duration_seconds > 0:
        duration_hours = total_video_duration_seconds / 3600.0
    elif ground_truth:
        max_sec = max(gt.end_second for gt in ground_truth)
        duration_hours = max(1.0 / 3600.0, max_sec / 3600.0)
    else:
        duration_hours = 1.0

    fa_per_hour = fp / duration_hours if duration_hours > 0 else 0.0

    unmatched_alerts = [
        {"id": a.id, "track_id": a.track_id, "timestamp": a.timestamp, "reason": a.reason}
        for a in alerts
        if a.id not in matched_alert_ids
    ]
    unmatched_gt = [
        {"scenario_id": g.scenario_id, "expected_second": g.expected_alert_second, "notes": g.notes}
        for g in positives
        if g.scenario_id not in matched_gt_ids
    ]

    return EvaluationReport(
        total_ground_truth_events=len(ground_truth),
        positive_ground_truth_count=len(positives),
        negative_ground_truth_count=len(negatives),
        total_alerts_logged=len(alerts),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1_score=round(f1, 4),
        mean_detection_delay_seconds=round(mean_delay, 2) if mean_delay is not None else None,
        delays=delays,
        false_alarms_per_hour=round(fa_per_hour, 2),
        evaluation_duration_hours=round(duration_hours, 4),
        matched_pairs=matched_pairs,
        unmatched_alerts=unmatched_alerts,
        unmatched_ground_truth=unmatched_gt,
    )


def print_report(rep: EvaluationReport) -> None:
    """Format and print an executive summary of the evaluation report."""
    print("=" * 70)
    print("       RAILGUARD VISION — EVALUATION HARNESS BENCHMARK REPORT")
    print("       FOR HUMAN REVIEW ONLY — NOT A CERTIFIED HAZARD DETECTOR")
    print("=" * 70)
    print(f"Total Ground Truth Scenarios: {rep.total_ground_truth_events}")
    print(f"  - Positive Scenarios (Unattended Bags): {rep.positive_ground_truth_count}")
    print(f"  - Negative Scenarios (Accompanied Bags): {rep.negative_ground_truth_count}")
    print(f"Total Alerts Logged in Database:       {rep.total_alerts_logged}")
    print("-" * 70)
    print(f"True Positives  (TP) : {rep.true_positives}")
    print(f"False Positives (FP) : {rep.false_positives}")
    print(f"False Negatives (FN) : {rep.false_negatives}")
    print(f"True Negatives  (TN) : {rep.true_negatives}")
    print("-" * 70)
    print(f"Precision            : {rep.precision * 100:.2f}%")
    print(f"Recall (Sensitivity) : {rep.recall * 100:.2f}%")
    print(f"F1-Score             : {rep.f1_score:.4f}")
    if rep.mean_detection_delay_seconds is not None:
        print(f"Mean Detection Delay : {rep.mean_detection_delay_seconds:.2f} seconds")
    else:
        print("Mean Detection Delay : N/A (No True Positives)")
    print(f"False Alarms/Hour    : {rep.false_alarms_per_hour:.2f}")
    print("=" * 70)

    if rep.matched_pairs:
        print("\nMatched Scenarios (True Positives):")
        for m in rep.matched_pairs:
            print(
                f"  Scenario '{m['scenario_id']}': Expected @ {m['expected_second']}s -> "
                f"Alert #{m['alert_id']} @ {m['actual_second']}s (Delay: {m['detection_delay_seconds']}s)"
            )

    if rep.unmatched_ground_truth:
        print("\nMissed Scenarios (False Negatives):")
        for u in rep.unmatched_ground_truth:
            print(f"  Scenario '{u['scenario_id']}': Expected @ {u['expected_second']}s - {u['notes']}")

    if rep.unmatched_alerts:
        print(f"\nSpurious Alerts (False Positives: {len(rep.unmatched_alerts)}):")
        for u in rep.unmatched_alerts[:5]:
            print(f"  Alert #{u['id']} (Track #{u['track_id']}): {u['reason']}")
        if len(rep.unmatched_alerts) > 5:
            print(f"  ... and {len(rep.unmatched_alerts) - 5} more.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RailGuard Vision alerts against ground-truth CSV.")
    parser.add_argument("--ground-truth", type=str, default="data/ground_truth/ground_truth_template.csv", help="Path to ground truth CSV")
    parser.add_argument("--db", type=str, default="data/security_alerts.db", help="Path to SQLite alerts database")
    parser.add_argument("--tolerance", type=float, default=4.0, help="Time tolerance in seconds for alert matching")
    parser.add_argument("--duration", type=float, default=None, help="Total footage duration in seconds")
    args = parser.parse_args()

    gt_path = Path(args.ground_truth).resolve()
    db_path = Path(args.db).resolve()

    if not gt_path.is_file():
        print(f"Error: Ground truth CSV not found at '{gt_path}'.")
        return

    if not db_path.is_file():
        print(f"Notice: Database file not found at '{db_path}'. Generating zero-alert evaluation report.")
        alerts: List[AlertRecord] = []
    else:
        db = AlertDatabaseManager(str(db_path))
        alerts = db.get_alerts(limit=10000)

    gt_events = parse_ground_truth_csv(gt_path)
    report = evaluate_alerts(
        ground_truth=gt_events,
        alerts=alerts,
        time_tolerance_seconds=args.tolerance,
        total_video_duration_seconds=args.duration,
    )
    print_report(report)


if __name__ == "__main__":
    main()
