"""Standalone migration utility: Cloud Firestore -> Firebase Realtime Database.

Safely copies alert records from Cloud Firestore to Firebase Realtime Database:
- Idempotent: Skips records already present in RTDB (unless --overwrite is passed).
- Dry-run mode: Evaluates and prints planned changes without writing to RTDB.
- Preserves all existing IDs, event UIDs, timestamps, review history, and evidence references.
- Non-destructive: NEVER deletes or modifies records in Cloud Firestore.
- Resume-capable: Can be safely interrupted and rerun.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from src.config import SystemConfig, load_config
from src.firebase_service import get_firebase_app, get_firestore_db, get_realtime_db
from src.realtime_database_repository import iso_to_epoch_ms, sanitize_rtdb_key

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [MIGRATION] %(message)s",
)
logger = logging.getLogger("firestore_to_rtdb_migration")


def transform_firestore_doc_to_rtdb(
    doc_id: str,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert Firestore document dict to Firebase Realtime Database schema."""
    event_uid = str(data.get("event_uid") or doc_id)
    safe_id = sanitize_rtdb_key(doc_id or event_uid)

    timestamp_str = str(data.get("timestamp") or "")
    created_at_str = str(
        data.get("created_at")
        or timestamp_str
        or datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    created_at_ms = iso_to_epoch_ms(created_at_str)
    timestamp_ms = iso_to_epoch_ms(timestamp_str) if timestamp_str else created_at_ms

    track_id_raw = data.get("track_id")
    track_id = 0
    if track_id_raw is not None:
        try:
            track_id = int(track_id_raw)
        except (ValueError, TypeError):
            track_id = 0

    duration_raw = data.get("event_duration")
    duration = 0.0
    if duration_raw is not None:
        try:
            duration = float(duration_raw)
        except (ValueError, TypeError):
            duration = 0.0

    cleared_ts = data.get("cleared_timestamp")
    status = "cleared" if cleared_ts else "active"

    review_status = str(data.get("review_status") or "unreviewed").lower()
    if review_status not in ("verified", "false_alarm", "unreviewed"):
        review_status = "unreviewed"

    rtdb_record: Dict[str, Any] = {
        "id": safe_id,
        "event_uid": event_uid,
        "status": status,
        "timestamp": timestamp_str,
        "timestamp_ms": timestamp_ms,
        "created_at": created_at_str,
        "created_at_ms": created_at_ms,
        "camera_id": str(data.get("camera_id") or "CAM-01"),
        "track_id": track_id,
        "object_class": str(data.get("object_class") or "luggage"),
        "event_duration": duration,
        "reason": str(data.get("reason") or "Unattended luggage detected"),
        "snapshot_key": data.get("snapshot_key"),
        "snapshot_url": data.get("snapshot_url") or f"/api/alerts/{safe_id}/snapshot",
        "clip_key": data.get("clip_key"),
        "clip_url": data.get("clip_url") or (f"/api/alerts/{safe_id}/clip" if data.get("clip_key") else None),
        "review_status": review_status,
        "reviewed_by": data.get("reviewed_by"),
        "reviewed_at": data.get("reviewed_at"),
        "cleared_timestamp": cleared_ts,
    }

    return rtdb_record


def run_migration(
    config: SystemConfig,
    dry_run: bool = True,
    overwrite: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute migration from Cloud Firestore to Firebase Realtime Database.
    
    Returns summary dictionary with counts.
    """
    logger.info("Initializing Firebase clients for migration...")
    logger.info("Project ID: %s", config.firebase.project_id)
    logger.info("RTDB URL: %s", config.firebase.database_url)
    logger.info("Dry Run Mode: %s", dry_run)
    logger.info("Overwrite Existing RTDB Records: %s", overwrite)

    try:
        firestore_db = get_firestore_db(config.firebase)
    except Exception as e:
        logger.error("Failed to connect to Cloud Firestore: %s", e)
        raise RuntimeError(f"Could not connect to Cloud Firestore: {e}") from e

    try:
        rtdb_root = get_realtime_db(config.firebase)
    except Exception as e:
        logger.error("Failed to connect to Firebase Realtime Database: %s", e)
        raise RuntimeError(f"Could not connect to Firebase Realtime Database: {e}") from e

    collection_name = config.firebase.alerts_collection
    base_path = config.firebase.rtdb_alerts_path.rstrip("/")
    if base_path.endswith("/alerts"):
        base_path = base_path[:-7].rstrip("/")
    base_path = base_path or "railguard"

    alerts_ref = rtdb_root.child(f"{base_path}/alerts")
    index_ref = rtdb_root.child(f"{base_path}/event_uid_index")

    logger.info("Reading documents from Firestore collection '%s'...", collection_name)
    coll_ref = firestore_db.collection(collection_name)
    if limit:
        query = coll_ref.limit(limit)
    else:
        query = coll_ref

    docs = list(query.stream())
    total_docs = len(docs)
    logger.info("Found %d Firestore alert records to process.", total_docs)

    summary = {
        "total_scanned": total_docs,
        "migrated": 0,
        "skipped": 0,
        "failed": 0,
        "dry_run": dry_run,
        "failures": [],
    }

    for idx, doc in enumerate(docs, start=1):
        doc_id = doc.id
        try:
            data = doc.to_dict() or {}
            transformed = transform_firestore_doc_to_rtdb(doc_id, data)
            safe_id = transformed["id"]
            event_uid = transformed["event_uid"]

            # Check if record already exists in RTDB
            existing = None
            if not dry_run and not overwrite:
                try:
                    existing = alerts_ref.child(safe_id).get()
                except Exception as check_err:
                    logger.warning("Error checking existence for %s: %s", safe_id, check_err)

            if existing and not overwrite:
                logger.info(
                    "[%d/%d] SKIPPED: %s (already present in RTDB and overwrite=False)",
                    idx,
                    total_docs,
                    safe_id,
                )
                summary["skipped"] += 1
                continue

            if dry_run:
                logger.info(
                    "[%d/%d] [DRY RUN] Would migrate alert: id=%s, event_uid=%s, status=%s, review=%s",
                    idx,
                    total_docs,
                    safe_id,
                    event_uid,
                    transformed["status"],
                    transformed["review_status"],
                )
                summary["migrated"] += 1
            else:
                # Write to RTDB alerts node
                alerts_ref.child(safe_id).set(transformed)
                # Write to event_uid_index node
                index_ref.child(safe_id).set(safe_id)
                logger.info(
                    "[%d/%d] MIGRATED: %s (event_uid: %s)",
                    idx,
                    total_docs,
                    safe_id,
                    event_uid,
                )
                summary["migrated"] += 1

        except Exception as item_err:
            logger.error("[%d/%d] FAILED: %s: %s", idx, total_docs, doc_id, item_err)
            summary["failed"] += 1
            summary["failures"].append({"id": doc_id, "error": str(item_err)})

    logger.info("==================================================")
    logger.info("MIGRATION SUMMARY:")
    logger.info("  Mode: %s", "DRY-RUN (No changes written)" if dry_run else "LIVE EXECUTION")
    logger.info("  Total Firestore Docs Scanned: %d", summary["total_scanned"])
    logger.info("  Migrated: %d", summary["migrated"])
    logger.info("  Skipped: %d", summary["skipped"])
    logger.info("  Failed: %d", summary["failed"])
    logger.info("==================================================")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate RailGuard Vision alerts from Cloud Firestore to Firebase Realtime Database."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Execute live migration (default is dry-run mode).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite records in RTDB if they already exist (default is to skip existing).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of Firestore records to migrate (for testing).",
    )

    args = parser.parse_args()
    dry_run = not args.live

    cfg = load_config()

    try:
        run_migration(
            config=cfg,
            dry_run=dry_run,
            overwrite=args.overwrite,
            limit=args.limit,
        )
    except Exception as e:
        logger.error("Migration halted due to error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
