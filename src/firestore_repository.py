"""Firestore Alert Repository for RailGuard Vision cloud persistence on Render."""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple

from src.firebase_service import get_firestore_db

logger = logging.getLogger("railguard_firestore")


class FirestoreAlertRepository:
    """Encapsulates all Cloud Firestore CRUD operations for RailGuard security alerts."""

    def __init__(
        self,
        collection_name: str = "railguard_alerts",
        db_client: Optional[Any] = None,
    ) -> None:
        self.collection_name = collection_name
        self._client = db_client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = get_firestore_db()
        return self._client

    @property
    def collection(self) -> Any:
        return self.client.collection(self.collection_name)

    def check_readiness(self) -> bool:
        """Check if Firestore collection is accessible."""
        try:
            _ = list(self.collection.limit(1).stream())
            return True
        except Exception as e:
            logger.warning("Firestore readiness check failed: %s", e)
            return False

    def create_or_update_alert(self, alert_data: Dict[str, Any]) -> Tuple[str, bool]:
        """Ingest an alert document into Firestore.
        
        Uses event_uid as document ID to guarantee idempotency during network retries.
        Returns:
            (document_id, is_newly_created: bool)
        """
        event_uid = alert_data.get("event_uid")
        if not event_uid:
            raise ValueError("event_uid is required for alert ingestion.")

        doc_ref = self.collection.document(event_uid)
        snapshot = doc_ref.get()

        if snapshot.exists:
            logger.info("Alert %s already exists in Firestore. Suppressing duplicate write.", event_uid)
            return event_uid, False

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        doc_payload = {
            "event_uid": event_uid,
            "timestamp": alert_data.get("timestamp", now_iso),
            "camera_id": alert_data.get("camera_id", "UNKNOWN"),
            "track_id": int(alert_data.get("track_id", 0)),
            "object_class": alert_data.get("object_class", "luggage"),
            "event_duration": float(alert_data.get("event_duration", 0.0)),
            "reason": alert_data.get("reason", "Unattended luggage detected"),
            "snapshot_url": alert_data.get("snapshot_url", ""),
            "clip_url": alert_data.get("clip_url", None),
            "review_status": alert_data.get("review_status", "unreviewed"),
            "cleared_timestamp": alert_data.get("cleared_timestamp", None),
            "created_at": now_iso,
            "disclaimer": (
                "FOR HUMAN REVIEW ONLY — Visible-light optical cameras cannot detect explosives or narcotics. "
                "Verify physical context before initiating security actions."
            ),
        }

        doc_ref.set(doc_payload)
        logger.info("Created alert document %s in collection %s", event_uid, self.collection_name)
        return event_uid, True

    def get_alert(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Fetch alert document by document ID (event_uid)."""
        doc = self.collection.document(alert_id).get()
        if not doc.exists:
            return None
        data = doc.to_dict()
        data["id"] = doc.id
        return data

    def list_alerts(
        self,
        limit: int = 50,
        review_status: Optional[str] = None,
        unreviewed_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """List alerts from Firestore with optional filtering.
        
        Gracefully handles missing Firestore composite indexes by catching
        query evaluation errors and falling back to filtering with in-memory sorting.
        """
        limit = max(1, min(limit, 500))
        query = self.collection

        target_status = "unreviewed" if unreviewed_only else review_status
        if target_status and target_status.lower() in ("all", ""):
            target_status = None

        if target_status:
            query = query.where("review_status", "==", target_status)

        # Attempt 1: Native indexed query with descending creation timestamp ordering
        try:
            ordered_query = query.order_by("created_at", direction="DESCENDING").limit(limit)
            docs = list(ordered_query.stream())
            results: List[Dict[str, Any]] = []
            for doc in docs:
                item = doc.to_dict() or {}
                item["id"] = doc.id
                results.append(item)
            return results
        except Exception as e:
            logger.warning(
                "Firestore ordered query failed (likely composite index required for (review_status, created_at)): %s. "
                "Falling back to unindexed query with in-memory descending sort.",
                e,
            )

        # Attempt 2: Fallback query without Firestore order_by; sort in Python memory
        try:
            fallback_query = query.limit(limit)
            docs = list(fallback_query.stream())
            results = []
            for doc in docs:
                item = doc.to_dict() or {}
                item["id"] = doc.id
                results.append(item)
            results.sort(
                key=lambda x: str(x.get("created_at") or x.get("timestamp") or ""),
                reverse=True,
            )
            return results
        except Exception as e:
            logger.exception("Firestore fallback query failed: %s", e)
            raise

    def update_review_status(
        self,
        alert_id: str,
        review_status: str,
        reviewed_by: Optional[str] = None,
    ) -> bool:
        """Update operator review state ('verified' or 'false_alarm') and record reviewer UID."""
        if review_status not in ("verified", "false_alarm", "unreviewed"):
            raise ValueError(f"Invalid review_status: {review_status}")

        doc_ref = self.collection.document(alert_id)
        doc = doc_ref.get()
        if not doc.exists:
            return False

        update_fields: Dict[str, Any] = {
            "review_status": review_status,
            "reviewed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        if reviewed_by:
            update_fields["reviewed_by"] = reviewed_by

        doc_ref.update(update_fields)
        logger.info("Updated review_status of %s to %s by %s", alert_id, review_status, reviewed_by)
        return True

    def update_cleared_timestamp(
        self,
        camera_id: str,
        track_id: int,
        cleared_timestamp: str,
    ) -> int:
        """Mark active alert cleared in Firestore when luggage is retrieved or moved."""
        query = (
            self.collection.where("camera_id", "==", camera_id)
            .where("track_id", "==", track_id)
            .where("cleared_timestamp", "==", None)
        )
        docs = query.stream()
        count = 0
        for doc in docs:
            doc.reference.update({"cleared_timestamp": cleared_timestamp})
            count += 1

        logger.info("Updated cleared timestamp for cam=%s trk=%d (%d documents)", camera_id, track_id, count)
        return count

    def get_false_alarm_rate(self) -> Tuple[Optional[float], int, int]:
        """Compute false alarm rate across all reviewed alerts in Firestore."""
        # Query verified and false_alarm records
        docs = self.collection.stream()
        total_reviewed = 0
        false_alarms = 0

        for doc in docs:
            status = doc.to_dict().get("review_status")
            if status in ("verified", "false_alarm"):
                total_reviewed += 1
                if status == "false_alarm":
                    false_alarms += 1

        if total_reviewed == 0:
            return None, 0, 0

        rate = false_alarms / total_reviewed
        return rate, total_reviewed, false_alarms
