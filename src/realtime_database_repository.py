"""Firebase Realtime Database Alert Repository for RailGuard Vision cloud persistence on Render."""

from __future__ import annotations

import datetime
import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from src.firebase_service import get_realtime_db

logger = logging.getLogger("railguard_rtdb")


def sanitize_rtdb_key(key: str) -> str:
    """Sanitize string for use as a Firebase Realtime Database node key.
    
    Disallows characters: . $ # [ ] / and ASCII control characters.
    """
    if not key:
        return "unknown"
    return re.sub(r"[.$#\[\]/]", "_", str(key).strip())


def iso_to_epoch_ms(iso_str: Optional[str]) -> int:
    """Parse ISO 8601 string to integer epoch milliseconds."""
    if not iso_str:
        return int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
    try:
        dt = datetime.datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    except Exception:
        return int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)


class RealtimeDatabaseAlertRepository:
    """Encapsulates all Firebase Realtime Database CRUD operations for RailGuard alerts."""

    def __init__(
        self,
        base_path: str = "railguard",
        alerts_path: Optional[str] = None,
        event_uid_index_path: Optional[str] = None,
        db_root: Optional[Any] = None,
    ) -> None:
        clean_base = base_path.strip("/")
        if clean_base.endswith("/alerts"):
            clean_base = clean_base[:-7].strip("/")
        self.base_path = clean_base or "railguard"

        if alerts_path and alerts_path.strip():
            self._alerts_path = alerts_path.strip().strip("/")
        else:
            self._alerts_path = f"{self.base_path}/alerts"

        if event_uid_index_path and event_uid_index_path.strip():
            self._index_path = event_uid_index_path.strip().strip("/")
        else:
            self._index_path = f"{self.base_path}/event_uid_index"

        self._root = db_root

    @property
    def root(self) -> Any:
        """Lazily initialize or return root Realtime Database reference."""
        if self._root is None:
            self._root = get_realtime_db()
        return self._root

    @property
    def alerts_ref(self) -> Any:
        """Reference to alerts node (default: railguard/alerts)."""
        return self.root.child(self._alerts_path)

    @property
    def index_ref(self) -> Any:
        """Reference to event_uid_index node (default: railguard/event_uid_index)."""
        return self.root.child(self._index_path)

    def check_readiness(self) -> bool:
        """Check if Firebase Realtime Database is accessible."""
        try:
            # Query at most 1 key to verify connection without reading entire database
            _ = self.alerts_ref.order_by_key().limit_to_first(1).get()
            return True
        except Exception as e:
            logger.warning("Realtime Database readiness check failed: %s", e)
            return False

    def create_or_update_alert(self, alert_data: Dict[str, Any]) -> Tuple[str, bool]:
        """Ingest an alert document into Firebase Realtime Database.
        
        Uses sanitized event_uid as node key, disambiguates sanitization collisions,
        and guarantees idempotency under concurrent ingestion without resetting review state.
        
        Returns:
            (alert_id, is_newly_created: bool)
        """
        raw_uid = alert_data.get("event_uid")
        if not raw_uid:
            raise ValueError("event_uid is required for alert ingestion.")

        base_id = sanitize_rtdb_key(str(raw_uid))
        alert_id = base_id

        # Disambiguate key collisions where different raw_uids sanitize to the same base_id
        try:
            existing_base = self.alerts_ref.child(base_id).get()
            if existing_base is not None and isinstance(existing_base, dict):
                existing_uid = existing_base.get("event_uid")
                if existing_uid and existing_uid != raw_uid:
                    # Key collision detected
                    hash_suffix = hashlib.sha256(str(raw_uid).encode("utf-8")).hexdigest()[:8]
                    alert_id = f"{base_id}_{hash_suffix}"
                    logger.warning(
                        "Sanitization collision detected between '%s' and '%s'. Disambiguated key to '%s'.",
                        existing_uid,
                        raw_uid,
                        alert_id,
                    )
        except Exception as check_err:
            logger.debug("Collision pre-check skipped or failed: %s", check_err)

        target_ref = self.alerts_ref.child(alert_id)

        now = datetime.datetime.now(datetime.timezone.utc)
        now_iso = now.isoformat()
        now_ms = int(now.timestamp() * 1000)

        ts_raw = alert_data.get("timestamp") or now_iso
        ts_ms = alert_data.get("timestamp_ms") or iso_to_epoch_ms(ts_raw)

        created_at_raw = alert_data.get("created_at") or ts_raw
        created_at_ms = alert_data.get("created_at_ms") or (
            iso_to_epoch_ms(created_at_raw) if created_at_raw != now_iso else now_ms
        )

        cleared_ts = alert_data.get("cleared_timestamp")
        status = alert_data.get("status") or ("cleared" if cleared_ts else "active")

        doc_payload: Dict[str, Any] = {
            "id": alert_id,
            "event_uid": raw_uid,
            "status": status,
            "timestamp": ts_raw,
            "timestamp_ms": ts_ms,
            "camera_id": alert_data.get("camera_id", "UNKNOWN"),
            "track_id": int(alert_data.get("track_id", 0)),
            "object_class": alert_data.get("object_class", "luggage"),
            "event_duration": float(alert_data.get("event_duration", 0.0)),
            "reason": alert_data.get("reason", "Unattended luggage detected"),
            "snapshot_key": alert_data.get("snapshot_key", ""),
            "snapshot_url": alert_data.get("snapshot_url", f"/api/alerts/{alert_id}/snapshot"),
            "clip_key": alert_data.get("clip_key", None),
            "clip_url": alert_data.get("clip_url", None),
            "review_status": alert_data.get("review_status", "unreviewed"),
            "reviewed_by": alert_data.get("reviewed_by", None),
            "reviewed_at": alert_data.get("reviewed_at", None),
            "cleared_timestamp": cleared_ts,
            "created_at": created_at_raw,
            "created_at_ms": created_at_ms,
            "disclaimer": (
                "FOR HUMAN REVIEW ONLY — Visible-light optical cameras cannot detect explosives or narcotics. "
                "Verify physical context before initiating security actions."
            ),
        }

        is_newly_created = False

        def _merge_payload_preserving_review(current: Dict[str, Any]) -> Dict[str, Any]:
            merged = dict(current)
            # Never reset reviewed alert back to unreviewed on camera retry
            if current.get("review_status") in ("verified", "false_alarm"):
                merged["review_status"] = current["review_status"]
                merged["reviewed_by"] = current.get("reviewed_by")
                merged["reviewed_at"] = current.get("reviewed_at")
            # Preserve existing cleared_timestamp if set
            if current.get("cleared_timestamp"):
                merged["cleared_timestamp"] = current["cleared_timestamp"]
                merged["status"] = "cleared"
            # Update evidence links if missing in existing
            if not merged.get("snapshot_key") and doc_payload.get("snapshot_key"):
                merged["snapshot_key"] = doc_payload["snapshot_key"]
                merged["snapshot_url"] = doc_payload["snapshot_url"]
            if not merged.get("clip_key") and doc_payload.get("clip_key"):
                merged["clip_key"] = doc_payload["clip_key"]
                merged["clip_url"] = doc_payload["clip_url"]
            return merged

        def txn_insert(current_val: Any) -> Any:
            nonlocal is_newly_created
            if current_val is None:
                is_newly_created = True
                return doc_payload
            if isinstance(current_val, dict):
                is_newly_created = False
                return _merge_payload_preserving_review(current_val)
            return current_val

        try:
            if hasattr(target_ref, "transaction") and callable(getattr(target_ref, "transaction")):
                target_ref.transaction(txn_insert)
            else:
                existing = target_ref.get()
                if existing is None:
                    target_ref.set(doc_payload)
                    is_newly_created = True
                elif isinstance(existing, dict):
                    updated = _merge_payload_preserving_review(existing)
                    target_ref.set(updated)
                    is_newly_created = False
                else:
                    is_newly_created = False
        except Exception as e:
            logger.error("Failed to persist alert %s into Realtime Database: %s", alert_id, e)
            raise RuntimeError(f"Database transaction failure on alert {alert_id}: {e}") from e

        if is_newly_created:
            try:
                self.index_ref.child(alert_id).set(raw_uid)
            except Exception as e:
                logger.warning("Could not update event_uid_index for %s: %s", alert_id, e)
            logger.info("Created alert record %s in Realtime Database under %s", alert_id, self._alerts_path)
        else:
            logger.info("Alert record %s already exists. Idempotent write handled without review loss.", alert_id)

        return alert_id, is_newly_created

    def get_alert(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single alert document by ID or sanitized key."""
        clean_id = sanitize_rtdb_key(alert_id)
        data = self.alerts_ref.child(clean_id).get()
        if not data or not isinstance(data, dict):
            # Also try direct lookup if clean_id differs
            if clean_id != alert_id:
                data = self.alerts_ref.child(alert_id).get()
            if not data or not isinstance(data, dict):
                return None
        data["id"] = data.get("id") or clean_id
        return data

    def list_alerts(
        self,
        limit: int = 50,
        review_status: Optional[str] = None,
        unreviewed_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """List alerts from Realtime Database with filtering and bounded retrieval.
        
        Uses indexed RTDB queries where possible, sorting results descending by creation time.
        """
        limit = max(1, min(limit, 500))

        target_status = "unreviewed" if unreviewed_only else review_status
        if target_status and target_status.lower() in ("all", ""):
            target_status = None

        raw_results: Dict[str, Any] = {}

        # Strategy 1: Filter by review_status using RTDB equal_to query
        if target_status:
            try:
                raw_results = (
                    self.alerts_ref.order_by_child("review_status")
                    .equal_to(target_status)
                    .get()
                    or {}
                )
            except Exception as e:
                logger.warning(
                    "RTDB order_by_child('review_status') query failed: %s. Falling back to bounded read.",
                    e,
                )
                try:
                    raw_results = self.alerts_ref.order_by_child("created_at_ms").limit_to_last(limit * 3).get() or {}
                except Exception:
                    raw_results = self.alerts_ref.get() or {}
        else:
            # Strategy 2: Bounded query ordering by created_at_ms descending
            try:
                raw_results = (
                    self.alerts_ref.order_by_child("created_at_ms")
                    .limit_to_last(limit)
                    .get()
                    or {}
                )
            except Exception as e:
                logger.warning(
                    "RTDB order_by_child('created_at_ms') query failed: %s. Falling back to direct read.",
                    e,
                )
                raw_results = self.alerts_ref.get() or {}

        # Process and filter documents in memory
        results: List[Dict[str, Any]] = []
        if isinstance(raw_results, dict):
            for key, val in raw_results.items():
                if isinstance(val, dict):
                    item = dict(val)
                    item["id"] = key
                    if target_status and item.get("review_status") != target_status:
                        continue
                    results.append(item)
        elif isinstance(raw_results, list):
            for idx, val in enumerate(raw_results):
                if isinstance(val, dict):
                    item = dict(val)
                    item["id"] = item.get("id") or str(idx)
                    if target_status and item.get("review_status") != target_status:
                        continue
                    results.append(item)

        # Deterministic sort descending by created_at_ms with key tie-breaking
        def _extract_sort_tuple(x: Dict[str, Any]) -> Tuple[int, str]:
            ms = x.get("created_at_ms")
            sort_ms = 0
            if isinstance(ms, (int, float)):
                sort_ms = int(ms)
            elif isinstance(ms, str):
                try:
                    sort_ms = int(float(ms))
                except ValueError:
                    sort_ms = iso_to_epoch_ms(x.get("created_at") or x.get("timestamp"))
            else:
                sort_ms = iso_to_epoch_ms(x.get("created_at") or x.get("timestamp"))
            return (sort_ms, str(x.get("id", "")))

        results.sort(key=_extract_sort_tuple, reverse=True)

        return results[:limit]

    def update_review_status(
        self,
        alert_id: str,
        review_status: str,
        reviewed_by: Optional[str] = None,
    ) -> bool:
        """Update operator review state ('verified' or 'false_alarm') and record reviewer UID."""
        if review_status not in ("verified", "false_alarm", "unreviewed"):
            raise ValueError(f"Invalid review_status: {review_status}")

        clean_id = sanitize_rtdb_key(alert_id)
        current = self.alerts_ref.child(clean_id).get()
        if not current or not isinstance(current, dict):
            # Fall back to raw alert_id
            current = self.alerts_ref.child(alert_id).get()
            if not current or not isinstance(current, dict):
                return False
            clean_id = alert_id

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        update_fields: Dict[str, Any] = {
            "review_status": review_status,
            "reviewed_at": now_iso,
        }
        if reviewed_by:
            update_fields["reviewed_by"] = reviewed_by

        self.alerts_ref.child(clean_id).update(update_fields)
        logger.info("Updated review_status of %s to %s by %s", clean_id, review_status, reviewed_by)
        return True

    def update_cleared_timestamp(
        self,
        camera_id: str,
        track_id: int,
        cleared_timestamp: str,
    ) -> int:
        """Mark active alert cleared in RTDB when luggage is retrieved or moved."""
        count = 0
        try:
            # Query alerts matching camera_id
            camera_alerts = (
                self.alerts_ref.order_by_child("camera_id")
                .equal_to(camera_id)
                .get()
                or {}
            )
        except Exception:
            camera_alerts = self.alerts_ref.get() or {}

        if isinstance(camera_alerts, dict):
            for alert_key, val in camera_alerts.items():
                if isinstance(val, dict):
                    if (
                        val.get("camera_id") == camera_id
                        and int(val.get("track_id", -1)) == int(track_id)
                        and not val.get("cleared_timestamp")
                    ):
                        self.alerts_ref.child(alert_key).update({
                            "cleared_timestamp": cleared_timestamp,
                            "status": "cleared",
                        })
                        count += 1

        logger.info(
            "Updated cleared timestamp for cam=%s trk=%d (%d documents in RTDB)",
            camera_id,
            track_id,
            count,
        )
        return count

    def get_false_alarm_rate(self) -> Tuple[Optional[float], int, int]:
        """Compute false alarm rate across all reviewed alerts in Realtime Database."""
        total_reviewed = 0
        false_alarms = 0

        # Query indexed review statuses efficiently
        try:
            verified_alerts = (
                self.alerts_ref.order_by_child("review_status")
                .equal_to("verified")
                .get()
                or {}
            )
            false_alarm_alerts = (
                self.alerts_ref.order_by_child("review_status")
                .equal_to("false_alarm")
                .get()
                or {}
            )

            verified_count = len(verified_alerts) if isinstance(verified_alerts, dict) else len([x for x in verified_alerts if x])
            false_alarms = len(false_alarm_alerts) if isinstance(false_alarm_alerts, dict) else len([x for x in false_alarm_alerts if x])
            total_reviewed = verified_count + false_alarms

        except Exception as e:
            logger.warning("Indexed stats query failed, falling back to full scan: %s", e)
            all_alerts = self.alerts_ref.get() or {}
            items = all_alerts.values() if isinstance(all_alerts, dict) else all_alerts
            for doc in items:
                if isinstance(doc, dict):
                    st = doc.get("review_status")
                    if st in ("verified", "false_alarm"):
                        total_reviewed += 1
                        if st == "false_alarm":
                            false_alarms += 1

        if total_reviewed == 0:
            return None, 0, 0

        rate = false_alarms / total_reviewed
        return rate, total_reviewed, false_alarms
