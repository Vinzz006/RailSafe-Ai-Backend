# RailGuard Vision — Firebase Firestore Migration & Render Free Deployment Plan

## 1. Root Cause Analysis of Render Startup Failure

### Incident Traceback
```text
PermissionError: [Errno 13] Permission denied: '/var/data'
  at os.makedirs(name, mode=mode, exist_ok=True)
  at src/db.py:35 in __init__ (AlertDatabaseManager)
  at app/api.py:50 in <module>
```

### Exact Mechanism of Failure
1. `app/api.py` unconditionally instantiated `AlertDatabaseManager(config.storage.db_path)` and ran `Path(config.storage.alerts_dir).mkdir(parents=True, exist_ok=True)` at the **module top-level**.
2. On Render Free Tier, no persistent disk is mounted at `/var/data` (persistent disks require a paid Starter plan). Linux container processes run without root permissions to create directories at the filesystem root `/`.
3. Because this code ran during top-level Python module loading, `uvicorn app.api:app` crashed before binding to `$PORT` or serving `/health`.

---

## 2. Target Architecture

```
┌────────────────────────────────────────────────────────┐
│               STATION LAPTOP (EDGE)                    │
│  • Laptop Webcam (cv2.VideoCapture(0)) / RTSP          │
│  • YOLOv8n Object Detection & ByteTrack Tracking       │
│  • Unattended Luggage Rules Engine                     │
│  • Local SQLite (data/security_alerts.db)              │
│  • Local Evidence (data/alerts/ & data/clips/)         │
│  • Outbound Cloud Forwarder (src/agent.py)             │
└───────────────────────────┬────────────────────────────┘
                            │ Outbound HTTPS (POST /api/ingest/alerts)
                            │ Header: Authorization: Bearer <RAILGUARD_AGENT_TOKEN>
                            ▼
┌────────────────────────────────────────────────────────┐
│           RENDER CLOUD BACKEND (FREE TIER)             │
│  • FastAPI Service (app/api.py) on $PORT               │
│  • NO /var/data dependency; NO local SQLite            │
│  • Ingestion Auth: RAILGUARD_AGENT_TOKEN               │
│  • Operator Auth: Firebase Auth (Firebase ID Token)    │
│  • Firestore Repository (src/firestore_repository.py)  │
│  • Collection: 'railguard_alerts'                      │
└─────────────┬────────────────────────────┬─────────────┘
              │ Read / Write               │ Verify Tokens
              ▼                            ▼
┌───────────────────────────┐ ┌──────────────────────────┐
│     CLOUD FIRESTORE       │ │  FIREBASE AUTHENTICATION │
│ • Document per Alert      │ │ • Operators / Officers   │
│ • Idempotent on event_uid │ │ • Cryptographic JWT      │
│ • State & Review Audit    │ │   Verification           │
└───────────────────────────┘ └──────────────────────────┘
              ▲
              │ Authenticated HTTPS (Bearer <Firebase_ID_Token>)
┌─────────────┴──────────────────────────────────────────┐
│              FLUTTER MOBILE APP (CLIENT)               │
│  • Signs in via Firebase Authentication                │
│  • Obtains ID Token: user.getIdToken()                 │
│  • RailGuardApiService calls Render Backend APIs       │
│  • Displays Alerts & Submits Verified/False Alarm      │
└────────────────────────────────────────────────────────┘
```

---

## 3. Implementation Phases

### Phase 1: Clean Decoupling of Cloud API from Local SQLite
- Remove `AlertDatabaseManager` and `/var/data` directory creation from `app/api.py`.
- Preserve `AlertDatabaseManager` and `src/db.py` for local edge buffering on the laptop.
- Cloud storage config will use standard container temp/fallback paths for any transient local buffers, never hardcoding or requiring `/var/data`.

### Phase 2: Cloud Firestore Repository (`src/firestore_repository.py`)
- Implement `FirestoreAlertRepository` providing:
  - `create_or_update_alert(...)` (idempotent on `event_uid` as Document ID)
  - `get_alert(alert_id: str)`
  - `list_alerts(limit=50, review_status=None, unreviewed_only=False, start_after=None)`
  - `update_review_status(alert_id: str, review_status: str, reviewed_by: str)`
  - `update_cleared_timestamp(camera_id: str, track_id: int, cleared_timestamp: str)`
  - `get_false_alarm_rate()`
- Document schema:
  - `event_uid`: Unique event key
  - `camera_id`: Camera identifier
  - `track_id`: Integer track ID
  - `object_class`: Luggage category
  - `event_duration`: Floating point seconds
  - `reason`: Alert rule explanation
  - `timestamp`: Event ISO-8601 string
  - `server_timestamp`: Firestore server timestamp
  - `review_status`: `'unreviewed' | 'verified' | 'false_alarm'`
  - `reviewed_by`: UID of reviewing operator
  - `reviewed_at`: ISO timestamp of review
  - `cleared_timestamp`: Timestamp when luggage retrieved/cleared
  - `snapshot_key` / `snapshot_url`: Evidence metadata
  - `clip_key` / `clip_url`: Evidence metadata
  - `disclaimer`: Mandatory safety boundary message

### Phase 3: Firebase Admin SDK & Authentication
- Initialize Firebase Admin SDK safely (singleton pattern).
- Support configuration via:
  - `FIREBASE_PROJECT_ID` (default: `"railsafe-ai-92d9f"`)
  - `FIREBASE_SERVICE_ACCOUNT_JSON` (raw JSON string secret on Render) or `FIREBASE_SERVICE_ACCOUNT_PATH`
  - ADC fallback for Google Cloud environments.
- Implement FastAPI security dependency:
  - Validates `Authorization: Bearer <Firebase_ID_Token>` using `firebase_admin.auth.verify_id_token`.
  - Rejects missing, invalid, or expired tokens with `HTTP 401`.
  - Extracts authenticated user UID and email.

### Phase 4: Separate Laptop Agent Authentication
- `POST /api/ingest/alerts` and `POST /api/ingest/alerts/clear` protected strictly via `RAILGUARD_AGENT_TOKEN`.
- The edge camera agent does not need to impersonate a human Firebase user.
- Local edge agent preserves its local SQLite logging and bounded upload queue with backoff.

### Phase 5: Cloud Evidence Handling
- Ingested snapshots and clips:
  - If Firebase Storage is initialized (via default bucket `railsafe-ai-92d9f.firebasestorage.app`), store files as Cloud Storage blobs and record storage paths/signed URLs.
  - If Firebase Storage credentials are not provided, store evidence safely in a non-privileged container directory (e.g. `Path(tempfile.gettempdir()) / "railguard_evidence"`) or provide an evidence abstraction layer.
  - Never attempt to write to `/var/data`.

### Phase 6: Flutter Integration
- Update `lib/services/railguard_api_service.dart` to take `AuthService` and retrieve the real user ID token dynamically via `_authService.getIdToken()`.
- Field officers reviewing an alert have their verified UID recorded on Firestore documents.

### Phase 7: Render Configuration & Documentation
- Update `render.yaml` to remove persistent disk requirements.
- Document Render Free tier environment variables:
  - `FIREBASE_PROJECT_ID=railsafe-ai-92d9f`
  - `FIREBASE_SERVICE_ACCOUNT_JSON` (secret)
  - `FIRESTORE_ALERTS_COLLECTION=railguard_alerts`
  - `RAILGUARD_AGENT_TOKEN` (secret)
- Update `RENDER_DEPLOYMENT.md` with complete setup instructions.

---

## 4. Completed Implementation & Verification Summary

### Implementation Deliverables
1. **`src/config.py`**: Added `FirebaseConfig` with environment variable overrides (`FIREBASE_PROJECT_ID`, `FIREBASE_SERVICE_ACCOUNT_JSON`, `FIRESTORE_ALERTS_COLLECTION`, `FIREBASE_STORAGE_BUCKET`, `FIREBASE_AUTH_DEV_BYPASS`).
2. **`src/firebase_service.py`**: Safe singleton Firebase Admin SDK initialization supporting raw JSON string secret, credential file path, and Google ADC. Cryptographic JWT verification for Firebase ID tokens.
3. **`src/firestore_repository.py`**: Complete repository layer decoupling cloud persistence from SQLite:
   - Idempotent alert creation on `event_uid`.
   - Alert retrieval, filtering by review status, pagination with cursors.
   - Status updates with operator UID audit trail.
   - Luggage clearing lifecycle timestamps.
   - False alarm rate analytics.
   - Protected readiness checks (`check_readiness()`).
4. **`app/api.py`**:
   - Zero `/var/data` or SQLite calls during module import or runtime.
   - Public lightweight `/health` endpoint for Render uptime monitoring.
   - Protected `/ready` endpoint verifying Firestore connectivity.
   - Dedicated `RAILGUARD_AGENT_TOKEN` for laptop edge ingest (`/api/ingest/alerts`).
   - Bearer Firebase ID token authentication for operator endpoints (`/api/alerts`, `/api/alerts/{id}/review`).
   - Hardened evidence streaming using container temporary storage (`tempfile.gettempdir() / "railguard_evidence"`), preventing path traversal.
5. **`requirements.txt`**: Added `firebase-admin>=6.2.0`.
6. **`render.yaml`**: Configured for Render `plan: free` with zero disk mounts; mapped environment variables.
7. **`lib/services/railguard_api_service.dart`**: Injected `AuthService` into Dart client; requests dynamically fetch and attach `Authorization: Bearer <token>`.
8. **`firestore.rules`**: Restricted direct client writes to Firestore; all mutations must route through validated FastAPI backend.
9. **`RENDER_DEPLOYMENT.md`**: Created detailed step-by-step deployment runbook.

### Automated Test Results
Test run executed with `python -m pytest -v`:
- **Total tests**: 41
- **Passed**: 41
- **Failed / Errors**: 0
- **Duration**: ~10.0 seconds
- **Covered suites**:
  - `test_api.py` (health, readiness, token auth, ingestion idempotency, validation rejection, review lifecycle, evidence delivery & path traversal defense)
  - `test_firestore_repository.py` (create idempotency, review updates, cleared timestamp, false alarm analytics)
  - `test_alerts.py`, `test_cloud_agent.py`, `test_config.py`, `test_db.py`, `test_evaluation.py`, `test_persistence.py`, `test_rules.py`, `test_tracker.py`

