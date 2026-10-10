"""FastAPI Cloud Backend for RailGuard Vision on Render.

Utilizes:
- Cloud Firestore for distributed alert persistence (No persistent disk required)
- Firebase Authentication for verified operator role verification (Firebase ID Token)
- Dedicated secret token (RAILGUARD_AGENT_TOKEN) for laptop edge camera ingestion
- Path-traversal safe evidence delivery and WebSocket live push notifications
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Set

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Security,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from src.config import SystemConfig, load_config
from src.firebase_service import get_firebase_app, get_firestore_db, verify_firebase_id_token
from src.firestore_repository import FirestoreAlertRepository

# Configure sanitized logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [RAILGUARD_API] %(message)s",
)
logger = logging.getLogger("railguard_api")

# Load configuration
config: SystemConfig = load_config()

# Standard container evidence storage (in /tmp or safe container directory, NOT /var/data)
CONTAINER_EVIDENCE_DIR = Path(tempfile.gettempdir()) / "railguard_evidence"
CONTAINER_SNAPSHOTS_DIR = CONTAINER_EVIDENCE_DIR / "alerts"
CONTAINER_CLIPS_DIR = CONTAINER_EVIDENCE_DIR / "clips"
CONTAINER_SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
CONTAINER_CLIPS_DIR.mkdir(parents=True, exist_ok=True)

# Security bearer scheme
security_scheme = HTTPBearer(auto_error=False)


# ============================================================================
# WebSocket Connection Manager
# ============================================================================
class WebSocketManager:
    """Manages active WebSocket connections for push alert broadcasting."""

    def __init__(self) -> None:
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.discard(websocket)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        dead_connections = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.add(connection)
        for dead in dead_connections:
            self.active_connections.discard(dead)


ws_manager = WebSocketManager()


# ============================================================================
# Lazy Repository Access
# ============================================================================
_repository_instance: Optional[FirestoreAlertRepository] = None


def get_repository() -> FirestoreAlertRepository:
    """Lazily initialize and return the Firestore repository instance."""
    global _repository_instance
    if _repository_instance is None:
        try:
            get_firebase_app(config.firebase)
            _repository_instance = FirestoreAlertRepository(
                collection_name=config.firebase.alerts_collection
            )
        except Exception as e:
            logger.error("Failed to connect to Cloud Firestore: %s", e)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cloud Firestore persistence layer is unavailable. Check server configuration.",
            )
    return _repository_instance


# ============================================================================
# Application Lifespan
# ============================================================================
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("RailGuard Vision Cloud API initialized (Render Free / Firestore Mode).")
    logger.info("Project ID: %s", config.firebase.project_id)
    logger.info("Collection: %s", config.firebase.alerts_collection)
    # Attempt non-blocking background initialization of Firebase
    try:
        get_firebase_app(config.firebase)
    except Exception as e:
        logger.warning("Firebase Admin initialization deferred: %s", e)
    yield
    logger.info("RailGuard Vision Cloud API shutting down.")


app = FastAPI(
    title="RailGuard Vision API",
    version=config.app.version,
    description="Early-warning unattended luggage screening assistance API with Firebase Firestore persistence.",
    lifespan=lifespan,
)

# Enable CORS for Flutter mobile integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Authentication Dependencies
# ============================================================================
def verify_firebase_operator(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security_scheme),
    token_query: Optional[str] = Query(None, alias="token"),
) -> Dict[str, Any]:
    """Verify Firebase ID token for mobile operators and screening officers."""
    raw_token = None
    if credentials:
        raw_token = credentials.credentials
    elif token_query:
        raw_token = token_query

    if not raw_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token. Please provide a Bearer Firebase ID token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Allow development / demo token bypass if explicitly enabled
    if config.firebase.allow_dev_auth_bypass and raw_token in (
        "DEMO_OFFICER_TOKEN",
        "dev-test-token",
        "test-operator-token",
        config.security.api_token,
    ):
        return {
            "uid": "dev-operator-uid",
            "email": "operator@railsafe.local",
            "name": "Screening Operator",
            "role": "operator",
        }

    try:
        decoded_claims = verify_firebase_id_token(
            token=raw_token,
            allow_dev_bypass=config.firebase.allow_dev_auth_bypass,
        )
        return decoded_claims
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def verify_agent_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security_scheme),
    token_query: Optional[str] = Query(None, alias="token"),
) -> str:
    """Verify dedicated secret token for station laptop camera ingestion agent."""
    provided_token = None
    if credentials:
        provided_token = credentials.credentials
    elif token_query:
        provided_token = token_query

    expected_agent = config.security.agent_token

    if not provided_token or provided_token != expected_agent:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid camera ingestion agent token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return provided_token


# ============================================================================
# Pydantic Schemas
# ============================================================================
class ReviewUpdateRequest(BaseModel):
    review_status: str = Field(..., description="'verified' or 'false_alarm'")


class ClearAlertRequest(BaseModel):
    camera_id: str
    track_id: int
    cleared_timestamp: str
    event_uid: Optional[str] = None


class AlertResponse(BaseModel):
    id: Any
    event_uid: str
    timestamp: str
    camera_id: str
    track_id: int
    object_class: str
    event_duration: float
    reason: str
    snapshot_url: str
    clip_url: Optional[str] = None
    review_status: str
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None
    cleared_timestamp: Optional[str] = None
    created_at: Optional[str] = None
    disclaimer: str = (
        "FOR HUMAN REVIEW ONLY — Does not detect explosives or narcotics. Verify physical context."
    )


def serialize_firestore_alert(doc_data: Dict[str, Any]) -> AlertResponse:
    """Serialize Firestore document data into standardized AlertResponse model."""
    doc_id = doc_data.get("id") or doc_data.get("event_uid", "")
    snap_url = doc_data.get("snapshot_url") or f"/api/alerts/{doc_id}/snapshot"
    clip_url = doc_data.get("clip_url") or (f"/api/alerts/{doc_id}/clip" if doc_data.get("clip_key") else None)

    return AlertResponse(
        id=doc_id,
        event_uid=doc_data.get("event_uid", str(doc_id)),
        timestamp=doc_data.get("timestamp", ""),
        camera_id=doc_data.get("camera_id", "CAM-01"),
        track_id=int(doc_data.get("track_id", 0)),
        object_class=doc_data.get("object_class", "luggage"),
        event_duration=float(doc_data.get("event_duration", 0.0)),
        reason=doc_data.get("reason", "Unattended luggage detected"),
        snapshot_url=snap_url,
        clip_url=clip_url,
        review_status=doc_data.get("review_status", "unreviewed"),
        reviewed_by=doc_data.get("reviewed_by"),
        reviewed_at=doc_data.get("reviewed_at"),
        cleared_timestamp=doc_data.get("cleared_timestamp"),
        created_at=doc_data.get("created_at"),
    )


# ============================================================================
# Safe Evidence Path Validation
# ============================================================================
def validate_safe_evidence_path(filepath_str: str) -> Path:
    """Ensure evidence path exists and is confined to the safe container directory."""
    raw_path = Path(filepath_str)
    if not raw_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evidence file not found on cloud storage.",
        )

    resolved = raw_path.resolve()
    try:
        resolved.relative_to(CONTAINER_EVIDENCE_DIR.resolve())
    except ValueError:
        logger.warning("Traversal attempt blocked on evidence: %s", filepath_str)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: Path outside permitted storage boundaries.",
        )
    return resolved


# ============================================================================
# Health & Status Endpoints
# ============================================================================
@app.get("/health", tags=["System"])
def health_check() -> Dict[str, Any]:
    """Lightweight health check for Render process monitoring (Independent of Firestore)."""
    return {
        "status": "healthy",
        "service": "RailGuard Vision Cloud Backend",
        "version": config.app.version,
        "persistence": "Cloud Firestore",
        "disclaimer": config.app.disclaimer,
    }


@app.get("/ready", tags=["System"])
def readiness_check(
    _user: Dict[str, Any] = Depends(verify_firebase_operator),
    repo: FirestoreAlertRepository = Depends(get_repository),
) -> Dict[str, Any]:
    """Protected readiness check confirming Firestore database accessibility."""
    if not repo.check_readiness():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Firestore database unavailable.",
        )
    return {
        "status": "ready",
        "firestore_connected": True,
        "project_id": config.firebase.project_id,
        "collection": config.firebase.alerts_collection,
    }


# ============================================================================
# Ingestion Endpoints (Station Laptop -> Cloud)
# ============================================================================
@app.post(
    "/api/ingest/alerts",
    tags=["Ingestion"],
    status_code=status.HTTP_201_CREATED,
)
async def ingest_alert(
    event_uid: str = Form(...),
    timestamp: str = Form(...),
    camera_id: str = Form(...),
    track_id: int = Form(...),
    object_class: str = Form(...),
    event_duration: float = Form(...),
    reason: str = Form(...),
    snapshot: UploadFile = File(...),
    clip: Optional[UploadFile] = File(None),
    _token: str = Depends(verify_agent_token),
    repo: FirestoreAlertRepository = Depends(get_repository),
) -> Dict[str, Any]:
    """Ingest an alert and evidence from the station laptop into Firestore.
    
    Guaranteed idempotent using event_uid as document key.
    """
    if not event_uid or len(event_uid) < 4:
        raise HTTPException(status_code=400, detail="Invalid event_uid.")
    if track_id < 0 or event_duration < 0:
        raise HTTPException(status_code=400, detail="Invalid track_id or event_duration.")

    # 1. Check for duplicate upload (Idempotency)
    existing = repo.get_alert(event_uid)
    if existing:
        logger.info("Alert %s already exists in Firestore. Suppressing duplicate insert.", event_uid)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "already_exists",
                "alert_id": event_uid,
                "event_uid": event_uid,
                "message": "Event previously ingested. Duplicate suppressed.",
            },
        )

    # 2. Validate snapshot image
    if not snapshot.content_type or not snapshot.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Snapshot must be a valid image file.")

    safe_cam = re.sub(r"[^a-zA-Z0-9_\-]", "_", camera_id)
    safe_uid = re.sub(r"[^a-zA-Z0-9_\-]", "_", event_uid)

    # Save snapshot in safe container storage
    snap_bytes = await snapshot.read()
    if len(snap_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Snapshot file exceeds 10MB limit.")

    snap_filename = f"snap_{safe_cam}_{safe_uid}.jpg"
    snap_dest = CONTAINER_SNAPSHOTS_DIR / snap_filename
    with open(snap_dest, "wb") as f:
        f.write(snap_bytes)

    clip_filename: Optional[str] = None
    if clip is not None and clip.filename:
        clip_bytes = await clip.read()
        if len(clip_bytes) > 30 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Clip file exceeds 30MB limit.")
        ext = ".mp4" if "mp4" in (clip.content_type or "") or clip.filename.endswith(".mp4") else ".avi"
        clip_filename = f"clip_{safe_cam}_{safe_uid}{ext}"
        clip_dest = CONTAINER_CLIPS_DIR / clip_filename
        with open(clip_dest, "wb") as f:
            f.write(clip_bytes)

    # 3. Create document in Cloud Firestore
    doc_data = {
        "event_uid": event_uid,
        "timestamp": timestamp,
        "camera_id": camera_id,
        "track_id": track_id,
        "object_class": object_class,
        "event_duration": event_duration,
        "reason": reason,
        "snapshot_key": snap_filename,
        "snapshot_url": f"/api/alerts/{event_uid}/snapshot",
        "clip_key": clip_filename,
        "clip_url": f"/api/alerts/{event_uid}/clip" if clip_filename else None,
        "review_status": "unreviewed",
        "cleared_timestamp": None,
    }

    doc_id, created = repo.create_or_update_alert(doc_data)

    # Broadcast new alert to connected mobile clients
    created_alert = repo.get_alert(doc_id)
    if created_alert:
        await ws_manager.broadcast({
            "event": "new_alert",
            "alert": serialize_firestore_alert(created_alert).model_dump(),
        })

    logger.info("Alert %s successfully ingested into Firestore.", doc_id)
    return {
        "status": "created",
        "alert_id": doc_id,
        "event_uid": event_uid,
        "message": "Alert successfully recorded in Cloud Firestore.",
    }


@app.post("/api/ingest/alerts/clear", tags=["Ingestion"])
async def ingest_alert_cleared(
    req: ClearAlertRequest,
    _token: str = Depends(verify_agent_token),
    repo: FirestoreAlertRepository = Depends(get_repository),
) -> Dict[str, Any]:
    """Update Firestore when luggage is retrieved or moved on the platform."""
    rows = repo.update_cleared_timestamp(
        camera_id=req.camera_id,
        track_id=req.track_id,
        cleared_timestamp=req.cleared_timestamp,
    )
    if rows > 0:
        await ws_manager.broadcast({
            "event": "alert_cleared",
            "camera_id": req.camera_id,
            "track_id": req.track_id,
            "cleared_timestamp": req.cleared_timestamp,
        })
    return {"status": "success", "rows_updated": rows}


# ============================================================================
# Mobile & Operator Alert Endpoints (Flutter Client)
# ============================================================================
@app.get("/api/alerts", tags=["Alerts"], response_model=List[AlertResponse])
def get_alerts(
    limit: int = Query(50, ge=1, le=500),
    unreviewed_only: bool = Query(False),
    status_filter: Optional[str] = Query(None, alias="status"),
    _user: Dict[str, Any] = Depends(verify_firebase_operator),
    repo: FirestoreAlertRepository = Depends(get_repository),
) -> List[AlertResponse]:
    """Retrieve security alerts from Cloud Firestore."""
    docs = repo.list_alerts(limit=limit, review_status=status_filter, unreviewed_only=unreviewed_only)
    return [serialize_firestore_alert(d) for d in docs]


@app.get("/api/alerts/{alert_id}", tags=["Alerts"], response_model=AlertResponse)
def get_alert_detail(
    alert_id: str,
    _user: Dict[str, Any] = Depends(verify_firebase_operator),
    repo: FirestoreAlertRepository = Depends(get_repository),
) -> AlertResponse:
    """Retrieve single alert details from Cloud Firestore."""
    doc = repo.get_alert(alert_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Alert not found.")
    return serialize_firestore_alert(doc)


@app.patch("/api/alerts/{alert_id}/review", tags=["Alerts"], response_model=AlertResponse)
@app.post("/api/alerts/{alert_id}/review", tags=["Alerts"], response_model=AlertResponse)
async def update_alert_review(
    alert_id: str,
    req: ReviewUpdateRequest,
    user: Dict[str, Any] = Depends(verify_firebase_operator),
    repo: FirestoreAlertRepository = Depends(get_repository),
) -> AlertResponse:
    """Record operator verification or false-alarm review with reviewer UID audit."""
    if req.review_status not in ("verified", "false_alarm"):
        raise HTTPException(
            status_code=400,
            detail="review_status must be either 'verified' or 'false_alarm'.",
        )

    operator_uid = user.get("uid", "operator")
    success = repo.update_review_status(
        alert_id=alert_id,
        review_status=req.review_status,
        reviewed_by=operator_uid,
    )
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found.")

    updated_doc = repo.get_alert(alert_id)
    assert updated_doc is not None

    await ws_manager.broadcast({
        "event": "alert_reviewed",
        "alert_id": alert_id,
        "review_status": req.review_status,
        "reviewed_by": operator_uid,
    })

    return serialize_firestore_alert(updated_doc)


@app.get("/api/alerts/stats/false-alarm-rate", tags=["Alerts"])
def get_false_alarm_rate(
    _user: Dict[str, Any] = Depends(verify_firebase_operator),
    repo: FirestoreAlertRepository = Depends(get_repository),
) -> Dict[str, Any]:
    """Retrieve false alarm rate computed from Firestore operator reviews."""
    rate, total, false_alarms = repo.get_false_alarm_rate()
    return {
        "false_alarm_rate": rate,
        "total_reviewed": total,
        "false_alarms": false_alarms,
        "note": "Rate is computed strictly from operator reviews.",
    }


# ============================================================================
# Evidence Streaming Endpoints
# ============================================================================
@app.get("/api/alerts/{alert_id}/snapshot", tags=["Evidence"])
def get_alert_snapshot(
    alert_id: str,
    _user: Dict[str, Any] = Depends(verify_firebase_operator),
    repo: FirestoreAlertRepository = Depends(get_repository),
) -> FileResponse:
    """Retrieve verified JPEG snapshot evidence for an alert."""
    doc = repo.get_alert(alert_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Alert not found.")

    snap_key = doc.get("snapshot_key")
    if not snap_key:
        raise HTTPException(status_code=404, detail="No snapshot registered for this alert.")

    snap_file = CONTAINER_SNAPSHOTS_DIR / snap_key
    safe_file = validate_safe_evidence_path(str(snap_file))

    return FileResponse(
        path=safe_file,
        media_type="image/jpeg",
        filename=safe_file.name,
    )


@app.get("/api/alerts/{alert_id}/clip", tags=["Evidence"])
def get_alert_clip(
    alert_id: str,
    _user: Dict[str, Any] = Depends(verify_firebase_operator),
    repo: FirestoreAlertRepository = Depends(get_repository),
) -> FileResponse:
    """Retrieve contextual MP4 video clip for an alert."""
    doc = repo.get_alert(alert_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Alert not found.")

    clip_key = doc.get("clip_key")
    if not clip_key:
        raise HTTPException(status_code=404, detail="No video clip available for this alert.")

    clip_file = CONTAINER_CLIPS_DIR / clip_key
    safe_file = validate_safe_evidence_path(str(clip_file))
    media_type = "video/mp4" if safe_file.suffix.lower() == ".mp4" else "video/x-msvideo"

    return FileResponse(
        path=safe_file,
        media_type=media_type,
        filename=safe_file.name,
    )


# ============================================================================
# Real-Time WebSocket Endpoint
# ============================================================================
@app.websocket("/api/ws/alerts")
async def websocket_alerts_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
) -> None:
    """WebSocket endpoint for real-time alert pushes to mobile apps."""
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    try:
        verify_firebase_id_token(token, allow_dev_bypass=config.firebase.allow_dev_auth_bypass)
    except Exception:
        # Check if matches agent token or static dev token
        if token != config.security.agent_token and token != config.security.api_token:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    logger.info("Starting RailGuard Vision Cloud Backend on 0.0.0.0:%d", port)
    uvicorn.run("app.api:app", host="0.0.0.0", port=port, reload=False)
