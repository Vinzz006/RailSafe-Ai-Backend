"""Firebase Admin SDK initialization and authentication service for RailGuard Vision."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import firebase_admin
from firebase_admin import auth as fb_auth, credentials, firestore, storage

from src.config import FirebaseConfig

logger = logging.getLogger("railguard_firebase")

_firebase_app: Optional[firebase_admin.App] = None


def get_firebase_app(config: Optional[FirebaseConfig] = None) -> Optional[firebase_admin.App]:
    """Initialize or retrieve the Firebase Admin app instance."""
    global _firebase_app

    if _firebase_app is not None:
        return _firebase_app

    if len(firebase_admin._apps) > 0:
        _firebase_app = firebase_admin.get_app()
        return _firebase_app

    project_id = config.project_id if config else os.getenv("FIREBASE_PROJECT_ID", "railsafe-ai-92d9f")
    sa_json = config.service_account_json if config else os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    sa_path = config.service_account_path if config else os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
    storage_bucket = (
        config.storage_bucket if config else os.getenv("FIREBASE_STORAGE_BUCKET", "railsafe-ai-92d9f.firebasestorage.app")
    )

    cred: Optional[credentials.Base] = None

    if sa_json:
        try:
            cert_dict = json.loads(sa_json)
            cred = credentials.Certificate(cert_dict)
            logger.info("Initializing Firebase Admin with service account JSON from environment secret.")
        except Exception as e:
            logger.error("Failed to parse FIREBASE_SERVICE_ACCOUNT_JSON: %s", e)
            raise ValueError(f"Invalid FIREBASE_SERVICE_ACCOUNT_JSON format: {e}")

    elif sa_path and os.path.isfile(sa_path):
        try:
            cred = credentials.Certificate(sa_path)
            logger.info("Initializing Firebase Admin with service account file: %s", sa_path)
        except Exception as e:
            logger.error("Failed to load service account certificate from %s: %s", sa_path, e)
            raise ValueError(f"Failed to read service account from {sa_path}: {e}")

    app_options = {"projectId": project_id}
    if storage_bucket:
        app_options["storageBucket"] = storage_bucket

    try:
        if cred:
            _firebase_app = firebase_admin.initialize_app(cred, options=app_options)
        else:
            # Attempt default application credentials (e.g. Cloud Run or Google Cloud environment)
            _firebase_app = firebase_admin.initialize_app(options=app_options)
            logger.info("Initialized Firebase Admin using default credentials for project: %s", project_id)
        return _firebase_app
    except Exception as e:
        logger.warning(
            "Firebase Admin SDK initialization deferred or failed: %s. "
            "Ensure FIREBASE_SERVICE_ACCOUNT_JSON is configured on Render.",
            e,
        )
        return None


def get_firestore_db(config: Optional[FirebaseConfig] = None) -> Any:
    """Return Firestore client instance. Raises RuntimeError if Firebase is uninitialized."""
    app = get_firebase_app(config)
    if not app:
        raise RuntimeError(
            "Firebase Admin SDK is not initialized. Please provide FIREBASE_SERVICE_ACCOUNT_JSON "
            "or FIREBASE_SERVICE_ACCOUNT_PATH."
        )
    return firestore.client(app=app)


def verify_firebase_id_token(token: str, allow_dev_bypass: bool = False) -> Dict[str, Any]:
    """Cryptographically verify Firebase Authentication ID token.
    
    Returns decoded token dictionary with user claims (uid, email, role, etc.).
    Raises ValueError on invalid, expired, or rejected tokens.
    """
    if not token or not token.strip():
        raise ValueError("Token is empty or missing.")

    # Optional development bypass token for offline local unit tests
    if allow_dev_bypass and token in (
        "DEMO_OFFICER_TOKEN",
        "dev-test-token",
        "test-valid-firebase-token",
    ):
        return {
            "uid": "dev-officer-uid",
            "email": "officer@railsafe.local",
            "name": "Screening Officer",
            "role": "operator",
            "auth_time": 1700000000,
        }

    # Verify via Firebase Admin SDK
    try:
        decoded = fb_auth.verify_id_token(token)
        return decoded
    except fb_auth.ExpiredIdTokenError:
        raise ValueError("Firebase ID token has expired. Please refresh the token.")
    except fb_auth.RevokedIdTokenError:
        raise ValueError("Firebase ID token has been revoked.")
    except fb_auth.InvalidIdTokenError as e:
        raise ValueError(f"Invalid Firebase ID token: {e}")
    except Exception as e:
        raise ValueError(f"Authentication failed: {e}")


def get_storage_bucket(config: Optional[FirebaseConfig] = None) -> Any:
    """Return default Firebase Storage bucket, or None if uninitialized or unavailable."""
    app = get_firebase_app(config)
    if not app:
        return None
    try:
        bucket_name = (
            config.storage_bucket
            if config and config.storage_bucket
            else os.getenv("FIREBASE_STORAGE_BUCKET", "railsafe-ai-92d9f.firebasestorage.app")
        )
        return storage.bucket(bucket_name, app=app)
    except Exception as e:
        logger.warning("Firebase Storage bucket access unavailable: %s", e)
        return None


def upload_evidence_blob(
    blob_path: str,
    data: bytes,
    content_type: str = "image/jpeg",
    config: Optional[FirebaseConfig] = None,
) -> bool:
    """Upload binary evidence to Firebase Storage bucket."""
    bucket = get_storage_bucket(config)
    if not bucket:
        return False
    try:
        blob = bucket.blob(blob_path)
        blob.upload_from_string(data, content_type=content_type)
        logger.info("Successfully uploaded evidence to Firebase Storage: %s", blob_path)
        return True
    except Exception as e:
        logger.warning("Failed to upload evidence to Firebase Storage (%s): %s", blob_path, e)
        return False


def download_evidence_blob(
    blob_path: str,
    config: Optional[FirebaseConfig] = None,
) -> Optional[bytes]:
    """Download binary evidence from Firebase Storage bucket."""
    bucket = get_storage_bucket(config)
    if not bucket:
        return None
    try:
        blob = bucket.blob(blob_path)
        if blob.exists():
            return blob.download_as_bytes()
        return None
    except Exception as e:
        logger.warning("Failed to download evidence from Firebase Storage (%s): %s", blob_path, e)
        return None

