# RailGuard Vision — Render Free Cloud Deployment with Firebase Firestore

> [!IMPORTANT]
> **FOR HUMAN REVIEW ONLY** — RailGuard Vision is an optical surveillance early-warning prototype engineered for Indian Railways station premises. Visible-light optical cameras **CANNOT detect explosives, narcotics, or chemical hazards**. Alerts prioritize locations for human security personnel (RPF/GRP) to deploy certified physical inspection tools (X-ray, Explosive Trace Detection / ETD, and canine squads). The system does not implement facial recognition, biometric profiling, or automated enforcement.

---

## 1. Resolution of the `/var/data` PermissionError

### Root Cause Analysis
During previous deployment attempts on Render, the backend process failed on startup with:
```text
PermissionError: [Errno 13] Permission denied: '/var/data'
```
**Why this occurred:**
`app/api.py` was previously instantiating a local SQLite database at `/var/data/security_alerts.db` at the module top level. On the **Render Free Tier**, no persistent disk exists at `/var/data` (disks require a paid Starter plan), and container processes run as a standard non-root user without permissions to create directories at the filesystem root `/`. This caused `uvicorn` to crash during module import before binding to `$PORT` or serving `/health`.

**How it is now resolved:**
1. **Decoupled Cloud from SQLite:** The cloud API on Render no longer touches `/var/data` or uses local SQLite.
2. **Cloud Firestore Persistence:** Cloud alert records, timestamps, and operator review states are persisted directly in **Cloud Firestore** (`railguard_alerts` collection).
3. **100% Render Free Tier Compatible:** No attached persistent disk is required on Render.
4. **Local Edge Preserved:** The station laptop continues using its local SQLite database (`data/security_alerts.db`) for real-time video buffering and offline resilience.

---

## 2. Distributed Architecture

```
┌──────────────────────────────────────┐
│       STATION LAPTOP (EDGE)          │
│ • Laptop USB Webcam / RTSP CCTV      │
│ • YOLOv8n + ByteTrack Multi-Tracking │
│ • Spatial-Temporal Rules Engine      │
│ • Local SQLite (security_alerts.db)  │
│ • Local Evidence (data/alerts/)      │
│ • Asynchronous Agent (src/agent.py)  │
└──────────────────┬───────────────────┘
                   │ Outbound Authenticated HTTPS (POST /api/ingest/alerts)
                   │ Header: Authorization: Bearer <RAILGUARD_AGENT_TOKEN>
                   ▼
┌──────────────────────────────────────┐
│     RENDER CLOUD BACKEND (FREE)      │
│ • FastAPI Web Service (app/api.py)   │
│ • Dynamic Port ($PORT)               │
│ • Free Tier Plan (No Disks Needed)   │
│ • Ingestion Auth: Agent Secret Token │
│ • Operator Auth: Firebase ID Token   │
│ • WebSockets (WSS /api/ws/alerts)    │
└──────────┬─────────────────┬─────────┘
           │                 │
           ▼                 ▼
┌──────────────────┐ ┌──────────────────┐
│ CLOUD FIRESTORE  │ │  FIREBASE AUTH   │
│ Collection:      │ │ Cryptographic    │
│ railguard_alerts │ │ Token Validation │
└──────────────────┘ └──────────────────┘
           ▲
           │ Authenticated HTTPS (Bearer <Firebase_ID_Token>)
┌──────────┴───────────────────────────┐
│         FLUTTER MOBILE APP           │
│ • Signs in via Firebase Auth         │
│ • RailGuardApiService calls Render   │
│ • RPF Field Officer Alerts Screen    │
│ • Submits Verified / False Alarm     │
└──────────────────────────────────────┘
```

---

## 3. Render Dashboard Setup (Step-by-Step)

Follow these steps in the Render Dashboard to deploy on the **Free Tier**:

### Step 3.1: Create Web Service
1. Log in to [render.com](https://render.com) and click **New +** > **Web Service**.
2. Select your connected repository: `RailSafe-AI`.
3. Configure the service settings:
   - **Name:** `railguard-vision-api`
   - **Region:** `Oregon (US West)` or `Singapore`
   - **Branch:** `main`
   - **Root Directory:** `RailGuard_Vision` *(Important: points to the subproject folder)*
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command:** `uvicorn app.api:app --host 0.0.0.0 --port $PORT`
   - **Plan:** `Free` *(No paid persistent disk needed)*

### Step 3.2: Set Health Check Path
- Under **Advanced Settings**, set **Health Check Path** to `/health`.
*(Note: `/health` runs without external dependencies so Render's health prober succeeds immediately).*

### Step 3.3: Configure Environment Variables
Under the **Environment** tab in Render, enter the following key-value pairs:

| KEY | VALUE | Description |
| :--- | :--- | :--- |
| `PYTHON_VERSION` | `3.10.13` | Recommended stable Python version |
| `FIREBASE_PROJECT_ID` | `railsafe-ai-92d9f` | Your Firebase Project ID |
| `FIRESTORE_ALERTS_COLLECTION` | `railguard_alerts` | Firestore collection for luggage alerts |
| `FIREBASE_STORAGE_BUCKET` | `railsafe-ai-92d9f.firebasestorage.app` | Default Firebase Storage bucket |
| `ALLOW_DEV_AUTH_BYPASS` | `false` | Disables testing bypass in production |
| `RAILGUARD_AGENT_TOKEN` | *(Click **Generate** or enter random string)* | Secret token for station laptop agent |
| `FIREBASE_SERVICE_ACCOUNT_JSON`| *(Paste JSON contents - see Step 3.4)* | Service account credentials secret |

> [!IMPORTANT]
> Do NOT set `RAILGUARD_DATA_DIR=/var/data` on the Render Free Tier.

### Step 3.4: How to Obtain `FIREBASE_SERVICE_ACCOUNT_JSON`
1. Go to the [Firebase Console](https://console.firebase.google.com/).
2. Select your project: **railsafe-ai-92d9f**.
3. Click the **Gear icon (Project settings)** > **Service accounts** tab.
4. Click **Generate new private key** and confirm by clicking **Generate key**.
5. Open the downloaded `.json` file in a text editor (e.g., Notepad).
6. Copy the **entire JSON text** and paste it as the value of `FIREBASE_SERVICE_ACCOUNT_JSON` in Render.

Click **Create Web Service**.

---

## 4. Verification & Health Checks

Once Render displays **Live**, verify backend operation using Windows PowerShell or curl:

### Health Check (Public, Process Liveness)
```powershell
curl -i https://railguard-vision-api.onrender.com/health
```
**Expected Response:**
```json
HTTP/1.1 200 OK
Content-Type: application/json

{
  "status": "healthy",
  "service": "RailGuard Vision Cloud Backend",
  "version": "1.0.0",
  "persistence": "Cloud Firestore",
  "disclaimer": "FOR HUMAN REVIEW ONLY — Does not detect explosives, chemical substances, or narcotics."
}
```

---

## 5. Starting the Station Laptop Detection Agent

On the laptop connected to the webcam on the railway platform:

### Windows PowerShell:
```powershell
cd "c:\Users\shanm\Downloads\RailSafe AI\RailGuard_Vision"

# Set cloud URL and your agent token from Render
$env:RAILGUARD_CLOUD_URL = "https://railguard-vision-api.onrender.com"
$env:RAILGUARD_AGENT_TOKEN = "<YOUR_RAILGUARD_AGENT_TOKEN>"

# Run the agent with live desktop visual display
python run_agent.py --display
```

### Alternatively via Batch File:
```cmd
cd "c:\Users\shanm\Downloads\RailSafe AI\RailGuard_Vision"
set RAILGUARD_CLOUD_URL=https://railguard-vision-api.onrender.com
set RAILGUARD_AGENT_TOKEN=<YOUR_RAILGUARD_AGENT_TOKEN>
run_agent.bat
```

### Edge-to-Cloud Behavior:
1. Laptop captures local frames from webcam `0`.
2. Runs YOLOv8n object detection and ByteTrack tracking locally.
3. Evaluates luggage stationarity and nearest-person heuristics.
4. When an unattended bag triggers `ALERT`, it logs to local SQLite and queues an upload task.
5. The background forwarder streams the alert and snapshot JPEG to Render via `POST /api/ingest/alerts`.
6. Render creates a document in Cloud Firestore (`railguard_alerts`).
7. If internet disconnects, the edge camera continues operating uninterrupted. Up to 100 alerts queue locally and retry with exponential backoff once reconnected.

---

## 6. Flutter Mobile App Configuration

The Flutter app interfaces with the cloud backend through [`lib/services/railguard_api_service.dart`](file:///c:/Users/shanm/Downloads/RailSafe%20AI/lib/services/railguard_api_service.dart).

1. In the Flutter app settings (or [`LocalStorageService`](file:///c:/Users/shanm/Downloads/RailSafe%20AI/lib/services/local_storage_service.dart)), configure:
   - **Backend Base URL:** `https://railguard-vision-api.onrender.com`
2. When the user signs in with email/password via Firebase Authentication, `RailGuardApiService` automatically retrieves the user's real Firebase ID token via `AuthService.getIdToken()` and attaches:
   ```text
   Authorization: Bearer <Firebase_ID_Token>
   ```
3. Operators can:
   - View recent unattended luggage alerts in real-time.
   - Review an alert as **Verified** or **False Alarm**.
   - The reviewing officer's Firebase UID is recorded on the Firestore document.

---

## 7. Cloud Firestore Security Rules

Ensure your `firestore.rules` has the following rule:

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {

    // Helper functions
    function isAuthenticated() {
      return request.auth != null;
    }

    // RailGuard Vision Alerts Collection
    // Authenticated railway security officers can read alerts; client-side direct writes are blocked.
    // All creation, updates, and reviews are mediated securely through FastAPI backend using Admin SDK.
    match /railguard_alerts/{alertId} {
      allow read: if isAuthenticated();
      allow write: if false;
    }

    // Default deny
    match /{document=**} {
      allow read, write: if false;
    }
  }
}
```

---

## 8. API Reference Summary

| Method | Endpoint | Auth Header | Description |
| :--- | :--- | :---: | :--- |
| `GET` | `/health` | None | Public process liveness check for Render |
| `GET` | `/ready` | Bearer Firebase ID Token | Protected Firestore database connectivity check |
| `POST`| `/api/ingest/alerts` | Bearer Agent Token | Ingests alert & snapshot into Firestore (Idempotent) |
| `POST`| `/api/ingest/alerts/clear` | Bearer Agent Token | Records cleared timestamp when bag is retrieved |
| `GET` | `/api/alerts` | Bearer Firebase ID Token | Lists alerts with limit and review_status filters |
| `GET` | `/api/alerts/{id}` | Bearer Firebase ID Token | Fetches alert details from Firestore |
| `PATCH`| `/api/alerts/{id}/review` | Bearer Firebase ID Token | Records Verified / False Alarm with reviewer UID |
| `GET` | `/api/alerts/{id}/snapshot`| Bearer Firebase ID Token | Streams JPEG evidence (path traversal guarded) |
| `GET` | `/api/alerts/{id}/clip`| Bearer Firebase ID Token | Streams MP4 video clip |
| `GET` | `/api/alerts/stats/false-alarm-rate`| Bearer Firebase ID Token | False alarm rate metrics from reviewed alerts |
| `WSS` | `/api/ws/alerts` | Token Query Param | Real-time WebSocket push notifications |

---

## 9. Security & Privacy Safeguards

- [x] **No `/var/data` Dependency:** Deploys on Render Free without filesystem permission issues.
- [x] **Cryptographic Auth:** Mobile operator requests are validated using Firebase Admin SDK public keys.
- [x] **Separated Agent Ingestion:** Laptop edge camera uses a dedicated `RAILGUARD_AGENT_TOKEN`.
- [x] **Idempotent Ingestion:** Duplicate event transmissions reuse `event_uid` as document ID, preventing duplicate records.
- [x] **Traversal Prevention:** Evidence delivery enforces strict containment within container storage.
- [x] **No Facial Recognition / Biometrics:** Completely strictly adheres to early-warning optical screening without demographic profiling.
