# RailGuard Vision — AI-Assisted Unattended Luggage Screening Assistance
### Indian Railways Early-Warning Surveillance & Decision-Support Prototype

---

## ⚠️ Mandatory Safety & Scientific Disclaimer

> **PROTOTYPE LIMITATION NOTICE:**  
> This prototype detects visible object categories and temporal patterns. **It cannot determine whether an object contains explosives or narcotics.**  
> Visible-light optical cameras cannot penetrate opaque luggage or perform chemical spectrometry. Alerts are fallible and are **FOR HUMAN REVIEW ONLY**.  
> The system does not trigger automatic security enforcement, evacuations, or passenger detentions. Alerts serve exclusively to notify human security personnel (RPF/GRP) where to direct certified secondary screening methods (such as Multi-Energy Dual-View X-Ray scanners, Explosive Trace Detectors, or Police Canine Squads).

---

## 1. System Architecture

RailGuard Vision processes live optical video feeds through a multi-stage modular pipeline:

```mermaid
flowchart TD
    Capture["Video Capture (Webcam / File / RTSP)"] --> RingBuf["Rolling 10s Circular Buffer"]
    Capture --> Detector["YOLOv8 Object Detection (Person & Bags)"]
    Detector --> Tracker["ByteTrack Multi-Object Association"]
    Tracker --> RulesEngine["Deterministic Spatial-Temporal Rules Engine"]
    RulesEngine --> StateMachine{"State Machine\nNORMAL ➔ WARNING ➔ ALERT"}
    StateMachine -->|Active Condition| AlertsMgr["Alert Service (JPEG Snapshot + 10s MP4 Clip)"]
    AlertsMgr --> SQLite["SQLite Database (security_alerts.db)"]
    StateMachine -->|Bag Retrieved / Moved| ClearHandler["Clear Active Alert (Timestamp Logged)"]
    ClearHandler --> SQLite
    SQLite --> Dashboard["Streamlit Operator HUD (Verification & Review)"]
```

---

## 2. Unattended Baggage State Machine

The system enforces a 3-tier deterministic state machine:

```text
       [ Owner Nearby or Bag Moving ]
                 │
                 ▼
          ┌──────────────┐
          │    NORMAL    │ ◄─── (Bag moved, picked up, or owner returned)
          └──────┬───────┘
                 │
                 │ (Bag stationary AND no passenger within 140px for > 4s)
                 ▼
          ┌──────────────┐
          │   WARNING    │
          └──────┬───────┘
                 │
                 │ (Condition persists for > 15s)
                 ▼
          ┌──────────────┐
          │    ALERT     │ ──► Generates Snapshot + 10s Clip + SQLite Record
          └──────────────┘     (Requires Human Operator Review)
```

- **NORMAL:** Bag is moving with a passenger or stationary with an accompanied person within `proximity_pixels` (default: 140px).
- **WARNING:** A stationary bag has no accompanying passenger within proximity for at least `warning_seconds` (default: 4s).
- **ALERT:** The unattended condition has persisted continuously for `unattended_seconds` (default: 15s). A snapshot and 10-second context clip are recorded and written to SQLite.

---

## 3. Technology Stack

- **Language:** Python 3.10+ (Tested on Python 3.13)
- **Object Detection:** Ultralytics YOLOv8 (`yolov8n.pt` CPU default)
- **Tracking Algorithm:** Ultralytics built-in ByteTrack
- **Image & Video Processing:** OpenCV (`opencv-python` 4.8+)
- **Operator Dashboard:** Streamlit
- **Persistence:** SQLite (`data/security_alerts.db`)
- **Configuration:** PyYAML (`configs/config.yaml`)
- **Testing:** pytest (Automated synthetic unit & integration tests)

---

## 4. Quick Start (One-Command Launch)

### On Windows
```bat
run.bat
```

### On Linux / macOS
```bash
chmod +x run.sh
./run.sh
```

### Manual Launch via Terminal
```bash
# 1. Navigate to project folder
cd RailGuard_Vision

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Streamlit Operator Dashboard
streamlit run app/dashboard.py
```
Open **`http://localhost:8501`** in your browser.

---

## 5. Configuration Reference (`configs/config.yaml`)

```yaml
video:
  source: 0                    # 0 for webcam, "path/to/video.mp4", or "rtsp://..."
  camera_id: "CAM-PLATFORM-01" # Unique station camera identifier
  target_width: 640            # Processing resolution width
  target_height: 480           # Processing resolution height
  frame_skip: 1                # 1 = every frame, 2 = every 2nd frame
  buffer_seconds: 10           # Rolling pre-alert context buffer duration

model:
  weights: "yolov8n.pt"        # Model weights path
  confidence_threshold: 0.35   # Minimum detection confidence
  device: "cpu"                # 'cpu' or 'cuda'
  input_size: 640

rules:
  stationary_pixels: 18        # Maximum centroid drift for stationary state
  stationary_window_seconds: 5 # Time window over which drift is computed
  proximity_pixels: 140        # Maximum Euclidean distance to consider a bag accompanied
  warning_seconds: 4           # Delay before entering WARNING state
  unattended_seconds: 15       # Delay before escalating to active ALERT
  track_drop_grace_seconds: 2.5# Tolerance for temporary visual detector occlusion
  owner_left_grace_seconds: 8  # Tolerance before resetting owner association
  loiter_seconds: 90           # Dwell threshold for noisy contextual loitering tag
  restricted_zone_enabled: false
  restricted_zone_polygon: [[20, 20], [300, 20], [300, 200], [20, 200]]

storage:
  db_path: "data/security_alerts.db"
  alerts_dir: "data/alerts"
  clips_dir: "data/clips"
```

### Environment Variable Overrides
To avoid committing passwords or sensitive RTSP links:
- `RAILGUARD_VIDEO_SOURCE`: Overrides `video.source` (e.g. RTSP URL with token).
- `RAILGUARD_CAMERA_ID`: Overrides camera identifier.
- `RAILGUARD_MODEL_WEIGHTS`: Overrides path to custom weights.
- `RAILGUARD_DB_PATH`: Overrides SQLite file path.

---

## 6. Video Source Configurations

### 1. USB Webcam
In `configs/config.yaml`, set `source: 0` (or select Webcam in the Streamlit UI sidebar).

### 2. Recorded Video File
Place test video in `data/videos/` and set `source: "data/videos/test.mp4"`.

### 3. Railway CCTV / RTSP Network Stream
Set `source: "rtsp://192.168.1.100:554/ch01"` or pass via environment variable:
```bash
export RAILGUARD_VIDEO_SOURCE="rtsp://admin:password@10.0.0.15:554/stream1"
```

---

## 7. Render Cloud Deployment & Edge Ingestion

RailGuard Vision supports distributed deployment: the station laptop runs optical computer vision locally, uploading authenticated alerts and evidence to a Render-hosted cloud backend for the Flutter mobile application.

```mermaid
flowchart LR
    Webcam["Station Laptop\n(YOLOv8 + ByteTrack)"] -->|Outbound HTTPS\nPOST /api/ingest/alerts| Render["Render Cloud Backend\n(FastAPI + /var/data)"]
    Render -->|HTTPS / WSS| Flutter["Flutter Mobile App\n(RPF Field Officer)"]
```

### 1. Launch Cloud Backend (Render)
- Deploy using `render.yaml` or Render Dashboard.
- Mount persistent disk at `/var/data` (holds SQLite database and evidence files).
- Start command: `uvicorn app.api:app --host 0.0.0.0 --port $PORT`
- Full instructions: [`RENDER_DEPLOYMENT.md`](file:///c:/Users/shanm/Downloads/RailSafe%20AI/RailGuard_Vision/RENDER_DEPLOYMENT.md)

### 2. Launch Local Laptop Detection Agent
On the station laptop connected to the webcam:
```cmd
cd RailGuard_Vision
set RAILGUARD_CLOUD_URL=https://railguard-vision-api.onrender.com
set RAILGUARD_AGENT_TOKEN=your-agent-token
run_agent.bat
```

### 3. Flutter Mobile App Client
The Flutter mobile application connects via [`RailGuardApiService`](file:///c:/Users/shanm/Downloads/RailSafe%20AI/lib/services/railguard_api_service.dart), fetching live alerts, reviewing events as `verified` or `false_alarm`, and streaming evidence snapshots and video clips securely over HTTPS.

---

## 8. Automated Test Suite

Run the complete test suite:
```bash
python -m pytest -v
```
Expected output:
```text
======================== 37 passed, 1 warning in 4.46s ========================
```

Test coverage includes:
- `test_config.py`: Configuration validation, overrides, and error handling.
- `test_rules.py`: Spatial rules, state transitions, owner suppression, return recovery, and loitering.
- `test_tracker.py`: ByteTrack trajectory histories, owner association heuristics, and dropout grace.
- `test_alerts.py`: Evidence saving (snapshots & MP4 clips) and duplicate alert suppression.
- `test_db.py`: SQLite operations, review updates, and false-alarm calculations.
- `test_evaluation.py`: Precision, recall, detection delay, and non-double-counting policy.
- `test_api.py`: FastAPI health checks, authentication, alert ingestion, idempotency, review updates, evidence streaming, and path traversal guards.
- `test_persistence.py`: Simulated restart across persistent directories, ensuring zero data loss.
- `test_cloud_agent.py`: Asynchronous edge forwarder, queue overflow protection, and bounded retry logic.

---

## 9. Evaluation Framework (`evaluate.py`)

Run benchmark comparisons between annotated ground truth scenarios and logged alerts:
```bash
python evaluate.py --ground-truth data/ground_truth/ground_truth_template.csv --db data/security_alerts.db --tolerance 4.0
```

Calculates and reports:
- **True Positives (TP)**, **False Positives (FP)**, **False Negatives (FN)**, **True Negatives (TN)**
- **Precision**, **Recall (Sensitivity)**, **F1-Score**
- **Mean Detection Delay** (seconds)
- **False Alarms per Hour**

---

## 10. Ethics, Privacy & Responsible AI Safeguards

1. **No Facial Recognition or Biometrics:** The system does not perform face detection, facial recognition, or biometric identification.
2. **No Demographic Inference:** Passengers are detected only under the general COCO `person` class. No age, gender, ethnicity, or appearance profiling is implemented.
3. **Temporary Tracking Identifiers:** Track IDs (e.g., Track #42) are discarded as soon as a person or object leaves the frame.
4. **Contextual Loitering Interpretation:** Indian Railway stations routinely experience delayed trains and resting commuters. Dwell time is treated as noisy context, never proof of malicious intent.
5. **Data Protection:** Evidence snapshots and clips are stored locally under `data/alerts` and `data/clips` (or `/var/data` in the cloud). Access must be restricted to authorized railway personnel. Operators should establish a 30-day data retention and secure deletion procedure in compliance with Indian IT and Railway Board security policies.

---

## 11. Future Work & Deployment Roadmap

- **NVIDIA Jetson Edge Deployment:** Porting YOLOv8 to TensorRT for 30+ FPS low-latency edge acceleration on station platforms.
- **Homography Ground-Plane Calibration:** Calibrating camera perspective matrices to measure physical distance in meters rather than image pixels.
- **Automated Privacy Blurring:** Applying Gaussian blur to bystander faces prior to saving evidence snapshots.
- **RPF Control Room Integration:** Secure webhook and MQTT alerts dispatched directly to station security monitors and hand-held terminals.
- **Verified Automatic Retention Purging:** Automated daemon purging snapshots older than the configured compliance retention window.

