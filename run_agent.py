"""CLI Entrypoint for RailGuard Vision Local Laptop Detection & Cloud Forwarding Agent.

Usage:
  # Normal run with live camera and cloud forwarding:
  python run_agent.py --source 0 --display --token <RENDER_AGENT_TOKEN>

  # Or using environment variables:
  $env:RAILGUARD_AGENT_TOKEN = "your_secret_token"
  python run_agent.py --source 0 --display

  # Safe synthetic cloud ingestion test (verifies Render + Firestore connectivity):
  python run_agent.py --test-ingest --token <RENDER_AGENT_TOKEN>

  # Offline purely local mode (camera + local SQLite only, no cloud upload):
  python run_agent.py --source 0 --display --offline
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from typing import Optional

import cv2
import httpx
import numpy as np

from src.agent import LocalDetectionAgent
from src.config import load_config


def run_synthetic_ingest_test(cloud_url: str, agent_token: Optional[str]) -> bool:
    """Send a safe synthetic unattended luggage alert to verify Render ingestion and Firestore persistence."""
    clean_url = cloud_url.rstrip("/")
    print("=" * 70)
    print("  RAILGUARD VISION — SYNTHETIC CLOUD INGESTION TEST")
    print("=" * 70)
    print(f"  • Target Cloud URL : {clean_url}")

    if not agent_token or not agent_token.strip():
        print("  • Auth Status      : FAILED — No agent token provided.")
        print("[ERROR] An agent token is required. Pass --token <SECRET> or set RAILGUARD_AGENT_TOKEN.")
        return False

    print("  • Auth Status      : Bearer token loaded (secret masked)")
    print("=" * 70)

    # 1. Generate synthetic test image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (180, 140), (460, 340), (0, 0, 255), 2)
    cv2.putText(
        img,
        "SYNTHETIC TEST ALERT",
        (190, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    _, encoded = cv2.imencode(".jpg", img)
    snap_bytes = encoded.tobytes()

    evt_uid = f"TEST_SYNTHETIC_{int(time.time())}"
    form_data = {
        "event_uid": evt_uid,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "camera_id": "CAM-TEST-DIAGNOSTIC",
        "track_id": "999",
        "object_class": "suitcase",
        "event_duration": "15.0",
        "reason": "Synthetic diagnostic alert for Render cloud ingestion verification",
    }
    files = {
        "snapshot": ("synthetic_test.jpg", io.BytesIO(snap_bytes), "image/jpeg"),
    }
    headers = {
        "Authorization": f"Bearer {agent_token.strip()}",
    }

    ingest_url = f"{clean_url}/api/ingest/alerts"
    print(f"\n[1/2] Sending synthetic alert to {ingest_url}...")
    try:
        with httpx.Client(timeout=15.0) as client:
            res = client.post(ingest_url, data=form_data, files=files, headers=headers)
            print(f"      Response Code: HTTP {res.status_code}")
            print(f"      Response Body: {res.text}")

            if res.status_code in (200, 201):
                print("      [PASS] Ingestion accepted! Document recorded in Cloud Firestore.")
            elif res.status_code in (401, 403):
                print("      [FAIL] Authentication rejected (HTTP 401/403).")
                print("             The agent token does not match RAILGUARD_AGENT_TOKEN configured in Render.")
                return False
            else:
                print(f"      [FAIL] Ingestion returned unexpected HTTP {res.status_code}.")
                return False

            # 2. Test alert clear event
            clear_url = f"{clean_url}/api/ingest/alerts/clear"
            print(f"\n[2/2] Sending clear lifecycle event to {clear_url}...")
            clear_payload = {
                "camera_id": "CAM-TEST-DIAGNOSTIC",
                "track_id": 999,
                "cleared_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            res_clr = client.post(clear_url, json=clear_payload, headers=headers)
            print(f"      Response Code: HTTP {res_clr.status_code}")
            print(f"      Response Body: {res_clr.text}")
            if res_clr.status_code == 200:
                print("      [PASS] Alert clear notification successfully recorded!")
            else:
                print(f"      [WARN] Clear notification returned HTTP {res_clr.status_code}.")

            print("\n" + "=" * 70)
            print("  END-TO-END INGESTION TEST PASSED SUCCESSFULLY!")
            print("=" * 70)
            return True

    except httpx.RequestError as e:
        print(f"\n[ERROR] Connection failed: {e}")
        print("Ensure the Render backend is live and reachable from this network.")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RailGuard Vision - Laptop Edge Detection & Cloud Ingestion Agent"
    )
    parser.add_argument(
        "--cloud-url",
        type=str,
        default=None,
        help="Base HTTPS URL of Render-hosted RailGuard Vision FastAPI backend (overrides RAILGUARD_CLOUD_URL)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Agent Bearer authentication token for cloud ingestion (overrides RAILGUARD_AGENT_TOKEN)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Override video source (e.g. 0 for webcam, rtsp://..., or video.mp4)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Render live annotated video in an OpenCV desktop window",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run in purely local edge mode (camera + local SQLite) without cloud forwarding",
    )
    parser.add_argument(
        "--test-ingest",
        action="store_true",
        help="Run a safe synthetic alert ingestion test against the cloud backend and exit",
    )
    args = parser.parse_args()

    # Precedence: CLI --cloud-url > RAILGUARD_CLOUD_URL env > live backend URL
    cloud_url = (
        args.cloud_url
        or os.getenv("RAILGUARD_CLOUD_URL")
        or "https://railsafe-ai-backend.onrender.com"
    ).rstrip("/")

    # Precedence: CLI --token > RAILGUARD_AGENT_TOKEN env > None
    raw_token = args.token or os.getenv("RAILGUARD_AGENT_TOKEN")
    agent_token = raw_token.strip() if raw_token else None

    # Handle synthetic test mode
    if args.test_ingest:
        success = run_synthetic_ingest_test(cloud_url=cloud_url, agent_token=agent_token)
        sys.exit(0 if success else 1)

    # Validate agent token if cloud forwarding is active
    enable_cloud = not args.offline
    if enable_cloud and not agent_token:
        print("=" * 70)
        print("  [ERROR] MISSING AGENT AUTHENTICATION TOKEN")
        print("=" * 70)
        print(f"  Target Cloud Backend : {cloud_url}")
        print("  Cloud ingestion requires an agent secret token to authenticate.")
        print()
        print("  Please provide the token using one of the following methods:")
        print(f"    1) CLI Argument: python run_agent.py --source 0 --display --token <SECRET>")
        print(f"    2) Environment : $env:RAILGUARD_AGENT_TOKEN = '<SECRET>' (PowerShell)")
        print()
        print("  Or run purely locally on this laptop without cloud uploads:")
        print(f"    python run_agent.py --source 0 --display --offline")
        print("=" * 70)
        sys.exit(1)

    cfg = load_config()
    if args.source is not None:
        cfg.video.source = int(args.source) if args.source.isdigit() else args.source

    auth_display = (
        "Disabled (Offline Mode)"
        if not enable_cloud
        else "Active (Bearer token loaded)"
    )

    print("=" * 70)
    print("  RAILGUARD VISION — LOCAL LAPTOP DETECTION & CLOUD INGESTION AGENT")
    print("  Indian Railways Optical Early-Warning Security Prototype")
    print("=" * 70)
    print(f"  • Video Source  : {cfg.video.source}")
    print(f"  • Camera ID     : {cfg.video.camera_id}")
    print(f"  • Cloud Backend : {cloud_url if enable_cloud else 'N/A (Offline)'}")
    print(f"  • Cloud Auth    : {auth_display}")
    print("  • Safety Notice : FOR HUMAN REVIEW ONLY — Does not detect explosives/narcotics")
    print("=" * 70)

    agent = LocalDetectionAgent(
        config=cfg,
        cloud_url=cloud_url,
        agent_token=agent_token,
        enable_cloud=enable_cloud,
    )

    if not agent.start():
        print("[ERROR] Failed to start local camera detection pipeline. Exiting.")
        sys.exit(1)

    print("[RUNNING] Local detection agent active. Press Ctrl+C in terminal (or 'q' in window) to stop.")
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            result = agent.step()
            frame_count += 1

            if result.is_eof:
                print("\n[INFO] End of video source reached.")
                break

            if args.display and result.annotated_frame is not None:
                cv2.imshow("RailGuard Vision - Station Edge Feed", result.annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n[INFO] 'q' pressed. Stopping agent...")
                    break

            if frame_count % 60 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                active_str = f" | Active Alerts: {len(result.active_alerts)}" if result.active_alerts else ""
                print(
                    f"\r[STATUS] Processed {frame_count} frames | FPS: {fps:.1f} | Tracks: {len(result.tracks)}{active_str}",
                    end="",
                    flush=True,
                )

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Shutting down...")
    finally:
        agent.stop()
        if args.display:
            cv2.destroyAllWindows()
        print("\n[COMPLETE] Agent terminated cleanly.")


if __name__ == "__main__":
    main()

