"""CLI Entrypoint for RailGuard Vision Local Laptop Detection & Cloud Forwarding Agent.

Usage:
  python run_agent.py --cloud-url https://railguard-vision.onrender.com --token <AGENT_TOKEN>
  python run_agent.py --display  # Show live OpenCV preview window
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2

from src.agent import LocalDetectionAgent
from src.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RailGuard Vision - Laptop Edge Detection & Cloud Ingestion Agent"
    )
    parser.add_argument(
        "--cloud-url",
        type=str,
        default=os.getenv("RAILGUARD_CLOUD_URL", "http://localhost:8000"),
        help="Base HTTPS URL of Render-hosted RailGuard Vision FastAPI backend",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("RAILGUARD_AGENT_TOKEN", None),
        help="Agent Bearer authentication token for cloud ingestion",
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
    args = parser.parse_args()

    cfg = load_config()
    if args.source is not None:
        cfg.video.source = int(args.source) if args.source.isdigit() else args.source

    print("=" * 70)
    print("  RAILGUARD VISION — LOCAL LAPTOP DETECTION & CLOUD INGESTION AGENT")
    print("  Indian Railways Optical Early-Warning Security Prototype")
    print("=" * 70)
    print(f"  • Video Source  : {cfg.video.source}")
    print(f"  • Camera ID     : {cfg.video.camera_id}")
    print(f"  • Cloud Backend : {args.cloud_url}")
    print("  • Safety Notice : FOR HUMAN REVIEW ONLY — Does not detect explosives/narcotics")
    print("=" * 70)

    agent = LocalDetectionAgent(
        config=cfg,
        cloud_url=args.cloud_url,
        agent_token=args.token,
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
