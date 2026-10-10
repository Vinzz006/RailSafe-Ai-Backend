"""RailGuard Vision - Streamlit Operator Console & Security Screening Dashboard."""

from __future__ import annotations

import datetime
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import pandas as pd
import streamlit as st

from src.config import SystemConfig, load_config
from src.db import AlertDatabaseManager
from src.pipeline import RailGuardPipeline


st.set_page_config(
    page_title="RailGuard Vision | Indian Railways",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state(cfg: SystemConfig) -> None:
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "db_manager" not in st.session_state:
        st.session_state.db_manager = AlertDatabaseManager(cfg.storage.db_path)


def main() -> None:
    cfg = load_config()
    init_session_state(cfg)
    db: AlertDatabaseManager = st.session_state.db_manager

    # Top Mandatory Safety Banner
    st.markdown(
        """
        <div style="background-color: #7f1d1d; border: 2px solid #ef4444; border-radius: 8px; padding: 14px 18px; margin-bottom: 20px;">
            <h4 style="color: #fef2f2; margin: 0; font-size: 1.15rem; font-weight: 700;">
                ⚠️ FOR HUMAN REVIEW ONLY — THIS SYSTEM DOES NOT DETECT EXPLOSIVES OR NARCOTICS
            </h4>
            <p style="color: #fecaca; margin: 6px 0 0 0; font-size: 0.92rem; line-height: 1.4;">
                RailGuard Vision is an AI-assisted optical surveillance early-warning prototype. 
                Visible-light optical cameras observe only external bounding geometries and temporal positions; 
                they cannot determine chemical composition or detect hidden substances. 
                Alerts assist trained RPF/Security personnel in determining where to apply secondary screening (X-ray, ETD, or canine units).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Controls
    with st.sidebar:
        st.title("🛡️ RailGuard Vision")
        st.caption("Indian Railways Security Screening Assistance")
        st.divider()

        st.subheader("Video Source")
        source_mode = st.radio("Select Input Mode", ["Webcam", "Video File", "RTSP Stream"])

        if source_mode == "Webcam":
            webcam_idx = st.number_input("Webcam Device Index", min_value=0, max_value=8, value=0, step=1)
            cfg.video.source = int(webcam_idx)
        elif source_mode == "Video File":
            video_path = st.text_input("Local Video Path", value="data/videos/sample.mp4")
            cfg.video.source = video_path
        else:
            rtsp_url = st.text_input(
                "RTSP Stream URL",
                value=os.getenv("RAILGUARD_RTSP_URL", "rtsp://127.0.0.1:8554/live"),
                help="Credentials should be supplied via environment variable or authenticated gateway.",
            )
            cfg.video.source = rtsp_url

        cfg.video.camera_id = st.text_input("Camera Identifier", value=cfg.video.camera_id)

        st.divider()
        st.subheader("Rules & Sensitivity")
        cfg.model.confidence_threshold = st.slider("Detection Confidence", 0.15, 0.90, float(cfg.model.confidence_threshold), 0.05)
        cfg.rules.proximity_pixels = float(st.slider("Owner Proximity (Pixels)", 50, 350, int(cfg.rules.proximity_pixels), 10))
        cfg.rules.warning_seconds = float(st.slider("Warning Delay (Seconds)", 2, 20, int(cfg.rules.warning_seconds), 1))
        cfg.rules.unattended_seconds = float(st.slider("Unattended Alert Threshold (Seconds)", 5, 60, int(cfg.rules.unattended_seconds), 1))
        cfg.rules.restricted_zone_enabled = st.checkbox("Restricted Zone Active", value=cfg.rules.restricted_zone_enabled)

        st.divider()
        st.subheader("Monitoring Controls")

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("▶️ Start Feed", use_container_width=True, type="primary"):
                if not st.session_state.is_running:
                    st.session_state.pipeline = RailGuardPipeline(cfg)
                    success = st.session_state.pipeline.start()
                    if success:
                        st.session_state.is_running = True
                        st.rerun()
                    else:
                        st.error(f"Failed to open source: {st.session_state.pipeline.capture.error_message}")

        with col_btn2:
            if st.button("⏹️ Stop Feed", use_container_width=True):
                if st.session_state.is_running and st.session_state.pipeline:
                    st.session_state.pipeline.stop()
                    st.session_state.pipeline = None
                st.session_state.is_running = False
                st.rerun()

        st.divider()
        st.subheader("Review Metrics")
        fa_rate, total_reviewed, fa_count = db.get_false_alarm_rate()

        if fa_rate is not None:
            st.metric("Operator False Alarm Rate", f"{fa_rate * 100:.1f}%", f"{fa_count} / {total_reviewed} reviewed")
            st.caption("Note: False alarm rate reflects historic reviews; not the probability that a new alert is erroneous.")
        else:
            st.info("Not enough reviewed alerts to calculate false-alarm rate.")

    # Main Area Tabs
    tab_live, tab_alerts, tab_docs = st.tabs(["📺 Live Monitoring Console", "📋 Alert History & Review", "📖 System Ethics & Limitations"])

    with tab_live:
        st.subheader(f"Station Surveillance: {cfg.video.camera_id}")

        frame_placeholder = st.empty()
        telemetry_placeholder = st.empty()

        if st.session_state.is_running and st.session_state.pipeline:
            pipeline: RailGuardPipeline = st.session_state.pipeline

            # Single processing iteration (or loop controlled by Streamlit)
            res = pipeline.step()

            if res.is_eof:
                st.warning("Video file reached End-of-File (EOF).")
                pipeline.stop()
                st.session_state.is_running = False
            elif res.error_message:
                st.error(f"Stream error: {res.error_message}")
                pipeline.stop()
                st.session_state.is_running = False
            else:
                rgb_frame = cv2.cvtColor(res.annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                with telemetry_placeholder.container():
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Pipeline FPS", f"{res.fps:.1f}")
                    c2.metric("Tracked Objects", len(res.tracks))
                    c3.metric("Active Alerts", len(res.active_alerts))
                    c4.metric("Frame Index", res.frame_index)

                    if res.active_alerts:
                        for alt in res.active_alerts:
                            st.error(
                                f"🚨 **ALERT #{alt.track_id} ({alt.object_class.upper()}):** {alt.reason} "
                                f"[Duration: {alt.duration:.1f}s] — **FOR HUMAN REVIEW ONLY**"
                            )

            # Re-trigger next frame automatically when active
            if st.session_state.is_running:
                time.sleep(0.01)
                st.rerun()
        else:
            frame_placeholder.info(
                f"Camera monitoring is currently standby. Click **▶️ Start Feed** in the sidebar to begin processing '{cfg.video.source}'."
            )

    with tab_alerts:
        st.subheader("Security Alert Log & Operator Review")
        alerts_list = db.get_alerts(limit=50)

        if not alerts_list:
            st.info("No security alerts have been recorded yet.")
        else:
            for alert in alerts_list:
                status_color = (
                    "🔴" if alert.review_status == "unreviewed"
                    else "🟢" if alert.review_status == "verified"
                    else "🟡"
                )
                cleared_str = f"Cleared: {alert.cleared_timestamp}" if alert.cleared_timestamp else "ACTIVE / PERSISTENT"

                with st.expander(
                    f"{status_color} Alert #{alert.id} | {alert.object_class.upper()} (Track #{alert.track_id}) | "
                    f"Status: {alert.review_status.upper()} | {alert.timestamp}",
                    expanded=(alert.review_status == "unreviewed"),
                ):
                    c_det, c_media = st.columns([1.2, 1.0])

                    with c_det:
                        st.markdown(f"**Camera:** `{alert.camera_id}`")
                        st.markdown(f"**Timestamp:** `{alert.timestamp}`")
                        st.markdown(f"**Unattended Duration:** `{alert.event_duration:.1f} seconds`")
                        st.markdown(f"**Reason:** {alert.reason}")
                        st.markdown(f"**Lifecycle:** `{cleared_str}`")
                        st.markdown(f"**Review Verdict:** `{alert.review_status.upper()}`")

                        st.write("---")
                        st.write("Operator Review Action:")
                        b_col1, b_col2 = st.columns(2)
                        with b_col1:
                            if st.button("✅ Mark Verified", key=f"btn_ver_{alert.id}"):
                                db.update_review_status(alert.id, "verified")
                                st.success("Marked as Verified Human Review.")
                                st.rerun()
                        with b_col2:
                            if st.button("❌ Mark False Alarm", key=f"btn_fa_{alert.id}"):
                                db.update_review_status(alert.id, "false_alarm")
                                st.warning("Marked as False Alarm.")
                                st.rerun()

                    with c_media:
                        if alert.snapshot_path and os.path.isfile(alert.snapshot_path):
                            st.image(alert.snapshot_path, caption=f"Snapshot (Track #{alert.track_id})", use_container_width=True)
                        else:
                            st.caption("Snapshot image not available.")

                        if alert.clip_path and os.path.isfile(alert.clip_path):
                            st.video(alert.clip_path)
                            st.caption(f"10-Second Context Clip: `{Path(alert.clip_path).name}`")

    with tab_docs:
        st.subheader("System Limitations, Ethics & Privacy Protocols")
        st.markdown(
            """
            ### 1. Mandatory Scientific & Physical Boundaries
            - **No Hazardous Material Detection:** Visible-light cameras observe only external bounding boxes and spatial positions. They cannot detect explosives, narcotics, or chemical compounds inside sealed luggage.
            - **Human Review Mandatory:** Alerts are cues for security personnel to inspect context, check whether an owner is nearby, and dispatch secondary physical screening (ETD/X-ray/canines).

            ### 2. Privacy & Ethical Safeguards
            - **No Facial Recognition:** No facial recognition, face matching, or biometric identification is implemented or permitted.
            - **No Demographic Profiling:** Bounding boxes classify general COCO classes (`person`, `backpack`, `handbag`, `suitcase`). No demographic, age, gender, or ethnic labels are ever inferred.
            - **Ephemeral Track IDs:** Tracking numbers (e.g. Track #42) are temporary frame trajectory identifiers, not persistent passenger identities.
            - **Loitering is Noisy:** Dwell time on platforms is heavily influenced by train delays and congestion. It is treated as noisy context, never evidence of malicious intent.

            ### 3. Edge Case Guidance for Operators
            - **Crowded Platforms:** People walking close to a bag can temporarily reset proximity timers.
            - **Brief Occlusions:** When trains or porters pass in front of a camera, temporary track dropout occurs; grace periods prevent false spikes.
            - **Camera Motion:** Ensure fixed overhead mounting. Handheld or moving cameras invalidate pixel stationarity assumptions.
            """
        )


if __name__ == "__main__":
    main()
