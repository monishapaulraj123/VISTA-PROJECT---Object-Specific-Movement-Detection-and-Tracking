"""
TrackNova Dashboard Page.

Displays real statistics computed from the active video tracking session:
video metadata, detection summary, selected-object movement summary,
and recent dataset sessions.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import COLORS, SUBTITLE, TAGLINE
from modules.analytics import compute_dashboard_metrics, sessions_to_dataframe
from utils.ui import render_empty_state, render_kpi_card, render_page_header


def render() -> None:
    render_page_header(
        "TrackNova Dashboard",
        f"{TAGLINE} — {SUBTITLE}",
        icon_name="dashboard",
    )

    session_result = st.session_state.get("processing_result")
    movement_summary = st.session_state.get("movement_summary")
    metrics = compute_dashboard_metrics(session_result, movement_summary)

    if not metrics["has_session"]:
        render_empty_state(
            title="No Video Dataset Processed Yet",
            description=(
                "Upload or select a video dataset in the Video Tracking workspace to run "
                "YOLO object detection, ByteTrack ID assignment, and object-specific movement analytics."
            ),
            icon_name="video",
            cta_label="Go to Video Tracking",
            cta_target_page="Video Tracking",
            key="dash_empty_goto_tracking",
        )
        return

    v = metrics["video_stats"]
    d = metrics["detection_stats"]

    # --- Section 1: Video Statistics ---
    st.markdown(f"<h4 style='color:{COLORS['light_indigo']};margin-bottom:0.5rem;'>Video Statistics</h4>", unsafe_allow_html=True)
    vc1, vc2, vc3, vc4, vc5 = st.columns(5)
    with vc1:
        render_kpi_card("Uploaded Video", v["file_name"], f"Resolution: {v['resolution']}")
    with vc2:
        render_kpi_card("Duration", v["duration_str"], f"Frame Rate: {v['fps']} FPS")
    with vc3:
        render_kpi_card("Total Frames", str(v["total_frames"]), f"Processed: {v['processed_frames']}")
    with vc4:
        render_kpi_card(
            "Processing FPS",
            f"{session_result.processing_fps:.2f}",
            f"Time: {session_result.processing_time_sec:.2f}s",
        )
    with vc5:
        status_color = COLORS["success"] if v["processing_status"] == "Completed" else COLORS["warning"]
        render_kpi_card("Status", v["processing_status"], session_result.model_info.weights_name, accent_color=status_color)

    st.markdown("<div style='height:0.9rem;'></div>", unsafe_allow_html=True)

    # --- Section 2: Detection Statistics ---
    st.markdown(f"<h4 style='color:{COLORS['light_indigo']};margin-bottom:0.5rem;'>Detection &amp; Tracking Summary</h4>", unsafe_allow_html=True)
    dc1, dc2, dc3, dc4 = st.columns(4)
    with dc1:
        render_kpi_card("Total Detections", str(d["total_detections"]), "Across processed frames")
    with dc2:
        render_kpi_card("Detected Classes", str(d["detected_classes_count"]), "Distinct object categories")
    with dc3:
        render_kpi_card("Unique Tracking IDs", str(d["unique_track_ids_count"]), "Assigned by ByteTrack")
    with dc4:
        render_kpi_card("Most Frequent Class", d["most_frequent_class"], "Highest detection count")

    st.markdown("<div style='height:0.9rem;'></div>", unsafe_allow_html=True)

    # --- Section 3: Selected Object Summary ---
    st.markdown(f"<h4 style='color:{COLORS['light_indigo']};margin-bottom:0.5rem;'>Selected Object Movement Summary</h4>", unsafe_allow_html=True)
    if not metrics["has_selected"]:
        st.markdown(
            f"""
            <div class="tn-alert-info">
                No tracking ID is currently selected. Open <strong>Video Tracking</strong> to choose a detected object ID
                and unlock displacement, speed, and trajectory metrics.
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Select Tracking ID in Video Tracking", type="primary", key="dash_select_id_btn"):
            st.session_state["current_page"] = "Video Tracking"
            st.rerun()
    else:
        s = metrics["selected_stats"]
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            render_kpi_card(
                "Selected Object",
                f"ID #{s['selected_id']}",
                f"Class: {s['object_class']}",
                accent_color=COLORS["trajectory_accent"],
            )
        with sc2:
            render_kpi_card(
                "Frames Tracked",
                str(s["frames_tracked"]),
                f"Tracking Duration: {s['tracking_duration_str']}",
            )
        with sc3:
            render_kpi_card(
                "Total Displacement",
                f"{s['total_displacement_px']:.1f} px",
                f"Direction: {s['current_direction']}",
                accent_color=COLORS["light_indigo"],
            )
        with sc4:
            render_kpi_card(
                "Avg / Max Speed",
                s["avg_speed_str"],
                f"Peak: {s['max_speed_str']}",
                accent_color=COLORS["trajectory_accent"],
            )

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    # --- Class Distribution & Recent Sessions ---
    left_col, right_col = st.columns([1, 1.3])
    with left_col:
        st.markdown(f"<h5 style='color:{COLORS['main_text']};'>Class Detection Breakdown</h5>", unsafe_allow_html=True)
        if d["class_counts"]:
            cls_df = pd.DataFrame(
                [{"Object Class": k, "Detections": v_cnt} for k, v_cnt in sorted(d["class_counts"].items(), key=lambda x: -x[1])]
            )
            st.dataframe(cls_df, use_container_width=True, hide_index=True)
        else:
            st.info("No detections recorded in this session.")

    with right_col:
        st.markdown(f"<h5 style='color:{COLORS['main_text']};'>Recent Processing Sessions</h5>", unsafe_allow_html=True)
        sessions_df = sessions_to_dataframe(st.session_state.get("dataset_sessions", []))
        if not sessions_df.empty:
            cols_to_show = [
                c for c in ["Video Name", "Resolution", "Total Detections", "Unique Track IDs", "Processing FPS", "Selected Object"]
                if c in sessions_df.columns
            ]
            st.dataframe(sessions_df[cols_to_show], use_container_width=True, hide_index=True)
        else:
            st.info("No session history recorded yet.")

    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)
    nav1, nav2, nav3 = st.columns(3)
    with nav1:
        if st.button("Open Video Tracking Workspace", type="primary", use_container_width=True, key="dash_nav_vt"):
            st.session_state["current_page"] = "Video Tracking"
            st.rerun()
    with nav2:
        if st.button("View Object Analysis", type="secondary", use_container_width=True, key="dash_nav_oa"):
            st.session_state["current_page"] = "Object Analysis"
            st.rerun()
    with nav3:
        if st.button("View Trajectory Graphs", type="secondary", use_container_width=True, key="dash_nav_ta"):
            st.session_state["current_page"] = "Trajectory Analysis"
            st.rerun()
