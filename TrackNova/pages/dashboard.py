"""
TrackNova Dashboard Page.

Displays real statistics computed from the active video tracking session:
video metadata, detection summary, selected-object movement summary,
and recent dataset sessions in a balanced, equal-height card grid.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import COLORS, SUBTITLE, TAGLINE
from modules.analytics import compute_dashboard_metrics, sessions_to_dataframe
from utils.ui import (
    render_empty_state,
    render_kpi_grid,
    render_page_header,
    render_section_title,
)


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
    render_section_title("Video Statistics", f"Model: {session_result.model_info.weights_name}")
    status_color = COLORS["success"] if v["processing_status"] == "Completed" else COLORS["warning"]
    render_kpi_grid(
        [
            {"label": "Uploaded Video", "value": v["file_name"], "subtext": f"Resolution: {v['resolution']}"},
            {"label": "Duration", "value": v["duration_str"], "subtext": f"Frame Rate: {v['fps']} FPS"},
            {"label": "Total Frames", "value": str(v["total_frames"]), "subtext": f"Processed: {v['processed_frames']}"},
            {
                "label": "Processing FPS",
                "value": f"{session_result.processing_fps:.2f}",
                "subtext": f"Time: {session_result.processing_time_sec:.2f}s",
            },
            {
                "label": "Status",
                "value": v["processing_status"],
                "subtext": session_result.model_info.weights_name,
                "accent_color": status_color,
            },
        ],
        columns=5,
    )

    # --- Section 2: Detection Statistics ---
    render_section_title("Detection & Tracking Summary")
    render_kpi_grid(
        [
            {"label": "Total Detections", "value": str(d["total_detections"]), "subtext": "Across processed frames"},
            {"label": "Detected Classes", "value": str(d["detected_classes_count"]), "subtext": "Distinct object categories"},
            {"label": "Unique Tracking IDs", "value": str(d["unique_track_ids_count"]), "subtext": "Assigned by ByteTrack"},
            {"label": "Most Frequent Class", "value": d["most_frequent_class"], "subtext": "Highest detection count"},
        ],
        columns=4,
    )

    # --- Section 3: Selected Object Summary ---
    render_section_title("Selected Object Movement Summary")
    if not metrics["has_selected"]:
        with st.container(border=True):
            c_msg, c_btn = st.columns([2.6, 1])
            with c_msg:
                st.markdown(
                    f"<div style='color:{COLORS['main_text']};font-size:0.9rem;padding-top:0.25rem;'>"
                    "No tracking ID is currently selected. Choose a detected object ID in "
                    "<strong>Video Tracking</strong> to compute displacement, speed, and trajectory metrics."
                    "</div>",
                    unsafe_allow_html=True,
                )
            with c_btn:
                if st.button("Select Tracking ID", type="primary", width="stretch", key="dash_select_id_btn"):
                    st.session_state["current_page"] = "Video Tracking"
                    st.rerun()
    else:
        s = metrics["selected_stats"]
        render_kpi_grid(
            [
                {
                    "label": "Selected Object",
                    "value": f"ID #{s['selected_id']}",
                    "subtext": f"Class: {s['object_class']}",
                    "accent_color": COLORS["trajectory_accent"],
                },
                {
                    "label": "Frames Tracked",
                    "value": str(s["frames_tracked"]),
                    "subtext": f"Duration: {s['tracking_duration_str']}",
                },
                {
                    "label": "Total Displacement",
                    "value": f"{s['total_displacement_px']:.1f} px",
                    "subtext": f"Direction: {s['current_direction']}",
                    "accent_color": COLORS["light_indigo"],
                },
                {
                    "label": "Avg / Max Speed",
                    "value": s["avg_speed_str"],
                    "subtext": f"Peak: {s['max_speed_str']}",
                    "accent_color": COLORS["trajectory_accent"],
                },
            ],
            columns=4,
        )

    # --- Section 4: Class Distribution & Recent Sessions ---
    render_section_title("Session Breakdown & History")
    left_col, right_col = st.columns([1, 1.35])
    with left_col:
        with st.container(border=True):
            st.markdown(f"<div style='color:{COLORS['main_text']};font-weight:700;font-size:0.95rem;margin-bottom:0.45rem;'>Class Detection Breakdown</div>", unsafe_allow_html=True)
            if d["class_counts"]:
                cls_df = pd.DataFrame(
                    [{"Object Class": k, "Detections": v_cnt} for k, v_cnt in sorted(d["class_counts"].items(), key=lambda x: -x[1])]
                )
                st.dataframe(cls_df, width="stretch", height=220, hide_index=True)
            else:
                st.info("No detections recorded in this session.")

    with right_col:
        with st.container(border=True):
            st.markdown(f"<div style='color:{COLORS['main_text']};font-weight:700;font-size:0.95rem;margin-bottom:0.45rem;'>Recent Processing Sessions</div>", unsafe_allow_html=True)
            sessions_df = sessions_to_dataframe(st.session_state.get("dataset_sessions", []))
            if not sessions_df.empty:
                cols_to_show = [
                    c for c in ["Video Name", "Resolution", "Total Detections", "Unique Track IDs", "Processing FPS", "Selected Object"]
                    if c in sessions_df.columns
                ]
                st.dataframe(sessions_df[cols_to_show], width="stretch", height=220, hide_index=True)
            else:
                st.info("No session history recorded yet.")

    # --- Quick Navigation ---
    nav1, nav2, nav3 = st.columns(3)
    with nav1:
        if st.button("Open Video Tracking Workspace", type="primary", width="stretch", key="dash_nav_vt"):
            st.session_state["current_page"] = "Video Tracking"
            st.rerun()
    with nav2:
        if st.button("View Object Analysis", type="secondary", width="stretch", key="dash_nav_oa"):
            st.session_state["current_page"] = "Object Analysis"
            st.rerun()
    with nav3:
        if st.button("View Trajectory Graphs", type="secondary", width="stretch", key="dash_nav_ta"):
            st.session_state["current_page"] = "Trajectory Analysis"
            st.rerun()
