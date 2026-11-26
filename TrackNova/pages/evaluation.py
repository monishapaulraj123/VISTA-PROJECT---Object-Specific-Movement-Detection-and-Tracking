"""
TrackNova Performance Evaluation & Multi-Dataset Testing Page.

Displays processing throughput, ID assignment, selected-object tracking
continuity, interruption events, and multi-video dataset comparisons.
"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from config import COLORS
from modules.analytics import sessions_to_dataframe
from modules.evaluation import evaluate_tracking_performance
from modules.export import export_comparison_csv
from utils.helpers import format_duration
from utils.ui import (
    render_empty_state,
    render_kpi_grid,
    render_page_header,
    render_section_title,
)


def render() -> None:
    render_page_header(
        "Performance Evaluation & Multi-Dataset Testing",
        "Pipeline throughput, tracking continuity metrics, and cross-video comparison",
        icon_name="gauge",
    )

    session_result = st.session_state.get("processing_result")
    movement_df = st.session_state.get("movement_df")
    movement_summary = st.session_state.get("movement_summary")

    if session_result is None:
        render_empty_state(
            title="No Evaluation Data Available",
            description="Process one or more video datasets in Video Tracking to evaluate throughput, tracking interruptions, and multi-video performance.",
            icon_name="gauge",
            cta_label="Go to Video Tracking",
            cta_target_page="Video Tracking",
            key="eval_empty_cta",
        )
        return

    eval_data = evaluate_tracking_performance(session_result, movement_df, movement_summary)
    st.session_state["evaluation_result"] = eval_data

    # --- Pipeline Throughput & Detection Coverage ---
    render_section_title("Pipeline Throughput & Detection Coverage", "Proxy-based runtime telemetry")
    render_kpi_grid(
        [
            {
                "label": "Frames Processed",
                "value": f"{eval_data['frames_processed']} / {eval_data['total_video_frames']}",
                "subtext": f"With detections: {eval_data['frames_with_detections']}",
            },
            {
                "label": "Processing Time",
                "value": f"{eval_data['processing_time_sec']:.2f} s",
                "subtext": "Inference + tracking runtime",
            },
            {
                "label": "Processing FPS",
                "value": f"{eval_data['processing_fps']:.2f}",
                "subtext": "Processed frames / sec",
                "accent_color": COLORS["trajectory_accent"],
            },
            {
                "label": "Unique Track IDs",
                "value": str(eval_data["unique_track_ids"]),
                "subtext": f"Total Detections: {eval_data['total_detections']}",
            },
            {
                "label": "ID Assignment Rate",
                "value": f"{eval_data['id_assignment_rate_pct']:.1f}%",
                "subtext": f"Mean Conf: {eval_data['mean_detection_confidence']:.3f}",
            },
        ],
        columns=5,
    )

    # --- Selected-Object Tracking Continuity ---
    render_section_title("Selected-Object Tracking Continuity")
    sel_eval = eval_data.get("selected_eval")
    if sel_eval is None:
        st.info("Select a tracking ID in Video Tracking to compute target-specific gap and visibility metrics.")
    else:
        dur = sel_eval.get("tracking_duration_sec")
        gap_col = COLORS["success"] if sel_eval["tracking_interruptions"] == 0 else COLORS["warning"]
        render_kpi_grid(
            [
                {
                    "label": "Target Tracked Frames",
                    "value": str(sel_eval["frames_containing_selected"]),
                    "subtext": f"Active span: F{sel_eval['first_frame']} – F{sel_eval['last_frame']}",
                    "accent_color": COLORS["trajectory_accent"],
                },
                {
                    "label": "Tracking Duration",
                    "value": format_duration(dur) if dur is not None else "N/A",
                    "subtext": f"Movement Data Ready: {'Yes' if sel_eval['movement_data_available'] else 'Insufficient'}",
                },
                {
                    "label": "Span Visibility",
                    "value": f"{sel_eval['span_visibility_pct']:.1f}%",
                    "subtext": f"Overall video visibility: {sel_eval['overall_visibility_pct']:.1f}%",
                    "accent_color": COLORS["light_indigo"],
                },
                {
                    "label": "Tracking Interruptions",
                    "value": str(sel_eval["tracking_interruptions"]),
                    "subtext": f"Longest gap: {sel_eval['max_gap_steps']} step(s) | Missing: {sel_eval['missing_steps_in_span']}",
                    "accent_color": gap_col,
                },
            ],
            columns=4,
        )

    # --- Observations & Interruption Events ---
    obs_html = "".join(
        f"<li style='margin-bottom:0.35rem;'>{html.escape(o)}</li>" for o in eval_data["observations"]
    )
    if sel_eval is not None and sel_eval["gap_events"]:
        obs_col, gap_col_ui = st.columns([1, 1])
        with obs_col:
            st.markdown(
                f"""
                <div class="tn-card">
                    <div style="color:{COLORS['trajectory_accent']};font-weight:700;font-size:0.95rem;margin-bottom:0.5rem;">
                        Tracking Observations
                    </div>
                    <ul style="color:{COLORS['main_text']};font-size:0.88rem;line-height:1.5;padding-left:1.2rem;margin:0;">{obs_html}</ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with gap_col_ui:
            with st.container(border=True):
                st.markdown(
                    f"<div style='color:{COLORS['light_indigo']};font-weight:700;font-size:0.95rem;margin-bottom:0.45rem;'>Detected Interruption Events</div>",
                    unsafe_allow_html=True,
                )
                st.dataframe(pd.DataFrame(sel_eval["gap_events"]), width="stretch", height=180, hide_index=True)
    else:
        st.markdown(
            f"""
            <div class="tn-card">
                <div style="color:{COLORS['trajectory_accent']};font-weight:700;font-size:0.95rem;margin-bottom:0.45rem;">
                    Tracking Observations
                </div>
                <ul style="color:{COLORS['main_text']};font-size:0.88rem;line-height:1.5;padding-left:1.2rem;margin:0;">{obs_html}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Multi-Dataset Testing Comparison ---
    render_section_title("Multi-Dataset Testing Comparison")
    comp_df = sessions_to_dataframe(st.session_state.get("dataset_sessions", []))
    with st.container(border=True):
        if comp_df.empty:
            st.info("No tested videos recorded in this session yet.")
        else:
            st.dataframe(comp_df, width="stretch", hide_index=True)
            csv_bytes, saved_csv = export_comparison_csv(comp_df)
            dl_c1, _ = st.columns([1.2, 2.8])
            with dl_c1:
                st.download_button(
                    label="Export Dataset Comparison to CSV",
                    data=csv_bytes,
                    file_name=saved_csv.name,
                    mime="text/csv",
                    width="stretch",
                    key="eval_dl_comp_csv",
                )
