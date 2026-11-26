"""
TrackNova Performance Evaluation & Multi-Dataset Testing Page.

Displays observational / proxy-based tracking evaluation (without fabricating
ground-truth MOTA/IDF1/accuracy percentages) and multi-video dataset comparisons.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import COLORS
from modules.analytics import sessions_to_dataframe
from modules.evaluation import evaluate_tracking_performance
from modules.export import export_comparison_csv
from utils.helpers import format_duration
from utils.ui import render_empty_state, render_kpi_card, render_page_header


def render() -> None:
    render_page_header(
        "Performance Evaluation & Multi-Dataset Testing",
        "Observational / proxy-based tracking analysis, processing throughput, and cross-video comparison",
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

    # Clear Ground-Truth Limitation Banner
    st.markdown(
        f"""
        <div class="tn-alert-warn">
            <strong>Evaluation Methodology: {eval_data['evaluation_type']}</strong><br/>
            Ground-truth object annotations are not attached to uploaded videos. Therefore, TrackNova reports factual runtime throughput and proxy tracking continuity indicators rather than fabricated MOTA, MOTP, IDF1, Precision, Recall, or Accuracy percentages.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # System & Detection Throughput Cards
    st.markdown(f"<h4 style='color:{COLORS['light_indigo']};margin-bottom:0.45rem;'>Pipeline Throughput &amp; Detection Coverage</h4>", unsafe_allow_html=True)
    e1, e2, e3, e4, e5 = st.columns(5)
    with e1:
        render_kpi_card("Frames Processed", f"{eval_data['frames_processed']} / {eval_data['total_video_frames']}", f"With detections: {eval_data['frames_with_detections']}")
    with e2:
        render_kpi_card("Processing Time", f"{eval_data['processing_time_sec']:.2f} s", "Wall-clock inference + tracking")
    with e3:
        render_kpi_card("Processing FPS", f"{eval_data['processing_fps']:.2f}", "Processed frames / sec", accent_color=COLORS["trajectory_accent"])
    with e4:
        render_kpi_card("Unique Track IDs", str(eval_data["unique_track_ids"]), f"Total Detections: {eval_data['total_detections']}")
    with e5:
        render_kpi_card("ID Assignment Rate", f"{eval_data['id_assignment_rate_pct']:.1f}%", f"Mean Conf: {eval_data['mean_detection_confidence']:.3f}")

    st.markdown("<div style='height:0.9rem;'></div>", unsafe_allow_html=True)

    # Selected Object Evaluation
    st.markdown(f"<h4 style='color:{COLORS['light_indigo']};margin-bottom:0.45rem;'>Selected-Object Tracking Continuity</h4>", unsafe_allow_html=True)
    sel_eval = eval_data.get("selected_eval")
    if sel_eval is None:
        st.info("Select a tracking ID in Video Tracking to compute target-specific gap and visibility metrics.")
    else:
        s1, s2, s3, s4 = st.columns(4)
        dur = sel_eval.get("tracking_duration_sec")
        with s1:
            render_kpi_card(
                "Target Tracked Frames",
                str(sel_eval["frames_containing_selected"]),
                f"Active span: F{sel_eval['first_frame']} – F{sel_eval['last_frame']}",
                accent_color=COLORS["trajectory_accent"],
            )
        with s2:
            render_kpi_card(
                "Tracking Duration",
                format_duration(dur) if dur is not None else "N/A",
                f"Movement Data Ready: {'Yes' if sel_eval['movement_data_available'] else 'Insufficient'}",
            )
        with s3:
            render_kpi_card(
                "Span Visibility",
                f"{sel_eval['span_visibility_pct']:.1f}%",
                f"Overall video visibility: {sel_eval['overall_visibility_pct']:.1f}%",
                accent_color=COLORS["light_indigo"],
            )
        with s4:
            gap_col = COLORS["success"] if sel_eval["tracking_interruptions"] == 0 else COLORS["warning"]
            render_kpi_card(
                "Tracking Interruptions",
                str(sel_eval["tracking_interruptions"]),
                f"Longest gap: {sel_eval['max_gap_steps']} step(s) | Missing in span: {sel_eval['missing_steps_in_span']}",
                accent_color=gap_col,
            )

        if sel_eval["gap_events"]:
            st.markdown(f"<h5 style='color:{COLORS['main_text']};margin-top:0.75rem;'>Detected Interruption Events</h5>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(sel_eval["gap_events"]), use_container_width=True, hide_index=True)

    # Observations & Limitations
    obs_col, lim_col = st.columns(2)
    with obs_col:
        obs_html = "".join(f"<li style='margin-bottom:0.4rem;'>{o}</li>" for o in eval_data["observations"])
        st.markdown(
            f"""
            <div class="tn-card">
                <h5 style="color:{COLORS['trajectory_accent']};margin-top:0;">Tracking Observations</h5>
                <ul style="color:{COLORS['main_text']};font-size:0.88rem;padding-left:1.2rem;margin-bottom:0;">{obs_html}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with lim_col:
        lim_html = "".join(f"<li style='margin-bottom:0.4rem;'>{l}</li>" for l in eval_data["limitations"])
        st.markdown(
            f"""
            <div class="tn-card">
                <h5 style="color:{COLORS['warning']};margin-top:0;">Documented Limitations</h5>
                <ul style="color:{COLORS['secondary_text']};font-size:0.88rem;padding-left:1.2rem;margin-bottom:0;">{lim_html}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Multi-Dataset Testing Comparison
    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='color:{COLORS['light_indigo']};margin-bottom:0.45rem;'>Multi-Dataset Testing Comparison</h4>", unsafe_allow_html=True)
    comp_df = sessions_to_dataframe(st.session_state.get("dataset_sessions", []))
    if comp_df.empty:
        st.info("No tested videos recorded in this session yet.")
    else:
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        csv_bytes, saved_csv = export_comparison_csv(comp_df)
        st.download_button(
            label="Export Dataset Comparison to CSV",
            data=csv_bytes,
            file_name=saved_csv.name,
            mime="text/csv",
            key="eval_dl_comp_csv",
        )
