"""
TrackNova Object Analysis Page.

Displays detailed real movement kinematics, displacement breakdown,
coordinate convention notes, and frame-by-frame tabular data for the
selected tracking ID.
"""

from __future__ import annotations

import streamlit as st

from config import COLORS
from utils.helpers import format_duration
from utils.ui import render_empty_state, render_kpi_card, render_page_header


def render() -> None:
    render_page_header(
        "Selected Object Movement Analysis",
        "Frame-by-frame centre coordinates, X/Y displacement, pixel speed, and 2D image-plane direction",
        icon_name="crosshair",
    )

    session_result = st.session_state.get("processing_result")
    movement_df = st.session_state.get("movement_df")
    summary = st.session_state.get("movement_summary")

    if session_result is None:
        render_empty_state(
            title="No Processed Video Data Available",
            description="Process a video in the Video Tracking workspace first, then select a tracking ID to inspect its movement kinematics.",
            icon_name="video",
            cta_label="Go to Video Tracking",
            cta_target_page="Video Tracking",
            key="oa_empty_no_session",
        )
        return

    if movement_df is None or movement_df.empty or not summary or not summary.get("has_data"):
        render_empty_state(
            title="No Object ID Selected",
            description="Select one of the detected ByteTrack IDs in Video Tracking to view detailed object-specific movement metrics.",
            icon_name="crosshair",
            cta_label="Select Object ID",
            cta_target_page="Video Tracking",
            key="oa_empty_no_id",
        )
        return

    # --- Top Metric Rows ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card(
            "Selected Tracking ID",
            f"ID #{summary['selected_id']}",
            f"Class: {summary['object_class']} (Conf: {summary['latest_confidence']:.2f})",
            accent_color=COLORS["trajectory_accent"],
        )
    with c2:
        render_kpi_card(
            "Current Centre (X, Y)",
            f"({summary['current_center_x']:.1f}, {summary['current_center_y']:.1f})",
            "Image plane pixels",
        )
    with c3:
        render_kpi_card(
            "Last Step DX / DY",
            f"{summary['last_dx']:+.1f} / {summary['last_dy']:+.1f} px",
            f"Net DX/DY: {summary['net_dx']:+.1f} / {summary['net_dy']:+.1f} px",
        )
    with c4:
        render_kpi_card(
            "Total Pixel Displacement",
            f"{summary['total_displacement_px']:.2f} px",
            f"Straight-line net: {summary['net_displacement_px']:.2f} px",
            accent_color=COLORS["light_indigo"],
        )

    st.markdown("<div style='height:0.85rem;'></div>", unsafe_allow_html=True)

    c5, c6, c7, c8 = st.columns(4)
    avg_spd = summary.get("avg_speed_px_s")
    max_spd = summary.get("max_speed_px_s")
    dur = summary.get("tracking_duration_sec")

    with c5:
        render_kpi_card(
            "Average Pixel Speed",
            f"{avg_spd:.2f} px/s" if avg_spd is not None else "N/A",
            "Image-plane pixels/sec",
            accent_color=COLORS["trajectory_accent"],
        )
    with c6:
        render_kpi_card(
            "Maximum Pixel Speed",
            f"{max_spd:.2f} px/s" if max_spd is not None else "N/A",
            "Excludes flagged gap jumps",
        )
    with c7:
        render_kpi_card(
            "Current Movement Direction",
            summary["current_direction"],
            f"Dominant: {summary['dominant_direction']}",
            accent_color=COLORS["light_indigo"],
        )
    with c8:
        render_kpi_card(
            "Frames Tracked & Gaps",
            f"{summary['frames_tracked']} frames",
            f"Duration: {format_duration(dur) if dur is not None else 'N/A'} | Interruptions: {summary['tracking_interruptions']}",
        )

    st.markdown("<div style='height:0.85rem;'></div>", unsafe_allow_html=True)

    # Coordinate Convention & Physical Orientation Clarification
    st.markdown(
        f"""
        <div class="tn-alert-info">
            <strong>Coordinate &amp; Direction Interpretation:</strong><br/>
            • <strong>Image Coordinates:</strong> Origin <code>(0, 0)</code> is top-left. Positive <code>DX</code> = Right, Negative <code>DX</code> = Left, Positive <code>DY</code> = Down, Negative <code>DY</code> = Up.<br/>
            • <strong>Movement Direction vs Physical Orientation:</strong> Estimated direction reflects 2D centre-point displacement across frames, <em>not</em> the object's true 3D physical facing orientation (which would require keypoints or pose estimation).<br/>
            • <strong>Pixel Speed Units:</strong> Reported strictly in <code>pixels/second (px/s)</code>; physical m/s requires camera calibration or a known reference scale.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not summary.get("fps_valid", True):
        st.markdown(
            '<div class="tn-alert-warn">Video FPS metadata was zero or invalid; pixel speed values could not be reliably derived from frame rate.</div>',
            unsafe_allow_html=True,
        )

    st.markdown(f"<h5 style='color:{COLORS['main_text']};'>Frame-by-Frame Movement Log</h5>", unsafe_allow_html=True)
    st.dataframe(movement_df, use_container_width=True, height=380, hide_index=True)
