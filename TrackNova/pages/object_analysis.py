"""
TrackNova Object Analysis Page.

Displays detailed real movement kinematics, displacement breakdown,
coordinate convention reference, and frame-by-frame tabular data for the
selected tracking ID.
"""

from __future__ import annotations

import streamlit as st

from config import COLORS
from utils.helpers import format_duration
from utils.ui import (
    render_empty_state,
    render_kpi_grid,
    render_page_header,
    render_section_title,
)


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

    avg_spd = summary.get("avg_speed_px_s")
    max_spd = summary.get("max_speed_px_s")
    dur = summary.get("tracking_duration_sec")

    render_section_title("Target Kinematics Summary", f"Tracking ID #{summary['selected_id']}")
    render_kpi_grid(
        [
            {
                "label": "Selected Tracking ID",
                "value": f"ID #{summary['selected_id']}",
                "subtext": f"Class: {summary['object_class']} (Conf: {summary['latest_confidence']:.2f})",
                "accent_color": COLORS["trajectory_accent"],
            },
            {
                "label": "Current Centre (X, Y)",
                "value": f"({summary['current_center_x']:.1f}, {summary['current_center_y']:.1f})",
                "subtext": "Image plane pixels",
            },
            {
                "label": "Last Step DX / DY",
                "value": f"{summary['last_dx']:+.1f} / {summary['last_dy']:+.1f} px",
                "subtext": f"Net: {summary['net_dx']:+.1f} / {summary['net_dy']:+.1f} px",
            },
            {
                "label": "Total Pixel Displacement",
                "value": f"{summary['total_displacement_px']:.2f} px",
                "subtext": f"Straight-line net: {summary['net_displacement_px']:.2f} px",
                "accent_color": COLORS["light_indigo"],
            },
            {
                "label": "Average Pixel Speed",
                "value": f"{avg_spd:.2f} px/s" if avg_spd is not None else "N/A",
                "subtext": "Image-plane pixels/sec",
                "accent_color": COLORS["trajectory_accent"],
            },
            {
                "label": "Maximum Pixel Speed",
                "value": f"{max_spd:.2f} px/s" if max_spd is not None else "N/A",
                "subtext": "Excludes flagged gap jumps",
            },
            {
                "label": "Current Movement Direction",
                "value": summary["current_direction"],
                "subtext": f"Dominant: {summary['dominant_direction']}",
                "accent_color": COLORS["light_indigo"],
            },
            {
                "label": "Frames Tracked & Gaps",
                "value": f"{summary['frames_tracked']} frames",
                "subtext": f"Duration: {format_duration(dur) if dur is not None else 'N/A'} | Gaps: {summary['tracking_interruptions']}",
            },
        ],
        columns=4,
    )

    if not summary.get("fps_valid", True):
        st.markdown(
            '<div class="tn-alert-warn">Video FPS metadata was zero or invalid; pixel speed values could not be reliably derived from frame rate.</div>',
            unsafe_allow_html=True,
        )

    # Compact 3-column reference bar inside a balanced card
    st.markdown(
        f"""
        <div class="tn-card" style="padding:0.85rem 1.1rem;">
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:0.85rem;font-size:0.83rem;line-height:1.45;">
                <div>
                    <div style="color:{COLORS['trajectory_accent']};font-weight:700;margin-bottom:0.2rem;">Image Coordinates</div>
                    <div style="color:{COLORS['secondary_text']};">Origin <code>(0, 0)</code> top-left; <code>+DX</code> Right, <code>-DX</code> Left, <code>+DY</code> Down, <code>-DY</code> Up.</div>
                </div>
                <div>
                    <div style="color:{COLORS['light_indigo']};font-weight:700;margin-bottom:0.2rem;">Direction vs Orientation</div>
                    <div style="color:{COLORS['secondary_text']};">Direction reflects 2D centre displacement, not physical 3D body orientation (requires pose keypoints).</div>
                </div>
                <div>
                    <div style="color:{COLORS['trajectory_accent']};font-weight:700;margin-bottom:0.2rem;">Pixel Speed Units</div>
                    <div style="color:{COLORS['secondary_text']};">Reported in <code>pixels/second (px/s)</code>; real-world <code>m/s</code> requires camera scale calibration.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_section_title("Frame-by-Frame Movement Log", f"{len(movement_df)} observations")
    with st.container(border=True):
        st.dataframe(movement_df, width="stretch", height=360, hide_index=True)
