"""
TrackNova Trajectory Analysis Page.

Generates all 5 Matplotlib movement & trajectory charts from real tracking data
for the active video session inside balanced card containers with PNG downloads.
"""

from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import streamlit as st

from config import COLORS
from modules.analytics import activate_selected_track_id, sync_active_tracking_state
from modules.trajectory import build_trajectory_data
from modules.visualization import (
    figure_to_png_bytes,
    plot_displacement_analysis,
    plot_speed_vs_frame,
    plot_x_vs_frame,
    plot_xy_trajectory,
    plot_y_vs_frame,
)
from utils.ui import (
    render_empty_state,
    render_kpi_grid,
    render_page_header,
    render_section_title,
)


def _render_chart_card(title: str, fig, download_stem: str, key: str) -> None:
    """Display a Matplotlib chart inside a bordered card with full-width PNG download."""
    with st.container(border=True):
        st.markdown(
            f"<div style='color:{COLORS['light_indigo']};font-weight:700;font-size:0.95rem;margin-bottom:0.45rem;'>{title}</div>",
            unsafe_allow_html=True,
        )
        st.pyplot(fig, width="stretch")
        png_bytes = figure_to_png_bytes(fig)
        plt.close(fig)
        st.download_button(
            label="Download Chart (PNG)",
            data=png_bytes,
            file_name=f"{download_stem}.png",
            mime="image/png",
            width="stretch",
            key=key,
        )


def render() -> None:
    render_page_header(
        "Trajectory & Movement Visualization",
        "Real-data Matplotlib position, speed, 2D image-plane path, and displacement analytics",
        icon_name="activity",
    )

    session_result, movement_df, summary = sync_active_tracking_state()

    if session_result is None:
        render_empty_state(
            title="No Trajectory Data Available",
            description="Process a video and select a ByteTrack object ID in Video Tracking to generate trajectory paths and movement graphs.",
            icon_name="activity",
            cta_label="Go to Video Tracking",
            cta_target_page="Video Tracking",
            key="ta_empty_cta",
        )
        return

    if not session_result.unique_track_ids:
        render_empty_state(
            title="No Persistent Tracking IDs Found",
            description=(
                f"Processed {session_result.total_frames_processed} frames from "
                f"'{st.session_state.get('active_video_name') or session_result.video_meta.file_name}' "
                f"({session_result.total_detections} raw detections), but ByteTrack did not establish persistent tracking IDs. "
                "Try lowering the confidence threshold or removing class filters in Video Tracking."
            ),
            icon_name="activity",
            cta_label="Adjust Settings in Video Tracking",
            cta_target_page="Video Tracking",
            key="ta_empty_no_tracks",
        )
        return

    active_name = st.session_state.get("active_video_name") or session_result.video_meta.file_name

    # Allow selecting/switching Target Tracking ID directly on Trajectory Analysis
    with st.container(border=True):
        sel_c1, sel_c2 = st.columns([2.2, 1])
        options: List[Optional[int]] = [None] + session_result.unique_track_ids
        cur_sel = st.session_state.get("selected_track_id")
        if cur_sel not in options:
            cur_sel = None

        def _fmt(val: Optional[int]) -> str:
            if val is None:
                return "-- Select a Tracking ID --"
            s = session_result.track_summaries[val]
            return f"ID #{val} — {s.class_name} ({s.frames_tracked} frames, conf {s.avg_confidence:.2f})"

        with sel_c1:
            chosen = st.selectbox(
                f"Active Target Tracking ID ({active_name})",
                options=options,
                index=options.index(cur_sel),
                format_func=_fmt,
                key="ta_track_id_select",
            )
            if chosen != cur_sel:
                activate_selected_track_id(
                    session_result, chosen, user_cleared=(chosen is None)
                )
                st.session_state["re_rendered_video_path"] = None
                st.rerun()
        with sel_c2:
            st.markdown(
                f"<div style='padding-top:1.7rem;color:{COLORS['secondary_text']};font-size:0.84rem;'>"
                f"Detected IDs: <strong style='color:{COLORS['trajectory_accent']};'>{len(session_result.unique_track_ids)}</strong> "
                f"| Resolution: <strong style='color:{COLORS['main_text']};'>{session_result.processed_width}x{session_result.processed_height}</strong>"
                f"</div>",
                unsafe_allow_html=True,
            )

    if movement_df is None or movement_df.empty or not summary or not summary.get("has_data"):
        render_empty_state(
            title="No Object ID Selected",
            description="Choose one of the detected ByteTrack IDs above (or in Video Tracking) to generate trajectory graphs.",
            icon_name="activity",
            cta_label="Go to Video Tracking",
            cta_target_page="Video Tracking",
            key="ta_empty_no_id",
        )
        return

    traj = build_trajectory_data(movement_df)
    tid = summary["selected_id"]
    lin = traj.get("linearity_ratio")

    render_section_title("Trajectory Summary", f"{active_name} • Target ID #{tid}")
    render_kpi_grid(
        [
            {
                "label": "Target Object",
                "value": f"ID #{tid}",
                "subtext": f"Class: {summary['object_class']}",
                "accent_color": COLORS["trajectory_accent"],
            },
            {
                "label": "Trajectory Points",
                "value": str(traj["point_count"]),
                "subtext": f"Start: ({traj['start_point'][0]:.0f}, {traj['start_point'][1]:.0f})",
            },
            {
                "label": "Path Length",
                "value": f"{traj['path_length_px']:.1f} px",
                "subtext": f"Net Displacement: {traj['net_displacement_px']:.1f} px",
                "accent_color": COLORS["light_indigo"],
            },
            {
                "label": "Path Linearity",
                "value": f"{lin:.3f}" if lin is not None else "N/A",
                "subtext": "1.0 = straight-line path",
            },
        ],
        columns=4,
    )

    render_section_title("Position & Speed Analytics")
    g1_col, g2_col = st.columns(2)
    with g1_col:
        fig1 = plot_x_vs_frame(movement_df)
        _render_chart_card("Graph 1 — X Position vs Frame Number", fig1, f"TrackNova_ID{tid}_X_vs_Frame", "dl_g1")
    with g2_col:
        fig2 = plot_y_vs_frame(movement_df)
        _render_chart_card("Graph 2 — Y Position vs Frame Number", fig2, f"TrackNova_ID{tid}_Y_vs_Frame", "dl_g2")

    g3_col, g4_col = st.columns(2)
    with g3_col:
        fig3 = plot_speed_vs_frame(movement_df)
        _render_chart_card("Graph 3 — Pixel Speed vs Frame Number", fig3, f"TrackNova_ID{tid}_Speed_vs_Frame", "dl_g3")
    with g4_col:
        fig4 = plot_xy_trajectory(
            movement_df,
            frame_width=session_result.processed_width,
            frame_height=session_result.processed_height,
        )
        _render_chart_card("Graph 4 — 2D X/Y Trajectory Path", fig4, f"TrackNova_ID{tid}_XY_Trajectory", "dl_g4")

    render_section_title("Displacement Analysis")
    fig5 = plot_displacement_analysis(movement_df)
    _render_chart_card(
        "Graph 5 — Horizontal, Vertical & Cumulative Displacement Analysis",
        fig5,
        f"TrackNova_ID{tid}_Displacement",
        "dl_g5",
    )
