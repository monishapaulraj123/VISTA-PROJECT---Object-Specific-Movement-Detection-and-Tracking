"""
TrackNova Trajectory Analysis Page.

Generates all 5 Matplotlib movement & trajectory charts from real tracking data
and provides individual PNG downloads for each graph.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from config import COLORS
from modules.trajectory import build_trajectory_data
from modules.visualization import (
    figure_to_png_bytes,
    plot_displacement_analysis,
    plot_speed_vs_frame,
    plot_x_vs_frame,
    plot_xy_trajectory,
    plot_y_vs_frame,
)
from utils.ui import render_empty_state, render_kpi_card, render_page_header


def _render_chart_card(title: str, fig, download_stem: str, key: str) -> None:
    """Display a Matplotlib chart inside a clean layout with PNG download button."""
    st.markdown(f"<h5 style='color:{COLORS['light_indigo']};margin-bottom:0.35rem;'>{title}</h5>", unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=True)
    png_bytes = figure_to_png_bytes(fig)
    plt.close(fig)
    st.download_button(
        label=f"Download {title} (PNG)",
        data=png_bytes,
        file_name=f"{download_stem}.png",
        mime="image/png",
        key=key,
    )


def render() -> None:
    render_page_header(
        "Trajectory & Movement Visualization",
        "Real-data Matplotlib position, speed, 2D image-plane path, and displacement analytics",
        icon_name="activity",
    )

    session_result = st.session_state.get("processing_result")
    movement_df = st.session_state.get("movement_df")
    summary = st.session_state.get("movement_summary")

    if session_result is None or movement_df is None or movement_df.empty or not summary or not summary.get("has_data"):
        render_empty_state(
            title="No Trajectory Data Available",
            description="Process a video and select a ByteTrack object ID in Video Tracking to generate trajectory paths and movement graphs.",
            icon_name="activity",
            cta_label="Go to Video Tracking",
            cta_target_page="Video Tracking",
            key="ta_empty_cta",
        )
        return

    traj = build_trajectory_data(movement_df)
    tid = summary["selected_id"]

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Target Object", f"ID #{tid}", f"Class: {summary['object_class']}", accent_color=COLORS["trajectory_accent"])
    with k2:
        render_kpi_card("Trajectory Points", str(traj["point_count"]), f"Start: ({traj['start_point'][0]:.0f}, {traj['start_point'][1]:.0f})")
    with k3:
        render_kpi_card("Path Length", f"{traj['path_length_px']:.1f} px", f"Net Displacement: {traj['net_displacement_px']:.1f} px", accent_color=COLORS["light_indigo"])
    with k4:
        lin = traj.get("linearity_ratio")
        render_kpi_card("Path Linearity", f"{lin:.3f}" if lin is not None else "N/A", "1.0 = perfectly straight path")

    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

    # Row 1: Graph 1 (X vs Frame) & Graph 2 (Y vs Frame)
    g1_col, g2_col = st.columns(2)
    with g1_col:
        fig1 = plot_x_vs_frame(movement_df)
        _render_chart_card("Graph 1 — X Position vs Frame Number", fig1, f"TrackNova_ID{tid}_X_vs_Frame", "dl_g1")
    with g2_col:
        fig2 = plot_y_vs_frame(movement_df)
        _render_chart_card("Graph 2 — Y Position vs Frame Number", fig2, f"TrackNova_ID{tid}_Y_vs_Frame", "dl_g2")

    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

    # Row 2: Graph 3 (Pixel Speed vs Frame) & Graph 4 (X/Y Trajectory)
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

    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

    # Row 3: Graph 5 (Displacement Analysis)
    fig5 = plot_displacement_analysis(movement_df)
    _render_chart_card("Graph 5 — Horizontal, Vertical & Cumulative Displacement Analysis", fig5, f"TrackNova_ID{tid}_Displacement", "dl_g5")
