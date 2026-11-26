"""
TrackNova Reports & Export Page.

Provides real exports built strictly from active tracking data:
A. CSV Export (frame-by-frame movement & multi-dataset comparison)
B. Processed Video Export
C. PNG Graph Exports
D. Summary Report Export (PDF and HTML)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from config import COLORS
from modules.analytics import sessions_to_dataframe
from modules.evaluation import evaluate_tracking_performance
from modules.export import (
    export_comparison_csv,
    export_movement_csv,
    export_summary_html,
    export_summary_pdf,
)
from modules.trajectory import build_trajectory_data
from modules.visualization import (
    figure_to_png_bytes,
    plot_displacement_analysis,
    plot_speed_vs_frame,
    plot_x_vs_frame,
    plot_xy_trajectory,
    plot_y_vs_frame,
    render_annotated_video,
)
from utils.ui import render_empty_state, render_page_header, show_friendly_error


def render() -> None:
    render_page_header(
        "Reports & Data Export",
        "Download frame-by-frame CSVs, annotated videos, Matplotlib PNG charts, and TrackNova PDF/HTML summary reports",
        icon_name="file-text",
    )

    session_result = st.session_state.get("processing_result")
    if session_result is None:
        render_empty_state(
            title="No Results Ready for Export",
            description="Process a video in Video Tracking first to generate downloadable CSVs, videos, graphs, and reports.",
            icon_name="file-text",
            cta_label="Go to Video Tracking",
            cta_target_page="Video Tracking",
            key="rep_empty_cta",
        )
        return

    movement_df = st.session_state.get("movement_df")
    movement_summary = st.session_state.get("movement_summary")
    has_selected = bool(movement_df is not None and not movement_df.empty and movement_summary and movement_summary.get("has_data"))
    eval_data = st.session_state.get("evaluation_result") or evaluate_tracking_performance(
        session_result, movement_df, movement_summary
    )
    traj_data = build_trajectory_data(movement_df) if has_selected else None

    # --- A. CSV Export ---
    st.markdown(f"<h4 style='color:{COLORS['light_indigo']};'>A. CSV Data Export</h4>", unsafe_allow_html=True)
    csv_c1, csv_c2 = st.columns(2)

    with csv_c1:
        if has_selected and movement_summary is not None:
            try:
                csv_bytes, csv_path = export_movement_csv(
                    movement_df,
                    session_result.video_meta.file_name,
                    movement_summary["selected_id"],
                )
                st.download_button(
                    label=f"Download Selected Object CSV (ID #{movement_summary['selected_id']})",
                    data=csv_bytes,
                    file_name=csv_path.name,
                    mime="text/csv",
                    type="primary",
                    use_container_width=True,
                    key="rep_dl_move_csv",
                )
                st.caption(f"Includes {len(movement_df)} frame rows with DX, DY, displacement, pixel speed, direction, and angle.")
            except Exception as exc:
                show_friendly_error(f"CSV export failed: {exc}", exc)
        else:
            st.info("Select a tracking ID in Video Tracking to export frame-by-frame movement CSV.")

    with csv_c2:
        comp_df = sessions_to_dataframe(st.session_state.get("dataset_sessions", []))
        if not comp_df.empty:
            try:
                comp_bytes, comp_path = export_comparison_csv(comp_df)
                st.download_button(
                    label="Download Multi-Dataset Comparison CSV",
                    data=comp_bytes,
                    file_name=comp_path.name,
                    mime="text/csv",
                    use_container_width=True,
                    key="rep_dl_comp_csv",
                )
                st.caption(f"Contains {len(comp_df)} processed dataset session(s).")
            except Exception as exc:
                show_friendly_error(f"Comparison CSV export failed: {exc}", exc)
        else:
            st.info("No multi-dataset comparison entries recorded yet.")

    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

    # --- B. Processed Video Export ---
    st.markdown(f"<h4 style='color:{COLORS['light_indigo']};'>B. Processed Video Export</h4>", unsafe_allow_html=True)
    sel_id = st.session_state.get("selected_track_id")
    vid_path_str = st.session_state.get("re_rendered_video_path") or st.session_state.get("processed_video_path")

    vcol1, vcol2 = st.columns([1, 1])
    with vcol1:
        if sel_id is not None and st.button(
            f"Render Fresh Annotated Video for ID #{sel_id}",
            type="secondary",
            use_container_width=True,
            key="rep_render_vid_btn",
        ):
            ls = st.session_state.get("last_settings", {})
            r_prog = st.progress(0.0, text="Rendering annotated video...")
            out_p, codec_used, err = render_annotated_video(
                session_result=session_result,
                selected_track_id=sel_id,
                movement_df=movement_df,
                show_confidence=bool(ls.get("show_confidence", True)),
                show_other_objects=bool(ls.get("show_other_objects", True)),
                target_width=ls.get("target_width"),
                enhance_contrast=bool(ls.get("enhance_contrast", False)),
                progress_callback=lambda cur, tot: r_prog.progress(
                    min(1.0, cur / max(1, tot)),
                    text=f"Rendering frame {cur}/{tot}...",
                ),
            )
            if err:
                show_friendly_error(err)
            else:
                st.session_state["re_rendered_video_path"] = str(out_p)
                vid_path_str = str(out_p)
                st.success(f"Video rendered using codec '{codec_used}'.")

    with vcol2:
        if vid_path_str and Path(vid_path_str).exists():
            vp = Path(vid_path_str)
            mime_type = "video/mp4" if vp.suffix.lower() == ".mp4" else "video/x-msvideo"
            st.download_button(
                label=f"Download Processed Video ({vp.name})",
                data=vp.read_bytes(),
                file_name=vp.name,
                mime=mime_type,
                type="primary",
                use_container_width=True,
                key="rep_dl_video",
            )
        else:
            st.info("No rendered video file on disk yet. Click 'Render Fresh Annotated Video' to generate one.")

    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

    # --- C. Graph Export (PNG) ---
    st.markdown(f"<h4 style='color:{COLORS['light_indigo']};'>C. Graph Export (PNG)</h4>", unsafe_allow_html=True)
    if has_selected and movement_summary is not None:
        tid = movement_summary["selected_id"]
        chart_specs = [
            ("X Position vs Frame", plot_x_vs_frame(movement_df), f"TrackNova_ID{tid}_X_vs_Frame"),
            ("Y Position vs Frame", plot_y_vs_frame(movement_df), f"TrackNova_ID{tid}_Y_vs_Frame"),
            ("Pixel Speed vs Frame", plot_speed_vs_frame(movement_df), f"TrackNova_ID{tid}_Speed_vs_Frame"),
            (
                "2D X/Y Trajectory",
                plot_xy_trajectory(
                    movement_df,
                    frame_width=session_result.processed_width,
                    frame_height=session_result.processed_height,
                ),
                f"TrackNova_ID{tid}_XY_Trajectory",
            ),
            ("Displacement Analysis", plot_displacement_analysis(movement_df), f"TrackNova_ID{tid}_Displacement"),
        ]
        g_cols = st.columns(5)
        for idx, (label, fig, stem) in enumerate(chart_specs):
            png_bytes = figure_to_png_bytes(fig)
            plt.close(fig)
            with g_cols[idx]:
                st.download_button(
                    label=f"PNG: {label}",
                    data=png_bytes,
                    file_name=f"{stem}.png",
                    mime="image/png",
                    use_container_width=True,
                    key=f"rep_dl_png_{idx}",
                )
    else:
        st.info("Select an object ID in Video Tracking to enable PNG graph exports.")

    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

    # --- D. Summary Reports (PDF & HTML) ---
    st.markdown(f"<h4 style='color:{COLORS['light_indigo']};'>D. Summary Report Export</h4>", unsafe_allow_html=True)
    r_c1, r_c2 = st.columns(2)
    with r_c1:
        try:
            pdf_bytes, pdf_path = export_summary_pdf(
                session_result, movement_summary, traj_data, eval_data
            )
            st.download_button(
                label="Download Summary Report (PDF)",
                data=pdf_bytes,
                file_name=pdf_path.name,
                mime="application/pdf",
                type="primary",
                use_container_width=True,
                key="rep_dl_pdf",
            )
        except Exception as exc:
            show_friendly_error(f"PDF report generation failed: {exc}", exc)

    with r_c2:
        try:
            html_bytes, html_path = export_summary_html(
                session_result, movement_summary, traj_data, eval_data
            )
            st.download_button(
                label="Download Summary Report (HTML)",
                data=html_bytes,
                file_name=html_path.name,
                mime="text/html",
                use_container_width=True,
                key="rep_dl_html",
            )
        except Exception as exc:
            show_friendly_error(f"HTML report generation failed: {exc}", exc)
