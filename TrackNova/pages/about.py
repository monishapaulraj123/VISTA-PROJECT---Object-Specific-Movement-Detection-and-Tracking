"""
TrackNova About Project Page.

Presents the TrackNova identity, project overview, technology stack,
and a connected top-to-bottom vertical workflow diagram for all 13 stages.
"""

from __future__ import annotations

import html

import streamlit as st

from config import APP_NAME, COLORS, PROJECT_DESCRIPTION, SUBTITLE
from utils.ui import render_page_header, render_section_title


def _connector_svg() -> str:
    """Render an actual UI vertical connector line and downward SVG arrow."""
    return (
        f'<div style="display:flex;flex-direction:column;align-items:center;margin:0.12rem 0;">'
        f'<div style="width:2px;height:14px;background:{COLORS["primary_indigo"]};"></div>'
        f'<svg width="16" height="10" viewBox="0 0 16 10" fill="none" '
        f'xmlns="http://www.w3.org/2000/svg" style="display:block;">'
        f'<path d="M2 2L8 8L14 2" stroke="{COLORS["trajectory_accent"]}" '
        f'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>'
        f"</svg></div>"
    )


def render() -> None:
    render_page_header(
        f"About {APP_NAME}",
        f"{APP_NAME} ({SUBTITLE})",
        icon_name="info",
    )

    tech_items = [
        ("Python 3.10+", "Core application runtime & scientific computing"),
        ("OpenCV", "Video decoding, preprocessing, bounding-box & trajectory rendering"),
        ("Ultralytics YOLO", "Multi-class object detection (Pretrained COCO & Custom weights)"),
        ("ByteTrack", "Persistent multi-object ID association across consecutive frames"),
        ("NumPy & Pandas", "Euclidean displacement, speed kinematics, and CSV data pipelines"),
        ("Matplotlib", "Dark-themed position, speed, trajectory, and displacement charts"),
        ("Streamlit", "Interactive modular web workspace and analytics dashboard"),
        ("ReportLab & Pillow", "PDF report generation and branded asset synthesis"),
    ]
    rows_html = "".join(
        f"<div style='padding:0.38rem 0; border-bottom:1px solid {COLORS['borders']}; font-size:0.86rem;'>"
        f"<span style='color:{COLORS['trajectory_accent']}; font-weight:600;'>{html.escape(name)}</span> "
        f"<span style='color:{COLORS['secondary_text']};'>— {html.escape(desc)}</span></div>"
        for name, desc in tech_items
    )

    # Equal-height two-column CSS Grid for Overview & Technology Stack
    st.markdown(
        f"""
        <div class="tn-two-col-grid">
            <div class="tn-card">
                <h4 style="color:{COLORS['light_indigo']}; margin-top:0; margin-bottom:0.65rem;">Project Overview</h4>
                <p style="color:{COLORS['main_text']}; line-height:1.55; font-size:0.9rem;">
                    <strong>{html.escape(APP_NAME)} ({html.escape(SUBTITLE)})</strong><br/>
                    {html.escape(PROJECT_DESCRIPTION)}
                </p>
                <div style="color:{COLORS['trajectory_accent']}; font-weight:700; font-size:0.88rem; margin-bottom:0.25rem;">Objective</div>
                <p style="color:{COLORS['secondary_text']}; line-height:1.55; font-size:0.88rem; margin-bottom:0.7rem;">
                    To detect multiple objects in video frames, assign unique tracking IDs, select a
                    particular object, and estimate its movement, displacement, speed, direction, and trajectory.
                </p>
                <div style="color:{COLORS['trajectory_accent']}; font-weight:700; font-size:0.88rem; margin-bottom:0.25rem;">Expected Outcome &amp; Evaluation Scope</div>
                <p style="color:{COLORS['secondary_text']}; line-height:1.55; font-size:0.88rem; margin-bottom:0;">
                    A modular Python &amp; OpenCV system for object-specific motion analysis and multi-format reporting.
                    Without ground-truth labels, performance evaluation uses observational proxy metrics
                    (processing FPS, ID assignment rate, and continuity gaps) rather than MOTA/IDF1 estimates;
                    speeds are reported in <code>px/s</code>.
                </p>
            </div>
            <div class="tn-card">
                <h4 style="color:{COLORS['light_indigo']}; margin-top:0; margin-bottom:0.55rem;">Technology Stack</h4>
                {rows_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_section_title(
        "Project Workflow Pipeline",
        "End-to-end 13-stage vertical process flow (Stage 01 to Stage 13)",
    )

    # Exact 13 stages in top-to-bottom order
    # Key processing milestones receive a subtle trajectory-accent left border
    stages = [
        ("01", "Python & OpenCV Fundamentals", "Basic programming and computer vision foundation.", False),
        ("02", "Video Dataset Input", "Upload and load video datasets.", False),
        ("03", "Frame Extraction & Preprocessing", "Extract video frames and prepare them for analysis.", False),
        ("04", "YOLO Object Detection", "Detect objects in individual frames.", True),
        ("05", "Object Classification", "Identify detected object categories such as person, car, bus, and bicycle.", False),
        ("06", "Object Tracking & ID Assignment", "Assign unique tracking IDs using ByteTrack.", True),
        ("07", "Object ID Selection", "Allow the user to select a specific detected object.", True),
        ("08", "Selected Object Tracking", "Track the selected object across consecutive frames.", False),
        ("09", "X/Y Pixel Displacement", "Calculate horizontal and vertical movement.", False),
        ("10", "Movement Direction & Orientation", "Analyse the object's movement direction and angle.", False),
        ("11", "Speed & Trajectory Calculation", "Calculate pixel-based speed and generate the object's trajectory.", True),
        ("12", "Movement Visualization & Graphs", "Visualize bounding boxes, IDs, arrows, trajectories, and graphs.", False),
        ("13", "Testing & Performance Evaluation", "Test different videos/datasets and evaluate system performance.", True),
    ]

    flow_nodes = []
    for idx, (num, title, desc, is_key_stage) in enumerate(stages):
        left_accent = COLORS["trajectory_accent"] if is_key_stage else COLORS["primary_indigo"]
        badge_bg = "rgba(72, 214, 194, 0.14)" if is_key_stage else "rgba(76, 100, 165, 0.24)"
        badge_border = COLORS["trajectory_accent"] if is_key_stage else COLORS["light_indigo"]
        badge_text = COLORS["trajectory_accent"] if is_key_stage else COLORS["light_indigo"]

        node_html = (
            f'<div style="width:100%;background:{COLORS["sidebar_bg"]};border:1px solid {COLORS["borders"]};'
            f'border-left:4px solid {left_accent};border-radius:12px;padding:0.68rem 1rem;'
            f'display:flex;align-items:center;gap:0.9rem;box-sizing:border-box;">'
            f'<div style="min-width:38px;height:38px;border-radius:10px;background:{badge_bg};'
            f'border:1px solid {badge_border};color:{badge_text};display:flex;align-items:center;'
            f'justify-content:center;font-weight:800;font-size:0.86rem;flex-shrink:0;">{num}</div>'
            f'<div style="min-width:0;flex:1;">'
            f'<div style="color:{COLORS["main_text"]};font-weight:700;font-size:0.93rem;line-height:1.3;">'
            f'{num} &ndash; {html.escape(title)}</div>'
            f'<div style="color:{COLORS["secondary_text"]};font-size:0.82rem;line-height:1.38;margin-top:0.12rem;">'
            f'{html.escape(desc)}</div>'
            f"</div></div>"
        )
        flow_nodes.append(node_html)
        if idx < len(stages) - 1:
            flow_nodes.append(_connector_svg())

    flow_html = "".join(flow_nodes)
    st.markdown(
        f"""
        <div class="tn-card" style="padding:1.15rem 1.25rem;">
            <div style="max-width:720px;width:100%;margin:0 auto;display:flex;flex-direction:column;align-items:stretch;">
                {flow_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
