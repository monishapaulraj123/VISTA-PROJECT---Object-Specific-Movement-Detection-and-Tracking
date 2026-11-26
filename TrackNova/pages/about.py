"""
TrackNova About Project Page.

Presents project identity, objective, technology stack, expected outcome,
and a visual 13-stage processing pipeline diagram.
"""

from __future__ import annotations

import streamlit as st

from config import APP_NAME, COLORS, FULL_TITLE, SUBTITLE, TAGLINE
from utils.ui import render_page_header


def render() -> None:
    render_page_header(
        f"About {APP_NAME}",
        f"{FULL_TITLE} — {TAGLINE}",
        icon_name="info",
    )

    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.markdown(
            f"""
            <div class="tn-card">
                <h4 style="color:{COLORS['light_indigo']}; margin-top:0;">Project Overview</h4>
                <p style="color:{COLORS['main_text']}; line-height:1.6;">
                    <strong>{APP_NAME}</strong> ({SUBTITLE}) is an AI-powered computer vision platform
                    built for <strong>{FULL_TITLE}</strong>.
                </p>
                <h5 style="color:{COLORS['trajectory_accent']}; margin-bottom:0.35rem;">Objective</h5>
                <p style="color:{COLORS['secondary_text']}; line-height:1.6;">
                    To detect multiple objects in video frames, assign unique tracking IDs, select a
                    particular object, and estimate its movement, displacement, speed, direction, and trajectory.
                </p>
                <h5 style="color:{COLORS['trajectory_accent']}; margin-bottom:0.35rem;">Expected Outcome</h5>
                <p style="color:{COLORS['secondary_text']}; line-height:1.6; margin-bottom:0;">
                    A functional, modular Python &amp; OpenCV system for object-specific video movement
                    detection, real-time visual overlay, quantitative motion analytics, proxy-based tracking
                    evaluation, and downloadable multi-format reports.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
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
            f"<div style='padding:0.42rem 0; border-bottom:1px solid {COLORS['borders']};'>"
            f"<span style='color:{COLORS['trajectory_accent']}; font-weight:600;'>{name}</span> "
            f"<span style='color:{COLORS['secondary_text']}; font-size:0.86rem;'>— {desc}</span></div>"
            for name, desc in tech_items
        )
        st.markdown(
            f"""
            <div class="tn-card">
                <h4 style="color:{COLORS['light_indigo']}; margin-top:0;">Technology Stack</h4>
                {rows_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Clean visual workflow diagram
    stages = [
        ("1", "Video Input", "Upload or choose dataset video"),
        ("2", "Frame Extraction", "Stream frames via OpenCV VideoCapture"),
        ("3", "Image Preprocessing", "Resize & optional contrast enhancement"),
        ("4", "YOLO Object Detection", "Detect bounding boxes & classes"),
        ("5", "ByteTrack Tracking", "Associate detections across frames"),
        ("6", "Unique ID Assignment", "Maintain persistent object IDs"),
        ("7", "Object ID Selection", "Dynamically select target object ID"),
        ("8", "Selected Object Tracking", "Isolate target across frames"),
        ("9", "Movement Calculation", "Compute Centre X/Y, DX, DY, Distance"),
        ("10", "Kinematics & Direction", "Estimate pixel speed (px/s) & 2D direction"),
        ("11", "Trajectory Visualization", "Overlay path, markers & direction arrows"),
        ("12", "Analytics & Evaluation", "Generate 5 graphs & proxy evaluation"),
        ("13", "Reports & Export", "Export CSV, PNG, Video, HTML & PDF"),
    ]

    cards_html = ""
    for idx, (num, title, desc) in enumerate(stages):
        arrow = (
            f"<div style='text-align:center; color:{COLORS['trajectory_accent']}; font-size:1.1rem; margin:0.15rem 0;'>&#8595;</div>"
            if idx < len(stages) - 1
            else ""
        )
        cards_html += (
            f"<div style='display:flex; align-items:center; gap:0.85rem; background:{COLORS['sidebar_bg']}; "
            f"border:1px solid {COLORS['borders']}; border-left:4px solid {COLORS['primary_indigo']}; "
            f"border-radius:10px; padding:0.55rem 0.9rem;'>"
            f"<span style='min-width:28px; height:28px; border-radius:50%; background:{COLORS['primary_indigo']}; "
            f"color:{COLORS['main_text']}; display:inline-flex; align-items:center; justify-content:center; "
            f"font-weight:700; font-size:0.82rem;'>{num}</span>"
            f"<div><div style='color:{COLORS['main_text']}; font-weight:600; font-size:0.93rem;'>{title}</div>"
            f"<div style='color:{COLORS['secondary_text']}; font-size:0.8rem;'>{desc}</div></div>"
            f"</div>{arrow}"
        )

    st.markdown(
        f"""
        <div class="tn-card">
            <h4 style="color:{COLORS['light_indigo']}; margin-top:0; margin-bottom:0.85rem;">TrackNova System Processing Workflow</h4>
            <div style="max-width:680px; margin:0 auto;">{cards_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
