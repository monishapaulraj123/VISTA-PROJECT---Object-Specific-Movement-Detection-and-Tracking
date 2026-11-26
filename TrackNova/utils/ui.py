"""
TrackNova Custom UI Styling & Reusable Dashboard Components.

Enforces the exact TrackNova dark navy / indigo / turquoise colour palette,
custom CSS overrides, KPI cards, section headers, and empty/error states.
"""

from __future__ import annotations

import html
from typing import Optional

import streamlit as st

from config import COLORS
from utils.helpers import lucide_svg


def inject_tracknova_css() -> None:
    """Inject custom CSS implementing the strict TrackNova design system."""
    st.markdown(
        f"""
        <style>
        /* App Canvas & Typography */
        .stApp {{
            background-color: {COLORS['main_bg']};
            color: {COLORS['main_text']};
            font-family: 'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        [data-testid="stHeader"] {{
            background-color: rgba(16, 21, 34, 0.88);
            backdrop-filter: blur(8px);
            border-bottom: 1px solid {COLORS['borders']};
        }}
        .block-container {{
            padding-top: 1.6rem;
            padding-bottom: 2.5rem;
            max-width: 1380px;
        }}

        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: {COLORS['sidebar_bg']};
            border-right: 1px solid {COLORS['borders']};
        }}
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p {{
            color: {COLORS['main_text']};
        }}

        /* Sidebar Navigation Buttons */
        div[data-testid="stSidebar"] .stButton > button {{
            width: 100%;
            justify-content: flex-start;
            text-align: left;
            background-color: transparent;
            color: {COLORS['secondary_text']};
            border: 1px solid transparent;
            border-radius: 10px;
            padding: 0.58rem 0.9rem;
            font-weight: 500;
            font-size: 0.92rem;
            transition: all 0.18s ease;
        }}
        div[data-testid="stSidebar"] .stButton > button:hover {{
            background-color: rgba(76, 100, 165, 0.18);
            color: {COLORS['main_text']};
            border-color: {COLORS['borders']};
        }}
        div[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
            background-color: {COLORS['primary_indigo']};
            color: {COLORS['main_text']};
            border: 1px solid {COLORS['light_indigo']};
            font-weight: 600;
            box-shadow: 0 4px 14px rgba(76, 100, 165, 0.35);
        }}
        div[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {{
            background-color: {COLORS['hover_button']};
        }}

        /* Main Area Buttons */
        .stButton > button[kind="primary"],
        .stDownloadButton > button {{
            background-color: {COLORS['primary_indigo']};
            color: {COLORS['main_text']};
            border: 1px solid {COLORS['light_indigo']};
            border-radius: 10px;
            font-weight: 600;
            transition: background-color 0.18s ease, transform 0.12s ease;
        }}
        .stButton > button[kind="primary"]:hover,
        .stDownloadButton > button:hover {{
            background-color: {COLORS['hover_button']};
            border-color: {COLORS['light_indigo']};
            color: {COLORS['main_text']};
        }}
        .stButton > button[kind="secondary"] {{
            background-color: {COLORS['card_bg']};
            color: {COLORS['main_text']};
            border: 1px solid {COLORS['borders']};
            border-radius: 10px;
        }}
        .stButton > button[kind="secondary"]:hover {{
            border-color: {COLORS['light_indigo']};
            background-color: #222B42;
        }}

        /* Cards & Containers */
        .tn-card {{
            background-color: {COLORS['card_bg']};
            border: 1px solid {COLORS['borders']};
            border-radius: 14px;
            padding: 1.05rem 1.2rem;
            box-shadow: 0 6px 20px rgba(5, 8, 18, 0.35);
            margin-bottom: 0.85rem;
        }}
        .tn-kpi-card {{
            background-color: {COLORS['card_bg']};
            border: 1px solid {COLORS['borders']};
            border-radius: 14px;
            padding: 1rem 1.15rem;
            box-shadow: 0 6px 18px rgba(5, 8, 18, 0.32);
            height: 100%;
        }}
        .tn-kpi-label {{
            color: {COLORS['secondary_text']};
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.35rem;
        }}
        .tn-kpi-value {{
            color: {COLORS['main_text']};
            font-size: 1.48rem;
            font-weight: 700;
            line-height: 1.2;
        }}
        .tn-kpi-sub {{
            color: {COLORS['light_indigo']};
            font-size: 0.78rem;
            margin-top: 0.3rem;
        }}
        .tn-header-bar {{
            background-color: {COLORS['sidebar_bg']};
            border: 1px solid {COLORS['borders']};
            border-left: 4px solid {COLORS['primary_indigo']};
            border-radius: 14px;
            padding: 1.15rem 1.4rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 8px 24px rgba(5, 8, 18, 0.35);
        }}
        .tn-header-title {{
            font-size: 1.5rem;
            font-weight: 700;
            color: {COLORS['main_text']};
            margin: 0;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }}
        .tn-header-sub {{
            color: {COLORS['secondary_text']};
            font-size: 0.9rem;
            margin-top: 0.25rem;
        }}
        .tn-empty-state {{
            background-color: {COLORS['card_bg']};
            border: 1px dashed {COLORS['borders']};
            border-radius: 16px;
            padding: 2.5rem 1.5rem;
            text-align: center;
            margin: 1rem 0;
        }}
        .tn-badge {{
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            border: 1px solid {COLORS['borders']};
            background-color: {COLORS['sidebar_bg']};
            color: {COLORS['trajectory_accent']};
        }}
        .tn-alert-warn {{
            background-color: rgba(251, 191, 36, 0.08);
            border: 1px solid rgba(251, 191, 36, 0.38);
            border-left: 4px solid {COLORS['warning']};
            border-radius: 10px;
            padding: 0.85rem 1rem;
            color: {COLORS['main_text']};
            font-size: 0.9rem;
            margin-bottom: 0.85rem;
        }}
        .tn-alert-error {{
            background-color: rgba(248, 113, 113, 0.08);
            border: 1px solid rgba(248, 113, 113, 0.42);
            border-left: 4px solid {COLORS['error']};
            border-radius: 10px;
            padding: 0.85rem 1rem;
            color: {COLORS['main_text']};
            font-size: 0.9rem;
            margin-bottom: 0.85rem;
        }}
        .tn-alert-info {{
            background-color: rgba(72, 214, 194, 0.08);
            border: 1px solid rgba(72, 214, 194, 0.35);
            border-left: 4px solid {COLORS['trajectory_accent']};
            border-radius: 10px;
            padding: 0.85rem 1rem;
            color: {COLORS['main_text']};
            font-size: 0.9rem;
            margin-bottom: 0.85rem;
        }}

        /* Inputs, Selectboxes, Expanders, DataFrames */
        [data-testid="stExpander"] {{
            background-color: {COLORS['card_bg']};
            border: 1px solid {COLORS['borders']};
            border-radius: 12px;
        }}
        [data-testid="stFileUploader"] {{
            background-color: {COLORS['card_bg']};
            border: 1px dashed {COLORS['borders']};
            border-radius: 12px;
            padding: 0.6rem;
        }}
        div[data-baseweb="select"] > div {{
            background-color: {COLORS['sidebar_bg']};
            border-color: {COLORS['borders']};
            color: {COLORS['main_text']};
        }}
        .stProgress > div > div > div > div {{
            background-color: {COLORS['trajectory_accent']};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, subtitle: str, icon_name: str = "crosshair") -> None:
    """Render a consistent top page header with Lucide SVG icon."""
    icon_html = lucide_svg(icon_name, size=24, color=COLORS["trajectory_accent"])
    st.markdown(
        f"""
        <div class="tn-header-bar">
            <div class="tn-header-title">{icon_html}<span>{html.escape(title)}</span></div>
            <div class="tn-header-sub">{html.escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(
    label: str,
    value: str,
    subtext: str = "",
    accent_color: Optional[str] = None,
) -> None:
    """Render a styled analytics metric card."""
    val_style = f"color:{accent_color};" if accent_color else ""
    sub_html = f'<div class="tn-kpi-sub">{html.escape(subtext)}</div>' if subtext else ""
    st.markdown(
        f"""
        <div class="tn-kpi-card">
            <div class="tn-kpi-label">{html.escape(label)}</div>
            <div class="tn-kpi-value" style="{val_style}">{html.escape(str(value))}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(
    title: str,
    description: str,
    icon_name: str = "video",
    cta_label: Optional[str] = None,
    cta_target_page: Optional[str] = None,
    key: str = "empty_cta",
) -> None:
    """Render a clean, helpful empty state with optional navigation CTA."""
    icon_html = lucide_svg(icon_name, size=38, color=COLORS["light_indigo"])
    st.markdown(
        f"""
        <div class="tn-empty-state">
            <div style="margin-bottom:0.75rem;">{icon_html}</div>
            <h3 style="margin:0 0 0.45rem 0; color:{COLORS['main_text']};">{html.escape(title)}</h3>
            <p style="margin:0 auto; max-width:560px; color:{COLORS['secondary_text']}; font-size:0.94rem;">
                {html.escape(description)}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if cta_label and cta_target_page:
        col1, col2, col3 = st.columns([2, 1.5, 2])
        with col2:
            if st.button(cta_label, type="primary", use_container_width=True, key=key):
                st.session_state["current_page"] = cta_target_page
                st.rerun()


def show_friendly_error(message: str, exc: Optional[Exception] = None) -> None:
    """Display user-friendly error message; show exception details only in Debug Mode."""
    st.markdown(
        f'<div class="tn-alert-error"><strong>Error:</strong> {html.escape(message)}</div>',
        unsafe_allow_html=True,
    )
    if exc is not None and st.session_state.get("debug_mode", False):
        st.exception(exc)
