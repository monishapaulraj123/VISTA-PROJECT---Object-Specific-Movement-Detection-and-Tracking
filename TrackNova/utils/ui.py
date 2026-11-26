"""
TrackNova Custom UI Styling & Reusable Dashboard Components.

Enforces the exact TrackNova dark navy / indigo / turquoise colour palette,
consistent responsive card grids, uniform container sizing, and clean states.
"""

from __future__ import annotations

import html
from typing import Any, Dict, List, Optional

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
            padding-top: 1.35rem;
            padding-bottom: 2rem;
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
            padding: 0.54rem 0.88rem;
            font-weight: 500;
            font-size: 0.9rem;
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
        .stButton > button,
        .stDownloadButton > button {{
            min-height: 2.5rem;
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.88rem;
            line-height: 1.25;
            white-space: normal;
            word-break: break-word;
            transition: background-color 0.18s ease, border-color 0.18s ease;
        }}
        .stButton > button[kind="primary"],
        .stDownloadButton > button {{
            background-color: {COLORS['primary_indigo']};
            color: {COLORS['main_text']};
            border: 1px solid {COLORS['light_indigo']};
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
        }}
        .stButton > button[kind="secondary"]:hover {{
            border-color: {COLORS['light_indigo']};
            background-color: #222B42;
        }}

        /* Streamlit Horizontal Columns & Bordered Containers */
        [data-testid="stHorizontalBlock"] {{
            align-items: stretch;
            gap: 0.9rem;
        }}
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background-color: {COLORS['card_bg']};
            border: 1px solid {COLORS['borders']} !important;
            border-radius: 14px !important;
            padding: 1rem 1.15rem !important;
            box-shadow: 0 6px 18px rgba(5, 8, 18, 0.28);
            box-sizing: border-box;
            overflow: hidden;
        }}

        /* Prevent Media, Plots & Tables From Overflowing Containers */
        [data-testid="stPyplot"],
        [data-testid="stImage"],
        [data-testid="stVideo"],
        [data-testid="stDataFrame"] {{
            max-width: 100%;
            overflow: hidden;
            box-sizing: border-box;
        }}
        [data-testid="stPyplot"] img,
        [data-testid="stImage"] img {{
            border-radius: 10px;
            max-width: 100%;
            height: auto;
        }}

        /* Section Titles */
        .tn-section-title {{
            color: {COLORS['light_indigo']};
            font-size: 1rem;
            font-weight: 700;
            letter-spacing: 0.01em;
            margin: 1rem 0 0.55rem 0;
            padding-bottom: 0.35rem;
            border-bottom: 1px solid {COLORS['borders']};
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 0.75rem;
        }}
        .tn-section-sub {{
            color: {COLORS['secondary_text']};
            font-size: 0.8rem;
            font-weight: 500;
        }}

        /* Responsive Equal-Height KPI Grid */
        .tn-kpi-grid {{
            display: grid;
            grid-template-columns: repeat(var(--tn-cols, 4), minmax(0, 1fr));
            gap: 0.85rem;
            align-items: stretch;
            margin-bottom: 0.85rem;
        }}
        @media (max-width: 1120px) {{
            .tn-kpi-grid {{
                grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            }}
        }}
        @media (max-width: 640px) {{
            .tn-kpi-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        /* Equal-Height Two-Column Card Grid */
        .tn-two-col-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.9rem;
            align-items: stretch;
            margin-bottom: 0.85rem;
        }}
        @media (max-width: 900px) {{
            .tn-two-col-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        /* Cards & KPI Cards */
        .tn-card {{
            background-color: {COLORS['card_bg']};
            border: 1px solid {COLORS['borders']};
            border-radius: 14px;
            padding: 1rem 1.15rem;
            box-shadow: 0 6px 18px rgba(5, 8, 18, 0.28);
            margin-bottom: 0.85rem;
            box-sizing: border-box;
            height: 100%;
        }}
        .tn-two-col-grid .tn-card {{
            margin-bottom: 0;
        }}
        .tn-kpi-card {{
            background-color: {COLORS['card_bg']};
            border: 1px solid {COLORS['borders']};
            border-radius: 14px;
            padding: 0.95rem 1.1rem;
            box-shadow: 0 6px 18px rgba(5, 8, 18, 0.28);
            min-height: 106px;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-sizing: border-box;
            overflow: hidden;
        }}
        .tn-kpi-label {{
            color: {COLORS['secondary_text']};
            font-size: 0.74rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.35rem;
            line-height: 1.25;
        }}
        .tn-kpi-value {{
            color: {COLORS['main_text']};
            font-size: 1.35rem;
            font-weight: 700;
            line-height: 1.25;
            overflow-wrap: anywhere;
            word-break: break-word;
        }}
        .tn-kpi-sub {{
            color: {COLORS['light_indigo']};
            font-size: 0.77rem;
            margin-top: 0.35rem;
            line-height: 1.3;
            overflow-wrap: anywhere;
        }}

        /* Page Header */
        .tn-header-bar {{
            background-color: {COLORS['sidebar_bg']};
            border: 1px solid {COLORS['borders']};
            border-left: 4px solid {COLORS['primary_indigo']};
            border-radius: 14px;
            padding: 1rem 1.3rem;
            margin-bottom: 1rem;
            box-shadow: 0 6px 20px rgba(5, 8, 18, 0.3);
        }}
        .tn-header-title {{
            font-size: 1.42rem;
            font-weight: 700;
            color: {COLORS['main_text']};
            margin: 0;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }}
        .tn-header-sub {{
            color: {COLORS['secondary_text']};
            font-size: 0.88rem;
            margin-top: 0.22rem;
        }}

        /* Empty & Alert States */
        .tn-empty-state {{
            background-color: {COLORS['card_bg']};
            border: 1px dashed {COLORS['borders']};
            border-radius: 14px;
            padding: 2.2rem 1.4rem;
            text-align: center;
            margin: 0.75rem 0 1rem 0;
        }}
        .tn-alert-warn {{
            background-color: rgba(251, 191, 36, 0.08);
            border: 1px solid rgba(251, 191, 36, 0.38);
            border-left: 4px solid {COLORS['warning']};
            border-radius: 10px;
            padding: 0.75rem 0.95rem;
            color: {COLORS['main_text']};
            font-size: 0.88rem;
            margin-bottom: 0.75rem;
        }}
        .tn-alert-error {{
            background-color: rgba(248, 113, 113, 0.08);
            border: 1px solid rgba(248, 113, 113, 0.42);
            border-left: 4px solid {COLORS['error']};
            border-radius: 10px;
            padding: 0.75rem 0.95rem;
            color: {COLORS['main_text']};
            font-size: 0.88rem;
            margin-bottom: 0.75rem;
        }}
        .tn-alert-info {{
            background-color: rgba(72, 214, 194, 0.08);
            border: 1px solid rgba(72, 214, 194, 0.35);
            border-left: 4px solid {COLORS['trajectory_accent']};
            border-radius: 10px;
            padding: 0.75rem 0.95rem;
            color: {COLORS['main_text']};
            font-size: 0.88rem;
            margin-bottom: 0.75rem;
        }}

        /* Inputs, Expanders, Uploaders */
        [data-testid="stExpander"] {{
            background-color: {COLORS['sidebar_bg']};
            border: 1px solid {COLORS['borders']};
            border-radius: 12px;
        }}
        [data-testid="stFileUploader"] {{
            background-color: {COLORS['sidebar_bg']};
            border: 1px dashed {COLORS['borders']};
            border-radius: 12px;
            padding: 0.55rem;
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


def render_section_title(title: str, subtitle: str = "") -> None:
    """Render a uniform section heading bar."""
    sub_html = f'<span class="tn-section-sub">{html.escape(subtitle)}</span>' if subtitle else ""
    st.markdown(
        f'<div class="tn-section-title"><span>{html.escape(title)}</span>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def _kpi_card_html(
    label: str,
    value: str,
    subtext: str = "",
    accent_color: Optional[str] = None,
) -> str:
    """Return HTML markup for a single equal-height KPI card."""
    val_style = f"color:{accent_color};" if accent_color else ""
    sub_html = f'<div class="tn-kpi-sub">{html.escape(subtext)}</div>' if subtext else ""
    safe_val = html.escape(str(value))
    return (
        f'<div class="tn-kpi-card" title="{safe_val}">'
        f'<div><div class="tn-kpi-label">{html.escape(label)}</div>'
        f'<div class="tn-kpi-value" style="{val_style}">{safe_val}</div></div>'
        f"{sub_html}</div>"
    )


def render_kpi_card(
    label: str,
    value: str,
    subtext: str = "",
    accent_color: Optional[str] = None,
) -> None:
    """Render a single styled analytics metric card."""
    st.markdown(_kpi_card_html(label, value, subtext, accent_color), unsafe_allow_html=True)


def render_kpi_grid(cards: List[Dict[str, Any]], columns: Optional[int] = None) -> None:
    """Render a responsive, equal-height grid of KPI cards."""
    if not cards:
        return
    col_count = columns or min(5, max(1, len(cards)))
    cards_html = "".join(
        _kpi_card_html(
            label=str(c.get("label", "")),
            value=str(c.get("value", "")),
            subtext=str(c.get("subtext", "")),
            accent_color=c.get("accent_color"),
        )
        for c in cards
    )
    st.markdown(
        f'<div class="tn-kpi-grid" style="--tn-cols:{col_count};">{cards_html}</div>',
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
    icon_html = lucide_svg(icon_name, size=36, color=COLORS["light_indigo"])
    st.markdown(
        f"""
        <div class="tn-empty-state">
            <div style="margin-bottom:0.65rem;">{icon_html}</div>
            <h3 style="margin:0 0 0.4rem 0; color:{COLORS['main_text']}; font-size:1.15rem;">{html.escape(title)}</h3>
            <p style="margin:0 auto; max-width:540px; color:{COLORS['secondary_text']}; font-size:0.9rem; line-height:1.5;">
                {html.escape(description)}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if cta_label and cta_target_page:
        col1, col2, col3 = st.columns([2, 1.4, 2])
        with col2:
            if st.button(cta_label, type="primary", width="stretch", key=key):
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
