"""
TrackNova — AI-Powered Object-Specific Movement Detection and Tracking System.

Main Streamlit entry point configuring page identity, custom dark-navy theme,
sidebar branding & navigation, and modular page routing.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from config import APP_NAME, COLORS, LOGO_PATH, NAV_PAGES, SUBTITLE, TAGLINE
from pages import (
    about,
    dashboard,
    evaluation,
    object_analysis,
    reports,
    trajectory_analysis,
    video_tracking,
)
from utils.helpers import get_logo_base64, init_session_state, reset_tracking_session
from utils.ui import inject_tracknova_css

# Must be the first Streamlit command
st.set_page_config(
    page_title=f"{APP_NAME} — {SUBTITLE}",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()
inject_tracknova_css()

PAGE_DISPATCH = {
    "Dashboard": dashboard.render,
    "Video Tracking": video_tracking.render,
    "Object Analysis": object_analysis.render,
    "Trajectory Analysis": trajectory_analysis.render,
    "Performance Evaluation": evaluation.render,
    "Reports & Export": reports.render,
    "About Project": about.render,
}


def _render_sidebar() -> None:
    """Render TrackNova logo, navigation menu, active session pill, and reset control."""
    with st.sidebar:
        logo_b64 = get_logo_base64()
        logo_img_html = (
            f'<img src="data:image/png;base64,{logo_b64}" '
            f'style="width:46px;height:46px;border-radius:10px;border:1px solid {COLORS["borders"]};" />'
            if logo_b64
            else ""
        )
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:0.75rem;padding:0.4rem 0.2rem 0.9rem 0.2rem;
                        border-bottom:1px solid {COLORS['borders']};margin-bottom:0.9rem;">
                {logo_img_html}
                <div>
                    <div style="font-size:1.22rem;font-weight:800;color:{COLORS['main_text']};letter-spacing:0.02em;">
                        {APP_NAME}
                    </div>
                    <div style="font-size:0.72rem;color:{COLORS['trajectory_accent']};font-weight:600;">
                        {TAGLINE}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<div style='font-size:0.72rem;font-weight:700;text-transform:uppercase;"
            f"letter-spacing:0.08em;color:{COLORS['secondary_text']};margin-bottom:0.45rem;'>Navigation</div>",
            unsafe_allow_html=True,
        )

        active_page = st.session_state.get("current_page", "Dashboard")
        for page_name in NAV_PAGES:
            btn_type = "primary" if page_name == active_page else "secondary"
            if st.button(page_name, type=btn_type, use_container_width=True, key=f"nav_{page_name}"):
                st.session_state["current_page"] = page_name
                st.rerun()

        st.markdown(f"<hr style='border-color:{COLORS['borders']};margin:1rem 0;' />", unsafe_allow_html=True)

        # Compact live session status panel
        res = st.session_state.get("processing_result")
        sel_id = st.session_state.get("selected_track_id")
        if res is not None:
            st.markdown(
                f"""
                <div style="background:{COLORS['card_bg']};border:1px solid {COLORS['borders']};
                            border-radius:10px;padding:0.7rem 0.85rem;font-size:0.8rem;">
                    <div style="color:{COLORS['light_indigo']};font-weight:700;margin-bottom:0.25rem;">Active Session</div>
                    <div style="color:{COLORS['secondary_text']};">Video: <span style="color:{COLORS['main_text']};">{res.video_meta.file_name}</span></div>
                    <div style="color:{COLORS['secondary_text']};">Track IDs: <span style="color:{COLORS['main_text']};">{len(res.unique_track_ids)}</span></div>
                    <div style="color:{COLORS['secondary_text']};">Target ID: <span style="color:{COLORS['trajectory_accent']};font-weight:700;">{f"#{sel_id}" if sel_id is not None else "None"}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:0.55rem;'></div>", unsafe_allow_html=True)

        if st.button("Reset Full Session", type="secondary", use_container_width=True, key="sidebar_reset_all"):
            reset_tracking_session(keep_dataset_history=False)
            st.session_state["current_page"] = "Dashboard"
            st.rerun()

        st.session_state["debug_mode"] = st.checkbox(
            "Developer Debug Mode",
            value=st.session_state.get("debug_mode", False),
            key="sidebar_debug_toggle",
        )


def main() -> None:
    _render_sidebar()
    active_page = st.session_state.get("current_page", "Dashboard")
    render_fn = PAGE_DISPATCH.get(active_page, dashboard.render)
    render_fn()


if __name__ == "__main__":
    main()
