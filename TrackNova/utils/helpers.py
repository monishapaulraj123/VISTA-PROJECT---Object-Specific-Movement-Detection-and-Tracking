"""
TrackNova Helper Utilities.

Shared formatting, sanitization, session-state initialization,
file management, and Lucide-style inline SVG icon helpers.
"""

from __future__ import annotations

import base64
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from config import COLORS, LOGO_PATH


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as MM:SS.s or HH:MM:SS."""
    if seconds is None or seconds < 0:
        return "00:00.0"
    if seconds >= 3600:
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:04.1f}"


def format_file_size(num_bytes: int) -> str:
    """Convert byte count to a readable string."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 ** 2:
        return f"{num_bytes / 1024:.1f} KB"
    if num_bytes < 1024 ** 3:
        return f"{num_bytes / (1024 ** 2):.2f} MB"
    return f"{num_bytes / (1024 ** 3):.2f} GB"


def sanitize_filename(name: str) -> str:
    """Return a filesystem-safe filename stem."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return cleaned.strip("._") or "tracknova_output"


def timestamp_slug() -> str:
    """Return a sortable timestamp string for exported files."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_logo_base64(path: Path = LOGO_PATH) -> str:
    """Read the TrackNova PNG logo as base64 for HTML embedding."""
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def lucide_svg(name: str, size: int = 18, color: str = COLORS["light_indigo"]) -> str:
    """Return a crisp Lucide-style inline SVG icon."""
    icons = {
        "dashboard": '<rect width="7" height="9" x="3" y="3" rx="1"/><rect width="7" height="5" x="14" y="3" rx="1"/><rect width="7" height="9" x="14" y="12" rx="1"/><rect width="7" height="5" x="3" y="16" rx="1"/>',
        "video": '<path d="m16 13 5.223 3.482a.5.5 0 0 0 .777-.416V7.87a.5.5 0 0 0-.752-.432L16 10.5"/><rect x="2" y="6" width="14" height="12" rx="2"/>',
        "crosshair": '<circle cx="12" cy="12" r="10"/><line x1="22" x2="18" y1="12" y2="12"/><line x1="6" x2="2" y1="12" y2="12"/><line x1="12" x2="12" y1="6" y2="2"/><line x1="12" x2="12" y1="22" y2="18"/>',
        "activity": '<path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/>',
        "gauge": '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
        "file-text": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/>',
        "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
        "upload": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/>',
        "play": '<polygon points="6 3 20 12 6 21 6 3"/>',
        "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    }
    body = icons.get(name, icons["crosshair"])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;">'
        f"{body}</svg>"
    )


def init_session_state() -> None:
    """Initialize all TrackNova session state keys safely."""
    defaults: dict[str, Any] = {
        "current_page": "Dashboard",
        "debug_mode": False,
        "stop_requested": False,
        "video_source_type": None,
        "active_video_path": None,
        "active_video_name": None,
        "video_source_id": None,
        "video_path": None,
        "video_meta": None,
        "preprocessed_preview": None,
        "_prev_dataset_choice": "-- Select a dataset video --",
        "_prev_upload_id": None,
        "_uploader_nonce": 0,
        "_pending_clear_dataset_select": False,
        "processing_status": "Idle",
        "processing_error": None,
        "processing_result": None,
        "selected_track_id": None,
        "movement_df": None,
        "movement_summary": None,
        "trajectory_points": [],
        "evaluation_result": None,
        "dataset_sessions": [],
        "processed_video_path": None,
        "re_rendered_video_path": None,
        "last_settings": {},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_tracking_session(keep_dataset_history: bool = True) -> None:
    """Clear current video processing results while optionally keeping dataset history."""
    st.session_state["stop_requested"] = False
    st.session_state["processing_status"] = "Idle"
    st.session_state["processing_error"] = None
    st.session_state["preprocessed_preview"] = None
    st.session_state["processing_result"] = None
    st.session_state["selected_track_id"] = None
    st.session_state["movement_df"] = None
    st.session_state["movement_summary"] = None
    st.session_state["trajectory_points"] = []
    st.session_state["evaluation_result"] = None
    st.session_state["processed_video_path"] = None
    st.session_state["re_rendered_video_path"] = None
    st.session_state["_live_partial"] = None
    if not keep_dataset_history:
        st.session_state["video_source_type"] = None
        st.session_state["active_video_path"] = None
        st.session_state["active_video_name"] = None
        st.session_state["video_source_id"] = None
        st.session_state["video_path"] = None
        st.session_state["video_meta"] = None
        st.session_state["_prev_upload_id"] = None
        st.session_state["_prev_dataset_choice"] = "-- Select a dataset video --"
        st.session_state["_pending_clear_dataset_select"] = True
        st.session_state["_uploader_nonce"] = int(st.session_state.get("_uploader_nonce", 0)) + 1
        st.session_state["dataset_sessions"] = []
