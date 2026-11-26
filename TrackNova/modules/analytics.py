"""
Analytics & Multi-Dataset Comparison Module.

Aggregates real statistics for the Dashboard and maintains structured
comparison rows across multiple tested videos in the session.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from config import DEFAULT_MOVEMENT_THRESHOLD_PX
from modules.evaluation import evaluate_tracking_performance
from modules.movement import calculate_selected_object_movement
from modules.tracking import TrackingSessionResult
from modules.trajectory import build_trajectory_data
from utils.helpers import format_duration, reset_tracking_session


def compute_dashboard_metrics(
    session_result: Optional[TrackingSessionResult],
    movement_summary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute real overview metrics for the Dashboard from current session state."""
    if session_result is None:
        return {"has_session": False}

    meta = session_result.video_meta
    class_counts = session_result.class_counts
    most_frequent_class = (
        max(class_counts.items(), key=lambda kv: kv[1])[0] if class_counts else "None"
    )

    video_stats = {
        "file_name": meta.file_name,
        "duration_str": format_duration(meta.duration_sec) if meta.fps_is_valid else "N/A (0 FPS)",
        "fps": f"{meta.fps:.2f}" if meta.fps_is_valid else "Invalid / 0",
        "resolution": f"{meta.width} x {meta.height}",
        "total_frames": meta.total_frames,
        "processed_frames": session_result.total_frames_processed,
        "processing_status": "Stopped Early" if session_result.stopped_early else "Completed",
    }

    detection_stats = {
        "total_detections": session_result.total_detections,
        "detected_classes_count": len(class_counts),
        "unique_track_ids_count": len(session_result.unique_track_ids),
        "most_frequent_class": most_frequent_class,
        "class_counts": class_counts,
    }

    has_selected = bool(movement_summary and movement_summary.get("has_data"))
    selected_stats: Optional[Dict[str, Any]] = None
    if has_selected and movement_summary is not None:
        dur = movement_summary.get("tracking_duration_sec")
        avg_spd = movement_summary.get("avg_speed_px_s")
        max_spd = movement_summary.get("max_speed_px_s")
        selected_stats = {
            "selected_id": movement_summary["selected_id"],
            "object_class": movement_summary["object_class"],
            "frames_tracked": movement_summary["frames_tracked"],
            "tracking_duration_str": format_duration(dur) if dur is not None else "N/A",
            "total_displacement_px": movement_summary["total_displacement_px"],
            "avg_speed_str": f"{avg_spd:.2f} px/s" if avg_spd is not None else "N/A",
            "max_speed_str": f"{max_spd:.2f} px/s" if max_spd is not None else "N/A",
            "current_direction": movement_summary["current_direction"],
        }

    return {
        "has_session": True,
        "video_stats": video_stats,
        "detection_stats": detection_stats,
        "has_selected": has_selected,
        "selected_stats": selected_stats,
    }


def build_dataset_comparison_record(
    session_result: TrackingSessionResult,
    movement_summary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Create a comparison record for multi-dataset testing."""
    meta = session_result.video_meta
    has_sel = bool(movement_summary and movement_summary.get("has_data"))

    sel_label = (
        f"ID #{movement_summary['selected_id']} ({movement_summary['object_class']})"
        if has_sel and movement_summary is not None
        else "None selected"
    )
    avg_spd = (
        round(float(movement_summary["avg_speed_px_s"]), 2)
        if has_sel and movement_summary and movement_summary.get("avg_speed_px_s") is not None
        else None
    )
    max_spd = (
        round(float(movement_summary["max_speed_px_s"]), 2)
        if has_sel and movement_summary and movement_summary.get("max_speed_px_s") is not None
        else None
    )
    total_disp = (
        round(float(movement_summary["total_displacement_px"]), 2)
        if has_sel and movement_summary
        else None
    )
    track_dur = (
        round(float(movement_summary["tracking_duration_sec"]), 2)
        if has_sel and movement_summary and movement_summary.get("tracking_duration_sec") is not None
        else None
    )

    return {
        "Video Name": meta.file_name,
        "Duration (s)": round(meta.duration_sec, 2) if meta.fps_is_valid else None,
        "Resolution": f"{meta.width}x{meta.height}",
        "Video FPS": round(meta.fps, 2) if meta.fps_is_valid else None,
        "Model": session_result.model_info.weights_name,
        "Total Detections": session_result.total_detections,
        "Unique Track IDs": len(session_result.unique_track_ids),
        "Processing Time (s)": round(session_result.processing_time_sec, 2),
        "Processing FPS": round(session_result.processing_fps, 2),
        "Selected Object": sel_label,
        "Avg Pixel Speed (px/s)": avg_spd,
        "Max Pixel Speed (px/s)": max_spd,
        "Total Displacement (px)": total_disp,
        "Tracking Duration (s)": track_dur,
    }


def upsert_dataset_session(
    sessions: List[Dict[str, Any]],
    record: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Update existing row for the same video/model run or append a new entry."""
    updated = [
        r for r in sessions
        if not (r.get("Video Name") == record.get("Video Name") and r.get("Model") == record.get("Model"))
    ]
    updated.append(record)
    return updated


def sessions_to_dataframe(sessions: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert multi-dataset session records into a clean Pandas DataFrame."""
    if not sessions:
        return pd.DataFrame()
    return pd.DataFrame(sessions)


def activate_selected_track_id(
    session_result: TrackingSessionResult,
    track_id: Optional[int],
    movement_threshold_px: Optional[float] = None,
    user_cleared: bool = False,
) -> None:
    """Compute movement, trajectory, evaluation, and comparison updates for `track_id`."""
    if movement_threshold_px is None:
        movement_threshold_px = float(
            st.session_state.get("last_settings", {}).get(
                "movement_threshold_px", DEFAULT_MOVEMENT_THRESHOLD_PX
            )
        )

    st.session_state["selected_track_id"] = track_id
    st.session_state["_user_cleared_selection"] = bool(user_cleared and track_id is None)

    if track_id is None:
        st.session_state["movement_df"] = None
        st.session_state["movement_summary"] = None
        st.session_state["trajectory_points"] = []
        st.session_state["evaluation_result"] = evaluate_tracking_performance(
            session_result, None, None
        )
    else:
        mdf, msum = calculate_selected_object_movement(
            session_result, track_id, movement_threshold_px=movement_threshold_px
        )
        traj = build_trajectory_data(mdf)
        st.session_state["movement_df"] = mdf
        st.session_state["movement_summary"] = msum
        st.session_state["trajectory_points"] = traj.get("points", [])
        st.session_state["evaluation_result"] = evaluate_tracking_performance(
            session_result, mdf, msum
        )

    rec = build_dataset_comparison_record(
        session_result, st.session_state.get("movement_summary")
    )
    st.session_state["dataset_sessions"] = upsert_dataset_session(
        st.session_state.get("dataset_sessions", []), rec
    )


def sync_active_tracking_state() -> Tuple[Optional[TrackingSessionResult], Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """
    Ensure `processing_result`, `movement_df`, and `movement_summary` correspond
    to the current `active_video_path` and selected tracking ID.
    """
    session_result: Optional[TrackingSessionResult] = st.session_state.get("processing_result")
    active_path = st.session_state.get("active_video_path")

    if session_result is None:
        return None, None, None

    if active_path:
        try:
            if Path(session_result.video_meta.file_path).resolve() != Path(active_path).resolve():
                reset_tracking_session(keep_dataset_history=True)
                return None, None, None
        except OSError:
            pass

    if session_result.unique_track_ids:
        sel_id = st.session_state.get("selected_track_id")
        user_cleared = bool(st.session_state.get("_user_cleared_selection", False))

        if (sel_id is None or sel_id not in session_result.unique_track_ids) and not user_cleared:
            sel_id = max(
                session_result.track_summaries.values(), key=lambda s: s.frames_tracked
            ).track_id
            activate_selected_track_id(session_result, sel_id)

        sel_id = st.session_state.get("selected_track_id")
        if sel_id is not None and sel_id in session_result.unique_track_ids:
            mdf = st.session_state.get("movement_df")
            msum = st.session_state.get("movement_summary")
            needs_recompute = (
                mdf is None
                or mdf.empty
                or not msum
                or not msum.get("has_data")
                or int(msum.get("selected_id", -1)) != int(sel_id)
                or int(mdf["tracking_id"].iloc[0]) != int(sel_id)
            )
            if needs_recompute:
                activate_selected_track_id(session_result, sel_id)

    return (
        st.session_state.get("processing_result"),
        st.session_state.get("movement_df"),
        st.session_state.get("movement_summary"),
    )
