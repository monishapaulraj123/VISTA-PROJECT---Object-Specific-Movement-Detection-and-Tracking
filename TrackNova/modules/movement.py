"""
Selected Object Movement Calculation Module.

Computes centre coordinates, X/Y displacement, Euclidean distance,
cumulative displacement, movement angle, movement direction labels,
and pixel-based speed (px/s) for a user-selected tracking ID.

Coordinate Convention (Image Plane):
- Origin (0, 0) is at the top-left of the video frame.
- Positive X = Right, Negative X = Left.
- Positive Y = Down, Negative Y = Up.

Note: Movement direction represents displacement in the 2D image plane,
not the physical 3D body orientation of the object.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import DEFAULT_MOVEMENT_THRESHOLD_PX, MAX_PLAUSIBLE_JUMP_FACTOR
from modules.tracking import FrameTrackingRecord, TrackingSessionResult


def estimate_direction_from_displacement(
    dx: float,
    dy: float,
    threshold_px: float = DEFAULT_MOVEMENT_THRESHOLD_PX,
) -> Tuple[str, Optional[float]]:
    """
    Estimate 2D image-plane movement direction from (DX, DY).

    Uses image coordinate convention:
    - +X = Right, -X = Left
    - +Y = Down,  -Y = Up

    Returns (direction_label, angle_degrees_or_None).
    Angle is measured in degrees in image-plane coordinates where
    0 deg = Right, -90 deg = Up, +90 deg = Down, +/-180 deg = Left.
    """
    dist = math.hypot(dx, dy)
    if dist < threshold_px:
        return "Stationary", None

    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)

    # 8 directional sectors of 45 degrees each around image-plane axes
    if -22.5 <= angle_deg < 22.5:
        label = "Right"
    elif 22.5 <= angle_deg < 67.5:
        label = "Down-Right"
    elif 67.5 <= angle_deg < 112.5:
        label = "Down"
    elif 112.5 <= angle_deg < 157.5:
        label = "Down-Left"
    elif angle_deg >= 157.5 or angle_deg < -157.5:
        label = "Left"
    elif -157.5 <= angle_deg < -112.5:
        label = "Up-Left"
    elif -112.5 <= angle_deg < -67.5:
        label = "Up"
    else:
        label = "Up-Right"

    return label, angle_deg


def calculate_selected_object_movement(
    session_result: TrackingSessionResult,
    selected_track_id: int,
    movement_threshold_px: float = DEFAULT_MOVEMENT_THRESHOLD_PX,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Calculate frame-by-frame movement metrics for `selected_track_id`.

    Safely handles missing frames, zero FPS, invalid timestamps, and
    abnormally large coordinate jumps caused by tracking interruptions.
    """
    fps = session_result.video_meta.fps
    fps_valid = session_result.video_meta.fps_is_valid and fps > 0.0
    frame_stride = max(1, session_result.frame_stride)

    rows: List[Dict[str, Any]] = []
    prev_obs: Optional[Dict[str, Any]] = None
    cumulative_disp = 0.0
    interruption_count = 0
    gap_events: List[Dict[str, int]] = []
    flagged_jump_count = 0

    for rec in session_result.frame_records:
        # Find detection matching selected_track_id in this frame
        matched = [d for d in rec.detections if d.track_id == selected_track_id]
        if not matched:
            continue

        # If duplicate ID ever appeared in same frame, take highest confidence
        det = max(matched, key=lambda d: d.confidence)
        x1, y1, x2, y2 = det.bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bbox_diag = math.hypot(max(1.0, x2 - x1), max(1.0, y2 - y1))

        # Timestamp resolution
        ts: Optional[float] = rec.timestamp_sec
        if ts is None and fps_valid:
            ts = rec.frame_idx / fps

        if prev_obs is None:
            dx = 0.0
            dy = 0.0
            step_dist = 0.0
            dt_sec: Optional[float] = 0.0 if fps_valid or ts is not None else None
            pixel_speed: Optional[float] = 0.0 if fps_valid else None
            direction = "Stationary"
            angle_deg: Optional[float] = None
            frame_gap = 0
            is_gap_jump = False
        else:
            frame_diff = rec.frame_idx - prev_obs["frame_idx"]
            expected_step = frame_stride
            frame_gap = max(0, frame_diff - expected_step)
            if frame_gap > 0:
                interruption_count += 1
                gap_events.append(
                    {
                        "from_frame": int(prev_obs["frame_idx"]),
                        "to_frame": int(rec.frame_idx),
                        "missing_processed_steps": int(round(frame_gap / expected_step)),
                        "frame_delta": int(frame_diff),
                    }
                )

            dx = cx - prev_obs["center_x"]
            dy = cy - prev_obs["center_y"]
            step_dist = math.hypot(dx, dy)

            # Time interval calculation using FPS or valid timestamps
            dt_sec = None
            if fps_valid and frame_diff > 0:
                dt_sec = frame_diff / fps
            elif ts is not None and prev_obs["timestamp_sec"] is not None:
                delta_t = ts - prev_obs["timestamp_sec"]
                if delta_t > 0:
                    dt_sec = delta_t

            # Check for implausible jump across a tracking gap to prevent misleading speed spikes
            is_gap_jump = (
                frame_gap > 0
                and step_dist > (MAX_PLAUSIBLE_JUMP_FACTOR * max(bbox_diag, prev_obs["bbox_diag"]))
            )
            if is_gap_jump:
                flagged_jump_count += 1

            if dt_sec is not None and dt_sec > 0.0 and not is_gap_jump:
                pixel_speed = step_dist / dt_sec
            else:
                pixel_speed = None

            direction, angle_deg = estimate_direction_from_displacement(
                dx, dy, threshold_px=movement_threshold_px
            )
            cumulative_disp += step_dist

        row = {
            "frame_number": int(rec.frame_idx),
            "timestamp_sec": round(ts, 4) if ts is not None else np.nan,
            "tracking_id": int(selected_track_id),
            "object_class": det.class_name,
            "confidence": round(det.confidence, 4),
            "x1": round(x1, 2),
            "y1": round(y1, 2),
            "x2": round(x2, 2),
            "y2": round(y2, 2),
            "center_x": round(cx, 2),
            "center_y": round(cy, 2),
            "dx": round(dx, 2),
            "dy": round(dy, 2),
            "displacement_px": round(step_dist, 2),
            "cumulative_displacement_px": round(cumulative_disp, 2),
            "time_interval_sec": round(dt_sec, 4) if dt_sec is not None else np.nan,
            "pixel_speed_px_s": round(pixel_speed, 2) if pixel_speed is not None else np.nan,
            "direction": direction,
            "movement_angle_deg": round(angle_deg, 2) if angle_deg is not None else np.nan,
            "frame_gap": int(frame_gap),
            "is_gap_jump": bool(is_gap_jump),
        }
        rows.append(row)

        prev_obs = {
            "frame_idx": rec.frame_idx,
            "timestamp_sec": ts,
            "center_x": cx,
            "center_y": cy,
            "bbox_diag": bbox_diag,
        }

    df = pd.DataFrame(rows)
    if df.empty:
        return df, {
            "selected_id": selected_track_id,
            "frames_tracked": 0,
            "has_data": False,
        }

    valid_speeds = df["pixel_speed_px_s"].dropna()
    # Exclude initial zero-speed row when computing active average if multiple rows exist
    active_speeds = valid_speeds.iloc[1:] if len(valid_speeds) > 1 else valid_speeds

    first_frame = int(df["frame_number"].iloc[0])
    last_frame = int(df["frame_number"].iloc[-1])
    if fps_valid:
        tracking_duration_sec = (last_frame - first_frame) / fps
    elif df["timestamp_sec"].notna().sum() >= 2:
        tracking_duration_sec = float(df["timestamp_sec"].iloc[-1] - df["timestamp_sec"].iloc[0])
    else:
        tracking_duration_sec = None

    net_dx = float(df["center_x"].iloc[-1] - df["center_x"].iloc[0])
    net_dy = float(df["center_y"].iloc[-1] - df["center_y"].iloc[0])
    net_displacement = math.hypot(net_dx, net_dy)

    non_stationary_dirs = df.loc[df["direction"] != "Stationary", "direction"]
    latest_direction = str(df["direction"].iloc[-1])
    dominant_direction = (
        str(non_stationary_dirs.mode().iloc[0]) if not non_stationary_dirs.empty else "Stationary"
    )

    summary = {
        "has_data": True,
        "selected_id": int(selected_track_id),
        "object_class": str(df["object_class"].mode().iloc[0]),
        "latest_confidence": float(df["confidence"].iloc[-1]),
        "avg_confidence": float(df["confidence"].mean()),
        "frames_tracked": int(len(df)),
        "first_frame": first_frame,
        "last_frame": last_frame,
        "tracking_duration_sec": tracking_duration_sec,
        "current_center_x": float(df["center_x"].iloc[-1]),
        "current_center_y": float(df["center_y"].iloc[-1]),
        "last_dx": float(df["dx"].iloc[-1]),
        "last_dy": float(df["dy"].iloc[-1]),
        "net_dx": net_dx,
        "net_dy": net_dy,
        "net_displacement_px": net_displacement,
        "total_displacement_px": float(df["cumulative_displacement_px"].iloc[-1]),
        "latest_speed_px_s": float(valid_speeds.iloc[-1]) if not valid_speeds.empty else None,
        "avg_speed_px_s": float(active_speeds.mean()) if not active_speeds.empty else None,
        "max_speed_px_s": float(valid_speeds.max()) if not valid_speeds.empty else None,
        "current_direction": latest_direction,
        "dominant_direction": dominant_direction,
        "tracking_interruptions": int(interruption_count),
        "gap_events": gap_events,
        "flagged_jump_count": int(flagged_jump_count),
        "fps_valid": fps_valid,
    }
    return df, summary
