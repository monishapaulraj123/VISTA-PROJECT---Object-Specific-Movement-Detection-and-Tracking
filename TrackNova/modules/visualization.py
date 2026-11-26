"""
Visualization Module (OpenCV Frame Overlays & Matplotlib Analytics Charts).

Draws bounding boxes, tracking IDs, object classes, confidence scores,
centre points, direction arrows, trajectory paths, and generates
dark-themed Matplotlib plots matching the TrackNova colour palette.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    BGR_COLORS,
    COLORS,
    GRAPHS_DIR,
    MAX_TRAJECTORY_POINTS,
    MUTED_TRACK_BGR_PALETTE,
    PROCESSED_VIDEOS_DIR,
)
from modules.detection import Detection
from modules.tracking import TrackingSessionResult
from modules.video_processor import (
    create_video_writer,
    iter_video_frames,
    preprocess_frame,
)
from utils.helpers import sanitize_filename, timestamp_slug

matplotlib.use("Agg")


def _muted_color_for_id(track_id: Optional[int]) -> Tuple[int, int, int]:
    """Return a consistent muted BGR colour for non-selected objects."""
    if track_id is None:
        return BGR_COLORS["secondary_text"]
    return MUTED_TRACK_BGR_PALETTE[int(track_id) % len(MUTED_TRACK_BGR_PALETTE)]


def _place_label_avoiding_overlap(
    x1: int,
    y1: int,
    y2: int,
    label_w: int,
    label_h: int,
    frame_w: int,
    frame_h: int,
    occupied_rects: List[Tuple[int, int, int, int]],
) -> Tuple[int, int]:
    """Choose a label top-left position that avoids overlapping existing labels."""
    candidates = [
        (x1, max(4, y1 - label_h - 6)),
        (x1, min(frame_h - label_h - 4, y2 + 4)),
        (x1, max(4, y1 + 4)),
        (x1, max(4, y1 - 2 * label_h - 10)),
    ]
    for cx, cy in candidates:
        cx = max(2, min(cx, max(2, frame_w - label_w - 4)))
        cy = max(2, min(cy, max(2, frame_h - label_h - 4)))
        rect = (cx, cy, cx + label_w, cy + label_h)
        overlaps = any(
            not (rect[2] < ox1 or rect[0] > ox2 or rect[3] < oy1 or rect[1] > oy2)
            for (ox1, oy1, ox2, oy2) in occupied_rects
        )
        if not overlaps:
            occupied_rects.append(rect)
            return cx, cy

    cx = max(2, min(x1, max(2, frame_w - label_w - 4)))
    cy = max(2, min(y1 - label_h - 6, max(2, frame_h - label_h - 4)))
    occupied_rects.append((cx, cy, cx + label_w, cy + label_h))
    return cx, cy


def _draw_label_pill(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    bg_bgr: Tuple[int, int, int],
    text_bgr: Tuple[int, int, int],
    font_scale: float = 0.48,
    thickness: int = 1,
) -> None:
    """Draw a rounded-style solid pill label with crisp readable text."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad_x, pad_y = 6, 4
    rx1, ry1 = max(0, x), max(0, y)
    rx2 = min(canvas.shape[1] - 1, rx1 + tw + 2 * pad_x)
    ry2 = min(canvas.shape[0] - 1, ry1 + th + baseline + 2 * pad_y)
    cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), bg_bgr, -1)
    cv2.putText(
        canvas,
        text,
        (rx1 + pad_x, ry1 + pad_y + th),
        font,
        font_scale,
        text_bgr,
        thickness,
        cv2.LINE_AA,
    )


def _draw_corner_brackets(
    canvas: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, int, int],
    thickness: int = 3,
) -> None:
    """Draw high-tech corner brackets around the selected object bounding box."""
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    arm = max(8, min(22, int(min(w, h) * 0.28)))
    # Top-left
    cv2.line(canvas, (x1, y1), (x1 + arm, y1), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x1, y1), (x1, y1 + arm), color, thickness, cv2.LINE_AA)
    # Top-right
    cv2.line(canvas, (x2, y1), (x2 - arm, y1), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x2, y1), (x2, y1 + arm), color, thickness, cv2.LINE_AA)
    # Bottom-left
    cv2.line(canvas, (x1, y2), (x1 + arm, y2), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x1, y2), (x1, y2 - arm), color, thickness, cv2.LINE_AA)
    # Bottom-right
    cv2.line(canvas, (x2, y2), (x2 - arm, y2), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x2, y2), (x2, y2 - arm), color, thickness, cv2.LINE_AA)


def draw_tracking_frame(
    frame: np.ndarray,
    detections: List[Detection],
    selected_track_id: Optional[int] = None,
    trajectory_points: Optional[List[Tuple[float, float]]] = None,
    show_confidence: bool = True,
    show_other_objects: bool = True,
    current_direction: Optional[str] = None,
    current_speed: Optional[float] = None,
    movement_vector: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Render detections, IDs, selected highlight, centre point, direction arrow,
    and trajectory path onto a copy of a BGR frame.
    """
    canvas = frame.copy()
    fh, fw = canvas.shape[:2]
    occupied_rects: List[Tuple[int, int, int, int]] = []
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 1. Draw selected-object trajectory path first so boxes sit cleanly above it
    if trajectory_points and len(trajectory_points) >= 1:
        pts_slice = trajectory_points[-MAX_TRAJECTORY_POINTS:]
        int_pts = [(int(round(px)), int(round(py))) for px, py in pts_slice]

        if len(int_pts) >= 2:
            poly = np.array(int_pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                canvas,
                [poly],
                isClosed=False,
                color=BGR_COLORS["trajectory_accent"],
                thickness=3,
                lineType=cv2.LINE_AA,
            )

        # Previous centre markers (sampled to avoid clutter)
        step = max(1, len(int_pts) // 35)
        for pt in int_pts[:-1:step]:
            cv2.circle(canvas, pt, 3, BGR_COLORS["light_indigo"], -1, cv2.LINE_AA)

        # Start point marker
        cv2.circle(canvas, int_pts[0], 6, BGR_COLORS["primary_indigo"], -1, cv2.LINE_AA)
        cv2.circle(canvas, int_pts[0], 6, BGR_COLORS["main_text"], 1, cv2.LINE_AA)

    # 2. Draw non-selected objects (dimmer context boxes)
    selected_det: Optional[Detection] = None
    for det in detections:
        if selected_track_id is not None and det.track_id == selected_track_id:
            selected_det = det
            continue
        if not show_other_objects:
            continue

        x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
        color = _muted_color_for_id(det.track_id)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

        id_part = f"ID {det.track_id}" if det.track_id is not None else "No ID"
        conf_part = f" {det.confidence:.2f}" if show_confidence else ""
        label = f"{id_part} | {det.class_name}{conf_part}"
        (tw, th), bl = cv2.getTextSize(label, font, 0.42, 1)
        lx, ly = _place_label_avoiding_overlap(
            x1, y1, y2, tw + 12, th + bl + 8, fw, fh, occupied_rects
        )
        _draw_label_pill(
            canvas,
            label,
            lx,
            ly,
            bg_bgr=(34, 26, 20),
            text_bgr=color,
            font_scale=0.42,
            thickness=1,
        )

    # 3. Draw selected object prominently
    if selected_det is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in selected_det.bbox]
        cx = int(round((selected_det.bbox[0] + selected_det.bbox[2]) / 2.0))
        cy = int(round((selected_det.bbox[1] + selected_det.bbox[3]) / 2.0))

        # Soft translucent highlight inside selected bbox
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), BGR_COLORS["primary_indigo"], -1)
        cv2.addWeighted(overlay, 0.14, canvas, 0.86, 0, canvas)

        # Prominent Light Indigo box + Turquoise corner reticle
        cv2.rectangle(canvas, (x1, y1), (x2, y2), BGR_COLORS["light_indigo"], 2, cv2.LINE_AA)
        _draw_corner_brackets(
            canvas, x1, y1, x2, y2, BGR_COLORS["trajectory_accent"], thickness=3
        )

        # Centre point
        cv2.circle(canvas, (cx, cy), 6, BGR_COLORS["trajectory_accent"], -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 9, BGR_COLORS["main_text"], 1, cv2.LINE_AA)

        # Direction arrow from movement vector
        if movement_vector is not None:
            dx, dy = movement_vector
            mag = math.hypot(dx, dy)
            if mag >= 1.5:
                scale = min(48.0, max(22.0, mag * 2.5)) / mag
                tip = (
                    int(round(cx + dx * scale)),
                    int(round(cy + dy * scale)),
                )
                cv2.arrowedLine(
                    canvas,
                    (cx, cy),
                    tip,
                    BGR_COLORS["trajectory_accent"],
                    2,
                    cv2.LINE_AA,
                    tipLength=0.35,
                )

        conf_part = f" | {selected_det.confidence:.2f}" if show_confidence else ""
        dir_part = f" | {current_direction}" if current_direction else ""
        spd_part = f" | {current_speed:.1f} px/s" if current_speed is not None else ""
        label = f"TARGET #{selected_det.track_id} ({selected_det.class_name}{conf_part}){dir_part}{spd_part}"

        (tw, th), bl = cv2.getTextSize(label, font, 0.48, 1)
        lx, ly = _place_label_avoiding_overlap(
            x1, y1, y2, tw + 12, th + bl + 8, fw, fh, occupied_rects
        )
        _draw_label_pill(
            canvas,
            label,
            lx,
            ly,
            bg_bgr=BGR_COLORS["primary_indigo"],
            text_bgr=BGR_COLORS["main_text"],
            font_scale=0.48,
            thickness=1,
        )
    elif selected_track_id is not None:
        # Selected object temporarily not detected in this frame
        notice = f"Selected ID #{selected_track_id} temporarily not detected"
        _draw_label_pill(
            canvas,
            notice,
            12,
            12,
            bg_bgr=(34, 21, 16),
            text_bgr=BGR_COLORS["warning"],
            font_scale=0.52,
            thickness=1,
        )

    return canvas


def render_annotated_video(
    session_result: TrackingSessionResult,
    selected_track_id: Optional[int] = None,
    movement_df: Optional[pd.DataFrame] = None,
    show_confidence: bool = True,
    show_other_objects: bool = True,
    target_width: Optional[int] = None,
    enhance_contrast: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[Optional[Path], Optional[str], Optional[str]]:
    """
    Re-render the processed video from source frames + stored tracking records,
    highlighting `selected_track_id` and its cumulative trajectory.

    Returns (output_path, codec_used, error_message).
    """
    meta = session_result.video_meta
    out_fps = (
        meta.fps / max(1, session_result.frame_stride)
        if meta.fps_is_valid and meta.fps > 0
        else 25.0
    )
    frame_size = (session_result.processed_width, session_result.processed_height)
    writer, out_path, codec = create_video_writer(
        base_name=meta.file_name,
        fps=out_fps,
        frame_size=frame_size,
        output_dir=PROCESSED_VIDEOS_DIR,
    )
    if writer is None or out_path is None:
        return (
            None,
            None,
            "Video writer initialization failed for all supported codecs (avc1, H264, mp4v, XVID, MJPG).",
        )

    records_by_frame = {rec.frame_idx: rec.detections for rec in session_result.frame_records}
    move_by_frame: Dict[int, Dict[str, object]] = {}
    if movement_df is not None and not movement_df.empty:
        for _, row in movement_df.iterrows():
            move_by_frame[int(row["frame_number"])] = row.to_dict()

    traj_accum: List[Tuple[float, float]] = []
    total_steps = max(1, len(session_result.frame_records))
    written = 0

    try:
        for idx, (frame_idx, _ts, raw_frame) in enumerate(
            iter_video_frames(
                meta.file_path,
                fps=meta.fps,
                frame_stride=session_result.frame_stride,
                max_frames=len(session_result.frame_records),
            )
        ):
            pre_frame = preprocess_frame(
                raw_frame, target_width=target_width, enhance_contrast=enhance_contrast
            )
            # Match writer dimensions strictly
            if pre_frame.shape[1] != frame_size[0] or pre_frame.shape[0] != frame_size[1]:
                pre_frame = cv2.resize(pre_frame, frame_size)

            dets = records_by_frame.get(frame_idx, [])
            m_info = move_by_frame.get(frame_idx)
            cur_dir: Optional[str] = None
            cur_spd: Optional[float] = None
            m_vec: Optional[Tuple[float, float]] = None

            if m_info is not None:
                traj_accum.append((float(m_info["center_x"]), float(m_info["center_y"])))
                cur_dir = str(m_info["direction"])
                spd_val = m_info["pixel_speed_px_s"]
                if spd_val is not None and not pd.isna(spd_val):
                    cur_spd = float(spd_val)
                m_vec = (float(m_info["dx"]), float(m_info["dy"]))

            annotated = draw_tracking_frame(
                pre_frame,
                detections=dets,
                selected_track_id=selected_track_id,
                trajectory_points=traj_accum if selected_track_id is not None else None,
                show_confidence=show_confidence,
                show_other_objects=show_other_objects,
                current_direction=cur_dir,
                current_speed=cur_spd,
                movement_vector=m_vec,
            )
            writer.write(annotated)
            written += 1
            if progress_callback is not None:
                progress_callback(idx + 1, total_steps)
    except Exception as exc:
        writer.release()
        return None, None, f"Failed while rendering annotated video: {exc}"
    finally:
        writer.release()

    if written == 0:
        return None, None, "No frames were written to the output video."
    return out_path, codec, None


# ---------------------------------------------------------------------------
# Matplotlib Dark-Themed Analytics Charts
# ---------------------------------------------------------------------------

def _apply_tracknova_axes_style(ax, title: str, xlabel: str, ylabel: str) -> None:
    """Apply consistent TrackNova dark navy styling to a Matplotlib Axes."""
    ax.set_facecolor(COLORS["card_bg"])
    ax.set_title(title, color=COLORS["main_text"], fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=COLORS["secondary_text"], fontsize=10, labelpad=8)
    ax.set_ylabel(ylabel, color=COLORS["secondary_text"], fontsize=10, labelpad=8)
    ax.tick_params(colors=COLORS["secondary_text"], labelsize=9)
    ax.grid(True, linestyle="--", linewidth=0.6, color=COLORS["borders"], alpha=0.8)
    for spine in ax.spines.values():
        spine.set_color(COLORS["borders"])


def _new_styled_figure(figsize: Tuple[float, float] = (8.0, 4.2)):
    fig, ax = plt.subplots(figsize=figsize, dpi=140)
    fig.patch.set_facecolor(COLORS["main_bg"])
    return fig, ax


def plot_x_vs_frame(movement_df: pd.DataFrame):
    """Graph 1: Horizontal Center X Position vs Frame Number."""
    fig, ax = _new_styled_figure()
    frames = movement_df["frame_number"]
    xs = movement_df["center_x"]
    ax.plot(frames, xs, color=COLORS["light_indigo"], linewidth=2.2, label="Center X (px)")
    ax.fill_between(frames, xs, xs.min(), color=COLORS["primary_indigo"], alpha=0.18)
    _apply_tracknova_axes_style(
        ax, "Horizontal Position (Center X) vs Frame Number", "Frame Number", "Center X (pixels)"
    )
    ax.legend(facecolor=COLORS["sidebar_bg"], edgecolor=COLORS["borders"], labelcolor=COLORS["main_text"])
    fig.tight_layout()
    return fig


def plot_y_vs_frame(movement_df: pd.DataFrame):
    """Graph 2: Vertical Center Y Position vs Frame Number."""
    fig, ax = _new_styled_figure()
    frames = movement_df["frame_number"]
    ys = movement_df["center_y"]
    ax.plot(frames, ys, color=COLORS["trajectory_accent"], linewidth=2.2, label="Center Y (px)")
    ax.fill_between(frames, ys, ys.min(), color=COLORS["trajectory_accent"], alpha=0.12)
    _apply_tracknova_axes_style(
        ax,
        "Vertical Position (Center Y) vs Frame Number (Image Y: Top=0)",
        "Frame Number",
        "Center Y (pixels)",
    )
    ax.legend(facecolor=COLORS["sidebar_bg"], edgecolor=COLORS["borders"], labelcolor=COLORS["main_text"])
    fig.tight_layout()
    return fig


def plot_speed_vs_frame(movement_df: pd.DataFrame):
    """Graph 3: Instantaneous Pixel Speed (px/s) vs Frame Number."""
    fig, ax = _new_styled_figure()
    valid = movement_df.dropna(subset=["pixel_speed_px_s"])
    if valid.empty:
        ax.text(
            0.5,
            0.5,
            "Pixel speed unavailable (invalid/zero video FPS or single observation)",
            ha="center",
            va="center",
            color=COLORS["warning"],
            transform=ax.transAxes,
        )
        _apply_tracknova_axes_style(
            ax, "Pixel Speed vs Frame Number", "Frame Number", "Speed (px/s)"
        )
        fig.tight_layout()
        return fig

    frames = valid["frame_number"]
    speeds = valid["pixel_speed_px_s"]
    mean_spd = float(speeds.mean())

    ax.plot(frames, speeds, color=COLORS["trajectory_accent"], linewidth=2.0, label="Instantaneous Speed")
    ax.axhline(
        mean_spd,
        color=COLORS["warning"],
        linestyle="--",
        linewidth=1.5,
        label=f"Mean Speed ({mean_spd:.1f} px/s)",
    )
    ax.fill_between(frames, speeds, 0, color=COLORS["trajectory_accent"], alpha=0.14)
    _apply_tracknova_axes_style(
        ax, "Selected Object Pixel Speed vs Frame Number", "Frame Number", "Pixel Speed (px/s)"
    )
    ax.legend(facecolor=COLORS["sidebar_bg"], edgecolor=COLORS["borders"], labelcolor=COLORS["main_text"])
    fig.tight_layout()
    return fig


def plot_xy_trajectory(
    movement_df: pd.DataFrame,
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None,
):
    """
    Graph 4: Complete 2D Image-Plane X/Y Trajectory.

    Y-axis is inverted so top-left (0, 0) matches OpenCV image coordinates.
    """
    fig, ax = _new_styled_figure(figsize=(8.0, 5.0))
    xs = movement_df["center_x"].to_numpy()
    ys = movement_df["center_y"].to_numpy()

    ax.plot(xs, ys, color=COLORS["trajectory_accent"], linewidth=2.4, label="Trajectory Path", zorder=2)
    ax.scatter(xs, ys, color=COLORS["light_indigo"], s=14, alpha=0.65, zorder=3)

    # Start & end markers
    ax.scatter(
        [xs[0]],
        [ys[0]],
        color=COLORS["primary_indigo"],
        edgecolors=COLORS["main_text"],
        s=95,
        zorder=5,
        label=f"Start (F{int(movement_df['frame_number'].iloc[0])})",
    )
    ax.scatter(
        [xs[-1]],
        [ys[-1]],
        color=COLORS["warning"],
        edgecolors=COLORS["main_text"],
        s=105,
        marker="D",
        zorder=5,
        label=f"End (F{int(movement_df['frame_number'].iloc[-1])})",
    )

    # Sample direction arrows along the trajectory
    if len(xs) >= 4:
        step = max(1, len(xs) // 8)
        for i in range(0, len(xs) - 1, step):
            dx = xs[i + 1] - xs[i]
            dy = ys[i + 1] - ys[i]
            if math.hypot(dx, dy) >= 2.0:
                ax.annotate(
                    "",
                    xy=(xs[i + 1], ys[i + 1]),
                    xytext=(xs[i], ys[i]),
                    arrowprops=dict(arrowstyle="->", color=COLORS["main_text"], lw=1.3),
                    zorder=4,
                )

    if frame_width and frame_height:
        ax.set_xlim(0, frame_width)
        ax.set_ylim(frame_height, 0)
    else:
        ax.invert_yaxis()

    ax.set_aspect("equal", adjustable="datalim")
    _apply_tracknova_axes_style(
        ax,
        "2D Image-Plane Trajectory (Origin Top-Left, Y Increases Downward)",
        "Center X (pixels)",
        "Center Y (pixels)",
    )
    ax.legend(facecolor=COLORS["sidebar_bg"], edgecolor=COLORS["borders"], labelcolor=COLORS["main_text"])
    fig.tight_layout()
    return fig


def plot_displacement_analysis(movement_df: pd.DataFrame):
    """Graph 5: Horizontal (DX), Vertical (DY), and Cumulative Total Displacement."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 5.8), dpi=140, sharex=True)
    fig.patch.set_facecolor(COLORS["main_bg"])

    frames = movement_df["frame_number"]
    ax1.plot(frames, movement_df["dx"], color=COLORS["light_indigo"], linewidth=1.8, label="DX (Horizontal, +Right)")
    ax1.plot(frames, movement_df["dy"], color=COLORS["warning"], linewidth=1.8, label="DY (Vertical, +Down)")
    ax1.axhline(0, color=COLORS["borders"], linewidth=1.0)
    _apply_tracknova_axes_style(
        ax1, "Step-wise X / Y Displacement vs Frame Number", "", "Displacement (px)"
    )
    ax1.legend(facecolor=COLORS["sidebar_bg"], edgecolor=COLORS["borders"], labelcolor=COLORS["main_text"])

    ax2.plot(
        frames,
        movement_df["cumulative_displacement_px"],
        color=COLORS["trajectory_accent"],
        linewidth=2.3,
        label="Cumulative Distance (px)",
    )
    ax2.fill_between(
        frames,
        movement_df["cumulative_displacement_px"],
        0,
        color=COLORS["trajectory_accent"],
        alpha=0.14,
    )
    _apply_tracknova_axes_style(
        ax2, "Cumulative Euclidean Displacement", "Frame Number", "Total Distance (px)"
    )
    ax2.legend(facecolor=COLORS["sidebar_bg"], edgecolor=COLORS["borders"], labelcolor=COLORS["main_text"])

    fig.tight_layout()
    return fig


def figure_to_png_bytes(fig, save_stem: Optional[str] = None) -> bytes:
    """Serialize a Matplotlib figure to PNG bytes and optionally save to outputs/graphs/."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    png_bytes = buf.getvalue()
    if save_stem:
        GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = GRAPHS_DIR / f"{sanitize_filename(save_stem)}_{timestamp_slug()}.png"
        out_path.write_bytes(png_bytes)
    return png_bytes
