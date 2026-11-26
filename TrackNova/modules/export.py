"""
Reports & Export Module.

Provides CSV exports (frame-by-frame movement & multi-dataset comparison),
HTML summary report generation, and PDF summary report generation using ReportLab.
"""

from __future__ import annotations

import html
import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from config import (
    APP_NAME,
    CSV_DIR,
    FULL_TITLE,
    REPORTS_DIR,
    SUBTITLE,
    TAGLINE,
)
from modules.tracking import TrackingSessionResult
from utils.helpers import format_duration, sanitize_filename, timestamp_slug


def export_movement_csv(
    movement_df: pd.DataFrame,
    video_name: str,
    track_id: int,
) -> Tuple[bytes, Path]:
    """Export selected-object frame-by-frame movement data as CSV bytes and file."""
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    stem = sanitize_filename(Path(video_name).stem)
    out_path = CSV_DIR / f"TrackNova_{stem}_ID{track_id}_movement_{timestamp_slug()}.csv"
    csv_bytes = movement_df.to_csv(index=False).encode("utf-8")
    out_path.write_bytes(csv_bytes)
    return csv_bytes, out_path


def export_comparison_csv(comparison_df: pd.DataFrame) -> Tuple[bytes, Path]:
    """Export multi-dataset comparison table as CSV bytes and file."""
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CSV_DIR / f"TrackNova_dataset_comparison_{timestamp_slug()}.csv"
    csv_bytes = comparison_df.to_csv(index=False).encode("utf-8")
    out_path.write_bytes(csv_bytes)
    return csv_bytes, out_path


def _build_report_context(
    session_result: TrackingSessionResult,
    movement_summary: Optional[Dict[str, Any]],
    trajectory_data: Optional[Dict[str, Any]],
    evaluation_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble factual, non-fabricated report fields from active session data."""
    meta = session_result.video_meta
    has_sel = bool(movement_summary and movement_summary.get("has_data"))

    sel_dur = movement_summary.get("tracking_duration_sec") if has_sel and movement_summary else None
    avg_spd = movement_summary.get("avg_speed_px_s") if has_sel and movement_summary else None
    max_spd = movement_summary.get("max_speed_px_s") if has_sel and movement_summary else None

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "app_name": APP_NAME,
        "full_title": FULL_TITLE,
        "tagline": TAGLINE,
        "subtitle": SUBTITLE,
        "video_name": meta.file_name,
        "resolution": f"{meta.width} x {meta.height}",
        "video_fps": f"{meta.fps:.2f}" if meta.fps_is_valid else "Invalid / 0",
        "video_duration": format_duration(meta.duration_sec) if meta.fps_is_valid else "N/A",
        "total_frames": meta.total_frames,
        "processed_frames": session_result.total_frames_processed,
        "model_weights": session_result.model_info.weights_name,
        "model_type": "Custom Trained" if session_result.model_info.is_custom else "Pretrained COCO",
        "confidence_threshold": session_result.confidence_threshold,
        "processing_time_sec": f"{session_result.processing_time_sec:.2f} s",
        "processing_fps": f"{session_result.processing_fps:.2f} FPS",
        "total_detections": session_result.total_detections,
        "unique_track_ids": len(session_result.unique_track_ids),
        "has_selected": has_sel,
        "selected_id": movement_summary["selected_id"] if has_sel and movement_summary else "None",
        "object_class": movement_summary["object_class"] if has_sel and movement_summary else "N/A",
        "frames_tracked": movement_summary["frames_tracked"] if has_sel and movement_summary else 0,
        "tracking_duration": format_duration(sel_dur) if sel_dur is not None else "N/A",
        "total_displacement": (
            f"{movement_summary['total_displacement_px']:.2f} px"
            if has_sel and movement_summary
            else "N/A"
        ),
        "net_displacement": (
            f"{movement_summary['net_displacement_px']:.2f} px"
            if has_sel and movement_summary
            else "N/A"
        ),
        "avg_speed": f"{avg_spd:.2f} px/s" if avg_spd is not None else "N/A",
        "max_speed": f"{max_spd:.2f} px/s" if max_spd is not None else "N/A",
        "current_direction": movement_summary["current_direction"] if has_sel and movement_summary else "N/A",
        "dominant_direction": movement_summary["dominant_direction"] if has_sel and movement_summary else "N/A",
        "trajectory_points": trajectory_data.get("point_count", 0) if trajectory_data else 0,
        "linearity_ratio": (
            f"{trajectory_data['linearity_ratio']:.3f}"
            if trajectory_data and trajectory_data.get("linearity_ratio") is not None
            else "N/A"
        ),
        "interruptions": (
            movement_summary["tracking_interruptions"] if has_sel and movement_summary else "N/A"
        ),
        "evaluation_type": (
            evaluation_result["evaluation_type"]
            if evaluation_result
            else "Observational / Proxy-Based Evaluation (No Ground Truth)"
        ),
        "observations": evaluation_result.get("observations", []) if evaluation_result else [],
        "limitations": evaluation_result.get("limitations", []) if evaluation_result else [],
    }


def export_summary_html(
    session_result: TrackingSessionResult,
    movement_summary: Optional[Dict[str, Any]],
    trajectory_data: Optional[Dict[str, Any]],
    evaluation_result: Optional[Dict[str, Any]],
) -> Tuple[bytes, Path]:
    """Generate a standalone TrackNova HTML summary report."""
    ctx = _build_report_context(session_result, movement_summary, trajectory_data, evaluation_result)
    esc = lambda v: html.escape(str(v))

    obs_html = "".join(f"<li>{esc(o)}</li>" for o in ctx["observations"])
    lim_html = "".join(f"<li>{esc(l)}</li>" for l in ctx["limitations"])

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{esc(ctx['app_name'])} Analysis Report — {esc(ctx['video_name'])}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #101522; color: #E8ECF5; margin: 0; padding: 32px; }}
  .container {{ max-width: 920px; margin: 0 auto; background: #151C2E; border: 1px solid #303B55; border-radius: 12px; padding: 28px 34px; }}
  .header {{ border-bottom: 1px solid #303B55; padding-bottom: 16px; margin-bottom: 22px; }}
  h1 {{ margin: 0; color: #91A7E8; font-size: 26px; }}
  .subtitle {{ color: #48D6C2; font-size: 13px; margin-top: 4px; font-weight: 600; }}
  .meta {{ color: #9BA8C0; font-size: 12px; margin-top: 6px; }}
  h2 {{ color: #91A7E8; font-size: 16px; margin-top: 24px; border-left: 4px solid #4C64A5; padding-left: 10px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 10px; background: #1B2235; border-radius: 8px; overflow: hidden; }}
  th, td {{ text-align: left; padding: 10px 14px; border-bottom: 1px solid #303B55; font-size: 13px; }}
  th {{ color: #9BA8C0; width: 38%; font-weight: 600; }}
  td {{ color: #E8ECF5; }}
  ul {{ color: #E8ECF5; line-height: 1.6; font-size: 13px; }}
  .note {{ background: #1B2235; border: 1px solid #303B55; border-left: 4px solid #FBBF24; padding: 12px 14px; border-radius: 6px; font-size: 12.5px; color: #E8ECF5; margin-top: 14px; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>{esc(ctx['app_name'])} — {esc(ctx['full_title'])}</h1>
    <div class="subtitle">{esc(ctx['tagline'])} | {esc(ctx['subtitle'])}</div>
    <div class="meta">Generated: {esc(ctx['generated_at'])}</div>
  </div>

  <h2>1. Video &amp; Processing Summary</h2>
  <table>
    <tr><th>Video Filename</th><td>{esc(ctx['video_name'])}</td></tr>
    <tr><th>Resolution</th><td>{esc(ctx['resolution'])}</td></tr>
    <tr><th>Video Frame Rate</th><td>{esc(ctx['video_fps'])}</td></tr>
    <tr><th>Video Duration</th><td>{esc(ctx['video_duration'])}</td></tr>
    <tr><th>Frames Processed / Total</th><td>{esc(ctx['processed_frames'])} / {esc(ctx['total_frames'])}</td></tr>
    <tr><th>Detection Model</th><td>{esc(ctx['model_weights'])} ({esc(ctx['model_type'])})</td></tr>
    <tr><th>Confidence Threshold</th><td>{esc(ctx['confidence_threshold'])}</td></tr>
    <tr><th>Processing Time / Throughput</th><td>{esc(ctx['processing_time_sec'])} ({esc(ctx['processing_fps'])})</td></tr>
    <tr><th>Total Detections / Unique Track IDs</th><td>{esc(ctx['total_detections'])} / {esc(ctx['unique_track_ids'])}</td></tr>
  </table>

  <h2>2. Selected Object Movement &amp; Trajectory</h2>
  <table>
    <tr><th>Selected Tracking ID</th><td>{esc(ctx['selected_id'])}</td></tr>
    <tr><th>Object Class</th><td>{esc(ctx['object_class'])}</td></tr>
    <tr><th>Frames Tracked / Duration</th><td>{esc(ctx['frames_tracked'])} ({esc(ctx['tracking_duration'])})</td></tr>
    <tr><th>Total Path Displacement</th><td>{esc(ctx['total_displacement'])}</td></tr>
    <tr><th>Net Straight-Line Displacement</th><td>{esc(ctx['net_displacement'])}</td></tr>
    <tr><th>Average Pixel Speed</th><td>{esc(ctx['avg_speed'])}</td></tr>
    <tr><th>Maximum Pixel Speed</th><td>{esc(ctx['max_speed'])}</td></tr>
    <tr><th>Current / Dominant Movement Direction</th><td>{esc(ctx['current_direction'])} / {esc(ctx['dominant_direction'])}</td></tr>
    <tr><th>Trajectory Points / Linearity Ratio</th><td>{esc(ctx['trajectory_points'])} / {esc(ctx['linearity_ratio'])}</td></tr>
    <tr><th>Tracking Interruptions</th><td>{esc(ctx['interruptions'])}</td></tr>
  </table>

  <h2>3. Evaluation Observations ({esc(ctx['evaluation_type'])})</h2>
  <ul>{obs_html}</ul>

  <h2>4. Methodological Scope &amp; Limitations</h2>
  <div class="note">
    <ul>{lim_html}</ul>
  </div>
</div>
</body>
</html>
"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = sanitize_filename(Path(ctx["video_name"]).stem)
    out_path = REPORTS_DIR / f"TrackNova_{stem}_report_{timestamp_slug()}.html"
    report_bytes = html_doc.encode("utf-8")
    out_path.write_bytes(report_bytes)
    return report_bytes, out_path


def export_summary_pdf(
    session_result: TrackingSessionResult,
    movement_summary: Optional[Dict[str, Any]],
    trajectory_data: Optional[Dict[str, Any]],
    evaluation_result: Optional[Dict[str, Any]],
) -> Tuple[bytes, Path]:
    """Generate a formatted TrackNova PDF summary report using ReportLab."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ctx = _build_report_context(session_result, movement_summary, trajectory_data, evaluation_result)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40,
        title=f"TrackNova Report - {ctx['video_name']}",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TNTitle",
        parent=styles["Heading1"],
        fontSize=18,
        textColor=colors.HexColor("#151C2E"),
        spaceAfter=4,
    )
    sub_style = ParagraphStyle(
        "TNSub",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#4C64A5"),
        spaceAfter=12,
    )
    h2_style = ParagraphStyle(
        "TNH2",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=colors.HexColor("#4C64A5"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "TNBody",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#1F2937"),
    )

    story = [
        Paragraph(f"{ctx['app_name']} — {ctx['full_title']}", title_style),
        Paragraph(f"{ctx['tagline']} | Generated: {ctx['generated_at']}", sub_style),
        Paragraph("1. Video & Processing Details", h2_style),
    ]

    def _kv_table(rows):
        t = Table(rows, colWidths=[195, 315])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF2FF")),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#111827")),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        return t

    video_rows = [
        ["Video Filename", str(ctx["video_name"])],
        ["Resolution", str(ctx["resolution"])],
        ["Video FPS", str(ctx["video_fps"])],
        ["Video Duration", str(ctx["video_duration"])],
        ["Processed / Total Frames", f"{ctx['processed_frames']} / {ctx['total_frames']}"],
        ["YOLO Model", f"{ctx['model_weights']} ({ctx['model_type']})"],
        ["Confidence Threshold", str(ctx["confidence_threshold"])],
        ["Processing Time & FPS", f"{ctx['processing_time_sec']} ({ctx['processing_fps']})"],
        ["Detections / Unique Track IDs", f"{ctx['total_detections']} / {ctx['unique_track_ids']}"],
    ]
    story.append(_kv_table(video_rows))
    story.append(Spacer(1, 8))
    story.append(Paragraph("2. Selected Object Movement & Trajectory", h2_style))

    obj_rows = [
        ["Selected Tracking ID", str(ctx["selected_id"])],
        ["Object Class", str(ctx["object_class"])],
        ["Frames Tracked / Duration", f"{ctx['frames_tracked']} ({ctx['tracking_duration']})"],
        ["Total Path Displacement", str(ctx["total_displacement"])],
        ["Net Straight-Line Displacement", str(ctx["net_displacement"])],
        ["Average Pixel Speed", str(ctx["avg_speed"])],
        ["Maximum Pixel Speed", str(ctx["max_speed"])],
        ["Current / Dominant Direction", f"{ctx['current_direction']} / {ctx['dominant_direction']}"],
        ["Trajectory Points / Linearity", f"{ctx['trajectory_points']} / {ctx['linearity_ratio']}"],
        ["Tracking Interruptions", str(ctx["interruptions"])],
    ]
    story.append(_kv_table(obj_rows))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"3. {ctx['evaluation_type']}", h2_style))

    for obs in ctx["observations"]:
        story.append(Paragraph(f"• {obs}", body_style))

    story.append(Paragraph("4. Methodological Limitations", h2_style))
    for lim in ctx["limitations"]:
        story.append(Paragraph(f"• {lim}", body_style))

    doc.build(story)
    pdf_bytes = buf.getvalue()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = sanitize_filename(Path(ctx["video_name"]).stem)
    out_path = REPORTS_DIR / f"TrackNova_{stem}_report_{timestamp_slug()}.pdf"
    out_path.write_bytes(pdf_bytes)
    return pdf_bytes, out_path
