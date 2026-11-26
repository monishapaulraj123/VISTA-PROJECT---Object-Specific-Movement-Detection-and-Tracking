"""
Observational / Proxy-Based Tracking Performance Evaluation Module.

Computes factual processing throughput, tracking gaps, visibility coverage,
and consistency observations without fabricating ground-truth metrics
(such as MOTA, MOTP, IDF1, or accuracy percentages).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from modules.tracking import TrackingSessionResult


def evaluate_tracking_performance(
    session_result: Optional[TrackingSessionResult],
    movement_df: Optional[pd.DataFrame],
    movement_summary: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Perform observational / proxy-based evaluation on the current tracking session.
    """
    if session_result is None:
        return None

    processed_frames = session_result.total_frames_processed
    proc_time = max(1e-6, session_result.processing_time_sec)
    proc_fps = processed_frames / proc_time

    # Overall detection and ID assignment observations
    all_confs: List[float] = []
    frames_with_any_det = 0
    for rec in session_result.frame_records:
        if rec.detections:
            frames_with_any_det += 1
            for d in rec.detections:
                all_confs.append(d.confidence)

    mean_conf = float(np.mean(all_confs)) if all_confs else 0.0
    id_assignment_rate = (
        ((session_result.total_detections - session_result.untracked_detections) / session_result.total_detections) * 100.0
        if session_result.total_detections > 0
        else 0.0
    )

    has_selected = bool(
        movement_df is not None
        and not movement_df.empty
        and movement_summary is not None
        and movement_summary.get("has_data")
    )

    selected_eval: Optional[Dict[str, Any]] = None
    observations: List[str] = []

    if session_result.total_detections == 0:
        observations.append(
            "No objects were detected under the current confidence threshold and class filter."
        )
    else:
        observations.append(
            f"Detected {session_result.total_detections} object instances across "
            f"{frames_with_any_det}/{processed_frames} processed frames, forming "
            f"{len(session_result.unique_track_ids)} unique ByteTrack IDs."
        )
        if session_result.untracked_detections > 0:
            observations.append(
                f"{session_result.untracked_detections} raw detections had no assigned tracking ID "
                "(expected during track initialization or low-overlap associations)."
            )

    if has_selected and movement_df is not None and movement_summary is not None:
        sel_id = int(movement_summary["selected_id"])
        tracked_count = int(movement_summary["frames_tracked"])
        first_f = int(movement_summary["first_frame"])
        last_f = int(movement_summary["last_frame"])
        stride = max(1, session_result.frame_stride)

        expected_span_steps = ((last_f - first_f) // stride) + 1 if last_f >= first_f else tracked_count
        span_visibility_pct = (tracked_count / max(1, expected_span_steps)) * 100.0
        overall_visibility_pct = (tracked_count / max(1, processed_frames)) * 100.0
        interruptions = int(movement_summary["tracking_interruptions"])
        gap_events = movement_summary.get("gap_events", [])
        max_gap = max((g["missing_processed_steps"] for g in gap_events), default=0)
        flagged_jumps = int(movement_summary.get("flagged_jump_count", 0))
        missing_steps_in_span = max(0, expected_span_steps - tracked_count)

        selected_eval = {
            "selected_id": sel_id,
            "object_class": movement_summary["object_class"],
            "frames_containing_selected": tracked_count,
            "first_frame": first_f,
            "last_frame": last_f,
            "expected_span_steps": expected_span_steps,
            "missing_steps_in_span": missing_steps_in_span,
            "span_visibility_pct": round(span_visibility_pct, 1),
            "overall_visibility_pct": round(overall_visibility_pct, 1),
            "tracking_duration_sec": movement_summary.get("tracking_duration_sec"),
            "tracking_interruptions": interruptions,
            "max_gap_steps": max_gap,
            "gap_events": gap_events,
            "flagged_jump_count": flagged_jumps,
            "movement_data_available": tracked_count >= 2,
        }

        if interruptions == 0:
            observations.append(
                f"Target ID #{sel_id} maintained continuous visibility across its entire active span "
                f"(frames {first_f}–{last_f}) with zero tracking interruptions."
            )
        else:
            observations.append(
                f"Target ID #{sel_id} experienced {interruptions} tracking interruption(s) "
                f"(longest gap: {max_gap} processed frame step(s)), yielding "
                f"{span_visibility_pct:.1f}% span visibility."
            )
        if flagged_jumps > 0:
            observations.append(
                f"Detected {flagged_jumps} large coordinate jump(s) immediately following a tracking gap; "
                "speed values for those transitions were excluded to prevent artificial spikes."
            )
    else:
        observations.append(
            "No tracking ID is currently selected, so target-specific gap and visibility analysis is pending."
        )

    limitations = [
        "Evaluation Type: Observational / Proxy-Based Evaluation (no ground-truth annotations loaded).",
        "Formal MOT benchmarks (MOTA, MOTP, IDF1, Precision, Recall, or Accuracy %) are NOT computed because ground-truth bounding-box labels are absent.",
        "Speed measurements are strictly in image-plane pixels/second (px/s); physical m/s conversion requires camera calibration or a known spatial reference scale.",
        "If an object becomes occluded and ByteTrack assigns a new ID upon re-appearance, proxy metrics reflect the single selected ID lifespan rather than cross-ID re-identification.",
    ]

    return {
        "evaluation_type": "Observational / Proxy-Based Evaluation (No Ground Truth)",
        "total_video_frames": session_result.video_meta.total_frames,
        "frames_processed": processed_frames,
        "processing_time_sec": session_result.processing_time_sec,
        "processing_fps": proc_fps,
        "total_detections": session_result.total_detections,
        "untracked_detections": session_result.untracked_detections,
        "id_assignment_rate_pct": round(id_assignment_rate, 1),
        "unique_track_ids": len(session_result.unique_track_ids),
        "mean_detection_confidence": round(mean_conf, 3),
        "frames_with_detections": frames_with_any_det,
        "selected_eval": selected_eval,
        "observations": observations,
        "limitations": limitations,
    }
