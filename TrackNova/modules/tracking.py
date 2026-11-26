"""
ByteTrack Multi-Object Tracking Module.

Integrates Ultralytics YOLO tracking with ByteTrack persistence across
consecutive video frames and aggregates per-ID tracking summaries.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from config import DEFAULT_IOU_THRESHOLD, DEFAULT_TRACKER
from modules.detection import Detection, ModelInfo
from modules.video_processor import VideoMetadata


@dataclass
class FrameTrackingRecord:
    """Compact tracking output for a single processed frame."""

    frame_idx: int
    timestamp_sec: Optional[float]
    detections: List[Detection] = field(default_factory=list)


@dataclass
class TrackSummary:
    """Aggregated lifecycle statistics for one unique tracking ID."""

    track_id: int
    class_name: str
    first_frame: int
    last_frame: int
    frames_tracked: int
    avg_confidence: float
    max_confidence: float


@dataclass
class TrackingSessionResult:
    """Complete output of a TrackNova video processing run."""

    video_meta: VideoMetadata
    model_info: ModelInfo
    processed_width: int
    processed_height: int
    frame_stride: int
    confidence_threshold: float
    class_filter: List[str]
    frame_records: List[FrameTrackingRecord]
    track_summaries: Dict[int, TrackSummary]
    total_frames_processed: int
    processing_time_sec: float
    processing_fps: float
    total_detections: int
    untracked_detections: int
    unique_track_ids: List[int]
    class_counts: Dict[str, int]
    stopped_early: bool = False


class ByteTrackProcessor:
    """Stateful Ultralytics + ByteTrack tracker for sequential video frames."""

    def __init__(
        self,
        model,
        conf_threshold: float = 0.35,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        class_filter_ids: Optional[List[int]] = None,
        tracker_cfg: str = DEFAULT_TRACKER,
    ) -> None:
        self.model = model
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.class_filter_ids = class_filter_ids
        self.tracker_cfg = tracker_cfg
        self._reset_predictor_tracker()

    def _reset_predictor_tracker(self) -> None:
        """Reset Ultralytics predictor & tracker state cleanly before a new video."""
        if hasattr(self.model, "predictor"):
            self.model.predictor = None

    def track_frame(self, frame: np.ndarray) -> List[Detection]:
        """Run YOLO detection + ByteTrack association on the next sequential frame."""
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_cfg,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.class_filter_ids,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        res = results[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections

        names = getattr(res, "names", None) or getattr(self.model, "names", {})
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        raw_ids = getattr(boxes, "id", None)
        if raw_ids is not None:
            ids_list: List[Optional[int]] = [int(v) for v in raw_ids.cpu().numpy().tolist()]
        else:
            ids_list = [None] * len(xyxy)

        for box, conf, cls_id, tid in zip(xyxy, confs, clss, ids_list):
            cname = names.get(int(cls_id), f"class_{int(cls_id)}")
            detections.append(
                Detection(
                    bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    class_id=int(cls_id),
                    class_name=str(cname),
                    confidence=float(conf),
                    track_id=tid,
                )
            )

        return detections


def summarize_tracks(frame_records: List[FrameTrackingRecord]) -> Dict[int, TrackSummary]:
    """Build per-ID tracking summaries from actual frame records."""
    grouped: Dict[int, List[tuple[int, Detection]]] = defaultdict(list)

    for rec in frame_records:
        for det in rec.detections:
            if det.track_id is not None:
                grouped[det.track_id].append((rec.frame_idx, det))

    summaries: Dict[int, TrackSummary] = {}
    for tid in sorted(grouped.keys()):
        items = grouped[tid]
        frames = [f for f, _ in items]
        confs = [d.confidence for _, d in items]
        # Majority vote for class label if classifier ever flickered
        class_counter = Counter(d.class_name for _, d in items)
        dominant_class = class_counter.most_common(1)[0][0]
        unique_frames = len(set(frames))

        summaries[tid] = TrackSummary(
            track_id=tid,
            class_name=dominant_class,
            first_frame=min(frames),
            last_frame=max(frames),
            frames_tracked=unique_frames,
            avg_confidence=float(np.mean(confs)),
            max_confidence=float(np.max(confs)),
        )

    return summaries
