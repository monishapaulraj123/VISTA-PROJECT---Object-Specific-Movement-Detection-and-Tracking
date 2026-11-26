"""
YOLO Object Detection Module.

Provides Ultralytics YOLO model loading, validation, custom weights support,
class mapping, and frame-level object detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

from config import MODELS_DIR


@dataclass
class Detection:
    """Single detected (and optionally tracked) object in a frame."""

    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    class_id: int
    class_name: str
    confidence: float
    track_id: Optional[int] = None


@dataclass
class ModelInfo:
    """Metadata describing the currently active YOLO model."""

    weights_name: str
    weights_path: str
    is_custom: bool
    class_names: Dict[int, str]


def _resolve_weights_location(weights_source: str | Path, is_custom: bool) -> Path | str:
    """Resolve pretrained weights into TrackNova/models/ or validate custom path."""
    if is_custom:
        custom_path = Path(weights_source)
        if not custom_path.exists() or not custom_path.is_file():
            raise FileNotFoundError(
                f"Custom model file not found: '{custom_path}'. Verify the path and try again."
            )
        if custom_path.suffix.lower() not in (".pt", ".onnx"):
            raise ValueError(
                f"Unsupported model extension '{custom_path.suffix}'. Provide a .pt or .onnx YOLO model."
            )
        return custom_path

    # Pretrained model: keep inside TrackNova/models/ if present, otherwise let Ultralytics fetch to models/
    local_target = MODELS_DIR / Path(weights_source).name
    if local_target.exists():
        return str(local_target)
    return str(local_target)


@st.cache_resource(show_spinner=False)
def load_yolo_model_cached(weights_source: str, is_custom: bool = False):
    """
    Load and cache an Ultralytics YOLO model across Streamlit reruns.

    Returns (model, ModelInfo).
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics package is not installed. Run `pip install ultralytics`."
        ) from exc

    resolved = _resolve_weights_location(weights_source, is_custom=is_custom)
    try:
        model = YOLO(str(resolved))
    except Exception as exc:
        if is_custom:
            raise RuntimeError(
                f"Failed to load custom YOLO weights '{weights_source}': {exc}"
            ) from exc
        raise RuntimeError(
            f"Unable to load or download YOLO model '{weights_source}'. "
            "Check your internet connection or place the .pt file in the TrackNova/models/ folder."
        ) from exc

    raw_names = getattr(model, "names", None) or {}
    if isinstance(raw_names, list):
        class_names = {idx: str(name) for idx, name in enumerate(raw_names)}
    else:
        class_names = {int(k): str(v) for k, v in raw_names.items()}

    info = ModelInfo(
        weights_name=Path(weights_source).name,
        weights_path=str(resolved),
        is_custom=is_custom,
        class_names=class_names,
    )
    return model, info


def save_uploaded_model(file_bytes: bytes, filename: str) -> Path:
    """Save a user-uploaded custom YOLO .pt model into TrackNova/models/."""
    if not file_bytes:
        raise ValueError("Uploaded model file is empty.")
    suffix = Path(filename).suffix.lower()
    if suffix != ".pt":
        raise ValueError("Only PyTorch YOLO '.pt' weight files are supported for upload.")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / Path(filename).name
    target.write_bytes(file_bytes)
    return target


def resolve_class_filter_ids(
    class_names_map: Dict[int, str], selected_classes: Optional[List[str]]
) -> Optional[List[int]]:
    """Map selected human-readable class names to integer YOLO class IDs."""
    if not selected_classes:
        return None
    normalized = {c.strip().lower() for c in selected_classes if c.strip()}
    ids = [cid for cid, cname in class_names_map.items() if cname.lower() in normalized]
    return ids if ids else None


def detect_objects_in_frame(
    model,
    frame: np.ndarray,
    conf_threshold: float = 0.35,
    class_filter_ids: Optional[List[int]] = None,
) -> List[Detection]:
    """Run pure YOLO object detection (without tracking) on a single BGR frame."""
    results = model.predict(
        source=frame,
        conf=float(conf_threshold),
        classes=class_filter_ids,
        verbose=False,
    )
    detections: List[Detection] = []
    if not results:
        return detections

    res = results[0]
    boxes = getattr(res, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return detections

    names = getattr(res, "names", None) or getattr(model, "names", {})
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)

    for box, conf, cls_id in zip(xyxy, confs, clss):
        cname = names.get(int(cls_id), f"class_{int(cls_id)}")
        detections.append(
            Detection(
                bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                class_id=int(cls_id),
                class_name=str(cname),
                confidence=float(conf),
                track_id=None,
            )
        )
    return detections
