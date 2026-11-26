"""
Video Input, Frame Extraction, Preprocessing, and Video Writing Module.

Uses OpenCV VideoCapture/VideoWriter to stream video frames without
loading entire video datasets into memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

from config import PROCESSED_VIDEOS_DIR, SUPPORTED_VIDEO_EXTENSIONS
from utils.helpers import sanitize_filename, timestamp_slug


@dataclass
class VideoMetadata:
    """Metadata extracted from an input video file."""

    file_name: str
    file_path: str
    width: int
    height: int
    fps: float
    fps_is_valid: bool
    total_frames: int
    duration_sec: float
    file_size_bytes: int
    fourcc_str: str


def save_uploaded_video(file_bytes: bytes, original_name: str, dest_dir: Path) -> Path:
    """Persist an uploaded video stream to disk cleanly."""
    if not file_bytes:
        raise ValueError("Uploaded video file is empty (0 bytes).")

    ext = Path(original_name).suffix.lower().lstrip(".")
    if ext not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported video format '.{ext}'. Supported formats: {', '.join(SUPPORTED_VIDEO_EXTENSIONS)}."
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    stem = sanitize_filename(Path(original_name).stem)
    target = dest_dir / f"{stem}.{ext}"
    if target.exists():
        # Avoid overwriting an identically named dataset file with different bytes
        if target.stat().st_size != len(file_bytes):
            target = dest_dir / f"{stem}_{timestamp_slug()}.{ext}"
    target.write_bytes(file_bytes)
    return target


def validate_and_inspect_video(video_path: str | Path) -> VideoMetadata:
    """Open a video file with OpenCV, verify readability, and extract metadata."""
    path = Path(video_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path.name}")

    file_size = path.stat().st_size
    if file_size == 0:
        raise ValueError(f"Video file '{path.name}' is empty (0 bytes).")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(
            f"Unable to decode '{path.name}'. The file may be corrupted or use an unsupported codec."
        )

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        fourcc_str = "".join(
            chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)
        ).strip("\x00") or "N/A"

        ok, first_frame = cap.read()
        if not ok or first_frame is None:
            raise ValueError(
                f"Video '{path.name}' contains no readable frames."
            )

        if width <= 0 or height <= 0:
            height, width = first_frame.shape[:2]

        # Some containers report total_frames=0 in header; count safely if needed
        if total_frames <= 0:
            count = 1
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                count += 1
            total_frames = count

        fps_is_valid = raw_fps > 0.0 and not np.isnan(raw_fps) and not np.isinf(raw_fps)
        fps = raw_fps if fps_is_valid else 0.0
        duration_sec = (total_frames / fps) if fps_is_valid and total_frames > 0 else 0.0

        return VideoMetadata(
            file_name=path.name,
            file_path=str(path),
            width=width,
            height=height,
            fps=fps,
            fps_is_valid=fps_is_valid,
            total_frames=total_frames,
            duration_sec=duration_sec,
            file_size_bytes=file_size,
            fourcc_str=fourcc_str,
        )
    finally:
        cap.release()


def preprocess_frame(
    frame: np.ndarray,
    target_width: Optional[int] = None,
    enhance_contrast: bool = False,
) -> np.ndarray:
    """
    Preprocess a BGR frame before object detection and tracking.

    - Optional aspect-ratio-preserving resize to target_width.
    - Optional CLAHE luminance contrast enhancement for low-light footage.
    """
    if frame is None or frame.size == 0:
        raise ValueError("Encountered an empty frame during preprocessing.")

    processed = frame
    h, w = processed.shape[:2]

    if target_width is not None and target_width > 0 and w != target_width:
        scale = target_width / float(w)
        new_h = max(2, int(round(h * scale)))
        # Ensure even dimensions for downstream video codecs
        new_w = target_width if target_width % 2 == 0 else target_width - 1
        new_h = new_h if new_h % 2 == 0 else new_h - 1
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        processed = cv2.resize(processed, (new_w, new_h), interpolation=interp)

    if enhance_contrast:
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_chan)
        merged = cv2.merge((l_eq, a_chan, b_chan))
        processed = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    return processed


def iter_video_frames(
    video_path: str | Path,
    fps: float = 0.0,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
) -> Generator[Tuple[int, Optional[float], np.ndarray], None, None]:
    """
    Stream frames sequentially from disk.

    Yields (frame_index, timestamp_seconds_or_None, bgr_frame).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video stream: {video_path}")

    stride = max(1, int(frame_stride))
    frame_idx = 0
    yielded = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % stride == 0:
                ts: Optional[float] = None
                if fps > 0.0:
                    ts = frame_idx / fps
                else:
                    pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                    if pos_msec is not None and pos_msec >= 0.0:
                        ts = pos_msec / 1000.0
                yield frame_idx, ts, frame
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break

            frame_idx += 1
    finally:
        cap.release()


def read_single_frame(video_path: str | Path, frame_idx: int = 0) -> Optional[np.ndarray]:
    """Read a specific frame from a video for previewing."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        if frame_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        return frame if ret else None
    finally:
        cap.release()


def create_video_writer(
    base_name: str,
    fps: float,
    frame_size: Tuple[int, int],
    output_dir: Path = PROCESSED_VIDEOS_DIR,
) -> Tuple[Optional[cv2.VideoWriter], Optional[Path], Optional[str]]:
    """
    Initialize an OpenCV VideoWriter trying compatible codecs in order.

    Returns (writer, output_path, codec_used). If all codecs fail, returns (None, None, None).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = sanitize_filename(Path(base_name).stem)
    slug = timestamp_slug()
    safe_fps = fps if fps and fps > 0.0 else 25.0

    # Ensure even width/height for H.264/MP4 compatibility
    w, h = frame_size
    w = w if w % 2 == 0 else w - 1
    h = h if h % 2 == 0 else h - 1
    size = (max(2, w), max(2, h))

    codec_candidates = [
        ("avc1", "mp4"),
        ("H264", "mp4"),
        ("mp4v", "mp4"),
        ("XVID", "avi"),
        ("MJPG", "avi"),
    ]

    for codec, ext in codec_candidates:
        out_path = output_dir / f"{safe_stem}_tracked_{slug}.{ext}"
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(out_path), fourcc, safe_fps, size)
        if writer is not None and writer.isOpened():
            return writer, out_path, codec
        if writer is not None:
            writer.release()
        if out_path.exists():
            try:
                out_path.unlink()
            except OSError:
                pass

    return None, None, None
