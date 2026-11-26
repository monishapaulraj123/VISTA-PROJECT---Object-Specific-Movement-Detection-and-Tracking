"""
TrackNova Configuration Module.

Centralizes brand identity, exact UI colour palette, directory paths,
YOLO/ByteTrack defaults, and movement analysis constants.
"""

from pathlib import Path
import os

# ---------------------------------------------------------------------------
# Project Identity
# ---------------------------------------------------------------------------
APP_NAME = "TrackNova"
FULL_TITLE = "Object-Specific Movement Detection and Tracking Using Video Datasets"
TAGLINE = "See Every Movement. Track Every Object."
SUBTITLE = "AI-Powered Object Tracking & Motion Intelligence"
APP_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Directory Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"
DATASET_VIDEOS_DIR = BASE_DIR / "dataset" / "videos"

OUTPUTS_DIR = BASE_DIR / "outputs"
PROCESSED_VIDEOS_DIR = OUTPUTS_DIR / "processed_videos"
GRAPHS_DIR = OUTPUTS_DIR / "graphs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
CSV_DIR = OUTPUTS_DIR / "csv"

# Ensure required runtime directories exist
for _dir in (
    MODELS_DIR,
    MODELS_DIR / ".ultralytics",
    ASSETS_DIR,
    DATASET_VIDEOS_DIR,
    PROCESSED_VIDEOS_DIR,
    GRAPHS_DIR,
    REPORTS_DIR,
    CSV_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)

# Keep Ultralytics settings self-contained inside TrackNova/models/.ultralytics
os.environ.setdefault("YOLO_CONFIG_DIR", str(MODELS_DIR / ".ultralytics"))

# ---------------------------------------------------------------------------
# Strict UI Colour Palette
# ---------------------------------------------------------------------------
COLORS = {
    "primary_indigo": "#4C64A5",
    "main_bg": "#101522",
    "sidebar_bg": "#151C2E",
    "card_bg": "#1B2235",
    "light_indigo": "#91A7E8",
    "trajectory_accent": "#48D6C2",
    "main_text": "#E8ECF5",
    "secondary_text": "#9BA8C0",
    "borders": "#303B55",
    "hover_button": "#607BC2",
    "warning": "#FBBF24",
    "error": "#F87171",
    "success": "#48D6C2",
}

# BGR equivalents for OpenCV rendering
BGR_COLORS = {
    "primary_indigo": (165, 100, 76),
    "light_indigo": (232, 167, 145),
    "trajectory_accent": (194, 214, 72),
    "main_text": (245, 236, 232),
    "secondary_text": (192, 168, 155),
    "dark_panel": (34, 21, 16),
    "warning": (36, 191, 251),
    "error": (113, 113, 248),
}

# Muted distinguishable BGR palette for non-selected tracked objects
MUTED_TRACK_BGR_PALETTE = [
    (155, 125, 95),
    (140, 155, 115),
    (170, 140, 120),
    (125, 145, 165),
    (150, 120, 155),
    (120, 160, 155),
    (165, 150, 110),
    (135, 130, 165),
]

# ---------------------------------------------------------------------------
# Video & Model Settings
# ---------------------------------------------------------------------------
SUPPORTED_VIDEO_EXTENSIONS = ("mp4", "avi", "mov", "mkv", "webm")

YOLO_MODEL_OPTIONS = {
    "Nano (yolov8n.pt) — Fast / Recommended": "yolov8n.pt",
    "Small (yolov8s.pt) — Balanced": "yolov8s.pt",
    "Medium (yolov8m.pt) — High Accuracy": "yolov8m.pt",
}

DEFAULT_MODEL_KEY = "Nano (yolov8n.pt) — Fast / Recommended"
DEFAULT_MODEL_WEIGHTS = "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.35
MIN_CONFIDENCE = 0.25
MAX_CONFIDENCE = 0.90
DEFAULT_IOU_THRESHOLD = 0.50
DEFAULT_TRACKER = "bytetrack.yaml"

# Common COCO classes highlighted in filters (pretrained model supports all 80)
PRIMARY_COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
]

# ---------------------------------------------------------------------------
# Movement & Speed Analysis Defaults
# ---------------------------------------------------------------------------
DEFAULT_MOVEMENT_THRESHOLD_PX = 2.0
MAX_TRAJECTORY_POINTS = 500
MAX_PLAUSIBLE_JUMP_FACTOR = 5.0  # Heuristic jump check relative to bbox diagonal

# Navigation Pages (exact order requested)
NAV_PAGES = [
    "Dashboard",
    "Video Tracking",
    "Object Analysis",
    "Trajectory Analysis",
    "Performance Evaluation",
    "Reports & Export",
    "About Project",
]
