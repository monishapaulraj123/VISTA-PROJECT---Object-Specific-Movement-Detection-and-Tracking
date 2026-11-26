# TrackNova

**Object-Specific Movement Detection and Tracking Using Video Datasets**

> **See Every Movement. Track Every Object.**  
> *AI-Powered Object Tracking & Motion Intelligence*

---

## 1. Project Overview

**TrackNova** is a modular, Python-based computer vision application built with **OpenCV**, **Ultralytics YOLO**, **ByteTrack**, **Matplotlib**, **Pandas**, and **Streamlit**. It processes uploaded video datasets frame by frame, detects and classifies multiple objects, assigns persistent tracking IDs using ByteTrack, enables dynamic selection of a specific target ID, and quantifies that object's movement through pixel displacement, 2D image-plane direction, pixel-based speed, and trajectory visualization.

### Project Objective

To detect multiple objects in video frames, assign unique tracking IDs, select a particular object, and estimate its movement, displacement, speed, direction, and trajectory.

---

## 2. Features & All 14 Project Tasks

1. **Python & OpenCV Pipeline** — Frame extraction, preprocessing, codec handling, and bounding-box/trajectory rendering.
2. **Video Dataset Input** — Supports uploading MP4, AVI, MOV, MKV, and WEBM videos or selecting stored files from `dataset/videos/`.
3. **Frame Extraction & Preprocessing** — Streams frames without loading entire videos into RAM, with optional aspect-preserving resizing and CLAHE contrast enhancement.
4. **YOLO Object Detection** — Ultralytics YOLO integration (`yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`) plus custom `.pt` model upload/path support.
5. **Object Classification** — Supports Person, Car, Bicycle, Motorcycle, Bus, Truck, and all pretrained COCO classes (or custom classes when using custom weights).
6. **ByteTrack Multi-Object Tracking** — Persistent frame-to-frame tracking IDs via `model.track(..., persist=True, tracker="bytetrack.yaml")`.
7. **Dynamic Object ID Selection** — Interactive dropdown & summary table populated directly from detected ByteTrack IDs.
8. **Selected Object Tracking** — Isolates movement calculations to the chosen ID while gracefully handling temporary disappearances without silently switching IDs.
9. **X / Y Displacement Calculation** — Computes centre coordinates `(X1+X2)/2`, `(Y1+Y2)/2`, `DX`, `DY`, Euclidean step distance, and cumulative displacement.
10. **Movement Direction Analysis** — Estimates 8-sector 2D image-plane movement direction (`Right`, `Left`, `Up`, `Down`, `Up-Right`, `Up-Left`, `Down-Right`, `Down-Left`, `Stationary`) using a configurable motion threshold.
11. **Pixel-Based Speed & Trajectory Calculation** — Calculates instantaneous, average, and maximum speed in `pixels/second (px/s)` and constructs ordered trajectory geometry.
12. **Movement Visualization** — Renders bounding boxes, tracking IDs, class labels, confidence scores, centre points, direction arrows, `#48D6C2` trajectories, and 5 Matplotlib charts.
13. **Multi-Dataset Testing** — Sequentially processes multiple videos and compares throughput, detection counts, and selected-object kinematics in a comparison table.
14. **Observational Performance Evaluation & Exports** — Computes processing FPS, tracking interruptions, visibility coverage, CSV exports, PNG charts, processed video files, and PDF/HTML reports.

---

## 3. Technology Stack

- **Python 3.10 / 3.11 / 3.12+**
- **Streamlit** — Web interface & session state management
- **OpenCV (`opencv-python`)** — Video decoding, preprocessing, visualization, and export
- **Ultralytics YOLO** — Object detection & ByteTrack integration
- **`lap`** — Linear assignment dependency for ByteTrack
- **NumPy & Pandas** — Kinematics math and CSV data processing
- **Matplotlib** — Dark-themed analytics charts
- **Pillow & ReportLab** — Logo rendering and PDF report generation

---

## 4. Project Architecture

```text
TrackNova/
│
├── app.py                      # Main Streamlit entry point & sidebar navigation
├── config.py                   # Branding, colour palette, paths, and model defaults
├── requirements.txt            # Compatible Python dependencies
├── README.md                   # Complete project documentation
│
├── .streamlit/
│   └── config.toml             # Dark navy Streamlit theme configuration
│
├── models/                     # Pretrained or custom-trained YOLO .pt weights
│
├── modules/
│   ├── video_processor.py      # OpenCV VideoCapture, preprocessing, VideoWriter
│   ├── detection.py            # Cached YOLO model loading & detection utilities
│   ├── tracking.py             # ByteTrack wrapper & per-ID track summarization
│   ├── movement.py             # Centre coordinates, DX/DY, direction & px/s speed
│   ├── trajectory.py           # Trajectory path geometry & linearity metrics
│   ├── visualization.py        # OpenCV overlays & 5 Matplotlib chart generators
│   ├── analytics.py            # Dashboard metrics & multi-dataset comparison
│   ├── evaluation.py           # Observational / proxy-based tracking evaluation
│   └── export.py               # CSV, HTML, and ReportLab PDF exports
│
├── pages/
│   ├── dashboard.py            # Overview statistics & session summary
│   ├── video_tracking.py       # Video upload, YOLO+ByteTrack run & ID selection
│   ├── object_analysis.py      # Detailed selected-object movement metrics & table
│   ├── trajectory_analysis.py  # 5 Matplotlib plots & PNG downloads
│   ├── evaluation.py           # Proxy evaluation & multi-video comparison table
│   ├── reports.py              # CSV, processed video, PNG, PDF & HTML exports
│   └── about.py                # Project info & 13-stage workflow diagram
│
├── assets/
│   └── logo.png                # TrackNova branded emblem
│
├── dataset/
│   └── videos/                 # Uploaded or local test video datasets
│
├── outputs/
│   ├── processed_videos/       # Annotated output videos
│   ├── graphs/                 # Exported PNG plots
│   ├── reports/                # Generated HTML & PDF summary reports
│   └── csv/                    # Frame-by-frame & dataset comparison CSVs
│
└── utils/
    ├── helpers.py              # Formatting, session state, SVG icon helpers
    └── ui.py                   # Custom CSS theme, KPI cards, empty/error states
```

---

## 5. Installation & Setup (Windows)

Navigate to the `TrackNova` project folder:

```powershell
cd TrackNova
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### YOLO Model & ByteTrack Setup

- By default, TrackNova uses **`yolov8n.pt`** (Nano), stored in `TrackNova/models/`. On first run with internet access, Ultralytics downloads `yolov8n.pt` automatically if it is not already present.
- ByteTrack uses Ultralytics' built-in `bytetrack.yaml` configuration paired with `lap`.
- **Drone Detection Note:** Standard pretrained COCO YOLO models (80 classes) do **not** include a `drone` class and should not be expected to reliably detect drones. To track drones, switch **Model Source** to **Custom Trained YOLO (.pt)** on the **Video Tracking** page and upload or reference a model trained on a drone dataset.

---

## 6. Running the Application

```bash
streamlit run app.py
```

---

## 7. User Guide

### Uploading & Processing a Video
1. Open **Video Tracking** from the sidebar.
2. Upload an `.mp4`, `.avi`, `.mov`, `.mkv`, or `.webm` video (or choose a video from `dataset/videos/`).
3. Inspect the extracted resolution, FPS, frame count, and duration.
4. Select your YOLO model size (Nano recommended for laptops), confidence threshold (`0.25–0.90`), optional class filter, and preprocessing options.
5. Click **Start Video Processing** to stream frames through YOLO + ByteTrack.

### Selecting an Object ID
1. Once processing finishes, review the **Detected ByteTrack IDs** table.
2. Choose a target ID from **Select Target Tracking ID**.
3. TrackNova immediately computes movement, trajectory, and evaluation metrics for that ID without re-running YOLO inference, and lets you render an annotated video highlighting that target.

### Understanding Coordinate System & Movement Metrics
- **2D Image Coordinate Convention:** Origin `(0, 0)` is the **top-left corner** of the frame.
  - `+DX` = Moving Right, `-DX` = Moving Left
  - `+DY` = Moving Down, `-DY` = Moving Up
- **Movement Direction vs. Physical Orientation:** Direction labels (`Up-Right`, `Down`, etc.) describe 2D displacement of the bounding-box centre across frames, **not** the physical 3D body orientation of the object (which requires pose estimation or keypoints).
- **Pixel-Based Speed (`px/s`):** Calculated as `Euclidean Displacement / (Frame Difference / FPS)`. Values are strictly in `pixels/second (px/s)`; converting to metres/second requires camera calibration or a known physical scale reference.

### Evaluation Limitations
- Because uploaded user videos do not include ground-truth bounding-box annotations, TrackNova performs **Observational / Proxy-Based Evaluation** (processing FPS, ID assignment rate, visibility percentage, and tracking interruption gaps).
- Formal MOT metrics (**MOTA, MOTP, IDF1, Precision, Recall, or Accuracy %**) are intentionally **not fabricated**.

### Downloading Reports & Exports
Navigate to **Reports & Export** to download:
- Frame-by-frame movement CSV for the selected object
- Multi-dataset comparison CSV
- Processed video with bounding boxes, IDs, centre points, direction arrows, and `#48D6C2` trajectory
- Individual PNG graphs from Matplotlib
- Formatted **PDF** and **HTML** summary reports

---

## 8. Troubleshooting

- **Model download fails offline:** Manually place `yolov8n.pt` inside `TrackNova/models/`.
- **Zero FPS metadata in video:** TrackNova safely skips speed division by zero and marks pixel speed as `N/A` while preserving displacement and trajectory plots.
- **Browser cannot play an exported `.avi` or `mp4v` video:** Use the download button in **Reports & Export** and open locally (e.g., VLC), or ensure H.264 (`avc1`) codec support is available.
- **Technical stack traces hidden:** Enable **Developer Debug Mode** in the bottom of the sidebar to inspect full exception tracebacks during development.

---

## 9. Future Enhancements

- Camera calibration & homography support for real-world `m/s` speed conversion.
- Ground-truth MOT evaluation importer (MOTChallenge / COCO format) for formal MOTA/IDF1 scoring.
- Multi-camera re-identification (Re-ID) across occlusion boundaries.
- Pose/keypoint integration for true physical object orientation estimation.
