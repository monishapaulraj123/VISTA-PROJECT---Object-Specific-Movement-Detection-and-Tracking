"""
TrackNova Video Tracking Workspace.

Single-source-of-truth video pipeline:
1. Select a dataset video from `dataset/videos/` OR upload a new video file.
2. Preview the exact active original video & inspect OpenCV-verified metadata.
3. Configure preprocessing, YOLO model, confidence, and ByteTrack parameters.
4. Run frame extraction & preprocessing verification or full YOLO + ByteTrack tracking.
5. Dynamically select any detected tracking ID for movement & trajectory focus.
"""

from __future__ import annotations

import html
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import pandas as pd
import streamlit as st

from config import (
    COLORS,
    DATASET_VIDEOS_DIR,
    DEFAULT_CONFIDENCE,
    DEFAULT_MODEL_KEY,
    DEFAULT_MOVEMENT_THRESHOLD_PX,
    MAX_CONFIDENCE,
    MIN_CONFIDENCE,
    PRIMARY_COCO_CLASSES,
    SUPPORTED_VIDEO_EXTENSIONS,
    UPLOADS_DIR,
    YOLO_MODEL_OPTIONS,
)
from modules.analytics import build_dataset_comparison_record, upsert_dataset_session
from modules.detection import (
    load_yolo_model_cached,
    resolve_class_filter_ids,
    save_uploaded_model,
)
from modules.evaluation import evaluate_tracking_performance
from modules.movement import calculate_selected_object_movement
from modules.tracking import (
    ByteTrackProcessor,
    FrameTrackingRecord,
    TrackingSessionResult,
    summarize_tracks,
)
from modules.trajectory import build_trajectory_data
from modules.video_processor import (
    compute_bytes_hash,
    compute_file_source_id,
    create_video_writer,
    get_browser_preview_bytes,
    iter_video_frames,
    preprocess_frame,
    read_single_frame,
    run_frame_preprocessing,
    save_uploaded_video,
    validate_and_inspect_video,
)
from modules.visualization import draw_tracking_frame, render_annotated_video
from utils.helpers import (
    format_duration,
    format_file_size,
    reset_tracking_session,
)
from utils.ui import (
    render_kpi_grid,
    render_page_header,
    render_section_title,
    show_friendly_error,
)

DATASET_PLACEHOLDER = "-- Select a dataset video --"


def _clear_active_video_source() -> None:
    """Clear active video source keys and all downstream processing results."""
    reset_tracking_session(keep_dataset_history=True)
    st.session_state["video_source_type"] = None
    st.session_state["active_video_path"] = None
    st.session_state["active_video_name"] = None
    st.session_state["video_source_id"] = None
    st.session_state["video_path"] = None
    st.session_state["video_meta"] = None


def _set_active_video_source(
    source_type: str,
    file_path: Path,
    display_name: str,
    source_id: str,
    video_meta,
) -> None:
    """Set the single active video source of truth and reset stale outputs."""
    reset_tracking_session(keep_dataset_history=True)
    resolved = str(Path(file_path).resolve())
    st.session_state["video_source_type"] = source_type
    st.session_state["active_video_path"] = resolved
    st.session_state["active_video_name"] = display_name
    st.session_state["video_source_id"] = source_id
    st.session_state["video_path"] = resolved
    st.session_state["video_meta"] = video_meta


def _activate_selected_track_id(
    session_result: TrackingSessionResult,
    track_id: Optional[int],
    movement_threshold_px: float = DEFAULT_MOVEMENT_THRESHOLD_PX,
) -> None:
    """Compute movement, trajectory, evaluation, and comparison updates for `track_id`."""
    st.session_state["selected_track_id"] = track_id
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


def _list_dataset_videos() -> List[Path]:
    """Return existing videos stored under dataset/videos/."""
    DATASET_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    return [
        p
        for p in sorted(DATASET_VIDEOS_DIR.iterdir())
        if p.is_file() and p.suffix.lower().lstrip(".") in SUPPORTED_VIDEO_EXTENSIONS
    ]


def render() -> None:
    render_page_header(
        "Video Tracking & Object Selection",
        "Select a dataset video or upload a file, preview and preprocess frames, and run YOLO + ByteTrack tracking",
        icon_name="video",
    )

    # Apply pending widget reset BEFORE instantiating widgets
    if st.session_state.pop("_pending_clear_dataset_select", False):
        st.session_state["vt_existing_select"] = DATASET_PLACEHOLDER
        st.session_state["_prev_dataset_choice"] = DATASET_PLACEHOLDER

    # ------------------------------------------------------------------
    # 1. Video Dataset Input (Independent Dataset & Upload Controls)
    # ------------------------------------------------------------------
    render_section_title("1. Video Dataset Input", "Choose from dataset/videos/ or upload a video file")
    with st.container(border=True):
        in_col1, in_col2 = st.columns([1, 1.15])

        with in_col1:
            existing_videos = _list_dataset_videos()
            existing_names = [DATASET_PLACEHOLDER] + [p.name for p in existing_videos]
            chosen_name = st.selectbox(
                "Option 1: Select Dataset Video (`dataset/videos/`)",
                options=existing_names,
                key="vt_existing_select",
            )

        with in_col2:
            uploader_key = f"vt_video_uploader_{st.session_state.get('_uploader_nonce', 0)}"
            uploaded_file = st.file_uploader(
                "Option 2: Upload Video (MP4, AVI, MOV, MKV, WEBM)",
                type=list(SUPPORTED_VIDEO_EXTENSIONS),
                key=uploader_key,
            )

        prev_dataset = st.session_state.get("_prev_dataset_choice", DATASET_PLACEHOLDER)
        dataset_changed = chosen_name != prev_dataset

        current_upload_id: Optional[str] = None
        upload_bytes: Optional[bytes] = None
        if uploaded_file is not None:
            upload_bytes = uploaded_file.getvalue()
            upload_hash = compute_bytes_hash(upload_bytes)
            current_upload_id = f"{uploaded_file.name}:{len(upload_bytes)}:{upload_hash}"

        prev_upload_id = st.session_state.get("_prev_upload_id")
        upload_changed = current_upload_id is not None and current_upload_id != prev_upload_id
        upload_cleared = (
            uploaded_file is None
            and prev_upload_id is not None
            and st.session_state.get("video_source_type") == "upload"
        )

        # Handle new file upload first (takes precedence when user uploads a file)
        if upload_changed and uploaded_file is not None and upload_bytes is not None:
            try:
                saved_path, short_hash = save_uploaded_video(
                    upload_bytes, uploaded_file.name, UPLOADS_DIR
                )
                meta = validate_and_inspect_video(saved_path)
                source_id = compute_file_source_id("upload", saved_path, content_hash=short_hash)
                _set_active_video_source(
                    source_type="upload",
                    file_path=saved_path,
                    display_name=uploaded_file.name,
                    source_id=source_id,
                    video_meta=meta,
                )
                st.session_state["_prev_upload_id"] = current_upload_id
                if chosen_name != DATASET_PLACEHOLDER:
                    st.session_state["_pending_clear_dataset_select"] = True
                    st.rerun()
            except Exception as exc:
                st.session_state["_prev_upload_id"] = current_upload_id
                show_friendly_error(f"Could not load uploaded video '{uploaded_file.name}': {exc}", exc)

        # Handle user clicking 'X' on the file uploader
        elif upload_cleared:
            st.session_state["_prev_upload_id"] = None
            _clear_active_video_source()
            st.rerun()

        # Handle user selecting or clearing a dataset video
        elif dataset_changed:
            st.session_state["_prev_dataset_choice"] = chosen_name
            if chosen_name == DATASET_PLACEHOLDER:
                if st.session_state.get("video_source_type") == "dataset":
                    _clear_active_video_source()
                    st.rerun()
            else:
                chosen_path = (DATASET_VIDEOS_DIR / chosen_name).resolve()
                try:
                    meta = validate_and_inspect_video(chosen_path)
                    source_id = compute_file_source_id("dataset", chosen_path)
                    _set_active_video_source(
                        source_type="dataset",
                        file_path=chosen_path,
                        display_name=chosen_name,
                        source_id=source_id,
                        video_meta=meta,
                    )
                    if prev_upload_id is not None or uploaded_file is not None:
                        st.session_state["_prev_upload_id"] = None
                        st.session_state["_uploader_nonce"] = int(st.session_state.get("_uploader_nonce", 0)) + 1
                        st.rerun()
                except Exception as exc:
                    show_friendly_error(f"Could not open dataset video '{chosen_name}': {exc}", exc)

        # Active source indicator inside Input section
        active_name = st.session_state.get("active_video_name")
        active_type = st.session_state.get("video_source_type")
        active_id = st.session_state.get("video_source_id")
        if active_name and active_type:
            badge_label = "Uploaded Video" if active_type == "upload" else "Dataset Video"
            st.markdown(
                f"""
                <div style="margin-top:0.35rem;padding:0.5rem 0.8rem;border-radius:8px;
                            background:{COLORS['sidebar_bg']};border:1px solid {COLORS['borders']};
                            display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.5rem;">
                    <span style="font-size:0.86rem;color:{COLORS['main_text']};">
                        <strong style="color:{COLORS['trajectory_accent']};">Active Input ({badge_label}):</strong>
                        {html.escape(active_name)}
                    </span>
                    <span style="font-size:0.76rem;color:{COLORS['secondary_text']};font-family:monospace;">
                        Source ID: {html.escape(str(active_id))}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    active_video_path = st.session_state.get("active_video_path")
    active_video_name = st.session_state.get("active_video_name")
    video_source_id = st.session_state.get("video_source_id")
    video_meta = st.session_state.get("video_meta")

    if not active_video_path or video_meta is None or not video_source_id:
        st.markdown(
            """
            <div class="tn-alert-info">
                Select a dataset video from <code>dataset/videos/</code> or upload a video file above to preview and process.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ------------------------------------------------------------------
    # 2. Original Video Metadata & Live Preview (Single Source of Truth)
    # ------------------------------------------------------------------
    render_section_title("2. Original Video & Preview", f"Active file: {active_video_name}")
    fps_disp = f"{video_meta.fps:.2f}" if video_meta.fps_is_valid else "Invalid / 0"
    dur_disp = format_duration(video_meta.duration_sec) if video_meta.fps_is_valid else "N/A"
    render_kpi_grid(
        [
            {
                "label": "Active Video",
                "value": active_video_name or video_meta.file_name,
                "subtext": f"{'Uploaded' if st.session_state.get('video_source_type') == 'upload' else 'Dataset'} • {format_file_size(video_meta.file_size_bytes)}",
                "accent_color": COLORS["trajectory_accent"],
            },
            {
                "label": "Resolution",
                "value": f"{video_meta.width} x {video_meta.height}",
                "subtext": f"Codec: {video_meta.fourcc_str}",
            },
            {
                "label": "FPS & Duration",
                "value": f"{fps_disp} FPS",
                "subtext": f"Duration: {dur_disp} | Frames: {video_meta.total_frames}",
            },
        ],
        columns=3,
    )

    if not video_meta.fps_is_valid:
        st.markdown(
            '<div class="tn-alert-warn">This video reports 0 or invalid FPS metadata. '
            'Tracking will still run, but pixel speed (px/s) requires valid FPS or timestamps.</div>',
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        st.markdown(
            f"<div style='color:{COLORS['light_indigo']};font-weight:700;font-size:0.95rem;margin-bottom:0.45rem;'>"
            f"Preview Original Video — {html.escape(active_video_name or video_meta.file_name)}</div>",
            unsafe_allow_html=True,
        )
        prev_vid_col, prev_frame_col = st.columns([1.15, 1])

        with prev_vid_col:
            try:
                preview_bytes, preview_mime = get_browser_preview_bytes(
                    active_video_path, video_meta, video_source_id
                )
                st.video(preview_bytes, format=preview_mime)
            except Exception as exc:
                show_friendly_error(
                    f"Unable to render browser video player for '{active_video_name}': {exc}",
                    exc,
                )

        with prev_frame_col:
            first_frame = read_single_frame(active_video_path, 0)
            if first_frame is not None:
                st.image(
                    cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB),
                    caption=f"OpenCV Verified Frame #0 ({video_meta.width}x{video_meta.height})",
                    width="stretch",
                )
            else:
                show_friendly_error(
                    f"OpenCV VideoCapture could not read Frame #0 from '{active_video_name}'."
                )

    # ------------------------------------------------------------------
    # 3. YOLO, ByteTrack & Preprocessing Settings
    # ------------------------------------------------------------------
    render_section_title("3. Detection, Tracking & Preprocessing Settings")
    with st.container(border=True):
        cfg_col1, cfg_col2, cfg_col3 = st.columns([1.1, 1, 1])

        with cfg_col1:
            model_mode = st.radio(
                "Model Source",
                options=["Pretrained YOLO (COCO)", "Custom Trained YOLO (.pt)"],
                horizontal=True,
                key="vt_model_mode",
            )
            is_custom_model = model_mode == "Custom Trained YOLO (.pt)"
            weights_source = YOLO_MODEL_OPTIONS[DEFAULT_MODEL_KEY]

            if not is_custom_model:
                selected_model_label = st.selectbox(
                    "YOLO Model Size",
                    options=list(YOLO_MODEL_OPTIONS.keys()),
                    index=0,
                    key="vt_pretrained_model",
                )
                weights_source = YOLO_MODEL_OPTIONS[selected_model_label]
                st.caption(
                    "Pretrained COCO (80 classes). Standard COCO models do not detect drones; "
                    "drone detection requires a custom-trained dataset/model."
                )
            else:
                custom_upload = st.file_uploader("Upload Custom .pt Weights", type=["pt"], key="vt_custom_pt_upload")
                custom_path_text = st.text_input(
                    "Or Enter Local Custom .pt Model Path",
                    value="",
                    placeholder="e.g. models/custom_drone_yolov8n.pt",
                    key="vt_custom_pt_path",
                )
                if custom_upload is not None:
                    try:
                        saved_pt = save_uploaded_model(custom_upload.getvalue(), custom_upload.name)
                        weights_source = str(saved_pt)
                        st.caption(f"Model Type: Custom Trained ({saved_pt.name})")
                    except Exception as exc:
                        show_friendly_error(f"Invalid custom model file: {exc}", exc)
                        weights_source = ""
                elif custom_path_text.strip():
                    weights_source = custom_path_text.strip()
                    st.caption(f"Model Type: Custom Trained ({Path(weights_source).name})")
                else:
                    weights_source = ""
                    st.info("Upload or specify a custom-trained .pt model path to use custom weights.")

        with cfg_col2:
            conf_threshold = st.slider(
                "Confidence Threshold",
                min_value=MIN_CONFIDENCE,
                max_value=MAX_CONFIDENCE,
                value=DEFAULT_CONFIDENCE,
                step=0.05,
                key="vt_conf_slider",
            )
            iou_threshold = st.slider(
                "ByteTrack / NMS IoU Threshold",
                min_value=0.30,
                max_value=0.80,
                value=0.50,
                step=0.05,
                key="vt_iou_slider",
            )
            movement_thresh = st.number_input(
                "Stationary Movement Threshold (px)",
                min_value=0.5,
                max_value=25.0,
                value=DEFAULT_MOVEMENT_THRESHOLD_PX,
                step=0.5,
                key="vt_move_thresh",
            )

        with cfg_col3:
            resize_choice = st.selectbox(
                "Preprocessing Resize Width",
                options=["Original Resolution", "960 px", "640 px"],
                index=0,
                key="vt_resize_choice",
            )
            target_width: Optional[int] = None
            if resize_choice == "960 px":
                target_width = 960
            elif resize_choice == "640 px":
                target_width = 640

            frame_stride = st.selectbox(
                "Frame Processing Stride",
                options=[1, 2, 3],
                index=0,
                format_func=lambda s: "Every frame (1)" if s == 1 else f"Every {s} frames",
                key="vt_frame_stride",
            )
            enhance_contrast = st.checkbox(
                "Enhance Low-Light Contrast (CLAHE)",
                value=False,
                key="vt_clahe",
            )
            show_confidence = st.checkbox("Show Confidence on Boxes", value=True, key="vt_show_conf")
            show_other_objects = st.checkbox("Show Non-Selected Objects", value=True, key="vt_show_others")
            save_video_output = st.checkbox("Save Processed Video Output", value=True, key="vt_save_video")

        # Load model (cached) to populate class filter dynamically
        loaded_model = None
        model_info = None
        if weights_source:
            try:
                with st.spinner("Loading YOLO model weights..."):
                    loaded_model, model_info = load_yolo_model_cached(
                        weights_source, is_custom=is_custom_model
                    )
            except Exception as exc:
                show_friendly_error(str(exc), exc)

        selected_classes: List[str] = []
        if model_info is not None:
            all_model_classes = list(model_info.class_names.values())
            ordered_classes = [c for c in PRIMARY_COCO_CLASSES if c in all_model_classes] + [
                c for c in all_model_classes if c not in PRIMARY_COCO_CLASSES
            ]
            selected_classes = st.multiselect(
                "Optional Class Filter (leave empty to detect all supported classes)",
                options=ordered_classes,
                default=[],
                key="vt_class_filter",
            )

    # ------------------------------------------------------------------
    # 4. Action Controls (Preprocess / Start Tracking / Stop / Reset)
    # ------------------------------------------------------------------
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1.05, 1.25, 0.9, 0.95])
    with btn_col1:
        preprocess_clicked = st.button(
            "Preprocess",
            type="secondary",
            width="stretch",
            key="vt_preprocess_btn",
        )
    with btn_col2:
        start_clicked = st.button(
            "Start Video Processing",
            type="primary",
            width="stretch",
            disabled=(loaded_model is None),
            key="vt_start_btn",
        )
    with btn_col3:
        stop_clicked = st.button(
            "Stop Processing",
            type="secondary",
            width="stretch",
            key="vt_stop_btn",
        )
    with btn_col4:
        reset_clicked = st.button(
            "Reset Session",
            type="secondary",
            width="stretch",
            key="vt_reset_btn",
        )

    if reset_clicked:
        reset_tracking_session(keep_dataset_history=True)
        st.rerun()

    # Dedicated Frame Extraction & Preprocessing Handler
    if preprocess_clicked:
        try:
            with st.spinner(f"Extracting and preprocessing frames from {active_video_name}..."):
                pre_result = run_frame_preprocessing(
                    video_path=active_video_path,
                    video_meta=video_meta,
                    video_source_id=video_source_id,
                    target_width=target_width,
                    enhance_contrast=enhance_contrast,
                    frame_stride=frame_stride,
                    max_samples=4,
                )
            st.session_state["preprocessed_preview"] = pre_result
        except Exception as exc:
            st.session_state["preprocessed_preview"] = None
            show_friendly_error(f"Preprocessing failed for '{active_video_name}': {exc}", exc)

    # Show Preprocessing Results ONLY if they match the current active video_source_id
    pre_preview = st.session_state.get("preprocessed_preview")
    if pre_preview is not None and pre_preview.get("video_source_id") == video_source_id:
        with st.container(border=True):
            ow, oh = pre_preview["orig_size"]
            pw, ph = pre_preview["proc_size"]
            st.markdown(
                f"<div style='color:{COLORS['trajectory_accent']};font-weight:700;font-size:0.93rem;margin-bottom:0.4rem;'>"
                f"Preprocessing Output — {html.escape(pre_preview['video_name'])} "
                f"({pre_preview['frames_extracted']} frames extracted | {ow}x{oh} &rarr; {pw}x{ph} | "
                f"CLAHE: {'On' if pre_preview['enhance_contrast'] else 'Off'})</div>",
                unsafe_allow_html=True,
            )
            samples = pre_preview.get("sample_frames", [])
            if samples:
                s_cols = st.columns(len(samples))
                for col, (f_idx, rgb_img) in zip(s_cols, samples):
                    with col:
                        st.image(rgb_img, caption=f"Frame #{f_idx}", width="stretch")

    if stop_clicked:
        partial = st.session_state.get("_live_partial")
        if partial and partial.get("frame_records"):
            st.session_state["stop_requested"] = True
            recs = partial["frame_records"]
            summaries = summarize_tracks(recs)
            elapsed = max(1e-4, time.perf_counter() - partial["start_time"])
            res = TrackingSessionResult(
                video_meta=video_meta,
                model_info=partial["model_info"],
                processed_width=partial["proc_w"],
                processed_height=partial["proc_h"],
                frame_stride=partial["frame_stride"],
                confidence_threshold=partial["conf_threshold"],
                class_filter=partial["class_filter"],
                frame_records=recs,
                track_summaries=summaries,
                total_frames_processed=len(recs),
                processing_time_sec=elapsed,
                processing_fps=len(recs) / elapsed,
                total_detections=partial["total_detections"],
                untracked_detections=partial["untracked_detections"],
                unique_track_ids=sorted(summaries.keys()),
                class_counts=dict(partial["class_counts"]),
                stopped_early=True,
            )
            st.session_state["processing_result"] = res
            st.session_state["processed_video_path"] = partial.get("out_path")
            st.session_state["_live_partial"] = None
            _activate_selected_track_id(res, None, movement_threshold_px=float(movement_thresh))
            st.warning("Processing stopped early. Partial tracking results were preserved.")
        else:
            st.info("No active frame processing loop to stop.")

    # ------------------------------------------------------------------
    # 5. Frame-by-Frame YOLO + ByteTrack Processing Loop
    # ------------------------------------------------------------------
    if start_clicked and loaded_model is not None and model_info is not None:
        st.session_state["stop_requested"] = False
        st.session_state["processing_error"] = None
        st.session_state["processing_status"] = "Processing"

        # Re-verify current active video file before processing
        try:
            verified_meta = validate_and_inspect_video(active_video_path)
            st.session_state["video_meta"] = verified_meta
            video_meta = verified_meta
        except Exception as exc:
            st.session_state["processing_status"] = "Error"
            show_friendly_error(f"Cannot process '{active_video_name}': {exc}", exc)
            return

        class_filter_ids = resolve_class_filter_ids(model_info.class_names, selected_classes)
        tracker = ByteTrackProcessor(
            model=loaded_model,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            class_filter_ids=class_filter_ids,
        )

        progress_bar = st.progress(0.0, text=f"Initializing stream for {active_video_name}...")
        status_placeholder = st.empty()
        preview_placeholder = st.empty()

        frame_records: List[FrameTrackingRecord] = []
        class_counter: Counter = Counter()
        total_dets = 0
        untracked_dets = 0
        start_time = time.perf_counter()

        writer = None
        out_path: Optional[Path] = None
        proc_w, proc_h = video_meta.width, video_meta.height
        expected_steps = max(1, (video_meta.total_frames + frame_stride - 1) // frame_stride)

        st.session_state["_live_partial"] = {
            "frame_records": frame_records,
            "model_info": model_info,
            "proc_w": proc_w,
            "proc_h": proc_h,
            "frame_stride": frame_stride,
            "conf_threshold": conf_threshold,
            "class_filter": list(selected_classes),
            "start_time": start_time,
            "total_detections": 0,
            "untracked_detections": 0,
            "class_counts": class_counter,
            "out_path": None,
        }

        try:
            for step_i, (frame_idx, ts_sec, raw_frame) in enumerate(
                iter_video_frames(
                    active_video_path,
                    fps=video_meta.fps,
                    frame_stride=frame_stride,
                )
            ):
                pre_frame = preprocess_frame(
                    raw_frame,
                    target_width=target_width,
                    enhance_contrast=enhance_contrast,
                )
                if step_i == 0:
                    proc_h, proc_w = pre_frame.shape[:2]
                    st.session_state["_live_partial"]["proc_w"] = proc_w
                    st.session_state["_live_partial"]["proc_h"] = proc_h
                    if save_video_output:
                        out_fps = (
                            video_meta.fps / frame_stride
                            if video_meta.fps_is_valid and video_meta.fps > 0
                            else 25.0
                        )
                        writer, out_path, _codec_used = create_video_writer(
                            base_name=active_video_name or video_meta.file_name,
                            fps=out_fps,
                            frame_size=(proc_w, proc_h),
                        )
                        st.session_state["_live_partial"]["out_path"] = str(out_path) if out_path else None
                        if writer is None:
                            st.warning(
                                "Could not initialize an OpenCV video writer codec; continuing analytics without video file export."
                            )

                detections = tracker.track_frame(pre_frame)
                frame_records.append(
                    FrameTrackingRecord(
                        frame_idx=frame_idx,
                        timestamp_sec=ts_sec,
                        detections=detections,
                    )
                )

                for det in detections:
                    total_dets += 1
                    class_counter[det.class_name] += 1
                    if det.track_id is None:
                        untracked_dets += 1

                st.session_state["_live_partial"]["total_detections"] = total_dets
                st.session_state["_live_partial"]["untracked_detections"] = untracked_dets

                vis_frame = draw_tracking_frame(
                    pre_frame,
                    detections=detections,
                    selected_track_id=None,
                    show_confidence=show_confidence,
                    show_other_objects=True,
                )
                if writer is not None:
                    if vis_frame.shape[1] != proc_w or vis_frame.shape[0] != proc_h:
                        vis_frame = cv2.resize(vis_frame, (proc_w, proc_h))
                    writer.write(vis_frame)

                frac = min(1.0, (step_i + 1) / expected_steps)
                elapsed_now = max(1e-4, time.perf_counter() - start_time)
                cur_fps = (step_i + 1) / elapsed_now
                progress_bar.progress(
                    frac,
                    text=f"Processing {active_video_name}: frame {frame_idx + 1}/{video_meta.total_frames} ({cur_fps:.1f} FPS) — Detections: {total_dets}",
                )

                if step_i == 0 or (step_i + 1) % 5 == 0 or (step_i + 1) == expected_steps:
                    preview_placeholder.image(
                        cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB),
                        caption=f"Live Tracking Preview ({active_video_name}) — Frame {frame_idx}",
                        width="stretch",
                    )
                    status_placeholder.caption(
                        f"Processed {step_i + 1} frames | Active detections in frame: {len(detections)}"
                    )
        except Exception as exc:
            st.session_state["processing_status"] = "Error"
            show_friendly_error(f"Video processing encountered an error: {exc}", exc)
            if writer is not None:
                writer.release()
            st.session_state["_live_partial"] = None
            return
        finally:
            if writer is not None:
                writer.release()

        st.session_state["_live_partial"] = None
        elapsed_total = max(1e-4, time.perf_counter() - start_time)
        summaries = summarize_tracks(frame_records)

        session_res = TrackingSessionResult(
            video_meta=video_meta,
            model_info=model_info,
            processed_width=proc_w,
            processed_height=proc_h,
            frame_stride=frame_stride,
            confidence_threshold=conf_threshold,
            class_filter=list(selected_classes),
            frame_records=frame_records,
            track_summaries=summaries,
            total_frames_processed=len(frame_records),
            processing_time_sec=elapsed_total,
            processing_fps=len(frame_records) / elapsed_total,
            total_detections=total_dets,
            untracked_detections=untracked_dets,
            unique_track_ids=sorted(summaries.keys()),
            class_counts=dict(class_counter),
            stopped_early=False,
        )

        st.session_state["processing_result"] = session_res
        st.session_state["processing_status"] = "Completed"
        st.session_state["processed_video_path"] = str(out_path) if out_path else None
        st.session_state["re_rendered_video_path"] = None
        st.session_state["last_settings"] = {
            "target_width": target_width,
            "enhance_contrast": enhance_contrast,
            "show_confidence": show_confidence,
            "show_other_objects": show_other_objects,
            "movement_threshold_px": float(movement_thresh),
        }

        default_tid = None
        if summaries:
            default_tid = max(summaries.values(), key=lambda s: s.frames_tracked).track_id
        _activate_selected_track_id(
            session_res, default_tid, movement_threshold_px=float(movement_thresh)
        )
        progress_bar.progress(1.0, text="Processing complete!")
        st.rerun()

    # ------------------------------------------------------------------
    # 6. Dynamic Object ID Selection & Post-Processing Visualization
    # ------------------------------------------------------------------
    session_result: Optional[TrackingSessionResult] = st.session_state.get("processing_result")
    if session_result is None:
        return

    # Guard against stale result mismatching current active_video_path
    if str(Path(session_result.video_meta.file_path).resolve()) != str(Path(active_video_path).resolve()):
        return

    render_section_title("4. Dynamic Tracking ID Selection & Visualization", session_result.video_meta.file_name)

    if session_result.total_detections == 0 or not session_result.unique_track_ids:
        st.markdown(
            '<div class="tn-alert-warn">No persistent ByteTrack object IDs were detected in this video under the current settings. '
            'Try lowering the confidence threshold or clearing class filters.</div>',
            unsafe_allow_html=True,
        )
        return

    id_rows = [
        {
            "Tracking ID": s.track_id,
            "Object Class": s.class_name,
            "First Frame": s.first_frame,
            "Last Frame": s.last_frame,
            "Frames Tracked": s.frames_tracked,
            "Avg Confidence": round(s.avg_confidence, 3),
        }
        for s in session_result.track_summaries.values()
    ]
    id_df = pd.DataFrame(id_rows).sort_values(by="Frames Tracked", ascending=False)

    sel_col, tbl_col = st.columns([1, 1.35])
    with sel_col:
        with st.container(border=True):
            options: List[Optional[int]] = [None] + session_result.unique_track_ids
            current_sel = st.session_state.get("selected_track_id")
            if current_sel not in options:
                current_sel = None

            def _fmt_id(val: Optional[int]) -> str:
                if val is None:
                    return "-- None (Clear Selection) --"
                s = session_result.track_summaries[val]
                return f"ID #{val} — {s.class_name} ({s.frames_tracked} frames)"

            chosen_tid = st.selectbox(
                "Select Target Tracking ID",
                options=options,
                index=options.index(current_sel),
                format_func=_fmt_id,
                key="vt_dynamic_id_select",
            )

            if chosen_tid != st.session_state.get("selected_track_id"):
                _activate_selected_track_id(
                    session_result,
                    chosen_tid,
                    movement_threshold_px=float(
                        st.session_state.get("last_settings", {}).get(
                            "movement_threshold_px", DEFAULT_MOVEMENT_THRESHOLD_PX
                        )
                    ),
                )
                st.session_state["re_rendered_video_path"] = None
                st.rerun()

            if st.button("Clear Trajectory & Selection", type="secondary", width="stretch", key="vt_clear_sel"):
                _activate_selected_track_id(session_result, None)
                st.session_state["re_rendered_video_path"] = None
                st.rerun()

            msum = st.session_state.get("movement_summary")
            if msum and msum.get("has_data"):
                avg_s = f"{msum['avg_speed_px_s']:.2f}" if msum["avg_speed_px_s"] is not None else "N/A"
                max_s = f"{msum['max_speed_px_s']:.2f}" if msum["max_speed_px_s"] is not None else "N/A"
                st.markdown(
                    f"""
                    <div style="background:{COLORS['sidebar_bg']};border:1px solid {COLORS['borders']};
                                border-radius:10px;padding:0.75rem 0.9rem;margin-top:0.55rem;">
                        <div style="color:{COLORS['trajectory_accent']};font-weight:700;font-size:0.9rem;margin-bottom:0.3rem;">
                            Active Target: ID #{msum['selected_id']} ({msum['object_class']})
                        </div>
                        <div style="color:{COLORS['secondary_text']};font-size:0.84rem;line-height:1.55;">
                            • Frames Tracked: <strong style="color:{COLORS['main_text']};">{msum['frames_tracked']}</strong><br/>
                            • Total Displacement: <strong style="color:{COLORS['main_text']};">{msum['total_displacement_px']:.2f} px</strong><br/>
                            • Avg / Max Speed: <strong style="color:{COLORS['main_text']};">{avg_s} / {max_s} px/s</strong><br/>
                            • Current Direction: <strong style="color:{COLORS['light_indigo']};">{msum['current_direction']}</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if msum.get("tracking_interruptions", 0) > 0:
                    st.markdown(
                        f'<div class="tn-alert-warn" style="margin-top:0.55rem;margin-bottom:0;">Selected object had {msum["tracking_interruptions"]} temporary tracking interruption(s) across the video.</div>',
                        unsafe_allow_html=True,
                    )

    with tbl_col:
        with st.container(border=True):
            st.markdown(
                f"<div style='color:{COLORS['light_indigo']};font-weight:700;font-size:0.92rem;margin-bottom:0.45rem;'>Detected ByteTrack IDs</div>",
                unsafe_allow_html=True,
            )
            st.dataframe(id_df, width="stretch", height=270, hide_index=True)

    vis_col1, vis_col2 = st.columns([1, 1])

    with vis_col1:
        with st.container(border=True):
            st.markdown(
                f"<div style='color:{COLORS['light_indigo']};font-weight:700;font-size:0.95rem;margin-bottom:0.4rem;'>Interactive Frame Inspector</div>",
                unsafe_allow_html=True,
            )
            recs = session_result.frame_records
            if recs:
                max_idx = len(recs) - 1
                step_choice = st.slider(
                    "Inspect Processed Frame Step",
                    min_value=0,
                    max_value=max_idx,
                    value=max_idx,
                    key="vt_inspect_slider",
                )
                rec = recs[step_choice]
                raw_f = read_single_frame(session_result.video_meta.file_path, rec.frame_idx)
                if raw_f is not None:
                    ls = st.session_state.get("last_settings", {})
                    pre_f = preprocess_frame(
                        raw_f,
                        target_width=ls.get("target_width"),
                        enhance_contrast=bool(ls.get("enhance_contrast", False)),
                    )
                    mdf = st.session_state.get("movement_df")
                    sel_id = st.session_state.get("selected_track_id")
                    traj_up_to = None
                    cur_dir = None
                    cur_spd = None
                    m_vec = None
                    if sel_id is not None and mdf is not None and not mdf.empty:
                        sub = mdf[mdf["frame_number"] <= rec.frame_idx]
                        if not sub.empty:
                            traj_up_to = [
                                (float(r["center_x"]), float(r["center_y"]))
                                for _, r in sub.iterrows()
                            ]
                            exact = sub[sub["frame_number"] == rec.frame_idx]
                            if not exact.empty:
                                last_row = exact.iloc[-1]
                                cur_dir = str(last_row["direction"])
                                if not pd.isna(last_row["pixel_speed_px_s"]):
                                    cur_spd = float(last_row["pixel_speed_px_s"])
                                m_vec = (float(last_row["dx"]), float(last_row["dy"]))

                    drawn = draw_tracking_frame(
                        pre_f,
                        detections=rec.detections,
                        selected_track_id=sel_id,
                        trajectory_points=traj_up_to,
                        show_confidence=bool(ls.get("show_confidence", True)),
                        show_other_objects=bool(ls.get("show_other_objects", True)),
                        current_direction=cur_dir,
                        current_speed=cur_spd,
                        movement_vector=m_vec,
                    )
                    st.image(
                        cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB),
                        caption=f"Frame #{rec.frame_idx} | Target ID: {sel_id if sel_id is not None else 'None'}",
                        width="stretch",
                    )

    with vis_col2:
        with st.container(border=True):
            st.markdown(
                f"<div style='color:{COLORS['light_indigo']};font-weight:700;font-size:0.95rem;margin-bottom:0.4rem;'>Processed Video Playback</div>",
                unsafe_allow_html=True,
            )
            sel_id = st.session_state.get("selected_track_id")
            if sel_id is not None:
                if st.button(
                    f"Render Highlighted Video for ID #{sel_id}",
                    type="primary",
                    width="stretch",
                    key="vt_rerender_btn",
                ):
                    ls = st.session_state.get("last_settings", {})
                    r_prog = st.progress(0.0, text="Rendering highlighted video...")
                    out_p, codec_used, err = render_annotated_video(
                        session_result=session_result,
                        selected_track_id=sel_id,
                        movement_df=st.session_state.get("movement_df"),
                        show_confidence=bool(ls.get("show_confidence", True)),
                        show_other_objects=bool(ls.get("show_other_objects", True)),
                        target_width=ls.get("target_width"),
                        enhance_contrast=bool(ls.get("enhance_contrast", False)),
                        progress_callback=lambda cur, tot: r_prog.progress(
                            min(1.0, cur / max(1, tot)),
                            text=f"Rendering frame {cur}/{tot}...",
                        ),
                    )
                    if err:
                        show_friendly_error(err)
                    else:
                        st.session_state["re_rendered_video_path"] = str(out_p)
                        r_prog.progress(1.0, text=f"Rendered with codec {codec_used}!")

            vid_to_show = st.session_state.get("re_rendered_video_path") or st.session_state.get("processed_video_path")
            if vid_to_show and Path(vid_to_show).exists():
                try:
                    vp = Path(vid_to_show)
                    mime = "video/mp4" if vp.suffix.lower() == ".mp4" else "video/x-msvideo"
                    st.video(vp.read_bytes(), format=mime)
                    st.caption(f"Saved file: `{vp.name}`")
                except Exception as exc:
                    show_friendly_error(
                        "Browser playback failed for this codec, but the video file can still be downloaded from Reports & Export.",
                        exc,
                    )
            else:
                st.info(
                    "Click 'Render Highlighted Video' above to export and preview the video with the selected object's trajectory overlay."
                )
