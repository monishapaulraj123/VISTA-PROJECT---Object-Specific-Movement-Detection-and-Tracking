"""
TrackNova Video Tracking Workspace.

Primary computer vision workflow:
1. Upload or select a video dataset.
2. Inspect video metadata via OpenCV.
3. Configure YOLO model (Nano/Small/Medium or custom weights), confidence,
   class filter, preprocessing, and ByteTrack parameters.
4. Stream-process video frames with live progress and preview.
5. Dynamically select a detected tracking ID for movement & trajectory focus.
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
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
    create_video_writer,
    iter_video_frames,
    preprocess_frame,
    read_single_frame,
    save_uploaded_video,
    validate_and_inspect_video,
)
from modules.visualization import draw_tracking_frame, render_annotated_video
from utils.helpers import (
    format_duration,
    format_file_size,
    reset_tracking_session,
)
from utils.ui import render_kpi_card, render_page_header, show_friendly_error


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
    files = [
        p
        for p in sorted(DATASET_VIDEOS_DIR.iterdir())
        if p.is_file() and p.suffix.lower().lstrip(".") in SUPPORTED_VIDEO_EXTENSIONS
    ]
    return files


def render() -> None:
    render_page_header(
        "Video Tracking & Object Selection",
        "Upload a video dataset, run YOLO + ByteTrack detection, and select any tracking ID for motion intelligence",
        icon_name="video",
    )

    # ------------------------------------------------------------------
    # 1. Video Dataset Input
    # ------------------------------------------------------------------
    st.markdown(f"<h4 style='color:{COLORS['light_indigo']};margin-bottom:0.4rem;'>1. Video Dataset Input</h4>", unsafe_allow_html=True)
    in_col1, in_col2 = st.columns([1.2, 1])

    with in_col1:
        uploaded_file = st.file_uploader(
            "Upload Video Dataset (MP4, AVI, MOV, MKV, WEBM)",
            type=list(SUPPORTED_VIDEO_EXTENSIONS),
            key="vt_video_uploader",
        )
        if uploaded_file is not None:
            upload_sig = f"{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.get("_last_upload_sig") != upload_sig:
                try:
                    saved_path = save_uploaded_video(
                        uploaded_file.getvalue(), uploaded_file.name, DATASET_VIDEOS_DIR
                    )
                    meta = validate_and_inspect_video(saved_path)
                    reset_tracking_session(keep_dataset_history=True)
                    st.session_state["video_path"] = str(saved_path)
                    st.session_state["video_meta"] = meta
                    st.session_state["_last_upload_sig"] = upload_sig
                    st.success(f"Loaded uploaded video: {meta.file_name}")
                except Exception as exc:
                    show_friendly_error(f"Could not load uploaded video: {exc}", exc)

    with in_col2:
        existing_videos = _list_dataset_videos()
        existing_names = ["-- Select a saved dataset video --"] + [p.name for p in existing_videos]
        chosen_name = st.selectbox(
            "Or Select Dataset Video from `dataset/videos/`",
            options=existing_names,
            key="vt_existing_select",
        )
        if chosen_name != "-- Select a saved dataset video --":
            chosen_path = DATASET_VIDEOS_DIR / chosen_name
            if st.session_state.get("video_path") != str(chosen_path):
                try:
                    meta = validate_and_inspect_video(chosen_path)
                    reset_tracking_session(keep_dataset_history=True)
                    st.session_state["video_path"] = str(chosen_path)
                    st.session_state["video_meta"] = meta
                except Exception as exc:
                    show_friendly_error(f"Could not inspect video '{chosen_name}': {exc}", exc)

    video_meta = st.session_state.get("video_meta")
    video_path = st.session_state.get("video_path")

    if video_meta is None or not video_path:
        st.markdown(
            f"""
            <div class="tn-alert-info">
                Upload a video file above or place test videos inside <code>TrackNova/dataset/videos/</code> to begin.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ------------------------------------------------------------------
    # 2. Video Information & Original Preview
    # ------------------------------------------------------------------
    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='color:{COLORS['light_indigo']};margin-bottom:0.4rem;'>2. Original Video &amp; Metadata</h4>", unsafe_allow_html=True)

    meta_col, preview_col = st.columns([1.15, 1])
    with meta_col:
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            render_kpi_card("File Name", video_meta.file_name, format_file_size(video_meta.file_size_bytes))
        with r1c2:
            render_kpi_card("Resolution", f"{video_meta.width} x {video_meta.height}", f"Codec: {video_meta.fourcc_str}")
        with r1c3:
            fps_disp = f"{video_meta.fps:.2f}" if video_meta.fps_is_valid else "Invalid / 0"
            dur_disp = format_duration(video_meta.duration_sec) if video_meta.fps_is_valid else "N/A"
            render_kpi_card("FPS & Duration", fps_disp, f"Duration: {dur_disp} | Frames: {video_meta.total_frames}")

        if not video_meta.fps_is_valid:
            st.markdown(
                '<div class="tn-alert-warn">This video reports 0 or invalid FPS metadata. '
                'Tracking will still run, but pixel speed (px/s) requires valid FPS or timestamps.</div>',
                unsafe_allow_html=True,
            )

    with preview_col:
        with st.expander("Preview Original Video", expanded=False):
            try:
                st.video(str(video_path))
            except Exception as exc:
                show_friendly_error("Unable to play original video in browser widget; OpenCV frame reading can still proceed.", exc)
                frame0 = read_single_frame(video_path, 0)
                if frame0 is not None:
                    st.image(cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB), caption="First Frame Preview", use_container_width=True)

    # ------------------------------------------------------------------
    # 3. YOLO & ByteTrack Processing Settings
    # ------------------------------------------------------------------
    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='color:{COLORS['light_indigo']};margin-bottom:0.4rem;'>3. Detection, Tracking &amp; Preprocessing Settings</h4>", unsafe_allow_html=True)

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
                "Model Type: Pretrained COCO (80 classes: person, bicycle, car, motorcycle, bus, truck, etc.). "
                "Note: Standard COCO models do not detect drones; drone detection requires a custom-trained dataset/model."
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
            format_func=lambda s: f"Every frame (1)" if s == 1 else f"Every {s} frames",
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
        # Put common classes first for quick access while retaining all supported classes
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
    # 4. Action Controls (Start / Stop / Reset)
    # ------------------------------------------------------------------
    btn_col1, btn_col2, btn_col3 = st.columns([1.2, 1, 1])
    with btn_col1:
        start_clicked = st.button(
            "Start Video Processing",
            type="primary",
            use_container_width=True,
            disabled=(loaded_model is None),
            key="vt_start_btn",
        )
    with btn_col2:
        stop_clicked = st.button(
            "Stop Processing",
            type="secondary",
            use_container_width=True,
            key="vt_stop_btn",
        )
    with btn_col3:
        reset_clicked = st.button(
            "Reset Processing Session",
            type="secondary",
            use_container_width=True,
            key="vt_reset_btn",
        )

    if reset_clicked:
        reset_tracking_session(keep_dataset_history=True)
        st.rerun()

    if stop_clicked:
        # Finalize any partial run if interrupted mid-stream
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

        class_filter_ids = resolve_class_filter_ids(model_info.class_names, selected_classes)
        tracker = ByteTrackProcessor(
            model=loaded_model,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            class_filter_ids=class_filter_ids,
        )

        progress_bar = st.progress(0.0, text="Initializing video stream...")
        status_placeholder = st.empty()
        preview_placeholder = st.empty()

        frame_records: List[FrameTrackingRecord] = []
        class_counter: Counter = Counter()
        total_dets = 0
        untracked_dets = 0
        start_time = time.perf_counter()

        writer = None
        out_path: Optional[Path] = None
        codec_used: Optional[str] = None
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
                    video_path,
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
                        writer, out_path, codec_used = create_video_writer(
                            base_name=video_meta.file_name,
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
                    text=f"Processing frame {frame_idx + 1}/{video_meta.total_frames} ({cur_fps:.1f} FPS) — Detections: {total_dets}",
                )

                # Update live preview periodically to keep UI responsive
                if step_i == 0 or (step_i + 1) % 5 == 0 or (step_i + 1) == expected_steps:
                    preview_placeholder.image(
                        cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB),
                        caption=f"Live Tracking Preview — Frame {frame_idx}",
                        use_container_width=True,
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

        # Automatically select the longest-tracked object if available, or leave selectable
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

    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='color:{COLORS['light_indigo']};margin-bottom:0.4rem;'>4. Dynamic Tracking ID Selection &amp; Visualization</h4>", unsafe_allow_html=True)

    if session_result.total_detections == 0 or not session_result.unique_track_ids:
        st.markdown(
            '<div class="tn-alert-warn">No persistent ByteTrack object IDs were detected in this video under the current settings. '
            'Try lowering the confidence threshold or clearing class filters.</div>',
            unsafe_allow_html=True,
        )
        return

    # Table of all detected IDs
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

        if st.button("Clear Trajectory & Selection", type="secondary", use_container_width=True, key="vt_clear_sel"):
            _activate_selected_track_id(session_result, None)
            st.session_state["re_rendered_video_path"] = None
            st.rerun()

        msum = st.session_state.get("movement_summary")
        if msum and msum.get("has_data"):
            st.markdown(
                f"""
                <div class="tn-card" style="margin-top:0.65rem;">
                    <div style="color:{COLORS['trajectory_accent']};font-weight:700;margin-bottom:0.35rem;">
                        Active Target: ID #{msum['selected_id']} ({msum['object_class']})
                    </div>
                    <div style="color:{COLORS['secondary_text']};font-size:0.88rem;line-height:1.6;">
                        • Frames Tracked: <strong style="color:{COLORS['main_text']};">{msum['frames_tracked']}</strong><br/>
                        • Total Displacement: <strong style="color:{COLORS['main_text']};">{msum['total_displacement_px']:.2f} px</strong><br/>
                        • Avg / Max Speed: <strong style="color:{COLORS['main_text']};">{f"{msum['avg_speed_px_s']:.2f}" if msum['avg_speed_px_s'] is not None else "N/A"} / {f"{msum['max_speed_px_s']:.2f}" if msum['max_speed_px_s'] is not None else "N/A"} px/s</strong><br/>
                        • Current Direction: <strong style="color:{COLORS['light_indigo']};">{msum['current_direction']}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if msum.get("tracking_interruptions", 0) > 0:
                st.markdown(
                    f'<div class="tn-alert-warn">Selected object had {msum["tracking_interruptions"]} temporary tracking interruption(s) across the video.</div>',
                    unsafe_allow_html=True,
                )

    with tbl_col:
        st.markdown(f"<strong style='color:{COLORS['secondary_text']};'>Detected ByteTrack IDs</strong>", unsafe_allow_html=True)
        st.dataframe(id_df, use_container_width=True, height=250, hide_index=True)

    # Render / Preview Selected Object Overlay
    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)
    vis_col1, vis_col2 = st.columns([1, 1])

    with vis_col1:
        st.markdown(f"<h5 style='color:{COLORS['main_text']};'>Interactive Frame Inspector</h5>", unsafe_allow_html=True)
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
                    use_container_width=True,
                )

    with vis_col2:
        st.markdown(f"<h5 style='color:{COLORS['main_text']};'>Processed Video Playback</h5>", unsafe_allow_html=True)
        sel_id = st.session_state.get("selected_track_id")
        if sel_id is not None:
            if st.button(
                f"Render Highlighted Video for ID #{sel_id}",
                type="primary",
                use_container_width=True,
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
                st.video(vid_to_show)
                st.caption(f"Saved file: `{Path(vid_to_show).name}`")
            except Exception as exc:
                show_friendly_error(
                    "Browser playback failed for this codec, but the video file can still be downloaded from Reports & Export.",
                    exc,
                )
        else:
            st.info(
                "Click 'Render Highlighted Video' above to export and preview the video with the selected object's trajectory overlay."
            )
