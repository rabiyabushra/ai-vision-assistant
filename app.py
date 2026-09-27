"""
app.py - AI Vision Assistant Streamlit User Interface
Course: B.E. CSE Mini-Project — Deep Learning
Assigned to: Member B (Interaction & Systems)

Two-column companion dashboard for visually-impaired support system.
Integrates Camera Stream, YOLOv8 Object Detection, EasyOCR, Rule-Based Scene Description,
Response Prioritization Manager, Non-blocking Text-To-Speech, and Speech/Keyboard Commands.
"""

import time
import cv2
import numpy as np
import streamlit as st

# Import project modules
from modules.camera import CameraStream
from modules.object_detection import ObjectDetector
from modules.ocr import OCRReader
from modules.scene_description import describe_scene
from modules.text_to_speech import Speaker
from modules.response_manager import ResponseManager
from modules.speech_to_text import SpeechRecognizer
from utils.performance import SystemPerformanceTracker

# Page Setup - Accessible, Wide Layout
st.set_page_config(
    page_title="AI Vision Assistant",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# Load heavy deep learning and system models once via Streamlit cache
@st.cache_resource
def initialize_system_components():
    detector = ObjectDetector(model_path="models/yolov8n.pt", conf=0.35, device="cpu")
    ocr_reader = OCRReader(min_conf=0.45)
    speaker = Speaker(rate=160, enabled=True)
    response_mgr = ResponseManager(cooldown_seconds=4.0, label_cooldown_seconds=6.0, min_conf=0.35)
    speech_rec = SpeechRecognizer()
    perf_tracker = SystemPerformanceTracker()
    camera_stream = CameraStream(camera_index=0, width=640, height=480)
    return detector, ocr_reader, speaker, response_mgr, speech_rec, perf_tracker, camera_stream


# Global initialization
detector, ocr_reader, speaker, response_mgr, speech_rec, perf_tracker, camera_stream = initialize_system_components()

# Session State Variables
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False
if "latest_spoken_text" not in st.session_state:
    st.session_state.latest_spoken_text = "Assistant ready. Click 'Start Camera' or use controls below."
if "current_detections" not in st.session_state:
    st.session_state.current_detections = []
if "speech_enabled" not in st.session_state:
    st.session_state.speech_enabled = True
if "manual_trigger" not in st.session_state:
    st.session_state.manual_trigger = None
if "voice_command_status" not in st.session_state:
    st.session_state.voice_command_status = ""
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "last_annotated_frame" not in st.session_state:
    st.session_state.last_annotated_frame = None


# Header Section
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("👁️ AI Vision Assistant")
    st.caption("Assistive Computer-Vision Prototype for Visually Impaired Navigation & Guidance")

with col_head2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.session_state.camera_running:
        st.markdown("#### 🟢 **Camera Active**")
    else:
        st.markdown("#### 🔴 **Camera Idle**")

st.divider()

# PDF Layout Specification: Two-Column Dashboard (Wider Left ~65%, Right ~35%)
left_col, right_col = st.columns([1.6, 1.0], gap="large")

# ==============================================================================
# LEFT COLUMN (~65%): Camera Live Feed + Primary Control Bar
# ==============================================================================
with left_col:
    st.subheader("Live Camera View")
    
    # Persistent Placeholder for Video Frame (prevents layout shifting)
    video_placeholder = st.empty()
    
    # Primary Action Buttons Bar directly beneath video feed
    btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = st.columns(5)
    
    with btn_col1:
        start_btn = st.button("▶ Start Camera", width="stretch", type="primary")
    with btn_col2:
        stop_btn = st.button("⏹ Stop Camera", width="stretch")
    with btn_col3:
        detect_btn = st.button("🔍 Detect", width="stretch")
    with btn_col4:
        ocr_btn = st.button("📝 Read Text", width="stretch")
    with btn_col5:
        scene_btn = st.button("🧠 Describe Scene", width="stretch")

    # Command input box (Keyboard Fallback)
    cmd_text = st.text_input("⌨️ Keyboard Voice-Command Fallback (Type 'start', 'stop', 'detect', 'read text', 'describe'):", key="cmd_input")
    if cmd_text:
        action = speech_rec.parse_command(cmd_text)
        if action:
            st.session_state.manual_trigger = action
            st.session_state.voice_command_status = f"Parsed command: '{action}'"

# ==============================================================================
# RIGHT COLUMN (~35%): Stacked Cards (Response, Detections, Voice Controls)
# ==============================================================================
with right_col:
    # --------------------------------------------------------------------------
    # Card 1: Latest Spoken Response (Large & Scannable for Companion/Tester)
    # --------------------------------------------------------------------------
    st.markdown("### 📢 Spoken Audio Output")
    spoken_box = st.empty()
    
    def update_spoken_card(text_str):
        spoken_box.markdown(
            f"""
            <div style="background-color: #e3f2fd; padding: 14px 18px; border-radius: 8px; 
                        border-left: 5px solid #1976d2; margin-bottom: 15px;">
                <span style="font-size: 0.9em; color: #1565c0; font-weight: bold; text-transform: uppercase;">Latest Response</span>
                <p style="font-size: 1.15em; font-weight: 600; color: #0d47a1; margin: 6px 0 0 0;">"{text_str}"</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    update_spoken_card(st.session_state.latest_spoken_text)

    # --------------------------------------------------------------------------
    # Card 2: Current Detections List with Confidence Badges
    # --------------------------------------------------------------------------
    st.markdown("### 📦 Detected Objects")
    detections_placeholder = st.empty()

    def render_detections_panel_stable(dets):
        """Build single HTML string for detections panel with fixed scrollable container to prevent UI glitching."""
        if not dets:
            detections_placeholder.markdown(
                """
                <div style="padding: 12px; background-color: #fafafa; border-radius: 6px; border: 1px solid #eee; color: #777; font-style: italic;">
                    No objects detected in current frame.
                </div>
                """,
                unsafe_allow_html=True
            )
            return

        # Sort detections by confidence descending
        sorted_dets = sorted(dets, key=lambda x: x.get("confidence", 0.0), reverse=True)
        
        items_html = []
        for d in sorted_dets:
            lbl = d.get("label", "Object")
            conf = d.get("confidence", 0.0)
            pos = d.get("position", "ahead of you")
            conf_pct = int(conf * 100)

            # Color-coded confidence pills (Green >= 75%, Amber >= 50%, Gray < 50%)
            if conf >= 0.75:
                pill_color = "#2e7d32"  # Green
                badge_bg = "#e8f5e9"
            elif conf >= 0.50:
                pill_color = "#f57f17"  # Amber
                badge_bg = "#fffde7"
            else:
                pill_color = "#616161"  # Gray
                badge_bg = "#f5f5f5"

            items_html.append(
                f"""
                <div style="display: flex; justify-content: space-between; align-items: center; 
                            padding: 6px 12px; margin-bottom: 6px; background-color: {badge_bg}; 
                            border-radius: 6px; border-left: 4px solid {pill_color};">
                    <div>
                        <strong>{lbl.capitalize()}</strong> <span style="font-size: 0.85em; color: #555;">({pos})</span>
                    </div>
                    <div style="background-color: {pill_color}; color: white; padding: 2px 8px; 
                                border-radius: 12px; font-weight: bold; font-size: 0.85em;">
                        {conf_pct}%
                    </div>
                </div>
                """
            )

        container_html = f"""
        <div style="max-height: 240px; overflow-y: auto; padding-right: 4px;">
            {''.join(items_html)}
        </div>
        """
        detections_placeholder.markdown(container_html, unsafe_allow_html=True)

    render_detections_panel_stable(st.session_state.current_detections)

    st.markdown("---")

    # --------------------------------------------------------------------------
    # Card 3: Speech & Voice Controls
    # --------------------------------------------------------------------------
    st.markdown("### ⚙️ Speech & Voice Controls")

    speech_toggle = st.checkbox("🔊 Enable Text-To-Speech Output", value=st.session_state.speech_enabled)
    if speech_toggle != st.session_state.speech_enabled:
        st.session_state.speech_enabled = speech_toggle
        speaker.set_enabled(speech_toggle)
        if not speech_toggle:
            speaker.stop_speech()

    listen_col, status_col = st.columns([1, 1])
    with listen_col:
        listen_btn = st.button("🎙️ Listen Command", width="stretch")
    with status_col:
        st.caption(speech_rec.get_status_description())

    if listen_btn:
        with st.spinner("Listening for voice command..."):
            text_rec, action_found = speech_rec.listen_voice_command(timeout=4)
            if action_found:
                st.session_state.manual_trigger = action_found
                st.success(f"Heard: '{text_rec}' -> Action: {action_found}")
            elif text_rec:
                st.warning(f"Heard: '{text_rec}' (No action matched)")
            else:
                st.info("No speech detected or microphone unavailable.")

    if st.session_state.voice_command_status:
        st.caption(st.session_state.voice_command_status)

# ==============================================================================
# BUTTON ACTION LOGIC & CAMERA FEED LOOP
# ==============================================================================

# Handle Start / Stop Camera triggers
if start_btn or st.session_state.manual_trigger == "start":
    st.session_state.camera_running = True
    st.session_state.manual_trigger = None
    if not camera_stream.is_opened():
        if not camera_stream.start():
            st.error("Could not open camera device 0. Please verify camera permissions.")
            st.session_state.camera_running = False

if stop_btn or st.session_state.manual_trigger == "stop":
    st.session_state.camera_running = False
    st.session_state.manual_trigger = None
    camera_stream.stop()

# Handle Manual OCR button or Voice command 'read_text'
if ocr_btn or st.session_state.manual_trigger == "read_text":
    st.session_state.manual_trigger = None
    ok, frame = camera_stream.get_frame()
    if not ok or frame is None:
        temp_cam = cv2.VideoCapture(0)
        ok, frame = temp_cam.read()
        temp_cam.release()

    if ok and frame is not None:
        with st.spinner("Reading text from frame..."):
            text, ocr_dets, latency_ms = ocr_reader.read(frame)
        voice_text = response_mgr.ocr_response(text)
        if voice_text:
            st.session_state.latest_spoken_text = voice_text
            update_spoken_card(voice_text)
            if st.session_state.speech_enabled:
                speaker.speak(voice_text)

# Handle Manual Scene Description button or Voice command 'describe_scene'
if scene_btn or st.session_state.manual_trigger == "describe_scene":
    st.session_state.manual_trigger = None
    ok, frame = camera_stream.get_frame()
    if not ok or frame is None:
        temp_cam = cv2.VideoCapture(0)
        ok, frame = temp_cam.read()
        temp_cam.release()

    if ok and frame is not None:
        with st.spinner("Analyzing scene layout..."):
            _, dets, _ = detector.detect(frame)
            desc = describe_scene(dets)
        voice_text = response_mgr.scene_response(desc)
        if voice_text:
            st.session_state.latest_spoken_text = voice_text
            update_spoken_card(voice_text)
            if st.session_state.speech_enabled:
                speaker.speak(voice_text)
            st.session_state.current_detections = dets
            render_detections_panel_stable(dets)

# Handle Manual Detect button trigger
if detect_btn or st.session_state.manual_trigger == "detect":
    st.session_state.manual_trigger = None
    response_mgr.reset_cooldown()

# Main Continuous Video Stream Loop when Camera is Running
if st.session_state.camera_running:
    if not camera_stream.is_opened():
        if not camera_stream.start():
            st.error("Could not open camera device 0.")
            st.session_state.camera_running = False

    # Smooth continuous frame rendering loop
    while st.session_state.camera_running:
        ok, frame = camera_stream.get_frame()
        if ok and frame is not None:
            t0 = time.perf_counter()
            st.session_state.frame_count += 1
            
            # Frame Skipping Optimization: Run YOLO detection every 2nd frame
            if st.session_state.frame_count % 2 == 1 or st.session_state.last_annotated_frame is None:
                annotated_frame, detections, latency_ms = detector.detect(frame)
                st.session_state.last_annotated_frame = annotated_frame
                st.session_state.current_detections = detections
            else:
                annotated_frame = st.session_state.last_annotated_frame
                detections = st.session_state.current_detections
                latency_ms = 0.0

            # Update stable detections HTML panel in place
            render_detections_panel_stable(detections)

            # Process detections through ResponseManager
            voice_response = response_mgr.object_response(detections)
            spoke_event = False
            suppressed_event = (voice_response == "" and len(detections) > 0)

            if voice_response:
                st.session_state.latest_spoken_text = voice_response
                update_spoken_card(voice_response)
                spoke_event = True
                if st.session_state.speech_enabled:
                    speaker.speak(voice_response)

            # Render annotated video frame smoothly
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, channels="RGB", width="stretch")

            # Record System Metrics
            frame_time_ms = (time.perf_counter() - t0) * 1000.0
            perf_tracker.record_frame(latency_ms=frame_time_ms, spoke=spoke_event, suppressed=suppressed_event)

            # 30ms sleep for smooth 30 FPS video streaming without script rerun locks
            time.sleep(0.03)
        else:
            time.sleep(0.05)

st.markdown("---")

# Accessibility & Safety Footnote
st.warning(
    "⚠️ **Assistive System Notice:** This software is an experimental computer-vision prototype. "
    "It must NOT be used as a replacement for guide dogs, human assistance, or certified navigation systems. "
    "Detections and descriptions may occasionally be wrong or incomplete."
)
