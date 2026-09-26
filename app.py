import streamlit as st
import cv2
from modules.object_detection import ObjectDetector
from modules.ocr import OCRReader
from modules.scene_description import describe_scene
from modules.text_to_speech import Speaker
from modules.response_manager import ResponseManager

st.set_page_config(page_title="AI Vision Assistant", page_icon="👁️", layout="wide")

@st.cache_resource
def load_components():
    detector = ObjectDetector()
    ocr = OCRReader()
    speaker = Speaker()
    responder = ResponseManager()
    return detector, ocr, speaker, responder

st.title("👁️ AI Vision Assistant")
st.caption("Assistive computer-vision prototype for visually impaired users")

detector, ocr, speaker, responder = load_components()

if "running" not in st.session_state:
    st.session_state.running = False
if "last_text" not in st.session_state:
    st.session_state.last_text = ""
if "spoken" not in st.session_state:
    st.session_state.spoken = ""

col1, col2, col3 = st.columns(3)
with col1:
    start = st.button("▶ Start Camera", use_container_width=True)
with col2:
    stop = st.button("⏹ Stop Camera", use_container_width=True)
with col3:
    speech_on = st.checkbox("🔊 Speech", value=True)

if start:
    st.session_state.running = True
if stop:
    st.session_state.running = False

frame_placeholder = st.empty()
info_placeholder = st.empty()

read_text = st.button("📝 Read Text from Current Frame")
describe = st.button("🧠 Describe Current Scene")

if st.session_state.running:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        st.error("Could not open the webcam. Check camera permissions and close other camera applications.")
    else:
        st.info("Camera is running. Press Stop Camera to finish.")
        # Process a small number of frames per Streamlit rerun.
        ok, frame = camera.read()
        camera.release()

        if ok:
            result_frame, detections = detector.detect(frame)
            frame_placeholder.image(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB),
                                    channels="RGB", use_container_width=True)

            response = responder.object_response(detections)
            if response:
                st.session_state.spoken = response
                if speech_on:
                    speaker.speak(response)

            with info_placeholder.container():
                st.subheader("Detected objects")
                if detections:
                    for d in detections:
                        st.write(f"**{d['label']}** — {d['confidence']:.0%}")
                else:
                    st.write("No supported objects detected.")
                if st.session_state.spoken:
                    st.success(f"Assistant: {st.session_state.spoken}")
        else:
            st.error("Unable to read a frame from the webcam.")

if read_text:
    camera = cv2.VideoCapture(0)
    ok, frame = camera.read()
    camera.release()
    if ok:
        with st.spinner("Reading text..."):
            text = ocr.read(frame)
        st.subheader("OCR result")
        st.write(text if text else "No readable text detected.")
        if text and speech_on:
            speaker.speak(f"I can read: {text}")

if describe:
    camera = cv2.VideoCapture(0)
    ok, frame = camera.read()
    camera.release()
    if ok:
        with st.spinner("Analyzing scene..."):
            _, detections = detector.detect(frame)
            description = describe_scene(detections)
        st.subheader("Scene description")
        st.write(description)
        if speech_on:
            speaker.speak(description)

st.warning(
    "Safety note: This is a prototype assistive system. It should not be treated as a "
    "replacement for a guide, caregiver, or reliable navigation system. Object detections "
    "can be wrong or incomplete."
)
