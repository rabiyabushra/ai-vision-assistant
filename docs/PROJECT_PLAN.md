# AI Vision Assistant - Project Phase Timeline & Ownership Status

---

## 📅 Phase-by-Phase Ownership Matrix

| Phase | Task Description | Owner | Status | Notes |
| :---: | :--- | :---: | :---: | :--- |
| **1** | Environment setup (`venv`, `requirements.txt`) | Both | **Completed** | PyTorch, Ultralytics, EasyOCR, pyttsx3, Streamlit ready |
| **2** | Camera capture test & threading wrapper | **Member B** | **Completed** | `modules/camera.py` (`CameraStream`) implemented & tested |
| **3** | YOLO detection (pretrained baseline) | Member A | **Completed** | `modules/object_detection.py` running YOLOv8n on CPU |
| **3b** | Fine-tune YOLO on Indoor Objects dataset | Member A | Pending (Colab) | Training in Google Colab GPU environment |
| **4** | Text-To-Speech output (non-blocking thread) | **Member B** | **Completed** | `modules/text_to_speech.py` (`Speaker`) non-blocking queue |
| **5** | OCR working (EasyOCR + CLAHE) | Member A | **Completed** | `modules/ocr.py` text detection & cleaning |
| **6** | Rule-based deterministic scene description | Member A | **Completed** | `modules/scene_description.py` zero-hallucination logic |
| **7** | Response prioritization + speech cooldown | **Member B** | **Completed** | `modules/response_manager.py` debouncing & priority rules |
| **8** | Voice commands + keyboard fallback | **Member B** | **Completed** | `modules/speech_to_text.py` voice & regex keyword parser |
| **9** | Streamlit UI companion dashboard | **Member B** | **Completed** | `app.py` 2-column layout, confidence pills, controls |
| **10** | Full pipeline end-to-end integration | Both | **Completed** | Camera ➔ Detector ➔ OCR ➔ Scene ➔ Response ➔ TTS |
| **11** | Testing on custom test photos | Both | **Completed** | Verified pipeline execution on indoor & signboard sets |
| **12** | Quantitative evaluation & metrics logging | Both | **In Progress** | Member A: Detection mAP \| Member B: System FPS & Latency |
| **13** | Performance optimization & frame skipping | **Member B** | **Completed** | Thread buffering, CPU tuning, latency logging |
| **14** | Report, README, PPT structure, Viva prep | Both | **Completed** | Documentation, PPT outline, Member B Viva prep guide |

---

## 🎯 Member B Key Milestones Delivered
1. **Threaded Camera Stream (`modules/camera.py`)**: OpenCV VideoCapture wrapper with background thread buffering to prevent video stream latency lag on CPU hardware.
2. **Non-Blocking TTS Speaker (`modules/text_to_speech.py`)**: Async queue worker thread preventing Streamlit UI locks and Windows COM threading crashes.
3. **Response Prioritization Engine (`modules/response_manager.py`)**: Hazard alerts, spatial directions, per-label cooldown, and duplicate speech suppression.
4. **Voice & Keyboard Command Recognizer (`modules/speech_to_text.py`)**: Microphone voice listener with regex word boundary keyboard fallback.
5. **Streamlit UI Dashboard (`app.py`)**: Two-column layout (~65% / ~35%), OpenCV bounding box rendering, confidence pills, and audio output monitor.
6. **System Performance Instrumentation (`utils/performance.py`)**: Real-time FPS, response latency, and speech suppression rate measurement.
