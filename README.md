# 👁️ AI Vision Assistant for Visually Impaired People

An assistive computer-vision prototype that uses real-time object detection, Optical Character Recognition (OCR), rule-based scene understanding, and text-to-speech feedback to aid visually impaired individuals.

---

## 🌟 System Overview & Features

- **Real-Time Webcam Streaming**: Thread-buffered video frame capture (`CameraStream`).
- **YOLOv8 Object Detection**: Pretrained COCO model (`yolov8n.pt`) with fine-tuning support for indoor objects (`yolov8n_indoor_finetuned.pt`).
- **OCR Text Reading**: EasyOCR integration with contrast enhancement (`preprocess_for_ocr`) for reading signage and printed text.
- **Rule-Based Scene Description**: Transparent, deterministic spatial reasoning without generative neural hallucination.
- **Intelligent Response Manager**: Spatial prioritization (Hazards > Directions > Workstation > General), debouncing, per-label cooldown, and duplicate speech suppression.
- **Non-Blocking Text-To-Speech (TTS)**: Background worker thread (`Speaker`) using `pyttsx3` with dynamic enable/disable controls.
- **Voice Commands & Keyboard Fallback**: Microphone command recognition (`SpeechRecognizer`) with regex word-boundary keyboard fallback.
- **Two-Column Companion Dashboard**: 65%/35% Streamlit UI with live annotated video feed, color-coded confidence pills, and audio output monitoring.
- **CPU-Optimized Performance**: Configured specifically for Intel Core i5 / 8 GB RAM laptops without dedicated GPUs.

---

## 🏗️ End-to-End System Architecture

```text
Camera Capture Stream (modules/camera.py)
   │
   ├──────► YOLOv8 Object Detector (modules/object_detection.py)
   │           │
   │           ├──────► Spatial Position Resolver (utils/helpers.py)
   │           │
   │           └──────► Bounding Box OpenCV Annotator
   │
   ├──────► EasyOCR Text Reader (modules/ocr.py)
   │           │
   │           └──────► CLAHE Contrast Preprocessing (utils/preprocessing.py)
   │
   └──────► Rule-Based Scene Reasoning Engine (modules/scene_description.py)
               │
               ▼
   Response Prioritization & Cooldown Manager (modules/response_manager.py)
               │
               ▼
   Non-Blocking Text-To-Speech Worker Thread (modules/text_to_speech.py)
               │
               ▼
   Voice Output / Audio Feedback to User
```

---

## 👥 Two-Person Work Split

### Member A — Vision & Models (Detection, OCR, Scene Understanding, Training)
- Fine-tuning YOLOv8n on Indoor Objects dataset in Google Colab (`training/train_colab.ipynb`).
- Pretrained vs fine-tuned model comparative analysis (mAP, Precision, Recall, Inference Time).
- `modules/object_detection.py` (YOLO inference, NMS thresholding, position resolution).
- `modules/ocr.py` (EasyOCR pipeline, confidence filtering, text cleaning).
- `modules/scene_description.py` (Rule-based deterministic scene reasoning).
- Dataset organization (`data/`, `training/`).

### Member B — Interaction & Systems (Speech, UI, Integration, Performance, Testing)
- `modules/camera.py` (Threaded frame capture, resolution configuration, lifecycle management).
- `modules/text_to_speech.py` (Non-blocking pyttsx3 queue worker, speech toggle, rate control).
- `modules/response_manager.py` (Spatial priority rules, per-label cooldown, repeat speech suppression).
- `modules/speech_to_text.py` (Voice command listener & regex word-boundary keyboard fallback).
- `app.py` (Streamlit 2-column companion dashboard, colored confidence pills, UI controls).
- `utils/performance.py` (System FPS, response latency, and speech suppression instrumentation).
- System integration, testing, documentation, architecture diagrams, and Viva preparation.

---

## 📂 Project Directory Structure

```text
AI_Vision_Assistant/
├── app.py                          # Streamlit UI Dashboard (Member B)
├── requirements.txt                # Core dependencies
├── requirements-optional.txt       # Voice recognition optional dependencies
├── README.md                       # Comprehensive documentation & architecture
├── .gitignore                      # Git ignore rules
│
├── modules/
│   ├── __init__.py
│   ├── camera.py                   # CameraStream hardware wrapper (Member B)
│   ├── object_detection.py         # YOLOv8 object detector (Member A)
│   ├── ocr.py                      # EasyOCR text reader (Member A)
│   ├── scene_description.py        # Rule-based scene reasoning engine (Member A)
│   ├── speech_to_text.py           # Speech & keyboard command recognizer (Member B)
│   ├── text_to_speech.py           # Non-blocking TTS speaker (Member B)
│   └── response_manager.py         # Priority & cooldown manager (Member B)
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py            # Image CLAHE contrast enhancement & letterboxing
│   ├── helpers.py                  # Spatial positioning & timer utilities
│   └── performance.py              # Real-time FPS, latency & speech instrumentation (Member B)
│
├── models/
│   ├── yolov8n.pt                  # Stock COCO pretrained weights
│   └── yolov8n_indoor_finetuned.pt # Fine-tuned indoor objects weights
│
├── training/                       # Fine-tuning dataset & Colab notebook
│   ├── indoor_objects/             # Kaggle Indoor Objects dataset (10 classes)
│   │   ├── data.yaml
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│   ├── train_colab.ipynb           # Colab notebook for GPU training
│   └── training_results/           # Exported Colab metrics & confusion matrix
│
├── data/
│   ├── coco_sample/                # Small val2017 validation subset
│   ├── textocr/                    # TextOCR reference annotations
│   └── custom_test_set/            # Real-world test photos (50-100 samples)
│       ├── indoor/
│       ├── outdoor/
│       └── signs_text/
│
├── results/                        # Evaluation metrics & sample detections
│   ├── system_performance.json     # Measured system-level metrics
│   ├── detection_metrics.csv
│   └── sample_detections/
│
└── docs/                           # Documentation & Viva prep
    ├── PROJECT_PLAN.md             # Project phase status & milestones
    ├── DATASET_PLAN.md             # Dataset decisions & specifications
    ├── VIVA_NOTES.md               # Team Viva preparation Q&A guide
    └── report/
        └── ppt_structure.md        # Presentation slide deck outline
```

---

## ⚡ Quick Start & Run Instructions

### 1. Navigate to Project Directory & Environment Setup (Windows PowerShell)

Open PowerShell in the workspace folder and enter the project folder:

```powershell
cd ai-vision-assistant
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

*(Optional voice recognition dependencies can be installed via `pip install -r requirements-optional.txt`)*

### 2. Run Streamlit Application

```powershell
# Make sure you are inside the ai-vision-assistant folder:
streamlit run app.py
```

*Alternatively, if running from the parent workspace root:*

```powershell
streamlit run ai-vision-assistant/app.py
```

- Click **▶ Start Camera** to initiate live video processing.
- Click **🔍 Detect**, **📝 Read Text**, or **🧠 Describe Scene** for manual scans.
- Type voice commands in the keyboard input box or click **🎙️ Listen Command**.

---

## 📊 Measured System Performance

*Measured on Intel Core i5 CPU, 8 GB RAM:*

| Metric | Measured Value | Verification Status |
| :--- | :--- | :--- |
| **Object Detection Latency** | ~45 - 65 ms / frame | Empirical Measurement |
| **Processing Frame Rate** | 12 - 16 FPS | Empirical Measurement |
| **OCR Latency** | ~1.8 - 3.2 s / scan | Empirical Measurement (On Demand) |
| **Speech Spam Suppression Rate** | > 85% duplicate frames suppressed | Empirical Measurement |
| **TTS Queue Latency** | < 10 ms (Non-blocking worker thread) | Empirical Measurement |

---

## 🛡️ Assistive Prototype Guardrails & Ethics

1. **Rule-Based Scene Description**: Scene captions are generated using deterministic spatial rules, NOT generative neural networks. This guarantees zero hallucination.
2. **Relative Distance Estimation**: Distances are communicated as relative directions (`ahead of you`, `on your left`, `on your right`) rather than exact meters without depth sensors.
3. **Assistive Prototype Notice**: This system is designed as an assistive research prototype and must NOT be used as a replacement for guide dogs, human mobility aids, or certified medical safety equipment.
