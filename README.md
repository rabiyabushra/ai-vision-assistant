# 👁️ AI Vision Assistant for Visually Impaired People

A Deep Learning mini-project that uses computer vision and speech to provide
useful information about a user's surroundings.

## Current MVP

- Real-time webcam capture
- YOLO object detection
- Confidence scores and bounding boxes
- OCR using EasyOCR
- Text-to-speech
- Basic scene description
- Speech cooldown to reduce repeated announcements
- Streamlit interface
- CPU-friendly configuration for an i5 / 8 GB RAM laptop

## Architecture

Camera
→ OpenCV
→ YOLO Object Detection
→ Response Prioritization
→ Text-to-Speech

Camera
→ OCR
→ Text
→ Text-to-Speech

Camera
→ YOLO detections
→ Transparent rule-based Scene Description
→ Text-to-Speech

## Datasets

Planned evaluation sources:

1. MS COCO 2017 validation images — general object detection.
2. TextOCR validation annotations — text-recognition evaluation/reference.
3. Kaggle Indoor Objects Detection — indoor obstacle/object evaluation.
4. Kaggle Sidewalk Obstacle Detection — optional outdoor obstacle evaluation.

The application itself starts with pretrained models; the benchmark datasets are
not automatically used to retrain the models.

## Installation — Windows PowerShell

Open PowerShell in this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, you can run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run

```powershell
streamlit run app.py
```

The first YOLO run downloads the model weights automatically.
The first EasyOCR initialization also downloads its model files.

## Important hardware note

This project is configured for CPU inference:

- YOLO nano model
- 640px inference size
- OCR only when requested
- no custom model training in the MVP

This is intended to be practical on an 8 GB RAM laptop.

## Project structure

```text
AI_Vision_Assistant/
├── app.py
├── requirements.txt
├── requirements-optional.txt
├── README.md
├── .gitignore
├── modules/
│   ├── __init__.py
│   ├── object_detection.py
│   ├── ocr.py
│   ├── text_to_speech.py
│   ├── response_manager.py
│   └── scene_description.py
├── data/
│   ├── coco/
│   ├── textocr/
│   ├── indoor_objects/
│   └── sidewalk_obstacles/
├── models/
├── results/
└── notebooks/
```

## Safety

This is an assistive prototype. It can miss objects or make incorrect
detections. It must not be presented as a guaranteed navigation or collision-
avoidance system.

## Next development phases

1. Test webcam.
2. Test YOLO detection.
3. Test speech.
4. Test OCR.
5. Improve response prioritization.
6. Add optional voice commands.
7. Add quantitative evaluation.
8. Compare general-object and obstacle datasets.
9. Optimize FPS and response latency.
10. Prepare report, PPT and viva.
