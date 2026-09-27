# AI Vision Assistant - Presentation Slide Deck (PPT) Structure

---

## Slide Outline & Content Allocation

| Slide # | Slide Title | Primary Content & Visuals | Presenter |
| :---: | :--- | :--- | :---: |
| **1** | Title Slide | Project Title, Team Members (Member A & B), Course & Department Details | Both |
| **2** | Problem Statement & Motivation | Challenges faced by visually impaired individuals in indoor/outdoor navigation; need for real-time auditory assistance. | Member A |
| **3** | System Architecture | End-to-End Pipeline Diagram: Camera Stream ➔ YOLOv8 ➔ EasyOCR ➔ Rule-Based Scene ➔ Response Manager ➔ Non-blocking TTS. | Member B |
| **4** | Dataset Stack & Decisions | Pretrained COCO weights, Kaggle Indoor Objects fine-tuning set, TextOCR reference, and custom 50-100 test photo set. | Member A |
| **5** | Object Detection Subsystem | YOLOv8 Nano architecture, CPU optimization, spatial position calculation (Left / Ahead / Right). | Member A |
| **6** | OCR & Scene Reasoning | EasyOCR CRAFT+CRNN pipeline with CLAHE contrast enhancement; deterministic rule-based scene logic. | Member A |
| **7** | Response Management & Cooldown | Spatial prioritization (Hazards > Directions > Workstation), per-label cooldown, and duplicate speech suppression logic. | Member B |
| **8** | Interaction, TTS & Voice Layer | Threaded camera capture (`CameraStream`), non-blocking TTS worker thread (`Speaker`), voice/keyboard fallback (`SpeechRecognizer`). | Member B |
| **9** | Companion Streamlit UI | Two-column dashboard design (~65% video feed / ~35% audio output card), color-coded confidence pills, accessible layout. | Member B |
| **10** | Experimental Results & Metrics | Model metrics (mAP, Precision, Recall) + System performance (12-16 FPS, ~45ms latency, >85% speech suppression). | Both |
| **11** | Live System Demo | Video clip or live execution of `streamlit run app.py` demonstrating camera, detection, OCR, and speech. | Both |
| **12** | Safety, Ethics & Future Work | Prototype guardrails (zero hallucination, assistive notice); future expansion to wearable depth cameras and edge devices. | Both |
| **13** | Conclusion & Q&A | Summary of deliverables and open floor for examiner questions. | Both |
