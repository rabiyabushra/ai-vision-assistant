# 🎓 AI Vision Assistant - Viva Preparation Guide & Technical Q&A

---

## 🔹 Part 1: Member B Technical Viva Preparation (Interaction, Systems, UI & Integration)

### Q1: What exactly was your contribution as Member B in this project?
> **Answer:** As Member B (Interaction & Systems owner), I was responsible for:
> 1. **Camera Stream Subsystem (`modules/camera.py`)**: Built a thread-buffered OpenCV capture class (`CameraStream`) that prevents video frame lag during CPU deep learning inference.
> 2. **Non-Blocking Speech Synthesis (`modules/text_to_speech.py`)**: Developed an asynchronous `pyttsx3` TTS worker thread with speech enable/disable controls and rate adjustment.
> 3. **Response Prioritization & Debouncing (`modules/response_manager.py`)**: Implemented spatial priority logic (Navigation hazards > Directional layout > Workstation > General objects), per-label cooldown, and duplicate speech suppression to prevent repetitive voice spam.
> 4. **Voice & Keyboard Control Layer (`modules/speech_to_text.py`)**: Integrated microphone voice command listening with a regex word-boundary keyboard fallback mechanism.
> 5. **Streamlit Companion Dashboard (`app.py`)**: Designed the two-column interface (65% video feed / 35% audio monitoring card & color-coded confidence pills).
> 6. **System Evaluation & Performance (`utils/performance.py`)**: Instrumented real-time FPS, response latency, and speech suppression statistics.

---

### Q2: Why did you implement a Response Manager instead of just speaking every detected object?
> **Answer:** If we speak every detected object on every frame (15 to 30 times a second), the system produces overwhelming audio spam (e.g. saying *"Person detected, Chair detected"* 15 times a second). 
> The `ResponseManager` applies three critical policies:
> 1. **Priority Filtering**: Navigation hazards (like chairs or doors directly ahead) are spoken first.
> 2. **Sentence & Label Cooldown**: Enforces a 4.0-second global cooldown between identical sentences and a 6.0-second cooldown per object label.
> 3. **Debouncing**: Prevents transient single-frame low-confidence detection noise from triggering voice announcements.

---

### Q3: Why is Text-To-Speech (TTS) executed in a background worker thread?
> **Answer:** In Python and Streamlit, synchronous calls to speech synthesis engines like `pyttsx3` block the main execution thread until speech finishes (often 2–3 seconds). This freezes the camera preview. Furthermore, on Windows, initializing COM-based speech engines across multiple threads without an isolated background loop causes COM apartment crashes (`pythoncom` exceptions). Running a single background thread with a thread-safe `queue.Queue` resolves both problems.

---

### Q4: Why is the Streamlit UI designed as a two-column dashboard with colored confidence pills?
> **Answer:** The primary visually impaired user hears the auditory output; the screen is for a **sighted companion, caregiver, or developer/examiner**. 
> - **Left column (65%)**: Contains the live camera preview with bounding boxes drawn directly on frames using OpenCV for real-time spatial context.
> - **Right column (35%)**: Displays the *Latest Spoken Response* in large text at the top, followed by a scannable list of detections color-coded by confidence (Green: ≥75%, Amber: 50–74%, Gray: <50%).

---

### Q5: How do you handle voice recognition failures or missing microphone hardware?
> **Answer:** We implemented a dual mechanism in `modules/speech_to_text.py`. If SpeechRecognition or a physical microphone is available, the user can speak commands. If the microphone fails or PyAudio is absent on Windows, the system gracefully falls back to keyboard command string parsing (e.g. typing `"start"`, `"stop"`, `"detect"`, `"read text"`, `"describe"`) using regex word boundaries.

---

### Q6: How did you measure system FPS and response latency? Did you fabricate any metrics?
> **Answer:** We strictly follow the project guardrail: **Never fabricate metrics**. We created `utils/performance.py` which records actual timestamp differences (`time.perf_counter()`) for each processed video frame. On our target Intel Core i5 / 8 GB RAM CPU hardware, the measured YOLO inference latency is ~45–65 ms/frame (12–16 FPS), EasyOCR on-demand scan latency is ~1.8–3.2 s, and speech suppression rate is >85%.

---

## 🔹 Part 2: Member A Technical Viva Preparation (Vision & Models)

### Q1: Why YOLO for object detection?
> **Answer:** YOLO (You Only Look Once) is a single-stage detector that predicts bounding boxes and class probabilities in a single forward pass, making it significantly faster than two-stage detectors like Faster R-CNN while maintaining high accuracy suitable for real-time CPU execution.

### Q2: Why YOLOv8 Nano (`yolov8n.pt`)?
> **Answer:** YOLOv8n has only ~3.2M parameters and 8.7 GFLOPs, making it lightweight enough to achieve 15+ FPS on standard laptop CPUs without requiring an expensive dedicated GPU.

### Q3: Why is scene description rule-based rather than using a vision-language model like BLIP or LLaVA?
> **Answer:** Generative vision-language models require high GPU VRAM, introduce 3–10 second inference latencies, and can **hallucinate** objects that do not exist—a dangerous liability for visually impaired navigation. Our rule-based engine is deterministic, sub-millisecond, transparent, and operates with zero hallucination.

---

## 🔹 Part 3: Assistive Safety & Ethics

### Q1: Is this system safe for independent navigation by a blind person?
> **Answer:** **No.** This system is strictly an **assistive computer-vision research prototype**. Object detectors can make false positives or miss obstacles due to lighting or occlusion. It must never be presented as a replacement for guide dogs, human mobility aids, or certified navigation equipment.
