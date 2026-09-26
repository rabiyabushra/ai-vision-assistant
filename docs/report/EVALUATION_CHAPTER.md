# Project Report — Evaluation and Results Chapter
**Project**: AI Vision Assistant for Visually Impaired People  
**Course**: B.E. Computer Science and Engineering — 7th Semester Mini-Project  
**Author**: Member A (Detection, OCR, Scene Understanding, Model Training, Dataset Organization, Evaluation)  
**Target Hardware for Inference**: Intel Core i5, 8 GB RAM, CPU-only (No local dedicated GPU)

---

## Section 21: Quantitative Object Detection Evaluation

### 21.1 Experimental Methodology & Metrics
To evaluate object detection performance on resource-constrained edge hardware, two models are evaluated:
1. **Baseline Model**: Pretrained YOLOv8n (`yolov8n.pt`, 80 COCO classes).
2. **Fine-Tuned Domain Model**: YOLOv8n fine-tuned on the 10-class Indoor Objects dataset (`yolov8n_indoor_finetuned.pt`).

Performance is quantified using standard computer vision evaluation metrics:
- **Precision ($P$)**: Ratio of true positive detections over all positive predictions:
  $$P = \frac{TP}{TP + FP}$$
- **Recall ($R$)**: Ratio of true positive detections over all ground truth objects:
  $$R = \frac{TP}{TP + FN}$$
- **$\text{mAP}@50$**: Mean Average Precision calculated at an Intersection-over-Union (IoU) threshold of $0.50$.
- **$\text{mAP}@50\text{--}95$**: Mean Average Precision averaged over 10 IoU thresholds from $0.50$ to $0.95$ with a step size of $0.05$.
- **Inference Latency ($t_{\text{inf}}$)**: Average execution time per frame in milliseconds on Intel Core i5 CPU.
- **Frames Per Second ($\text{FPS}$)**: Throughput on CPU, computed as $\frac{1000}{t_{\text{inf}}}$.

### 21.2 Comparative Performance Table

> [!NOTE]
> Values marked as **"to be measured"** will be updated directly from the completed Google Colab test split validation once fine-tuning is completed. Measured values reflect actual benchmarks captured on the local Intel Core i5 test machine.

| Model Checkpoint | Evaluation Dataset | Precision ($P$) | Recall ($R$) | $\text{mAP}@50$ | $\text{mAP}@50\text{--}95$ | Inference Time (CPU) | Throughput (FPS) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv8n Baseline (COCO)** | Sample test image (`000000000139.jpg`) | *to be measured* | *to be measured* | *to be measured* | *to be measured* | **80.59 ms** | **12.4 FPS** |
| **YOLOv8n Baseline (COCO)** | COCO val sample (100 images) | *to be measured* | *to be measured* | *to be measured* | *to be measured* | *to be measured* | *to be measured* |
| **YOLOv8n Fine-Tuned** | Indoor Objects Test Set (107 images) | *to be measured* | *to be measured* | *to be measured* | *to be measured* | *to be measured* | *to be measured* |
| **YOLOv8n Fine-Tuned** | Custom Test Set (Indoor campus photos) | *to be measured* | *to be measured* | *to be measured* | *to be measured* | *to be measured* | *to be measured* |

### 21.3 Confusion Matrix & Per-Class Error Analysis
*Target Location for Artifact*: `results/confusion_matrix.png` (exported from Colab training run).
- Evaluates class confusions among visually similar indoor categories (e.g., distinguishing `cabinetDoor` from `door`, or `chair` from `couch`).
- Background false positive rate is verified to ensure empty corridors do not trigger false hazard alarms.

---

## Section 22: OCR & Text Recognition Performance

### 22.1 Two-Stage Pipeline Overview
Text extraction is performed via **EasyOCR**, combining a deep convolutional text detector (**CRAFT**) with a sequence recognizer (**ResNet + BiLSTM + CTC**):
1. **Preprocessing (CLAHE)**: Local contrast equalization improves edge gradients of signage letters under poor or glare-heavy indoor lighting.
2. **Confidence Filtering**: Any detected word with $\text{confidence} < 0.45$ is discarded. This suppresses false character hallucinations caused by door wood grains and wall tile seams.
3. **Text Cleanup**: Whitespace is trimmed, isolated non-alphanumeric punctuation artifacts are eliminated, and duplicate detections are unified.

### 22.2 Measured Benchmark on Signage Sample
*Sample Image*: `data/custom_test_set/signs_text/sample_lab_sign.jpg`  
*Visual Artifact*: `results/ocr_examples/sample_lab_sign_ocr.jpg`

- **Processing Latency on CPU**: $\approx 1,354\text{ ms}$ ($\approx 0.74\text{ FPS}$).
- **Detected Text Blocks**:
  1. `"ROOM 204"` — Confidence: $1.00$
  2. `"COMPUTER SCIENCE LAB"` — Confidence: $0.79$
  3. `"AUTHORIZED ENTRY ONLY"` — Confidence: $0.99$
- **Generated Speech String**: `"ROOM 204. COMPUTER SCIENCE LAB. AUTHORIZED ENTRY ONLY"`
- **Analysis**: Because OCR execution takes $> 1\text{ second}$ on CPU, it is intentionally decoupled from real-time live navigation and triggered asynchronously on-demand when the user requests text reading.

---

## Section 23: Scene Understanding & Architectural Analysis

### 23.1 Rule-Based vs. Generative Captioning Architecture
> [!IMPORTANT]
> **Explicit Architectural Distinction**: The scene understanding module in this project is **STRICTLY RULE-BASED AND DETERMINISTIC**. It is **NOT** a generative neural captioning model (such as BLIP, LLaVA, or Show-and-Tell).

**Rationale for Rule-Based Reasoning**:
1. **Elimination of Hallucinations**: Generative Vision-Language Models (VLMs) frequently hallucinate objects that do not exist in the frame. For a visually impaired user navigating physical space, hallucinating an open doorway or failing to report an obstacle poses safety hazards.
2. **Computational Feasibility**: Large Vision-Language Models require $> 6\text{ GB}$ of VRAM or several seconds of CPU latency per token. Our rule-based module executes in $< 1\text{ ms}$ on CPU with zero additional RAM footprint.
3. **Explainability**: Every generated sentence is traceable to specific detected bounding boxes and spatial coordinates.

### 23.2 Spatial Sector Partitioning
Objects are mapped to relative horizontal sectors based on bounding box horizontal centroids ($x_{\text{center}} = \frac{x_1 + x_2}{2}$):
- $\frac{x_{\text{center}}}{W} < 0.35 \implies$ **"on your left"**
- $0.35 \le \frac{x_{\text{center}}}{W} \le 0.65 \implies$ **"ahead of you"**
- $\frac{x_{\text{center}}}{W} > 0.65 \implies$ **"on your right"**

*Metric Distance Disclaimer*: Exact metric distances (e.g. "2.4 meters away") are intentionally not asserted, as single monocular 2D RGB cameras cannot compute calibrated depth without stereoscopic cameras, LiDAR, or hardware-calibrated depth sensors.

### 23.3 Demonstrated Rule Scenarios
- **Compound Co-occurrence**: `person` + `table` + `laptop` $\implies$ *"A person is at a table with a laptop."*
- **Navigation Waypoint**: `openedDoor` $\implies$ *"An open doorway is detected ahead of you."*
- **Multi-Object Directional Guidance**: `chair` (left) + `table` (center) + `door` (right) $\implies$ *"Visible: table ahead of you; chair on your left; door on your right."*
- **Clear Pathway**: No detected objects $\implies$ *"Path appears clear. No supported objects detected."*
