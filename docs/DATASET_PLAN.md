# AI Vision Assistant - Final Dataset Decision & Strategy Plan

---

## 1. Official Dataset Resolutions (Source of Truth)

| File / Source | Contents & Description | Project Plan Verdict |
| :--- | :--- | :--- |
| **`archive__3_.zip`** | Indoor Objects Detection (Roboflow), 10 classes, YOLO format, train/valid/test split | **Keep & Extract** — Fine-tuning dataset for custom indoor classes |
| **`archive__4_.zip`** | Identical copy of `archive__3_.zip` | **Delete Duplicate Copy** — Keep only `archive__3_.zip` |
| **`textOCR_0_1_val.json`** | TextOCR JSON annotations (OpenImages dataset) | **Keep as Reference** — Format reference / EasyOCR sanity check |
| **Custom Test Set (50-100 Photos)** | Real photos captured across indoor rooms, outdoor paths, and signboards | **Primary Demo & Report Test Set** — Real-world evaluation |

---

## 2. Final Dataset Stack

1. **COCO Pretrained Weights (`models/yolov8n.pt`)**:
   - Ships pretrained on COCO's 80 general object classes (person, chair, laptop, bottle, backpack, etc.).
   - Used for stock baseline detection without downloading full 18 GB image archive.

2. **Indoor Objects Dataset (`training/indoor_objects/`)**:
   - Extracted from `archive__3_.zip` into `training/indoor_objects/` (1,012 training images).
   - Used to fine-tune `yolov8n.pt` for indoor-specific classes: `door`, `window`, `cabinet`, `couch`, `table`, `chair`, `pole`.
   - Fine-tuning executed in Google Colab (`training/train_colab.ipynb`) for 15–30 epochs to generate `models/yolov8n_indoor_finetuned.pt`.

3. **TextOCR Reference JSON (`data/textocr/TextOCR_0_1_val.json`)**:
   - Kept as a format reference for text annotation structure.
   - Core OCR execution uses EasyOCR (CRAFT + CRNN) directly on input images.

4. **Custom 50-100 Real Photo Set (`data/custom_test_set/`)**:
   - Split into `indoor/`, `outdoor/`, and `signs_text/`.
   - Used for end-to-end integration testing, companion UI verification, and final presentation demo.
