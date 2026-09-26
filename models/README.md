# Models Directory

This directory contains the YOLO object detection model weights:

1. **`yolov8n.pt`**:
   - Pretrained COCO baseline model (80 classes, ~6.5MB).
   - Used as the starting checkpoint for fine-tuning and baseline quantitative evaluation.

2. **`yolov8n_indoor_finetuned.pt`**:
   - *Placeholder* (Generated after running Google Colab training in Phase 3b).
   - Trained on the 10-class Indoor Objects dataset.
   - Download `best.pt` from Google Colab (`runs/detect/indoor_yolov8n/weights/best.pt`) and rename it to `yolov8n_indoor_finetuned.pt` here.
