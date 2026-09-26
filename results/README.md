# Results Directory

This directory stores evaluation metrics, output charts, sample visual detections, and OCR predictions:

- `detection_metrics.csv`: Comparative table of Pretrained vs Fine-tuned models (latency, FPS, mAP@50, mAP@50-95, precision, recall).
- `confusion_matrix.png`: Fine-tuned indoor model confusion matrix (to be copied from Colab training run).
- `fps_comparison.png`: Latency and throughput bar chart on CPU (i5, 8GB RAM).
- `sample_detections/`: Annotated images output by YOLOv8n illustrating correct detections, spatial positioning, and edge cases.
- `ocr_examples/`: Cropped / annotated frames showing EasyOCR bounding boxes and extracted text strings.
