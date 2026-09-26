"""
object_detection.py - YOLOv8 Real-Time Object Detection Module
Course: B.E. CSE Mini-Project — Deep Learning
Assigned to: Member A

Handles YOLO model initialization, CPU inference, bounding box extraction,
and spatial position resolution for assistive guidance.
"""

from pathlib import Path
import time
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from ultralytics import YOLO

try:
    from utils.helpers import get_horizontal_position
except ImportError:
    # Fallback if imported from another working directory
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from utils.helpers import get_horizontal_position


class ObjectDetector:
    """
    Object detection engine using YOLOv8 Nano architecture.
    Optimized for CPU-only inference on Intel Core i5 with 8GB RAM.
    """

    def __init__(
        self,
        model_path: str = "models/yolov8n.pt",
        conf: float = 0.35,
        iou: float = 0.45,
        device: str = "cpu"
    ):
        """
        Initialize the YOLO detector.

        Args:
            model_path: Path to .pt weights file (relative to project root or absolute).
            conf: Confidence threshold (default 0.35).
            iou: Non-Maximum Suppression (NMS) IoU threshold.
            device: 'cpu' for local laptop execution.
        """
        self.device = device
        self.conf = conf
        self.iou = iou

        # Resolve model path relative to project root if needed
        resolved_path = Path(model_path)
        if not resolved_path.is_absolute() and not resolved_path.exists():
            # Check relative to AI_Vision_Assistant project root
            project_root = Path(__file__).resolve().parent.parent
            candidate = project_root / model_path
            if candidate.exists():
                resolved_path = candidate

        self.model_path = str(resolved_path)
        self.model = YOLO(self.model_path)

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict], float]:
        """
        Run object detection on a single image frame (BGR format).

        Args:
            frame: OpenCV BGR image numpy array.

        Returns:
            Tuple of:
            - annotated_frame: Frame with drawn bounding boxes and labels.
            - detections: List of dicts with keys ['label', 'confidence', 'bbox', 'position'].
            - latency_ms: Inference time in milliseconds.
        """
        t0 = time.perf_counter()

        # Run YOLO inference
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            imgsz=640,
            device=self.device
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        result = results[0]
        annotated = result.plot()
        frame_width = frame.shape[1]

        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                coords = [float(x) for x in box.xyxy[0].tolist()]

                # Determine relative horizontal direction for assistive voice feedback
                position = get_horizontal_position(coords, frame_width)

                detections.append({
                    "label": names[cls_id],
                    "confidence": round(confidence, 3),
                    "bbox": [round(c, 1) for c in coords],
                    "position": position
                })

        return annotated, detections, latency_ms

    def detect_image_file(
        self,
        image_path: str,
        save_annotated_to: Optional[str] = None
    ) -> Tuple[List[Dict], float]:
        """
        Convenience method to test inference on a static image file on disk.

        Args:
            image_path: Path to input image (.jpg, .png).
            save_annotated_to: Optional path to save visual detection output.
        """
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Could not load image at path: {image_path}")

        annotated, detections, latency_ms = self.detect(frame)

        if save_annotated_to:
            out_path = Path(save_annotated_to)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), annotated)

        return detections, latency_ms


if __name__ == "__main__":
    import os
    print("=" * 60)
    print("AI Vision Assistant - YOLOv8n Static Image Test (Phase 3)")
    print("=" * 60)

    project_root = Path(__file__).resolve().parent.parent
    model_file = project_root / "models" / "yolov8n.pt"

    # Select a static test image from data/coco_sample or indoor test set
    sample_dir = project_root / "data" / "coco_sample"
    sample_images = list(sample_dir.glob("*.jpg"))

    if not sample_images:
        # Fallback to indoor test images if coco_sample is empty
        sample_dir = project_root / "training" / "indoor_objects" / "test" / "images"
        sample_images = list(sample_dir.glob("*.jpg"))

    if not sample_images:
        print("[!] No sample images found to test.")
        sys.exit(1)

    test_image_path = sample_images[0]
    output_image_path = project_root / "results" / "sample_detections" / "baseline_detection_test.jpg"

    print(f"Loading Model: {model_file.name}")
    print(f"Test Image   : {test_image_path.name}")

    detector = ObjectDetector(model_path=str(model_file), conf=0.35, device="cpu")

    # Warmup run (CPU cache warm-up)
    dummy_frame = cv2.imread(str(test_image_path))
    _ = detector.detect(dummy_frame)

    # Timed inference run
    detections, latency_ms = detector.detect_image_file(
        str(test_image_path),
        save_annotated_to=str(output_image_path)
    )

    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

    print("-" * 60)
    print(f"Inference Latency: {latency_ms:.2f} ms ({fps:.1f} FPS on CPU)")
    print(f"Detected Objects : {len(detections)}")
    for i, d in enumerate(detections, 1):
        print(f"  {i}. [{d['label']}] (conf: {d['confidence']:.2f}) -> {d['position']} | bbox: {d['bbox']}")

    print("-" * 60)
    print(f"[OK] Annotated detection image saved to: {output_image_path}")
    print("=" * 60)
