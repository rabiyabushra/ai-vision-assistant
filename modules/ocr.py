"""
ocr.py - EasyOCR Text Detection & Recognition Module
Course: B.E. CSE Mini-Project — Deep Learning
Assigned to: Member A

Handles optical character recognition from video frames or static images,
applying CLAHE contrast preprocessing, confidence thresholding, and text cleanup.
"""

from pathlib import Path
import re
import time
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
import easyocr

try:
    from utils.preprocessing import preprocess_for_ocr
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from utils.preprocessing import preprocess_for_ocr


class OCRReader:
    """
    Robust Optical Character Recognition (OCR) pipeline.
    Combines CRAFT text detection with a CRNN recognizer on CPU.
    """

    def __init__(self, languages: List[str] = None, gpu: bool = False, min_conf: float = 0.45):
        """
        Initialize the EasyOCR reader.

        Args:
            languages: List of language codes, defaults to ['en'].
            gpu: False for CPU execution on i5 laptop.
            min_conf: Minimum confidence threshold to accept recognized text.
        """
        if languages is None:
            languages = ["en"]
        self.languages = languages
        self.gpu = gpu
        self.min_conf = min_conf

        # Downloads or loads EasyOCR pretrained weights (CRAFT detector + ResNet/CRNN recognizer)
        self.reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=False)

    def clean_text(self, text: str) -> str:
        """
        Clean raw OCR text:
        - Strip extraneous leading/trailing symbols
        - Remove non-printable or isolated punctuation artifacts
        """
        cleaned = text.strip()
        # Discard isolated punctuation artifacts like '.', ',', '~', '|'
        if len(cleaned) == 1 and not cleaned.isalnum():
            return ""
        return cleaned

    def read(
        self,
        frame: np.ndarray,
        preprocess: bool = True
    ) -> Tuple[str, List[Dict], float]:
        """
        Extract text from an image frame.

        Args:
            frame: OpenCV BGR image numpy array.
            preprocess: Whether to apply CLAHE contrast enhancement before OCR.

        Returns:
            Tuple of:
            - combined_text: Cleaned, deduplicated string ready for Speech synthesis.
            - detailed_results: List of dicts [{'text', 'confidence', 'bbox'}].
            - latency_ms: Processing time in milliseconds.
        """
        t0 = time.perf_counter()

        # Image preprocessing
        if preprocess:
            processed = preprocess_for_ocr(frame)
        else:
            processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # Run CRAFT detection and CRNN recognition
        raw_results = self.reader.readtext(processed, detail=1, paragraph=False)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        filtered_results = []
        extracted_texts = []

        for bbox, text, confidence in raw_results:
            if confidence >= self.min_conf:
                cleaned = self.clean_text(text)
                if cleaned:
                    # Convert bounding box to flat integer format [[x1,y1], [x2,y2], ...]
                    int_bbox = [[int(pt[0]), int(pt[1])] for pt in bbox]
                    filtered_results.append({
                        "text": cleaned,
                        "confidence": round(float(confidence), 3),
                        "bbox": int_bbox
                    })
                    extracted_texts.append(cleaned)

        # Deduplicate sequential identical strings while preserving order
        unique_texts = list(dict.fromkeys(extracted_texts))
        combined_text = ". ".join(unique_texts) if unique_texts else ""

        return combined_text, filtered_results, latency_ms

    def read_image_file(
        self,
        image_path: str,
        save_annotated_to: Optional[str] = None
    ) -> Tuple[str, List[Dict], float]:
        """
        Convenience method to test OCR on a static image file on disk.

        Args:
            image_path: Path to image.
            save_annotated_to: Optional path to save visual overlay image.
        """
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Could not load image at path: {image_path}")

        combined_text, results, latency_ms = self.read(frame)

        if save_annotated_to:
            annotated = frame.copy()
            for item in results:
                pts = np.array(item["bbox"], np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                x, y = item["bbox"][0]
                label = f"{item['text']} ({item['confidence']:.2f})"
                cv2.putText(
                    annotated,
                    label,
                    (x, max(15, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )
            out_path = Path(save_annotated_to)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), annotated)

        return combined_text, results, latency_ms


if __name__ == "__main__":
    print("=" * 60)
    print("AI Vision Assistant - EasyOCR Module Test (Phase 5)")
    print("=" * 60)

    project_root = Path(__file__).resolve().parent.parent
    ocr_reader = OCRReader(min_conf=0.45)

    # Create a synthetic signage test image to verify OCR end-to-end immediately
    test_img = np.ones((300, 700, 3), dtype=np.uint8) * 240  # Off-white background
    # Add room sign text
    cv2.rectangle(test_img, (30, 30), (670, 270), (40, 40, 40), 3)
    cv2.putText(test_img, "ROOM 204", (70, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (20, 20, 180), 4)
    cv2.putText(test_img, "COMPUTER SCIENCE LAB", (70, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (30, 30, 30), 3)
    cv2.putText(test_img, "AUTHORIZED ENTRY ONLY", (70, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 30, 30), 2)

    temp_test_path = project_root / "data" / "custom_test_set" / "signs_text" / "sample_lab_sign.jpg"
    temp_test_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(temp_test_path), test_img)

    output_annotated_path = project_root / "results" / "ocr_examples" / "sample_lab_sign_ocr.jpg"

    print(f"Input Signboard Image : {temp_test_path.name}")
    print("Running EasyOCR pipeline (Detection -> Recognition -> Filtering)...")

    combined_text, items, latency_ms = ocr_reader.read_image_file(
        str(temp_test_path),
        save_annotated_to=str(output_annotated_path)
    )

    print("-" * 60)
    print(f"OCR Latency       : {latency_ms:.2f} ms ({1000.0 / latency_ms:.1f} FPS on CPU)")
    print(f"Recognized Blocks : {len(items)}")
    for i, it in enumerate(items, 1):
        print(f"  {i}. \"{it['text']}\" (confidence: {it['confidence']:.2f})")
    print("-" * 60)
    print(f"Final TTS Output  : \"{combined_text}\"")
    print(f"[OK] Annotated visual saved to: {output_annotated_path}")
    print("=" * 60)
