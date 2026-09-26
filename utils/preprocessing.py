"""
preprocessing.py - Image Preprocessing Utilities
Assigned to: Member A

Functions for frame resizing, normalization, contrast enhancement (CLAHE),
and OCR image cleaning.
"""
import numpy as np


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for text extraction:
    - Converts to grayscale if 3-channel
    - Applies subtle contrast stretching
    """
    # Note: OpenCV is imported lazily to avoid import crashes before opencv-python is installed
    import cv2

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced


def resize_letterbox(image: np.ndarray, target_size: int = 640) -> np.ndarray:
    """
    Resize image preserving aspect ratio with black padding (letterbox),
    standard for YOLO model inputs.
    """
    import cv2

    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    nw, nh = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    top = (target_size - nh) // 2
    left = (target_size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas
