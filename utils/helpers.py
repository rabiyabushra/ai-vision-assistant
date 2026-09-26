"""
helpers.py - General Helper Utilities
Assigned to: Member A

Helper functions for bounding box conversions, spatial positioning,
timing/latency tracking, and visualization formatting.
"""
import time
from typing import List, Dict, Tuple


class LatencyTimer:
    """Simple timer context manager to record execution latency in milliseconds."""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed_ms = (self.end - self.start) * 1000.0


def get_horizontal_position(bbox: List[float], frame_width: int) -> str:
    """
    Determine approximate horizontal sector of a detected object:
    'left', 'center', or 'right'.
    Useful for assistive spatial guidance.
    """
    x1, _, x2, _ = bbox
    center_x = (x1 + x2) / 2.0
    relative_x = center_x / max(frame_width, 1)

    if relative_x < 0.35:
        return "on your left"
    elif relative_x > 0.65:
        return "on your right"
    else:
        return "ahead of you"


def format_detections_summary(detections: List[Dict]) -> str:
    """Format detection list into a clean readable summary string."""
    if not detections:
        return "No objects detected."
    lines = [f"- {d['label']}: confidence {d['confidence']:.2f}" for d in detections]
    return "\n".join(lines)
