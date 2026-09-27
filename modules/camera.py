"""
camera.py - Camera and Frame Capture Module
Course: B.E. CSE Mini-Project — Deep Learning
Assigned to: Member B (Interaction & Systems)

Handles video stream capture, camera hardware initialization, frame buffering,
and device cleanup optimized for real-time CPU processing.
"""

import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np


class CameraStream:
    """
    Webcam video stream capture wrapper.
    Uses a background thread to continuously buffer the latest frame,
    preventing OpenCV frame buffer lag during CPU inference.
    """

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        """
        Initialize camera settings.

        Args:
            camera_index: OpenCV VideoCapture device index (0 for default webcam).
            width: Target frame width in pixels.
            height: Target frame height in pixels.
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    def start(self) -> bool:
        """
        Initialize the camera capture device and start frame reading thread.

        Returns:
            bool: True if camera initialized successfully, False otherwise.
        """
        if self.running and self.cap is not None and self.cap.isOpened():
            return True

        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW if cv2.os.name == 'nt' else cv2.CAP_ANY)
        if not self.cap.isOpened():
            # Fallback without specific API backend if CAP_DSHOW fails
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            return False

        # Set capture parameters
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        return True

    def _update_loop(self):
        """Background thread loop to continuously read fresh frames."""
        while self.running and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._latest_frame = frame
            else:
                time.sleep(0.01)

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get the most recent video frame.

        Returns:
            Tuple of (success_flag, frame_bgr_array).
        """
        with self._lock:
            if self._latest_frame is not None:
                return True, self._latest_frame.copy()

        # Fallback to direct read if thread hasn't captured yet
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._latest_frame = frame
                return True, frame
        return False, None

    def is_opened(self) -> bool:
        """Check if camera device is opened and active."""
        return self.running and self.cap is not None and self.cap.isOpened()

    def stop(self):
        """Stop frame reading thread and release camera resources cleanly."""
        self.running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None

        if self.cap is not None:
            if self.cap.isOpened():
                self.cap.release()
            self.cap = None

        with self._lock:
            self._latest_frame = None

    def release(self):
        """Alias for stop()."""
        self.stop()


if __name__ == "__main__":
    print("=" * 60)
    print("AI Vision Assistant - CameraStream Test (Phase 2)")
    print("=" * 60)

    stream = CameraStream(camera_index=0, width=640, height=480)
    print("Attempting to open camera index 0...")

    if stream.start():
        print("[OK] Camera started successfully.")
        time.sleep(0.5)  # Allow buffer to fill

        success, frame = stream.get_frame()
        if success and frame is not None:
            print(f"[OK] Captured frame with shape: {frame.shape}")
        else:
            print("[!] Camera started but failed to retrieve frame.")

        stream.stop()
        print("[OK] Camera released cleanly.")
    else:
        print("[!] Could not open camera (No camera attached or occupied).")

    print("=" * 60)
