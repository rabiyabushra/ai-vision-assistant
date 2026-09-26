"""
camera.py - Camera and Frame Capture Module
Assigned to: Member B

Handles video stream capture, camera hardware initialization, and frame buffering.
"""
import cv2


class CameraStream:
    """Wrapper for camera hardware capture (webcam or external USB camera)."""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None

    def start(self) -> bool:
        """Initialize camera capture device. Returns True if opened successfully."""
        self.cap = cv2.VideoCapture(self.camera_index)
        return self.cap.isOpened()

    def get_frame(self):
        """Read a single frame (BGR numpy array) from the camera."""
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

    def release(self):
        """Release the camera device resources."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
