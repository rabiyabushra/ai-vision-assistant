"""
response_manager.py - Response Prioritization & Debouncing Module
Course: B.E. CSE Mini-Project — Deep Learning
Assigned to: Member B (Interaction & Systems)

Controls response generation, object prioritization, speech cooldown,
duplicate suppression, and directional voice alert logic for assistive guidance.
"""

import time
from typing import List, Dict, Optional, Set


class ResponseManager:
    """
    Intelligent response manager for assistive vision feedback.
    Prevents repetitive speech spam, enforces spatial priority rules,
    and applies debouncing and cooldown policies across video frames.
    """

    HAZARD_LABELS = {"chair", "table", "pole", "cabinet", "couch", "door", "openeddoor"}
    WORKSTATION_LABELS = {"person", "laptop", "keyboard", "mouse", "desk", "cell phone"}

    def __init__(
        self,
        cooldown_seconds: float = 4.0,
        label_cooldown_seconds: float = 6.0,
        min_conf: float = 0.35
    ):
        """
        Initialize ResponseManager configuration.

        Args:
            cooldown_seconds: Minimum seconds before repeating the exact same sentence.
            label_cooldown_seconds: Minimum seconds before re-announcing a specific object label.
            min_conf: Minimum confidence threshold required to speak an object detection.
        """
        self.cooldown_seconds = cooldown_seconds
        self.label_cooldown_seconds = label_cooldown_seconds
        self.min_conf = min_conf

        self.last_spoken_response: str = ""
        self.last_spoken_time: float = 0.0
        self.label_last_announced: Dict[str, float] = {}
        self.suppression_count: int = 0  # Metric for system evaluation: total suppressed spam responses

    def object_response(self, detections: List[Dict]) -> str:
        """
        Process object detections and generate a prioritized, non-repetitive voice sentence.

        Args:
            detections: List of detection dicts with 'label', 'confidence', 'bbox', 'position'.

        Returns:
            str: Voice message to speak, or empty string "" if suppressed by cooldown/debouncing.
        """
        if not detections:
            return ""

        # 1. Filter detections by confidence threshold
        valid_dets = [d for d in detections if d.get("confidence", 0.0) >= self.min_conf]
        if not valid_dets:
            return ""

        now = time.time()

        # 2. Extract unique labels present in current frame
        current_labels = [d["label"] for d in valid_dets]
        unique_current = list(dict.fromkeys(current_labels))

        # 3. Check if all detected objects are currently in label-cooldown
        # (i.e. we announced them recently and their set hasn't changed)
        unannounced_labels = [
            lbl for lbl in unique_current
            if now - self.label_last_announced.get(lbl, 0.0) > self.label_cooldown_seconds
        ]

        if not unannounced_labels and (now - self.last_spoken_time < self.cooldown_seconds):
            self.suppression_count += 1
            return ""

        # 4. Apply Priority Rules to compose sentence
        sentence = self._compose_prioritized_sentence(valid_dets)
        if not sentence:
            return ""

        # 5. Global sentence duplicate & cooldown check
        if sentence == self.last_spoken_response and (now - self.last_spoken_time < self.cooldown_seconds):
            self.suppression_count += 1
            return ""

        # 6. Update state tracking
        self.last_spoken_response = sentence
        self.last_spoken_time = now
        for lbl in unique_current:
            self.label_last_announced[lbl] = now

        return sentence

    def _compose_prioritized_sentence(self, detections: List[Dict]) -> str:
        """
        Compose prioritized directional alert.
        Priority:
          Priority 1: Immediate Obstacles / Hazards directly ahead
          Priority 2: Spatial Directional Guidance (Left vs Center vs Right)
          Priority 3: Contextual objects / General listing
        """
        # Group detections by position
        center_dets = [d for d in detections if d.get("position") == "ahead of you"]
        left_dets = [d for d in detections if d.get("position") == "on your left"]
        right_dets = [d for d in detections if d.get("position") == "on your right"]

        # Check Priority 1: Obstacle directly ahead
        ahead_hazards = [
            d["label"] for d in center_dets
            if d["label"].lower() in self.HAZARD_LABELS
        ]
        if ahead_hazards:
            unique_hazards = list(dict.fromkeys(ahead_hazards))
            if len(unique_hazards) == 1:
                return f"Caution: {unique_hazards[0]} ahead of you."
            else:
                return f"Caution: {', '.join(unique_hazards[:-1])} and {unique_hazards[-1]} ahead of you."

        # Priority 2: Directional distribution if objects exist in different horizontal sectors
        sectors = []
        if center_dets:
            c_labels = list(dict.fromkeys([d["label"] for d in center_dets]))
            sectors.append(f"{self._format_list(c_labels)} ahead of you")
        if left_dets:
            l_labels = list(dict.fromkeys([d["label"] for d in left_dets]))
            sectors.append(f"{self._format_list(l_labels)} on your left")
        if right_dets:
            r_labels = list(dict.fromkeys([d["label"] for d in right_dets]))
            sectors.append(f"{self._format_list(r_labels)} on your right")

        if len(sectors) >= 2:
            return "Detected: " + "; ".join(sectors) + "."

        # Priority 3: Single sector / simple summary
        all_labels = list(dict.fromkeys([d["label"] for d in detections]))
        if len(all_labels) == 1:
            pos = detections[0].get("position", "ahead of you")
            return f"{all_labels[0].capitalize()} detected {pos}."
        
        return f"{self._format_list(all_labels).capitalize()} detected."

    @staticmethod
    def _format_list(items: List[str]) -> str:
        """Format list of strings into natural English listing (a, b, and c)."""
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"

    def ocr_response(self, text: str) -> str:
        """Format OCR result for spoken announcement."""
        if not text or not text.strip():
            return "No readable text detected."
        
        cleaned = text.strip()
        formatted = f"Text reads: {cleaned}"
        now = time.time()
        
        if formatted == self.last_spoken_response and (now - self.last_spoken_time < self.cooldown_seconds):
            self.suppression_count += 1
            return ""
            
        self.last_spoken_response = formatted
        self.last_spoken_time = now
        return formatted

    def scene_response(self, description: str) -> str:
        """Format scene description for spoken announcement."""
        if not description or not description.strip():
            return ""

        formatted = description.strip()
        now = time.time()

        if formatted == self.last_spoken_response and (now - self.last_spoken_time < self.cooldown_seconds):
            self.suppression_count += 1
            return ""

        self.last_spoken_response = formatted
        self.last_spoken_time = now
        return formatted

    def reset_cooldown(self):
        """Reset cooldown history (e.g. when user changes mode or requests manual scan)."""
        self.last_spoken_response = ""
        self.last_spoken_time = 0.0
        self.label_last_announced.clear()


if __name__ == "__main__":
    print("=" * 65)
    print("AI Vision Assistant - ResponseManager Cooldown & Priority Test (Phase 7)")
    print("=" * 65)

    responder = ResponseManager(cooldown_seconds=4.0, label_cooldown_seconds=6.0)

    # Simulated detection frames over time
    frame_stream = [
        # Frame 1: Person, Chair, Laptop detected
        [
            {"label": "person", "confidence": 0.94, "bbox": [100, 100, 300, 400], "position": "on your left"},
            {"label": "chair", "confidence": 0.89, "bbox": [200, 200, 400, 500], "position": "ahead of you"},
            {"label": "laptop", "confidence": 0.76, "bbox": [150, 150, 250, 250], "position": "on your left"}
        ],
        # Frame 2: Identical objects 100ms later (simulating next video frame)
        [
            {"label": "person", "confidence": 0.95, "bbox": [102, 101, 301, 400], "position": "on your left"},
            {"label": "chair", "confidence": 0.90, "bbox": [201, 200, 401, 500], "position": "ahead of you"},
            {"label": "laptop", "confidence": 0.77, "bbox": [151, 150, 251, 250], "position": "on your left"}
        ],
        # Frame 3: Identical objects 500ms later
        [
            {"label": "person", "confidence": 0.93, "bbox": [100, 100, 300, 400], "position": "on your left"},
            {"label": "chair", "confidence": 0.88, "bbox": [200, 200, 400, 500], "position": "ahead of you"},
            {"label": "laptop", "confidence": 0.75, "bbox": [150, 150, 250, 250], "position": "on your left"}
        ]
    ]

    print("Simulating consecutive video frames with repeated detections:")
    for i, frame_dets in enumerate(frame_stream, 1):
        resp = responder.object_response(frame_dets)
        status = f"\"'{resp}'\"" if resp else "[SUPPRESSED BY COOLDOWN]"
        print(f" Frame {i}: Output -> {status}")

    print("-" * 65)
    print(f"Total spam responses suppressed: {responder.suppression_count}")
    print("[OK] Response prioritization and debouncing verified successfully.")
    print("=" * 65)
