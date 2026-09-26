"""
scene_description.py - Rule-Based Scene Understanding Module
Course: B.E. CSE Mini-Project — Deep Learning
Assigned to: Member A

================================================================================
CRITICAL ARCHITECTURAL DISTINCTION:
This module is STRICTLY A RULE-BASED, DETERMINISTIC HEURISTIC ENGINE.
It parses structured object detection lists (labels, bounding boxes, horizontal positions)
and applies geometric and contextual relational rules to compose natural language descriptions.

THIS IS NOT A GENERATIVE VISION-LANGUAGE MODEL (e.g., BLIP, LLaVA, Show-and-Tell).
It uses zero generative neural captioning to guarantee predictable, zero-hallucination,
and sub-millisecond execution on resource-constrained CPU hardware.
Exact distance in meters is NOT estimated without calibrated depth sensors.
================================================================================
"""

from typing import List, Dict, Optional


class RuleBasedSceneDescriber:
    """
    Deterministic rule-based scene reasoning engine for assistive guidance.
    Converts 2D object detections into human-interpretable contextual descriptions.
    """

    def __init__(self):
        # Known contextual group patterns
        self.workstation_items = {"laptop", "keyboard", "mouse", "desk", "table", "chair"}
        self.door_items = {"door", "openedDoor", "cabinetDoor", "refrigeratorDoor"}

    def describe(
        self,
        detections: List[Dict],
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None
    ) -> str:
        """
        Generate a concise, rule-based natural language description from detection outputs.

        Args:
            detections: List of detection dictionaries, each containing:
                        - 'label': str (e.g. 'chair', 'person', 'door')
                        - 'confidence': float
                        - 'bbox': [x1, y1, x2, y2]
                        - 'position': str (e.g. 'ahead of you', 'on your left', 'on your right')
            frame_width: Width of image in pixels (optional, for relative size checking).
            frame_height: Height of image in pixels (optional, for relative size checking).

        Returns:
            str: Deterministic, rule-based natural language sentence.
        """
        if not detections:
            return "Path appears clear. No supported objects detected."

        labels = [d["label"].lower() for d in detections]
        unique_labels = list(dict.fromkeys(labels))

        # Categorize detections by horizontal direction
        left_items = [d["label"] for d in detections if d.get("position") == "on your left"]
        center_items = [d["label"] for d in detections if d.get("position") == "ahead of you"]
        right_items = [d["label"] for d in detections if d.get("position") == "on your right"]

        # Rule Group 1: Immediate Pathway / Obstacle Alerts (Directly Ahead)
        immediate_hazards = [
            label for label in center_items
            if label in ["chair", "table", "pole", "cabinet", "couch"]
        ]

        # If objects span across multiple sectors (e.g., both left and right or ahead and left),
        # prioritize spatial directional guidance for assistive mobility
        sectors_active = sum([bool(left_items), bool(center_items), bool(right_items)])
        if sectors_active >= 2 and not ("person" in labels and "laptop" in labels):
            parts = []
            if center_items:
                center_summary = ", ".join(list(dict.fromkeys(center_items)))
                parts.append(f"{center_summary} ahead of you")
            if left_items:
                left_summary = ", ".join(list(dict.fromkeys(left_items)))
                parts.append(f"{left_summary} on your left")
            if right_items:
                right_summary = ", ".join(list(dict.fromkeys(right_items)))
                parts.append(f"{right_summary} on your right")
            return "Visible: " + "; ".join(parts) + "."

        # Rule Group 2: Contextual Compound Scene Rules (Semantic Co-occurrence)
        # Case A: Person working / seated at a table
        if "person" in labels and ("laptop" in labels or "table" in labels or "chair" in labels):
            has_laptop = "laptop" in labels
            has_table = "table" in labels or "chair" in labels
            if has_laptop and has_table:
                return "A person is at a table with a laptop."
            elif has_laptop:
                return "A person with a laptop is visible."
            else:
                return "A person is near a chair or table."

        # Case B: Doorway and Navigation State
        if "openeddoor" in labels:
            pos = next((d.get("position", "ahead of you") for d in detections if d["label"].lower() == "openeddoor"), "ahead of you")
            return f"An open doorway is detected {pos}."
        elif "door" in labels:
            pos = next((d.get("position", "ahead of you") for d in detections if d["label"].lower() == "door"), "ahead of you")
            return f"A closed door is {pos}."

        # Case C: Furniture / Seating area
        if ("chair" in labels or "couch" in labels) and "table" in labels:
            return "A seating arrangement with a table and chairs is visible."

        # Case D: Kitchenette / Office Storage
        if "refrigeratordoor" in labels or "refrigerator" in labels:
            return "A refrigerator is present in the scene."

        # Rule Group 3: Directional Guidance if single sector
        parts = []
        if center_items:
            center_summary = ", ".join(list(dict.fromkeys(center_items)))
            parts.append(f"{center_summary} directly ahead")
        if left_items:
            left_summary = ", ".join(list(dict.fromkeys(left_items)))
            parts.append(f"{left_summary} on your left")
        if right_items:
            right_summary = ", ".join(list(dict.fromkeys(right_items)))
            parts.append(f"{right_summary} on your right")

        if parts:
            return "Visible: " + "; ".join(parts) + "."

        # Rule Group 4: Fallback Single / Multi-Item Listing
        if len(unique_labels) == 1:
            return f"A {unique_labels[0]} is visible in the scene."

        return "Scene contains " + ", ".join(unique_labels) + "."


# Module-level convenience function preserving backward compatibility with app.py
_global_describer = RuleBasedSceneDescriber()

def describe_scene(detections: List[Dict]) -> str:
    """Module-level entry point called by app.py."""
    return _global_describer.describe(detections)


if __name__ == "__main__":
    print("=" * 65)
    print("AI Vision Assistant - Rule-Based Scene Understanding Test (Phase 6)")
    print("NOTE: Strictly rule-based heuristics -- NOT a generative neural captioner.")
    print("=" * 65)

    describer = RuleBasedSceneDescriber()

    # Scenario 1: Workstation compound
    scen1 = [
        {"label": "person", "confidence": 0.88, "bbox": [100, 100, 300, 400], "position": "ahead of you"},
        {"label": "table", "confidence": 0.75, "bbox": [80, 250, 400, 450], "position": "ahead of you"},
        {"label": "laptop", "confidence": 0.91, "bbox": [150, 200, 250, 280], "position": "ahead of you"}
    ]
    print(f"Scenario 1 (Person + Table + Laptop):")
    print(f" -> \"{describer.describe(scen1)}\"\n")

    # Scenario 2: Indoor Doorway Navigation
    scen2 = [
        {"label": "openedDoor", "confidence": 0.82, "bbox": [200, 50, 450, 500], "position": "ahead of you"}
    ]
    print(f"Scenario 2 (Open Door):")
    print(f" -> \"{describer.describe(scen2)}\"\n")

    # Scenario 3: Multi-directional layout
    scen3 = [
        {"label": "chair", "confidence": 0.79, "bbox": [20, 200, 150, 400], "position": "on your left"},
        {"label": "table", "confidence": 0.84, "bbox": [200, 200, 420, 450], "position": "ahead of you"},
        {"label": "door", "confidence": 0.71, "bbox": [500, 100, 620, 480], "position": "on your right"}
    ]
    print(f"Scenario 3 (Directional distribution):")
    print(f" -> \"{describer.describe(scen3)}\"\n")

    # Scenario 4: Empty scene
    print(f"Scenario 4 (Clear path):")
    print(f" -> \"{describer.describe([])}\"")
    print("=" * 65)
