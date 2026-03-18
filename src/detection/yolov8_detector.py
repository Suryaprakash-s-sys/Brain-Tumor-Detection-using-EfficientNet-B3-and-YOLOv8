"""
YOLOv8 Brain Tumor Detector
Localizes tumor regions in MRI images using bounding boxes.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """Represents a single tumor detection result."""
    label: str
    confidence: float
    bbox: List[float]          # [x1, y1, x2, y2] in pixel coords
    cropped_image: Optional[np.ndarray] = field(default=None, repr=False)


class TumorDetector:
    """High-level interface for YOLOv8-based brain tumor detection."""

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
    ):
        self.model = YOLO(weights)
        self.conf = conf_threshold
        self.iou = iou_threshold

    def predict(self, image_path: str) -> List[Detection]:
        """
        Run tumor detection on an MRI image.

        Args:
            image_path: Path to the MRI image.

        Returns:
            List of Detection objects.
        """
        results = self.model(
            image_path,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )

        image = cv2.imread(image_path)
        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label = result.names[int(box.cls)]
                confidence = float(box.conf)
                cropped = image[y1:y2, x1:x2]

                detections.append(Detection(
                    label=label,
                    confidence=confidence,
                    bbox=[x1, y1, x2, y2],
                    cropped_image=cropped,
                ))

        return detections

    def visualize(self, image_path: str, output_path: str = None) -> np.ndarray:
        """
        Draw bounding boxes on image and optionally save result.

        Args:
            image_path: Path to input MRI image.
            output_path: If provided, saves the annotated image here.

        Returns:
            Annotated image as numpy array.
        """
        detections = self.predict(image_path)
        image = cv2.imread(image_path)

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            label = f"{det.label} {det.confidence:.0%}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

        if output_path:
            cv2.imwrite(output_path, image)

        return image
