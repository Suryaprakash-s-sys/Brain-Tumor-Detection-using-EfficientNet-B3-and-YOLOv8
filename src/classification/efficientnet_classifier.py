"""
EfficientNet-B3 Brain Tumor Classifier
Classifies MRI images into: Glioma, Meningioma, Pituitary Tumor, No Tumor
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
from pathlib import Path


CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]
IMG_SIZE = 300  # EfficientNet-B3 optimal input size


class EfficientNetB3Classifier(nn.Module):
    """EfficientNet-B3 model fine-tuned for brain tumor classification."""

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        self.model = models.efficientnet_b3(
            weights="IMAGENET1K_V1" if pretrained else None
        )
        # Replace classifier head
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class TumorClassifier:
    """High-level interface for brain tumor classification."""

    def __init__(self, weights: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EfficientNetB3Classifier(num_classes=len(CLASSES))

        if weights and Path(weights).exists():
            state = torch.load(weights, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"Loaded weights from {weights}")

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def predict(self, image_path: str) -> dict:
        """
        Predict tumor class from an MRI image.

        Args:
            image_path: Path to the MRI image file.

        Returns:
            dict with keys: label, confidence, probabilities
        """
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        pred_idx = int(np.argmax(probs))
        return {
            "label": CLASSES[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {cls: float(p) for cls, p in zip(CLASSES, probs)},
        }

    def predict_batch(self, image_paths: list) -> list:
        """Predict on a list of image paths."""
        return [self.predict(p) for p in image_paths]
