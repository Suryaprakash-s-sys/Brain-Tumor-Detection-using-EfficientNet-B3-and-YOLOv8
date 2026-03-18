"""
Unit tests for Brain Tumor Detection system.
Run with: pytest tests/
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── Preprocessing Tests ───────────────────────────────────────────────────────

class TestPreprocessing:
    def test_normalize_range(self):
        from utils.preprocessing import normalize
        img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        result = normalize(img)
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        assert result.dtype == np.float32

    def test_resize_output_shape(self):
        from utils.preprocessing import resize_image
        img = np.zeros((100, 150, 3), dtype=np.uint8)
        resized = resize_image(img, (300, 300))
        assert resized.shape == (300, 300, 3)

    def test_anisotropic_filter_preserves_shape(self):
        from utils.preprocessing import anisotropic_diffusion_filter
        img = np.random.rand(64, 64, 3).astype(np.float32)
        filtered = anisotropic_diffusion_filter(img, num_iter=3)
        assert filtered.shape == img.shape
        assert filtered.min() >= 0.0
        assert filtered.max() <= 1.0


# ── Classifier Tests ──────────────────────────────────────────────────────────

class TestEfficientNetClassifier:
    def test_model_instantiation(self):
        from classification.efficientnet_classifier import EfficientNetB3Classifier, CLASSES
        model = EfficientNetB3Classifier(num_classes=len(CLASSES), pretrained=False)
        assert model is not None

    def test_model_forward_shape(self):
        import torch
        from classification.efficientnet_classifier import EfficientNetB3Classifier, CLASSES
        model = EfficientNetB3Classifier(num_classes=len(CLASSES), pretrained=False)
        model.eval()
        dummy = torch.randn(2, 3, 300, 300)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (2, len(CLASSES))

    def test_classes_defined(self):
        from classification.efficientnet_classifier import CLASSES
        assert "glioma" in CLASSES
        assert "meningioma" in CLASSES
        assert "pituitary" in CLASSES
        assert "no_tumor" in CLASSES
        assert len(CLASSES) == 4


# ── Detector Tests ────────────────────────────────────────────────────────────

class TestTumorDetector:
    def test_detection_result_structure(self):
        from detection.yolov8_detector import Detection
        det = Detection(
            label="glioma",
            confidence=0.91,
            bbox=[10, 20, 100, 120],
        )
        assert det.label == "glioma"
        assert 0.0 <= det.confidence <= 1.0
        assert len(det.bbox) == 4

    def test_detector_instantiation(self):
        """TumorDetector should initialise without raising (uses base YOLOv8 weights)."""
        try:
            from detection.yolov8_detector import TumorDetector
            detector = TumorDetector(weights="yolov8n.pt")
            assert detector is not None
        except Exception as e:
            pytest.skip(f"Model download not available in test env: {e}")


# ── Augmentation Tests ────────────────────────────────────────────────────────

class TestAugmentation:
    def test_train_transforms_output_type(self):
        import torch
        from utils.augmentation import get_train_transforms
        transform = get_train_transforms(img_size=300)
        img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        result = transform(image=img)["image"]
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 300, 300)

    def test_val_transforms_no_augmentation(self):
        import torch
        from utils.augmentation import get_val_transforms
        transform = get_val_transforms(img_size=300)
        img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        # Same image should always produce same tensor
        r1 = transform(image=img.copy())["image"]
        r2 = transform(image=img.copy())["image"]
        assert torch.allclose(r1, r2)
