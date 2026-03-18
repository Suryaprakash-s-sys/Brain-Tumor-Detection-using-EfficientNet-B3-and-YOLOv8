"""
Preprocessing utilities for MRI brain tumor images.
Handles resizing, normalization, filtering, and format conversion.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path


def load_mri(image_path: str) -> np.ndarray:
    """Load an MRI image and convert to 3-channel RGB."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_image(image: np.ndarray, size: tuple) -> np.ndarray:
    """Resize image to target (width, height)."""
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def normalize(image: np.ndarray) -> np.ndarray:
    """Normalize pixel values to [0, 1]."""
    return image.astype(np.float32) / 255.0


def anisotropic_diffusion_filter(
    image: np.ndarray,
    num_iter: int = 10,
    kappa: float = 50,
    gamma: float = 0.1,
) -> np.ndarray:
    """
    Apply anisotropic diffusion (Perona-Malik) to reduce speckle noise
    while preserving tumor boundary edges.

    Args:
        image: Grayscale or RGB image as float in [0, 1].
        num_iter: Number of diffusion iterations.
        kappa: Conduction coefficient (edge sensitivity).
        gamma: Step size (stability: keep <= 0.25).

    Returns:
        Filtered image of same shape.
    """
    img = image.copy().astype(np.float64)

    for _ in range(num_iter):
        # Compute gradients in 4 directions
        delta_n = np.roll(img, -1, axis=0) - img
        delta_s = np.roll(img, 1, axis=0) - img
        delta_e = np.roll(img, -1, axis=1) - img
        delta_w = np.roll(img, 1, axis=1) - img

        # Perona-Malik conduction function
        c_n = np.exp(-(delta_n / kappa) ** 2)
        c_s = np.exp(-(delta_s / kappa) ** 2)
        c_e = np.exp(-(delta_e / kappa) ** 2)
        c_w = np.exp(-(delta_w / kappa) ** 2)

        img += gamma * (c_n * delta_n + c_s * delta_s + c_e * delta_e + c_w * delta_w)

    return np.clip(img, 0, 1).astype(np.float32)


def preprocess_for_efficientnet(image_path: str) -> np.ndarray:
    """Full preprocessing pipeline for EfficientNet-B3 (300x300)."""
    img = load_mri(image_path)
    img = resize_image(img, (300, 300))
    img = normalize(img)
    img = anisotropic_diffusion_filter(img)
    return img


def preprocess_for_yolov8(image_path: str) -> np.ndarray:
    """Full preprocessing pipeline for YOLOv8 (640x640)."""
    img = load_mri(image_path)
    img = resize_image(img, (640, 640))
    img = normalize(img)
    return img


def batch_preprocess(image_dir: str, output_dir: str, model: str = "efficientnet"):
    """Preprocess all images in a directory and save results."""
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [p for p in image_dir.rglob("*") if p.suffix.lower() in extensions]
    print(f"Processing {len(images)} images...")

    for img_path in images:
        try:
            if model == "efficientnet":
                processed = preprocess_for_efficientnet(str(img_path))
            else:
                processed = preprocess_for_yolov8(str(img_path))

            out_path = output_dir / img_path.name
            processed_uint8 = (processed * 255).astype(np.uint8)
            Image.fromarray(processed_uint8).save(out_path)
        except Exception as e:
            print(f"  Skipped {img_path.name}: {e}")

    print(f"Done. Saved to {output_dir}")
