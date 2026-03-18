"""
Visualization utilities for model predictions and evaluation metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional
import cv2


CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]
CLASS_COLORS = {
    "glioma": "#e74c3c",
    "meningioma": "#f39c12",
    "no_tumor": "#2ecc71",
    "pituitary": "#3498db",
}


def plot_detections(
    image_path: str,
    detections: list,
    title: str = "Tumor Detection",
    save_path: Optional[str] = None,
):
    """
    Plot YOLOv8 detections with colored bounding boxes on the MRI image.

    Args:
        image_path:  Path to the original MRI image.
        detections:  List of Detection objects from TumorDetector.predict().
        title:       Plot title.
        save_path:   If provided, saves the figure here.
    """
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = CLASS_COLORS.get(det.label, "#ffffff")
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 6,
            f"{det.label}  {det.confidence:.0%}",
            color="white", fontsize=10, fontweight="bold",
            bbox=dict(facecolor=color, alpha=0.8, pad=2, edgecolor="none"),
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = CLASSES,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
):
    """Plot a labelled confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_curves(
    train_losses: List[float],
    val_accuracies: List[float],
    save_path: Optional[str] = None,
):
    """Plot training loss and validation accuracy curves side by side."""
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, "b-o", markersize=4, label="Train Loss")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, val_accuracies, "g-o", markersize=4, label="Val Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle("EfficientNet-B3 Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_class_distribution(
    class_counts: Dict[str, int],
    title: str = "Dataset Class Distribution",
    save_path: Optional[str] = None,
):
    """Bar chart showing number of samples per class."""
    labels = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = [CLASS_COLORS.get(l, "#888") for l in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Images")
    ax.set_xlabel("Tumor Class")

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            str(count), ha="center", va="bottom", fontweight="bold",
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_sample_predictions(
    image_paths: List[str],
    predictions: List[Dict],
    ncols: int = 3,
    save_path: Optional[str] = None,
):
    """Grid of MRI images with predicted labels overlaid."""
    nrows = (len(image_paths) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes

    for ax, img_path, pred in zip(axes, image_paths, predictions):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        label = pred.get("label", "unknown")
        conf = pred.get("confidence", 0)
        color = CLASS_COLORS.get(label, "white")
        ax.set_title(f"{label}\n{conf:.0%}", color=color, fontweight="bold")
        ax.axis("off")

    for ax in axes[len(image_paths):]:
        ax.set_visible(False)

    plt.suptitle("Sample Predictions", fontsize=16, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
