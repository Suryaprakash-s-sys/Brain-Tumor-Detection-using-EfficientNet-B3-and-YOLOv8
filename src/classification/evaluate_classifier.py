"""
Evaluation script for EfficientNet-B3 Brain Tumor Classifier.
Computes accuracy, precision, recall, F1-score, and confusion matrix.
"""

import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent))

from efficientnet_classifier import EfficientNetB3Classifier, CLASSES, IMG_SIZE

sys.path.append(str(Path(__file__).parent.parent / "utils"))
from visualization import plot_confusion_matrix


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_ds = datasets.ImageFolder(Path(args.data_dir) / "test", transform=transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    # Model
    model = EfficientNetB3Classifier(num_classes=len(CLASSES))
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="weighted")
    rec = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)

    print("\n" + "=" * 50)
    print("  EfficientNet-B3 Evaluation Results")
    print("=" * 50)
    print(f"  Accuracy : {acc * 100:.2f}%")
    print(f"  Precision: {prec * 100:.2f}%")
    print(f"  Recall   : {rec * 100:.2f}%")
    print(f"  F1-Score : {f1 * 100:.2f}%")
    print("\n" + classification_report(all_labels, all_preds, target_names=CLASSES))

    plot_confusion_matrix(cm, class_names=CLASSES, save_path=args.save_cm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Dataset root directory")
    parser.add_argument("--weights", required=True, help="Path to model weights (.pt)")
    parser.add_argument("--save_cm", default=None, help="Save confusion matrix to path")
    args = parser.parse_args()
    evaluate(args)
