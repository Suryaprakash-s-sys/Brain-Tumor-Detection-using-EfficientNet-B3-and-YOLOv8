"""
Training script for YOLOv8 Brain Tumor Detector
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train(args):
    """Fine-tune YOLOv8 on the brain tumor dataset."""
    model = YOLO(args.model)

    results = model.train(
        data=args.config,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        device=args.device,
        project="../../models",
        name="yolov8_brain_tumor",
        save=True,
        val=True,
        verbose=True,
    )

    print("\n✅ Training complete!")
    print(f"Best model saved to: {results.save_dir}/weights/best.pt")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 tumor detector")
    parser.add_argument("--config", type=str, default="../../configs/tumor.yaml",
                        help="Path to dataset YAML config")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Base model: yolov8n/s/m/l/x.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="",
                        help="Device: '' for auto, 'cpu', '0' for GPU 0")
    args = parser.parse_args()
    train(args)
