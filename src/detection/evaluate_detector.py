"""
Evaluation script for YOLOv8 Brain Tumor Detector.
Computes mAP@0.5, IoU, precision, and recall using Ultralytics built-in validation.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def evaluate(args):
    model = YOLO(args.weights)

    metrics = model.val(
        data=args.config,
        imgsz=args.img_size,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=True,
        plots=True,
        save_json=True,
    )

    print("\n" + "=" * 50)
    print("  YOLOv8 Evaluation Results")
    print("=" * 50)
    print(f"  mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"  Precision    : {metrics.box.mp:.4f}")
    print(f"  Recall       : {metrics.box.mr:.4f}")
    print(f"\nDetailed results saved to: {metrics.save_dir}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to YOLOv8 .pt weights")
    parser.add_argument("--config", default="../../configs/tumor.yaml", help="Dataset YAML")
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()
    evaluate(args)
