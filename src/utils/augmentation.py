"""
Data augmentation pipeline for MRI brain tumor images.
Uses albumentations for efficient, GPU-friendly transforms.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from pathlib import Path
import os


def get_train_transforms(img_size: int = 300) -> A.Compose:
    """Augmentation pipeline used during training."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),
        A.GridDistortion(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 300) -> A.Compose:
    """Minimal transforms for validation/testing — no augmentation."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def augment_dataset(
    input_dir: str,
    output_dir: str,
    num_augments: int = 5,
    img_size: int = 300,
):
    """
    Offline augmentation: generate augmented copies of all images in a directory.
    Useful for balancing under-represented classes (e.g., No Tumor: 327 images).

    Args:
        input_dir:     Source directory with class subdirectories.
        output_dir:    Destination directory; mirrors source structure.
        num_augments:  Number of augmented copies per image.
        img_size:      Target image size.
    """
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, p=0.7),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(var_limit=(10, 40), p=0.3),
        A.ElasticTransform(p=0.2),
    ])

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    extensions = {".jpg", ".jpeg", ".png"}

    for class_dir in input_dir.iterdir():
        if not class_dir.is_dir():
            continue
        out_class_dir = output_dir / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        images = [p for p in class_dir.iterdir() if p.suffix.lower() in extensions]
        print(f"  {class_dir.name}: {len(images)} images → generating {num_augments}x copies")

        for img_path in images:
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Copy original
            out_orig = out_class_dir / img_path.name
            cv2.imwrite(str(out_orig), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # Generate augmented copies
            for i in range(num_augments):
                augmented = transform(image=img)["image"]
                stem = img_path.stem
                out_aug = out_class_dir / f"{stem}_aug{i}{img_path.suffix}"
                cv2.imwrite(str(out_aug), cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

    print(f"\nAugmentation complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Offline dataset augmentation")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_augments", type=int, default=5)
    parser.add_argument("--img_size", type=int, default=300)
    args = parser.parse_args()
    augment_dataset(args.input_dir, args.output_dir, args.num_augments, args.img_size)
