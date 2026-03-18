# data/

This directory holds the MRI brain tumor dataset.
Raw images are **not committed to Git** (see `.gitignore`).

## Structure

```
data/
├── raw/               ← Original downloaded images (ignored by Git)
│   ├── train/
│   │   ├── glioma/         (746 images)
│   │   ├── meningioma/     (810 images)
│   │   ├── no_tumor/       (327 images)
│   │   └── pituitary/      (807 images)
│   └── test/
│       └── ...
├── processed/         ← Preprocessed images (ignored by Git)
│   ├── train/
│   ├── val/
│   └── test/
└── annotations/       ← YOLO format label .txt files
    └── train/
```

## Download

Download the dataset from Kaggle:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Then run preprocessing:
```bash
python src/utils/preprocessing.py --input_dir data/raw --output_dir data/processed
```
