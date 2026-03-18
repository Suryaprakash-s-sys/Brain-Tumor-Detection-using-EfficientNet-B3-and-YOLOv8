# Setup & Usage Guide

## 1. Environment Setup

### System Requirements
- Python 3.8 or higher
- 8 GB RAM minimum (16 GB recommended for training)
- NVIDIA GPU with CUDA support (recommended for training)
- 10 GB free disk space for dataset and model weights

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/brain-tumor-detection.git
cd brain-tumor-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Dataset Preparation

### Download Dataset
Download the Brain Tumor MRI dataset from Kaggle:
```
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
```

Extract and place the data in:
```
data/raw/
├── train/
│   ├── glioma/
│   ├── meningioma/
│   ├── no_tumor/
│   └── pituitary/
└── test/
    ├── glioma/
    ├── meningioma/
    ├── no_tumor/
    └── pituitary/
```

### Preprocess Images

```bash
# For EfficientNet-B3 (300×300)
python src/utils/preprocessing.py \
    --input_dir data/raw/train \
    --output_dir data/processed/train \
    --model efficientnet

# For YOLOv8 (640×640) — after annotating with bounding boxes
python src/utils/preprocessing.py \
    --input_dir data/raw/train \
    --output_dir data/processed/train \
    --model yolov8
```

### Data Augmentation (optional — for balancing classes)

```bash
python src/utils/augmentation.py \
    --input_dir data/processed/train \
    --output_dir data/processed/train_augmented \
    --num_augments 5
```

---

## 3. Training

### Train EfficientNet-B3

```bash
python src/classification/train_classifier.py \
    --data_dir data/processed \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001
```

### Train YOLOv8

First, annotate your images in YOLO format (use [LabelImg](https://github.com/HumanSignal/labelImg) or [Roboflow](https://roboflow.com)).

Then train:
```bash
python src/detection/train_detector.py \
    --config configs/tumor.yaml \
    --model yolov8n.pt \
    --epochs 100 \
    --img_size 640
```

---

## 4. Evaluation

```bash
# Evaluate classifier
python src/classification/evaluate_classifier.py \
    --data_dir data/processed \
    --weights models/efficientnet_b3_brain_tumor.pt \
    --save_cm docs/confusion_matrix.png

# Evaluate detector
python src/detection/evaluate_detector.py \
    --weights models/yolov8_brain_tumor/weights/best.pt \
    --config configs/tumor.yaml
```

---

## 5. Running the Web Interface

### Start the server
```bash
cd src/web_interface
python app.py
```

Open `http://localhost:5000` in your browser.

### Environment variables (optional)

Create a `.env` file in `src/web_interface/`:
```
YOLO_WEIGHTS=../../models/yolov8_brain_tumor/weights/best.pt
EFFNET_WEIGHTS=../../models/efficientnet_b3_brain_tumor.pt
MONGO_URI=mongodb://localhost:27017/
PORT=5000
```

---

## 6. Running Tests

```bash
pytest tests/ -v
```

---

## 7. Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

Notebooks:
- `01_data_exploration.ipynb` — EDA and class distribution
- `02_efficientnet_training.ipynb` — Interactive training
- `03_yolov8_training.ipynb` — YOLOv8 fine-tuning
