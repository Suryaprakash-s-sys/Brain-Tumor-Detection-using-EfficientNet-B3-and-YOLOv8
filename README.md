# 🧠 Brain Tumor Detection using EfficientNet-B3 and YOLOv8

> A hybrid deep learning system for automated brain tumor detection and classification in MRI images.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-brightgreen)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Project Overview

This project presents a **hybrid deep learning framework** that integrates:
- **EfficientNet-B3** — for high-accuracy MRI image classification (tumor vs. non-tumor)
- **YOLOv8** — for real-time tumor localization and bounding box prediction

The system assists radiologists by automating the detection and classification of brain tumors in MRI scans with high accuracy and speed.

**Final Year B.E. Project — Electronics and Communication Engineering**  
K.L.N. College of Engineering, Anna University (April 2025)

---

## 🏆 Performance Results

| Model | Metric | Score |
|-------|--------|-------|
| EfficientNet-B3 | Accuracy | **96.4%** |
| EfficientNet-B3 | Precision | 95.7% |
| EfficientNet-B3 | Recall | 96.1% |
| EfficientNet-B3 | F1-Score | 96.0% |
| YOLOv8 | mAP@0.5 | **94.8%** |
| YOLOv8 | Inference Time | ~12 ms/image |

---

## 🔧 Features

- ✅ Automated MRI brain tumor detection using YOLOv8
- ✅ Multi-class tumor classification (Glioma, Meningioma, Pituitary, No Tumor)
- ✅ Transfer learning for efficient training on limited medical data
- ✅ Data augmentation pipeline for improved model robustness
- ✅ Web interface (Flask/FastAPI) for easy clinical use
- ✅ MongoDB integration for storing patient results
- ✅ Real-time inference (~12ms per image)

---

## 🗂️ Dataset

| Category | Images |
|----------|--------|
| Total | 2,690 |
| Training (80%) | 2,152 |
| Testing (20%) | 538 |
| Glioma | 746 |
| Meningioma | 810 |
| Pituitary Tumor | 807 |
| No Tumor | 327 |

**Source:** [Kaggle Brain MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) / Figshare / BraTS

---

## 🏗️ Architecture

```
MRI Image Input
      │
      ▼
┌─────────────────┐
│  Preprocessing  │  ← Resize, Normalize, Augment
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ EfficientNet-B3 │  ← Classification: Tumor / No Tumor
└────────┬────────┘
         │ (If Tumor Detected)
         ▼
┌─────────────────┐
│    YOLOv8       │  ← Localization: Bounding Box + Class
└────────┬────────┘
         │
         ▼
   Output Result
  (Class + Location)
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
CUDA 11.8+ (recommended for GPU training)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/brain-tumor-detection.git
cd brain-tumor-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```python
from src.detection.yolov8_detector import TumorDetector
from src.classification.efficientnet_classifier import TumorClassifier

# Initialize models
detector = TumorDetector(weights='models/yolov8_brain_tumor.pt')
classifier = TumorClassifier(weights='models/efficientnet_b3_brain_tumor.pt')

# Run inference on an MRI image
result = detector.predict('path/to/mri_image.jpg')
classification = classifier.predict(result.cropped_tumor)

print(f"Tumor Type: {classification.label} ({classification.confidence:.2%})")
```

---

## 📁 Project Structure

```
brain-tumor-detection/
├── src/
│   ├── classification/          # EfficientNet-B3 model
│   │   ├── efficientnet_classifier.py
│   │   ├── train_classifier.py
│   │   └── evaluate_classifier.py
│   ├── detection/               # YOLOv8 model
│   │   ├── yolov8_detector.py
│   │   ├── train_detector.py
│   │   └── evaluate_detector.py
│   ├── web_interface/           # Flask web app
│   │   ├── app.py
│   │   ├── static/
│   │   └── templates/
│   └── utils/                   # Shared utilities
│       ├── preprocessing.py
│       ├── augmentation.py
│       └── visualization.py
├── data/
│   ├── raw/                     # Original MRI images
│   ├── processed/               # Preprocessed images
│   └── annotations/             # YOLO format labels
├── models/                      # Saved model weights
├── notebooks/                   # Jupyter notebooks for EDA & training
│   ├── 01_data_exploration.ipynb
│   ├── 02_efficientnet_training.ipynb
│   └── 03_yolov8_training.ipynb
├── configs/
│   ├── tumor.yaml               # YOLOv8 dataset config
│   └── train_config.yaml        # Training hyperparameters
├── tests/                       # Unit tests
├── docs/                        # Documentation
├── requirements.txt
└── README.md
```

---

## 🧪 Training

### Train EfficientNet-B3 Classifier

```bash
python src/classification/train_classifier.py \
    --data_dir data/processed/ \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001
```

### Train YOLOv8 Detector

```bash
python src/detection/train_detector.py \
    --config configs/tumor.yaml \
    --model yolov8n.pt \
    --epochs 100 \
    --img_size 640
```

---

## 🌐 Web Interface

```bash
cd src/web_interface
python app.py
```

Then open `http://localhost:5000` in your browser to upload MRI images and get real-time predictions.

---

## 👥 Team

| Name | Roll Number |
|------|-------------|
| Sankarram G | 910621103073 |
| Sharann PG | 910621103078 |
| Surya Prakash S | 910621103088 |

**Supervisor:** Mrs. R. Angayarkanni, M.E., Assistant Professor, Dept. of ECE  
**HOD:** Dr. V. Kejalakshmi, M.E., Ph.D., Professor & Head, Dept. of ECE

---

## 📚 References

1. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for CNNs. *ICML 2019*.
2. Jocher, G. et al. (2023). Ultralytics YOLOv8. [GitHub](https://github.com/ultralytics/ultralytics)
3. Cheng, J. (2017). Brain Tumor Dataset. *Figshare*.
4. Menze, B. H. et al. (2015). The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS). *IEEE TMI*.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

We extend our gratitude to the faculty and management of K.L.N. College of Engineering for their support and guidance throughout this project.
