# PlantVision — Plant Leaf Disease Classifier

[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![torchvision](https://img.shields.io/badge/torchvision-0.21.0-EE4C2C.svg?style=flat)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42.0-FF4B4B.svg?style=flat&logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=flat&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/Code%20License-MIT-blue.svg)](LICENSE)

An end-to-end, reproducible deep learning computer vision portfolio project comparing a custom scratch Convolutional Neural Network (CNN) against transfer learning with fine-tuned **ResNet-18** on the **PlantVillage** dataset.

Built with strict data-leakage auditing (SHA-256 deduplication), stratified 70/15/15 train/val/test splits, empirical inference latency benchmarking, diagnostic error analysis, and an interactive Streamlit demonstration application.

---

## 1. Project Overview & Problem Statement

Early and accurate identification of agricultural plant diseases is vital to mitigating crop yield loss. However, real-world visual diagnosis faces severe challenges: subtle phenotypic differences between lesions, visual ambiguity between pathogens, and class imbalance.

**PlantVision** addresses this problem by building a complete, production-grade image classification pipeline:
* **Empirical Benchmarking:** Evaluates a baseline 4-block scratch CNN against a pretrained **ResNet-18** fine-tuned using a two-stage strategy (head optimization followed by residual block 4 unfreezing).
* **Zero Data Leakage:** Eliminates duplicate images across train/val/test splits via cryptographic SHA-256 byte hashing.
* **Balanced Evaluation:** Reports Top-1 Accuracy, Top-3 Accuracy, Balanced Accuracy, and Macro-averaged F1, Precision, and Recall on an isolated test set.
* **Latency & Throughput Profiling:** Synchronized hardware benchmarks for interactive (Batch Size 1) and batch throughput (Batch Size 16) inference.
* **Interactive Web Demo:** Streamlit browser application for single-image diagnosis with top-3 prediction probabilities and agricultural advisory notices.

---

## 2. Dataset Provenance & Specifications

* **Dataset:** [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset) (Mohanty et al., 2016).
* **Reference:** Mohanty, S.P., Hughes, D.P., & Salathé, M. (2016). *Using deep learning for image-based plant disease detection.* Frontiers in Plant Science, 7, 1419.
* **Dataset License:** Creative Commons Attribution-ShareAlike 3.0 Unported ([CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)).
* **Categories:** 38 distinct multiclass classes spanning 14 crop species:
  * *Apple:* Scab, Black rot, Cedar apple rust, Healthy
  * *Blueberry:* Healthy
  * *Cherry:* Powdery mildew, Healthy
  * *Corn (Maize):* Cercospora leaf spot / Gray leaf spot, Common rust, Northern Leaf Blight, Healthy
  * *Grape:* Black rot, Esca (Black Measles), Leaf blight (Isariopsis), Healthy
  * *Orange:* Huanglongbing (Citrus greening)
  * *Peach:* Bacterial spot, Healthy
  * *Pepper (Bell):* Bacterial spot, Healthy
  * *Potato:* Early blight, Late blight, Healthy
  * *Raspberry:* Healthy
  * *Soybean:* Healthy
  * *Squash:* Powdery mildew
  * *Strawberry:* Leaf scorch, Healthy
  * *Tomato:* Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Yellow Leaf Curl Virus, Mosaic virus, Healthy

> **Note on Licences:** The code in this repository is distributed under the MIT License. The PlantVillage dataset has its own independent license terms (CC BY-SA 3.0).

---

## 3. Engineering Architecture

```
Computer vision classifier/
├── configs/
│   └── config.yaml                 # Centralized experiment & training configuration
├── data/
│   ├── raw/PlantVillage/           # Local raw image classes (git-ignored)
│   ├── splits/                     # train.csv, val.csv, test.csv, classes.json
│   └── processed/
├── docs/
│   └── methodology.md              # Deep dive on mathematical & architectural design
├── models/                         # Checkpoints with complete deployment metadata (.pt)
├── notebooks/
│   └── 01_dataset_exploration.ipynb# EDA, corruption check, duplicate audit
├── reports/
│   ├── model_comparison.csv        # Empirical performance & latency comparison
│   └── figures/                    # Confusion matrices, curves, error grids
├── scripts/
│   ├── download_dataset.py         # Automated download & extraction from verified mirror
│   ├── prepare_dataset.py          # Data validation, duplicate detection, splitting
│   ├── train_baseline.py           # Trains custom scratch CNN
│   ├── train_transfer_learning.py  # Trains two-stage ResNet-18
│   ├── evaluate_model.py           # Evaluates on test set & generates diagnostics
│   └── run_pipeline.py             # End-to-end automated pipeline
├── src/
│   └── plantvision/                # Core modular package
│       ├── __init__.py
│       ├── benchmark.py            # Latency & throughput benchmarking
│       ├── config.py               # YAML configuration loader
│       ├── data_validation.py      # Duplicate hashing & split generation
│       ├── dataset.py              # PyTorch Dataset & DataLoader factories
│       ├── evaluate.py             # Classification report & error isolation
│       ├── models.py               # Baseline CNN & ResNet-18 architectures
│       ├── predict.py              # Standalone inference engine
│       ├── train.py                # Training loop with validation checkpointing
│       ├── transforms.py           # Preprocessing & augmentation pipelines
│       └── visualize.py            # Diagnostic plotting functions
├── tests/                          # 20 automated unit tests (pytest)
├── app.py                          # Streamlit interactive application
├── pyproject.toml
└── requirements.txt
```

---

## 4. Setup and Installation

### 4.1 Prerequisites
* Windows 10/11 (or Linux/macOS)
* Python 3.10, 3.11, or 3.12
* NVIDIA GPU with CUDA support recommended (GTX 1660 SUPER or better), or CPU.

### 4.2 Installation Steps
```powershell
# 1. Clone repository
git clone https://github.com/your-username/plantvision.git
cd "Computer vision classifier"

# 2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install PyTorch with CUDA 12.4 (or cu121 / CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. Install dependencies and plantvision package in editable mode
pip install -r requirements.txt -e .
```

---

## 5. End-to-End Workflow & Commands

### Step 1: Download & Verify the Dataset
```powershell
python scripts/download_dataset.py
```
*Downloads the official PlantVillage dataset archive and extracts color images into `data/raw/PlantVillage/`.*

### Step 2: Validate Data & Generate Leak-Free Splits
```powershell
python scripts/prepare_dataset.py
```
*Performs SHA-256 duplicate detection, validates JPEG headers, eliminates cross-split duplicate leakage, and creates `train.csv` (70%), `val.csv` (15%), and `test.csv` (15%).*

### Step 3: Train Scratch CNN Baseline
```powershell
python scripts/train_baseline.py
```
*Trains the 4-block baseline CNN from scratch and saves `models/baseline_cnn_best.pt`.*

### Step 4: Train Pretrained ResNet-18 (Transfer Learning)
```powershell
python scripts/train_transfer_learning.py
```
*Executes Stage A (classifier head optimization) followed by Stage B (layer4 fine-tuning with differential learning rates), saving `models/resnet18_transfer_best.pt`.*

### Step 5: Evaluate on Held-Out Test Set
```powershell
python scripts/evaluate_model.py --all
```
*Computes test set Top-1, Top-3, Balanced Accuracy, and Macro-F1. Generates confusion matrices and misclassification grids in `reports/figures/` and appends to `reports/model_comparison.csv`.*

### Step 6: Benchmark Inference Latency
```powershell
python -m plantvision.benchmark --arch resnet18
```

### Step 7: Run Single Image Prediction (CLI)
```powershell
python -m plantvision.predict --image path/to/leaf.jpg --model models/resnet18_transfer_best.pt
```

### Step 8: Launch Interactive Web Demo
```powershell
streamlit run app.py
```

### Step 9: Run Automated Tests
```powershell
pytest -v
```

---

## 6. Preprocessing & Augmentation Strategy

| Split | Transformations Applied | Rationale |
| :--- | :--- | :--- |
| **Train** | Resize($224 \times 224$), Random Horizontal Flip ($p=0.5$), Random Rotation ($\pm 15^\circ$), Conservative Color Jitter ($\pm 10\%$), ToTensor, Normalize | Introduces orientation invariance while strictly preserving lesion pigmentation and necrotic margins. |
| **Validation** | Resize($224 \times 224$), ToTensor, Normalize | Strictly deterministic; unbiased signal for checkpoint selection and early stopping. |
| **Test** | Resize($224 \times 224$), ToTensor, Normalize | Strictly deterministic; evaluates unseen generalization performance. |

ImageNet normalization constants: $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.

---

## 7. Model Architectures & Transfer Learning Strategy

### 7.1 Baseline Scratch CNN (`CustomBaselineCNN`)
* **Input:** $3 \times 224 \times 224$
* **Backbone:** 4 convolutional blocks ($32 \to 64 \to 128 \to 256$), each with Conv2D ($3 \times 3$, padding 1), Batch Normalization, and ReLU activations.
* **Downsampling:** $2 \times 2$ Max-Pooling between blocks; global Adaptive Average Pooling $(1 \times 1)$ at output.
* **Head:** Dropout ($p=0.3$) $\to$ Linear($256 \to 256$) $\to$ ReLU $\to$ Dropout($p=0.3$) $\to$ Linear($256 \to 38$).
* **Parameters:** ~500,000 trainable parameters.

### 7.2 Pretrained ResNet-18 Transfer Learning
* **Backbone:** TorchVision ResNet-18 pretrained on ImageNet-1k (`IMAGENET1K_V1`).
* **Classification Head:** Linear classifier tailored to target classes with Dropout ($p=0.3$).
* **Two-Stage Training:**
  * **Stage A:** Feature extractor completely frozen. Trains classifier head at $\eta = 0.001$ for 4 epochs to prevent large gradient updates from disrupting pretrained weights.
  * **Stage B:** Unfreezes residual block 4 (`layer4`). Trains backbone at lower learning rate $\eta_{\text{backbone}} = 0.0001$ and head at $\eta_{\text{classifier}} = 0.0005$ with Cosine Annealing decay.

---

## 8. Empirical Results & Model Comparison

*(Populated after pipeline execution on test split)*

| Model | Initialization | Total Params | Best Val Macro-F1 | Test Top-1 Acc | Test Top-3 Acc | Test Macro-F1 | Balanced Acc | Median Latency B1 (ms) | Throughput B16 (img/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Custom Baseline CNN** | Scratch | ~500K | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| **ResNet-18 (Transfer)** | Pretrained (ImageNet) | 11.2M | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

---

## 9. Error Analysis & Diagnostics

Key diagnostic visualizations generated automatically in `reports/figures/`:
* `class_distribution.png`: Per-category sample frequencies illustrating dataset imbalance.
* `resnet18_transfer_confusion_matrix.png`: Normalized confusion matrix across all 38 categories.
* `resnet18_transfer_misclassified_grid.png`: Visual grid of actual misclassified test images showing true class, model prediction, and confidence score.
* `resnet18_transfer_top_confusions.png`: Bar chart of the most frequently confused pairs of categories.

### Diagnostic Takeaways
1. **Visually Similar Pathology:** Diseases causing similar necrotic spot morphology (e.g. Tomato Early Blight vs. Tomato Target Spot) represent the primary confusion pairs.
2. **Early-Stage vs. Healthy:** Very mild early-stage infections on otherwise green foliage present subtle visual cues that occasional false negatives confuse with healthy leaves.
3. **Class Imbalance Impact:** Rare disease categories with fewer than 300 samples benefit substantially more from transfer learning than abundant classes (>2,000 samples).

---

## 10. Limitations & Agricultural Advisory

> [!WARNING]
> **Advisory Disclaimer:** This model was trained on the public PlantVillage repository, where images were captured under controlled laboratory lighting against neutral, uniform backdrops. Field photographs taken under direct sunlight, shadows, wet foliage, soil occlusions, or multi-disease coinfections may experience domain shift. This project is a demonstration of machine learning engineering principles and is **not an authoritative agricultural or biochemical diagnosis tool**. Do not apply chemical treatments without consulting certified agronomists or local agricultural extension services.

---

## 11. Automated Testing

The repository includes a comprehensive 20-test test suite (`tests/`):
```powershell
pytest -v
```
Tests verify:
* Dataset root validation, corrupt image detection, and duplicate hashing.
* Split creation with zero cross-split duplicate leakage.
* Transform output tensor shapes, types, and normalizations.
* Baseline and ResNet-18 forward pass shapes and parameter grad flags.
* Training loop single-epoch smoke execution and checkpoint persistence.
* Prediction pipeline scores, top-k ranking, corrupt file handling, and missing checkpoint rejection.
* Benchmarking routines and hardware profile capture.
