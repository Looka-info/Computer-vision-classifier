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
│       ├── calibration.py          # Multiclass ECE, Brier score, selective classification
│       ├── config.py               # YAML configuration loader
│       ├── data_validation.py      # Duplicate hashing & split generation
│       ├── dataset.py              # PyTorch Dataset & DataLoader factories
│       ├── evaluate.py             # Classification report & error isolation
│       ├── models.py               # Baseline CNN & ResNet-18 architectures
│       ├── predict.py              # Standalone inference engine with warmup & timing breakdown
│       ├── train.py                # Training loop with validation checkpointing
│       ├── transforms.py           # Preprocessing & augmentation pipelines
│       └── visualize.py            # Diagnostic plotting functions
├── tests/                          # 26 automated unit tests (pytest)
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

## 8. Empirical Results & Latency Benchmark Reconciliation

### 8.1 Empirical Performance on Held-Out Test Split
Both models were evaluated on the exact same isolated held-out test split (**8,143 images**, 15% of dataset) on an NVIDIA GeForce GTX 1660 SUPER (CUDA 12.4):

| Model | Initialization | Total Params | Best Val Macro-F1 | Test Top-1 Acc | Test Top-3 Acc | Test Macro-F1 | Balanced Acc | Median Latency B1 (ms) | Throughput B16 (img/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Custom Baseline CNN** | Scratch | 464,454 | 0.8575 | **89.33%** | 98.19% | 0.8645 | 89.69% | **2.53 ms** | **1,704.0** |
| **ResNet-18 (Transfer)** | Pretrained (ImageNet-1k) | 11,196,006 | **0.9859** | **99.02%** | **99.94%** | **0.9864** | **98.87%** | 3.92 ms | 1,127.0 |

*Detailed machine-readable metrics are saved in [reports/model_comparison.csv](file:///reports/model_comparison.csv).*

---

### 8.2 Inference Latency Reconciliation (Priority 1 Audit)
A critical discrepancy was investigated between two previously reported ResNet-18 batch-1 timing figures:
* **Median inference latency (benchmark script):** `3.92 ms`
* **CLI prediction latency (single-image command):** `204.42 ms`

#### Empirical 5-Stage Latency Breakdown
To determine exactly what each measurement includes, a 5-stage profiler with explicit CUDA stream synchronization (`torch.cuda.synchronize()`) was executed on an **AMD Ryzen 5 5600 (6-core/12-thread)** and **NVIDIA GeForce GTX 1660 SUPER (6GB VRAM)** on Windows 11:

| Pipeline Stage | ResNet-18 Transfer (224×224) | Baseline CNN (64×64) | Measurement Scope & Execution Mechanics |
| :--- | :--- | :--- | :--- |
| **1. Checkpoint Loading & Init** | **260.79 ms** | **9.66 ms** | Cold disk I/O, weights deserialization (`torch.load`), allocation to VRAM, state dict loading. One-time startup cost. |
| **2. Image Decode & Preprocessing** | **6.07 ms** | **1.97 ms** | Disk file read, PIL image decoding, RGB conversion, resizing, tensor conversion, and normalization. |
| **3. Cold Forward Pass (1st Kernel)** | **230.74 ms** | **5.17 ms** | First CUDA forward pass; triggers lazy CUDA runtime context initialization, cuDNN kernel selection/autotuning, and memory pool creation. |
| **4. Warm Forward Pass (Median)** | **3.50 ms** *(p95: 5.42 ms)* | **1.02 ms** *(p95: 1.53 ms)* | Steady-state model forward pass on pre-allocated tensors after 15 warm-up iterations. |
| **5. Postprocessing (Softmax & Top-k)** | **11.57 ms** | **0.16 ms** | Softmax activation, tensor transfer from GPU to host memory, Top-k sorting, and label mapping. |
| **End-to-End Warm Pipeline** | **21.14 ms** *(~47 FPS)* | **3.15 ms** *(~317 FPS)* | Stages 2 + 4 + 5 per image in a persistent server process (e.g. Streamlit, FastAPI). |
| **End-to-End Cold Execution** | **509.17 ms** | **16.96 ms** | Stages 1 + 2 + 3 + 5 complete cold CLI invocation from cold process. |

*Saved in [reports/latency_breakdown.json](file:///reports/latency_breakdown.json).*

#### Exact Reconciliation
1. **The 3.92 ms benchmark measurement** isolated **Stage 4 only** (pure model forward pass on warm GPU tensors with synchronized timing after 15 warmup passes).
2. **The 204.42 ms CLI measurement** captured **Stage 3** (the cold first forward pass on CUDA). In PyTorch, lazy initialization causes the first CUDA kernel invocation to bear the entire overhead of context creation and cuDNN autotuning (~200–230 ms).
3. **Serving Implication:** In production deployment (Streamlit app, REST API), a single dummy warmup pass at startup guarantees all subsequent user requests execute at the steady-state **~21 ms end-to-end latency** (~47 FPS).

---

## 9. Model Calibration & Selective Classification Audit (Priority 2)

### 9.1 Confidence vs. Error Analysis
We audited all 8,143 held-out test predictions to verify whether model confidence reliably separates correct predictions from misclassifications:

| Prediction Outcome | Sample Count ($N$) | Mean Confidence | Median Confidence | 25th Percentile ($P_{25}$) | 75th Percentile ($P_{75}$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Correct Classifications** | **8,063** (99.02%) | **99.05%** | **99.99%** | **99.91%** | **99.99%** |
| **Misclassified Images** | **80** (0.98%) | **70.36%** | **71.33%** | **55.35%** | **86.16%** |

*Verified: Misclassified images exhibit dramatically lower confidence (median 71.33%) than correct predictions (median 99.99%). See [resnet18_confidence_distribution.png](file:///reports/figures/resnet18_confidence_distribution.png).*

---

### 9.2 Expected Calibration Error (ECE) and Multiclass Brier Score
Model calibration was evaluated using 15 equal-width confidence bins across both validation and held-out test splits:

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

$$\text{Multiclass Brier Score} = \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C (p_{ic} - y_{ic})^2 \quad \in [0, 2]$$

| Split | Sample Count | Expected Calibration Error (ECE) | Maximum Calibration Error (MCE) | Multiclass Brier Score | Reliability Diagram |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Validation Split** | 8,143 | **0.00288** | 0.27087 | **0.01713** | [resnet18_validation_reliability_diagram.png](file:///reports/figures/resnet18_validation_reliability_diagram.png) |
| **Held-Out Test Split** | 8,143 | **0.00364** | 0.26173 | **0.01503** | [resnet18_test_reliability_diagram.png](file:///reports/figures/resnet18_test_reliability_diagram.png) |

*The fine-tuned ResNet-18 exhibits outstanding calibration ($ECE < 0.004$), meaning output probabilities closely match empirical accuracy.*

---

### 9.3 Selective Classification (Rejection Gating)
To protect agricultural stakeholders from false positives on borderline images, a confidence rejection mechanism was developed:
* **Operating Threshold Selection:** Strict evaluation protocol (Priority 4) requires selecting $\tau^*$ **exclusively on validation data**.
* **Validation Target:** Targeting $\ge 99.5\%$ selective accuracy on validation data selected $\tau^* = 0.72$ (Validation Coverage: 98.39%, Validation Selective Accuracy: 99.54%).
* **Held-Out Test Evaluation at $\tau^* = 0.72$:**
  * **Test Coverage:** **98.71%** (only 105 ambiguous samples, or 1.29%, referred to agronomists).
  * **Test Selective Accuracy:** **99.53%** (misclassifications reduced from 80 down to 38 — a **52.5% error reduction**).

| Confidence Threshold ($\tau$) | Validation Coverage | Validation Accuracy | Test Coverage | Test Selective Accuracy | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0.00 (Unfiltered)** | 100.00% | 98.88% | 100.00% | 99.02% | Standard baseline |
| **0.50** | 99.83% | 98.98% | 99.74% | 99.11% | Minimal rejection |
| **0.70** | 98.65% | 99.43% | 98.88% | 99.47% | High coverage |
| **0.72 ($\tau^*$ Selected)** | **98.39%** | **99.54%** | **98.71%** | **99.53%** | **Optimal validation-selected threshold** |
| **0.80** | 97.75% | 99.60% | 98.01% | 99.66% | Conservative |
| **0.90** | 96.33% | 99.77% | 96.48% | 99.81% | High precision |
| **0.99** | 88.68% | 99.96% | 89.11% | 99.96% | Extreme certainty |

*Complete data saved in [reports/selective_classification_comparison.csv](file:///reports/selective_classification_comparison.csv) and [reports/calibration_audit.json](file:///reports/calibration_audit.json).*

---

### 9.4 In-Distribution Confidence vs. Out-of-Distribution (OOD) Detection
> [!IMPORTANT]
> **Critical Distiction:** High in-distribution calibration on PlantVillage **does not equal reliable out-of-distribution (OOD) detection**. 
> Standard deep neural networks with Softmax heads frequently produce high confidence (>90%) on completely foreign images (e.g., weeds, soil, non-plant items, hands) due to feature map activations on unfamiliar textures.
> Safe deployment in agricultural hardware requires dedicated OOD detection (e.g., Mahalanobis distance in embedding space, energy-based OOD gating, or a background non-leaf class), which is documented as an essential future release requirement.

---

## 10. Perceptual Near-Duplicate Audit Across Splits (Priority 3)

The dataset was initially verified for zero exact byte duplicates via SHA-256 hashing. To audit potential visual near-duplicate leakage across splits, a 64-bit difference hash (`dHash`) audit was executed across **10,652 images**:

* **Perceptual Hash Collisions Across Splits:** **0 collisions** ($d_{\text{Hamming}} = 0$).
* **Candidate Near-Matches ($d_{\text{Hamming}} \le 1$):** **3 candidate pairs** identified:
  * 2 pairs in `Corn_(maize)___Common_rust_` (train vs. test)
  * 1 pair in `Strawberry___Leaf_scorch` (train vs. test)
* **Visual Inspection & Pixel Verification:**
  * Candidate pairs were reviewed and pixel Mean Absolute Error (MAE) was calculated: Corn pair MAE = **42.98**; Strawberry pair MAE = **24.70**.
  * Inspection confirmed that the candidate pairs are **distinct biological leaves** that share similar overall leaf orientations against neutral background cards, **not duplicated photographs of the same leaf**.
* **Audit Verdict:** No cross-split visual duplication exists. Split manifests remain 100% leak-free and valid. Audit log saved in [reports/near_duplicates_audit.csv](file:///reports/near_duplicates_audit.csv).

---

## 11. Error Analysis & Diagnostics

All diagnostic figures are generated automatically in `reports/figures/`:
* [class_distribution.png](file:///reports/figures/class_distribution.png): Complete 38-class frequency distribution showing class counts ranging from 275 (Potato Healthy) to over 5,000 (Soybean/Tomato).
* [baseline_training_curves.png](file:///reports/figures/baseline_training_curves.png): Convergence curves for the scratch CNN over 4 epochs.
* [resnet18_training_curves.png](file:///reports/figures/resnet18_training_curves.png): Two-stage progression (Stage A head training $\to$ Stage B residual block 4 fine-tuning).
* [baseline_cnn_confusion_matrix.png](file:///reports/figures/baseline_cnn_confusion_matrix.png): Baseline normalized confusion matrix across all 38 categories.
* [resnet18_transfer_confusion_matrix.png](file:///reports/figures/resnet18_transfer_confusion_matrix.png): High-resolution normalized confusion matrix for fine-tuned ResNet-18.
* [resnet18_transfer_misclassified_grid.png](file:///reports/figures/resnet18_transfer_misclassified_grid.png): Visual grid of actual misclassified test images showing true class, model prediction, and confidence score.
* [resnet18_transfer_top_confusions.png](file:///reports/figures/resnet18_transfer_top_confusions.png): Bar chart of the most frequently confused pairs of categories.

### Diagnostic Observations
1. **Biological Pathogen Overlap:** The primary remaining confusion pairs occur between pathogens with near-identical necrotic symptoms (e.g. *Tomato Early Blight* vs. *Tomato Target Spot*, or *Corn Cercospora* vs. *Corn Northern Leaf Blight*).
2. **Cross-Species Pathology:** *Phytophthora infestans* (Late Blight) affects both Tomato and Potato; misclassifications occasionally occur across these botanically related Solanaceae species under subtle lesions.

---

## 12. Limitations & Real-World Agricultural Deployment Advisory

> [!WARNING]
> **Controlled Lab Dataset vs. Real-World Field Performance:**
> The 99.02% test accuracy achieved here reflects performance on the **PlantVillage dataset**, which was captured under strictly controlled laboratory conditions:
> * Individual excised leaves placed on uniform neutral gray or black cardboard backdrops.
> * Fixed, diffuse studio lighting without direct solar glare or dynamic shadows.
> * Single-leaf composition with no canopy clutter, soil, or background foliage.
>
> **Field Reality & Covariate Shift:**
> When deployed on real-world mobile or drone photography in agricultural fields, computer vision models suffer significant performance degradation due to:
> 1. Variable natural illumination (harsh direct sunlight, deep shadows, lens flare).
> 2. Wet leaves with specular reflections from dew or overhead irrigation.
> 3. Severe occlusions from adjacent branches, weeds, soil, and insect damage.
> 4. Multi-pathogen co-infections appearing on the same crop simultaneously.
>
> **Field Validation Notice:**
> PlantVision has **not been externally validated on uncurated agricultural field imagery**. High lab-dataset accuracy must not be conflated with field diagnostic efficacy. This software is an engineering reference implementation, **not a certified diagnostic device**. Always corroborate predictions with certified agronomic extension services before applying chemical pesticides or fungicides.

---

## 13. Automated Testing & Verification Suite

The repository includes a comprehensive 26-test automated verification suite (`tests/`):
```powershell
pytest -v
```
Tests systematically verify:
* **Data Validation & Integrity:** Root verification, corrupt image decoding rejection, exact SHA-256 duplicate detection, perceptual dHash invariance, and Hamming distance metric properties.
* **Leak-Free Splitting:** Cryptographic proof of zero duplicate hash leakage across train, val, and test splits.
* **Transforms:** Output tensor shapes, types, normalizations, and deterministic validation transforms.
* **Model Architectures:** Baseline CNN and ResNet-18 forward pass output shapes, parameter counts, and freezing/unfreezing grad states.
* **Inference Pipeline & Benchmarking:** Predictor output types, top-k ranking, corrupt file handling, warmup execution, and 5-stage latency profiling.
* **Calibration & Reliability:** Multiclass Brier score mathematical bounds ($[0, 2]$), ECE calculation, selective classification monotonicity, and validation threshold selection.
