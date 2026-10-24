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
* **Exact Distribution Source:** Acquired via [`scripts/download_dataset.py`](file:///scripts/download_dataset.py) from the open-access Hugging Face distribution [`mohanty/PlantVillage`](https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/data.zip), maintained by Sharada P. Mohanty.
* **Academic Reference:** Mohanty, S.P., Hughes, D.P., & Salathé, M. (2016). *Using deep learning for image-based plant disease detection.* Frontiers in Plant Science, 7, 1419. [doi:10.3389/fpls.2016.01419](https://doi.org/10.3389/fpls.2016.01419).
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

> [!IMPORTANT]
> **Independent Licensing Notice:** The MIT License applied to this repository strictly covers the engineering software, custom model code, prediction APIs, training pipelines, and audit scripts written for PlantVision. The PlantVillage dataset content retains its own independent Creative Commons CC BY-SA 3.0 terms and is not automatically covered or relicensed under the repository's MIT License.

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
  * **Stage B:** Unfreezes residual block 4 (`layer4`). Trains backbone at lower learning rate $\eta_{\text{backbone}} = 0.0001$ and head at $\eta_{\text{classifier}} = 0.0005$ with Cosine Anne| Model | Initialization | Total Params | Best Val Macro-F1 | Test Top-1 Acc | Test Top-3 Acc | Test Macro-F1 | Balanced Acc | Median Latency B1 (ms) | Throughput B16 (img/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Custom Baseline CNN** | Scratch | 464,454 | 0.8575 | **88.96%** | 98.06% | 0.8587 | 89.62% | **2.54 ms** | **1,726.5** |
| **ResNet-18 (Transfer)** | Pretrained (ImageNet-1k) | 11,196,006 | **0.9859** | **99.03%** | **99.94%** | **0.9871** | **98.93%** | 3.28 ms | 1,101.0 |

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

## 9. Model Calibration & Selective Classification Audit

### 9.1 Confidence vs. Error Analysis
We audited all 8,133 held-out test predictions to verify whether model confidence reliably separates correct predictions from misclassifications:

| Prediction Outcome | Sample Count ($N$) | Mean Confidence | Median Confidence | 25th Percentile ($P_{25}$) | 75th Percentile ($P_{75}$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Correct Classifications** | **8,054** (99.03%) | **98.97%** | **99.99%** | **99.90%** | **99.99%** |
| **Misclassified Images** | **79** (0.97%) | **70.69%** | **71.08%** | **57.29%** | **84.91%** |

*Verified: Misclassified images exhibit dramatically lower confidence (median 71.08%) than correct predictions (median 99.99%). See [resnet18_confidence_distribution.png](file:///reports/figures/resnet18_confidence_distribution.png).*

---

### 9.2 Expected Calibration Error (ECE) and Multiclass Brier Score

Calibration measures how well predicted softmax probabilities reflect true empirical likelihood. Two complementary metrics were evaluated:

1. **Top-Label Expected Calibration Error (Top-Label ECE):** Evaluates whether the confidence of the top predicted class $\hat{p}_i = \max_c p_{ic}$ matches empirical accuracy across $M=15$ fixed equal-width confidence bins:
   $$\text{Top-Label ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
   *Note on Metric Dependence:* Top-Label ECE depends on the choice of bin count ($M=15$) and partitions $[0, 1]$ into equal widths of $1/15 \approx 0.0667$. It measures top-prediction confidence calibration rather than the full joint probability vector.

2. **Multiclass Brier Score:** A strictly proper scoring rule evaluating the entire multiclass probability vector without binning:
   $$\text{Multiclass Brier Score} = \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C (p_{ic} - y_{ic})^2 \quad \in [0, 2]$$

| Split | Sample Count | Top-Label ECE (15 Bins) | Maximum Calibration Error (MCE) | Multiclass Brier Score | Reliability Diagram |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Validation Split** | 8,153 | **0.0029** | 0.2617 | **0.0168** | [resnet18_validation_reliability_diagram.png](file:///reports/figures/resnet18_validation_reliability_diagram.png) |
| **Held-Out Test Split** | 8,133 | **0.0040** | 0.2709 | **0.0154** | [resnet18_test_reliability_diagram.png](file:///reports/figures/resnet18_test_reliability_diagram.png) |

---

### 9.3 Selective Classification (Confidence Rejection Gating)
To explore trade-offs between coverage and diagnostic precision, a confidence rejection mechanism was audited:
* **Operating Threshold Selection:** To preserve evaluation integrity, the rejection threshold $\tau^*$ was **derived strictly on validation data**.
* **Validation Selection Procedure:** Sweeping candidate thresholds on validation predictions aiming for $\ge 99.5\%$ validation selective accuracy identified $\mathbf{\tau^* = 0.72}$ (Validation Coverage: 98.58%, Validation Selective Accuracy: 99.50%). Held-out test results were **not** used to tune or select this threshold.
* **Held-Out Test Outcome at $\mathbf{\tau^* = 0.72}$:**
  * **Images Flagged for Review:** Under $\tau^* = 0.72$, exactly **120 of 8,133 test images** (1.48% rejection rate) would be referred to expert human review.
  * **Error Reduction vs. False Rejections:**
    * **44 of the 120 rejected test images** were errors under the model's raw predictions (cutting test errors by 55.7% from 79 down to 35).
    * **76 of the 120 rejected test images** were correct predictions that happened to fall below the 0.72 confidence cutoff.
    * **35 errors remained** among the 8,013 accepted images (selective test accuracy: $7,978 / 8,013 = \mathbf{99.56\%}$, up from 99.03% unfiltered accuracy).
  * **Evaluation Precision:** Rejected images are **not** described as known ambiguities, and we make no claim that human agronomists reviewed them; they are purely algorithmic threshold rejections based on model confidence falling below 0.72.

| Confidence Threshold ($\tau$) | Validation Coverage | Validation Accuracy | Test Coverage | Test Selective Accuracy | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0.00 (Unfiltered)** | 100.00% | 98.87% | 100.00% | 99.03% | Standard baseline (79 test errors) |
| **0.50** | 99.82% | 98.97% | 99.75% | 99.12% | Minimal rejection |
| **0.60** | 99.29% | 99.23% | 99.24% | 99.37% | Balanced threshold |
| **0.70** | 98.80% | 99.42% | 98.73% | 99.48% | High coverage |
| **0.72 ($\tau^*$ Selected)** | **98.58%** | **99.50%** | **98.52%** | **99.56%** | **Validation-derived threshold (35 test errors remaining)** |
| **0.75** | 98.38% | 99.53% | 98.25% | 99.61% | Conservative |
| **0.80** | 97.95% | 99.60% | 97.81% | 99.66% | Strict precision |
| **0.90** | 96.41% | 99.80% | 96.40% | 99.78% | High precision |
| **0.99** | 89.30% | 99.97% | 88.48% | 99.94% | Extreme certainty |

*Complete data saved in [reports/selective_classification_comparison.csv](file:///reports/selective_classification_comparison.csv) and [reports/calibration_audit.json](file:///reports/calibration_audit.json).*

---

### 9.4 In-Distribution Confidence vs. Out-of-Distribution (OOD) Detection
> [!IMPORTANT]
> **Critical Distinction:** High in-distribution calibration on PlantVillage **does not equal reliable out-of-distribution (OOD) detection**. 
> Standard deep neural networks with Softmax heads frequently produce high confidence (>90%) on completely foreign images (e.g., weeds, soil, non-plant items, hands) due to feature map activations on unfamiliar textures.
> Safe deployment in agricultural hardware requires dedicated OOD detection (e.g., Mahalanobis distance in embedding space, energy-based OOD gating, or a background non-leaf class), which is documented as an essential future release requirement.

---

## 10. Perceptual Near-Duplicate Audit & Specimen Split Integrity

### 10.1 Audit Coverage Reconciliation
* The near-duplicate audit was executed across the **full dataset (54,284 images)**:
  * **Train:** 37,998 images (70.00%)
  * **Validation:** 8,153 images (15.02%)
  * **Test:** 8,133 images (14.98%)
* **Methodology:** 64-bit difference hashing (`dHash`) indexed via Multi-Index Hashing (MIH) into four 16-bit blocks. This guaranteed exhaustive evaluation of all $2,289,941$ candidate cross-split pairs sharing at least one 16-bit block.

### 10.2 Empirical Findings & Reconciled Measurements
All counts are recalculated directly from recorded measurements in [`reports/near_duplicates_audit.csv`](file:///reports/near_duplicates_audit.csv) and [`reports/near_duplicates_summary.json`](file:///reports/near_duplicates_summary.json):
* **Perceptual Hash Collisions ($d_{\text{Hamming}} = 0$):** **6 pairs** across splits. All 6 pairs were inspected and exhibited pixel Mean Absolute Error (MAE) between 20.36 and 48.09, confirming they are distinct physical leaves photographed in standard vertical orientations against neutral studio backgrounds. They are classified as **perceptual hash collisions**, not exact image duplicates.
* **Near-Matches ($d_{\text{Hamming}} \le 1$):** **22 pairs** across splits ($d=1$).
* **Exact Pixel MAE Distribution (Total 28 Flagged Pairs):**
  * **$\text{MAE} \le 15.0$ (Likely Duplicates):** Exactly **1 pair** (`Soybean___healthy`, `RS_HL 6169` in train vs `6170` in val, pixel MAE = **13.06**).
  * **$15.0 < \text{MAE} \le 20.0$ (Intermediate):** Exactly **1 pair** (`Pepper,_bell___healthy`, `JR_HL 8552` in train vs `8553` in val, pixel MAE = **16.71**).
  * **$\text{MAE} > 20.0$ (Distinct Leaves):** Exactly **26 pairs** (6 hash collisions at $d=0$ + 20 near-matches at $d=1$, with MAE values ranging from 20.36 to 59.68).
* **Cross-Split Duplicate Leakage Breakdown ($\text{MAE} \le 15.0$):**
  * **Train vs. Test Leakage:** **0 pairs**. Every candidate pair between train and test had MAE $> 20.0$; manual review confirms distinct biological leaves with different lesion distributions and margins.
  * **Val vs. Test Overlap:** **0 pairs**. Resolved through grouped evaluation assignment.
  * **Train vs. Val Leakage:** **1 pair** (`Soybean___healthy`, MAE = 13.06).

---

### 10.3 Resolution of the Validation-Test Same-Leaf Specimen Pair
1. **Identification:** An earlier audit identified a Cherry Powdery Mildew image in validation (`FREC_Pwd.M 0324`) and test (`FREC_Pwd.M 0323`) with pixel MAE = 5.58.
2. **Metadata Verification:** Inspection of the official PlantVillage specimen grouping repository ([`leaf-map.json`](https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/leaf_grouping/leaf-map.json)) confirmed that both files map to physical specimen `Cherry_(including_sour)___Powdery_mildew:::5.0` (22 total images across the dataset).
3. **Grouped Evaluation Partitioning:** To guarantee split isolation without perturbing the trained model weights, the 16,286 evaluation images (`val` + `test`) were re-partitioned using per-class greedy grouped assignment in [`scripts/resolve_evaluation_splits.py`](file:///scripts/resolve_evaluation_splits.py):
   * All 9 evaluation images of `Cherry leaf 5.0` were consolidated into the `test` split (0 in `val`).
   * Overlapping physical leaf IDs between validation and test was reduced from **2,118 down to EXACTLY ZERO**.
   * Split sizes were preserved at 8,153 validation images and 8,133 test images.
   * All 38 classes maintain representation in both evaluation splits with a maximum delta of 1 image.
   * `train.csv` (37,998 images) was untouched, preserving model training integrity.
   * Test evaluations and calibration thresholds were rerun on the corrected splits without tuning on test data.

### 10.4 Unresolved Dataset Limitation: Specimen-Level Independence
> [!WARNING]
> **Source Metadata Incompleteness & Benchmark Specimen Overlap:**
> While validation and test are now **100% specimen-independent** (0 overlapping physical leaves), full specimen-level independence across the entire dataset (train vs. test) **cannot be claimed**, due to two intrinsic properties of the benchmark PlantVillage dataset:
> 1. **Incomplete Specimen Metadata:** Official `leaf-map.json` annotations cover only **40,483 of 54,284 images (74.58%)** across 25 classes. **13,801 images (25.42%)** across 13 classes/subclasses (including all Corn classes, Squash Powdery Mildew, Tomato Mosaic Virus, Tomato Target Spot) have no physical leaf annotations.
> 2. **Benchmark Burst Capture Overlap:** The original dataset collection protocol photographed each physical leaf across 5–25 burst frames before random splitting. Consequently, 2,983 leaf IDs in validation and 2,989 leaf IDs in test have sibling photos in the historical training set.
> 
> True specimen-level independence across train and test would require either discarding 25.42% of the benchmark dataset or retraining on a non-standard subset. PlantVision documents this as an authoritative, transparent dataset limitation.

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
> The 99.03% test accuracy achieved here reflects performance on the **PlantVillage dataset**, which was captured under strictly controlled laboratory conditions:
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

The repository includes a comprehensive 29-test automated verification suite (`tests/`):
```powershell
pytest -v
```
Tests systematically verify:
* **Data Validation & Integrity:** Root verification, corrupt image decoding rejection, exact SHA-256 duplicate detection, perceptual dHash invariance, and Hamming distance metric properties.
* **Leak-Free Splitting & Specimen Isolation:** Proof of zero duplicate hash leakage across splits, zero physical specimen overlap between validation and test, and isolation of Cherry Powdery Mildew leaf 5.0 images.
* **Near-Duplicate Audit Reconciliation:** Verification that `reports/near_duplicates_summary.json` exactly matches the record counts and MAE distributions in `reports/near_duplicates_audit.csv`.
* **Transforms:** Output tensor shapes, types, normalizations, and deterministic validation transforms.
* **Model Architectures:** Baseline CNN and ResNet-18 forward pass output shapes, parameter counts, and freezing/unfreezing grad states.
* **Inference Pipeline & Benchmarking:** Predictor output types, top-k ranking, corrupt file handling, warmup execution, and 5-stage latency profiling.
* **Calibration & Reliability:** Multiclass Brier score mathematical bounds ($[0, 2]$), ECE calculation, selective classification monotonicity, and validation threshold selection.

---

## 14. Licensing, Provenance & Checkpoint Redistribution Governance

PlantVision maintains strict separation between software code licensing, dataset licensing, and model-weight redistribution assumptions:

### 14.1 Application Software License
* **License:** [MIT License](LICENSE)
* **Scope:** All custom source code in `src/plantvision/`, training and evaluation scripts in `scripts/`, automated tests in `tests/`, Streamlit application (`app.py`), and documentation written for PlantVision.

### 14.2 Dataset Provenance & Terms
* **Dataset:** [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
* **Authors:** David P. Hughes and Marcel Salathé (Frontiers in Plant Science, 2016).
* **License:** Creative Commons Attribution-ShareAlike 3.0 Unported ([CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)).
* **Conditions:** Any use or redistribution of the raw or processed image files requires proper attribution to Hughes & Salathé and must be distributed under the same or compatible CC BY-SA terms.

### 14.3 Backbone Architecture & Pretrained Weights
* **Backbone:** ResNet-18 (`IMAGENET1K_V1`)
* **Origin:** PyTorch / TorchVision (`torchvision.models.resnet18`)
* **License:** BSD 3-Clause / Apache 2.0.

### 14.4 Model Checkpoint Redistribution & Legal Governance
* **Checkpoints:** [`models/resnet18_transfer_best.pt`](file:///models/resnet18_transfer_best.pt) and [`models/baseline_cnn_best.pt`](file:///models/baseline_cnn_best.pt).
* **Redistribution Governance & Legal Uncertainty:**
  * The model checkpoints contain statistical weight matrices produced by numerical gradient optimization on training images from the CC BY-SA 3.0 PlantVillage dataset.
  * In current intellectual property and open-source law across various jurisdictions, there remains active legal debate regarding whether trained neural network parameters constitute a "derivative work" / "adaptation" of the underlying training imagery.
  * To observe maximum compliance and ethical governance, **anyone redistributing the fine-tuned model checkpoint weights publicly should observe CC BY-SA 3.0 ShareAlike provisions** by providing attribution to the PlantVillage creators and licensing checkpoint redistributions under CC BY-SA 3.0 compatible terms.
  * The repository code's MIT license does not override or supersede the CC BY-SA 3.0 obligations governing the underlying training data or derivative weight artifacts.
