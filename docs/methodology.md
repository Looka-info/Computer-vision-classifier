# PlantVision — Computer Vision Methodology and Engineering Architecture

## 1. Problem Formulation and Objective
PlantVision is an image classification system designed to diagnose plant leaf diseases across crop species from digital photographs. Given an input RGB image of a leaf $x \in \mathbb{R}^{H \times W \times 3}$, the classifier predicts a discrete probability distribution $\hat{y} \in [0, 1]^C$ across $C$ categories (healthy and diseased states for various crops), parameterized by a neural network $f_\theta(x)$.

The primary engineering goal is to benchmark an empirical baseline (a CNN trained strictly from scratch) against a transfer-learning pipeline utilizing a pretrained ResNet-18 architecture, evaluating trade-offs in generalization accuracy, data efficiency, training convergence, and inference throughput.

---

## 2. Dataset Provenance and License Verification
* **Dataset:** PlantVillage Dataset (Mohanty, Hughes, & Salathé, 2016).
* **Reference Publication:** Mohanty, S.P., Hughes, D.P., & Salathé, M. (2016). *Using deep learning for image-based plant disease detection.* Frontiers in Plant Science, 7, 1419.
* **License:** Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0).
* **Composition:** Contains 54,306 images spanning 14 crop species and 26 disease conditions (38 total multiclass categories).
* **Acquisition Mirror:** Verified open-access repository hosted on Hugging Face (`mohanty/PlantVillage`) and GitHub (`spMohanty/PlantVillage-Dataset`).

---

## 3. Data Integrity, Validation, and Leakage Prevention
A persistent failure mode in computer vision research is cross-split data leakage caused by duplicate images, near-duplicate crops, or improper random splitting. PlantVision implements a multi-step data audit:

1. **File Decoding and Dimension Verification:** Every raw image is opened via PIL and decoded. Corrupt files, non-RGB channel counts, or malformed JPEG headers are flagged and logged to `reports/corrupt_images.csv`.
2. **Cryptographic Content Hashing:** A SHA-256 hash is computed for every valid image file to identify exact duplicate copies within and across class subdirectories.
3. **Deduplication:** Redundant copies are pruned prior to splitting so duplicate images cannot exist in different splits.
4. **Stratified Splitting:** Data is partitioned into:
   * **Training split (70%):** Model parameter optimization and backpropagation.
   * **Validation split (15%):** Hyperparameter tuning, checkpoint selection, and early stopping.
   * **Held-out Test split (15%):** Strictly isolated final evaluation. The test split is never observed during training, model selection, or augmentation design.
5. **Leakage Verification:** Cross-split set intersections ($\text{Hash}_{\text{train}} \cap \text{Hash}_{\text{val}} = \emptyset$, $\text{Hash}_{\text{train}} \cap \text{Hash}_{\text{test}} = \emptyset$, $\text{Hash}_{\text{val}} \cap \text{Hash}_{\text{test}} = \emptyset$) are assert-tested before manifest generation.

---

## 4. Image Preprocessing and Augmentation Pipeline

### Preprocessing
* **Resolution:** Standardized to $224 \times 224$ pixels.
* **Channel Normalization:** Standard ImageNet statistics:
  $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
  Applied as: $x_{\text{norm}} = \frac{x - \mu}{\sigma}$.

### Augmentation Design (Training Split Only)
Leaf disease diagnosis relies heavily on localized lesions, textural chlorosis, and necrotic spot color. Excessive color distortion can destroy pathological signals:
* **Random Horizontal Flip:** ($p = 0.5$) — Leaf morphology is orientation-invariant horizontally.
* **Mild Random Rotation:** ($-15^\circ$ to $+15^\circ$) — Simulates variable leaf orientation.
* **Conservative Color Jitter:** Brightness ($\pm 10\%$), Contrast ($\pm 10\%$), Saturation ($\pm 10\%$), Hue ($\pm 2\%$). Preserves color semantics while introducing lighting invariance.
* **Validation & Test Sets:** Strictly deterministic resize and normalization; zero stochastic augmentation.

---

## 5. Model Architectures and Transfer Learning Strategy

### 5.1 Baseline: Custom Scratch CNN (`CustomBaselineCNN`)
* **Architecture:** 4 convolutional blocks ($32 \to 64 \to 128 \to 256$ channels), each utilizing $3 \times 3$ convolutions, Batch Normalization, and ReLU activations.
* **Pooling:** $2 \times 2$ Max-Pooling between blocks with global Adaptive Average Pooling $(1 \times 1)$ before the classification head.
* **Regularization:** Dropout ($p = 0.3$) in the fully connected dense layers ($256 \to 256 \to C$).
* **Role:** Establishes an empirical baseline demonstrating what a small model learns from scratch without pre-existing visual representations.

### 5.2 Transfer Learning: ResNet-18 (`resnet18_transfer`)
* **Backbone:** Deep Residual Network with 18 layers pretrained on ImageNet-1k (1.28 million images, 1,000 categories). Residual connections mitigate vanishing gradients:
  $$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$
* **Two-Stage Fine-Tuning Strategy:**
  * **Stage A (Classifier Head Optimization):** Backbone feature extractor is frozen ($\text{requires\_grad} = \text{False}$). Only the replacement classification head ($\text{Dropout}(0.3) \to \text{Linear}(512 \to C)$) is trained with learning rate $\eta = 10^{-3}$. Prevents large random gradient updates from corrupting pretrained features.
  * **Stage B (Differential Fine-Tuning):** The high-level residual block (`layer4`) is unfrozen. A lower learning rate ($\eta_{\text{backbone}} = 10^{-4}$) is applied to pretrained weights, while the classifier head trains at $\eta_{\text{classifier}} = 5 \times 10^{-4}$.

---

## 6. Training and Optimization Protocol
* **Loss Function:** Cross-Entropy with inverse-frequency class weights:
  $$w_c = \frac{N}{C \cdot N_c}$$
* **Optimizer:** AdamW with weight decay $\lambda = 10^{-4}$.
* **Learning Rate Schedule:** Cosine Annealing learning rate decay down to $5\%$ of initial $\eta$.
* **Checkpoint Selection Rule:** Strictly selected by maximum Validation Macro-F1 score ($\text{val\_macro\_f1}$), never training loss or held-out test metrics.
* **Early Stopping:** Triggered if validation macro-F1 does not improve for 3 consecutive epochs.

---

## 7. Comprehensive Evaluation Metrics
Because real-world plant disease datasets often feature class imbalance, overall Accuracy is insufficient on its own. The evaluation pipeline computes:

1. **Top-1 Accuracy:** Standard classification hit rate.
2. **Top-3 Accuracy:** Fraction of test samples where the ground truth is among the top 3 predicted logits.
3. **Balanced Accuracy:** Average recall obtained across all individual classes.
4. **Macro-Averaged Precision, Recall, and F1-Score:** Unweighted mean across all classes, treating rare diseases with equal importance to abundant categories.
5. **Confusion Matrix Heatmap:** Normalized matrix displaying row-wise classification distribution.
6. **Error Analysis Visual Grids:** Qualitative audit of misclassified leaf samples displaying ground truth, predicted class, and model confidence.

---

## 8. Inference Latency Benchmarking
Measurements follow rigorous hardware benchmarking practices:
* **Device Synchronization:** `torch.cuda.synchronize()` ensures asynchronous CUDA execution completes before reading high-precision timers (`time.perf_counter()`).
* **Warmup Iterations:** 15 forward passes discarded prior to timing to populate caches and initialize CUDA kernels.
* **Metrics:** Steady-state median latency per image (ms), 95th percentile latency (ms), and sustained throughput (images per second) across Batch Size 1 (interactive latency) and Batch Size 16 (throughput).
* **Isolation:** Startup and model loading time are measured separately from per-inference latency.

---

## 9. Limitations and Ethical Considerations
* **Controlled Lab Bias:** PlantVillage images were captured under controlled lighting against uniform, flat laboratory backdrops. Real farm photographs include variable sunlight, soil clutter, occluded foliage, and shadows.
* **Calibration Limitations:** Softmax probabilities are normalized output scores reflecting model confidence relative to training distributions, not epidemiological disease likelihood.
* **Advisory Role:** PlantVision is designed as a computer vision diagnostic demonstration; field treatment should always involve qualified agricultural specialists.
