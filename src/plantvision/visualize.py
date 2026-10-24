"""Visualization utilities for training curves, confusion matrices, and error analysis grids."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

logger = logging.getLogger(__name__)


def plot_training_curves(
    history: Dict[str, List[float]],
    output_path: Path | str,
    title: str = "Training and Validation Performance",
) -> None:
    """Plot Loss and Accuracy/Macro-F1 curves across epochs."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss subplot
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#2563eb", marker="o", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#dc2626", marker="s", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cross Entropy Loss", fontsize=11, fontweight="bold")
    ax1.set_title("Loss Progression", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    # Metrics subplot
    ax2.plot(epochs, history["train_accuracy"], label="Train Accuracy", color="#2563eb", linestyle="--")
    ax2.plot(epochs, history["val_accuracy"], label="Val Accuracy", color="#059669", marker="^", linewidth=2)
    if "val_macro_f1" in history and history["val_macro_f1"]:
        ax2.plot(epochs, history["val_macro_f1"], label="Val Macro-F1", color="#7c3aed", marker="D", linewidth=2)

    ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax2.set_title("Validation Accuracy & Macro-F1", fontsize=12, fontweight="bold")
    ax2.set_ylim([0.0, 1.05])
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right")

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved training curves to %s", out_file)


def plot_confusion_matrix_heatmap(
    cm: np.ndarray | List[List[int]],
    classes: List[str],
    output_path: Path | str,
    title: str = "Test Set Normalized Confusion Matrix",
    normalize: bool = True,
) -> None:
    """Render high-resolution confusion matrix heatmap with clean labels."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    cm_arr = np.array(cm, dtype=float)
    fmt = "d"
    if normalize:
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        # Avoid division by zero for classes with 0 true labels
        row_sums[row_sums == 0] = 1.0
        cm_arr = cm_arr / row_sums
        fmt = ".2f"

    display_labels = [c.replace("___", "\n").replace("_", " ") for c in classes]
    fig_size = max(10, len(classes) * 0.45)
    plt.figure(figsize=(fig_size, fig_size))

    sns.heatmap(
        cm_arr,
        annot=len(classes) <= 15,
        fmt=fmt,
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
        cbar=True,
        linewidths=0.5,
    )

    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Class", fontsize=12, fontweight="bold")
    plt.ylabel("Actual True Class", fontsize=12, fontweight="bold")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved confusion matrix heatmap to %s", out_file)


def plot_error_analysis_grid(
    misclassified_records: List[Dict[str, Any]],
    output_path: Path | str,
    max_images: int = 12,
    title: str = "Diagnostic Error Analysis: Misclassified Test Images",
) -> None:
    """Plot grid of misclassified test examples showing true class, prediction, and confidence."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not misclassified_records:
        # Create an informative placeholder if zero errors occurred
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "Zero misclassifications in test set!\nModel achieved 100% test accuracy.",
                 ha="center", va="center", fontsize=14, color="#059669")
        plt.axis("off")
        plt.savefig(out_file, dpi=200)
        plt.close()
        return

    num_samples = min(len(misclassified_records), max_images)
    cols = 4 if num_samples >= 4 else num_samples
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.8, rows * 4.2))
    axes_flat = np.array(axes).reshape(-1)

    for i in range(num_samples):
        rec = misclassified_records[i]
        ax = axes_flat[i]
        try:
            img = Image.open(rec["file_path"]).convert("RGB")
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "Image Load Error", ha="center", va="center")

        actual = rec["actual_class"].replace("___", "\n")
        pred = rec["predicted_class"].replace("___", "\n")
        conf = rec["confidence"]

        caption = f"True: {actual}\nPred: {pred}\nConf: {conf:.1%}"
        ax.set_title(caption, fontsize=8.5, color="#dc2626", fontweight="bold")
        ax.axis("off")

    for j in range(num_samples, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved misclassified grid to %s", out_file)


def plot_top_confused_pairs(
    confused_pairs: List[Dict[str, Any]],
    output_path: Path | str,
    top_n: int = 8,
) -> None:
    """Plot horizontal bar chart of most frequently confused class pairs."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not confused_pairs:
        return

    top_pairs = confused_pairs[:top_n]
    labels = [f"True: {p['actual_class'].replace('___', ' ')}\n-> Pred: {p['predicted_class'].replace('___', ' ')}" for p in top_pairs]
    counts = [p["count"] for p in top_pairs]

    plt.figure(figsize=(10, max(5, len(labels) * 0.7)))
    y_pos = np.arange(len(labels))
    plt.barh(y_pos, counts, color="#ea580c", edgecolor="#c2410c", alpha=0.85)
    plt.yticks(y_pos, labels, fontsize=8.5)
    plt.xlabel("Number of Misclassified Images", fontsize=11, fontweight="bold")
    plt.title("Most Frequently Confused Class Pairs", fontsize=13, fontweight="bold", pad=12)
    plt.gca().invert_yaxis()
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    for i, v in enumerate(counts):
        plt.text(v + 0.1, i, f" {v}", va="center", fontsize=9, fontweight="bold", color="#1e293b")

    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()
    logger.info("Saved top confused pairs plot to %s", out_file)
