"""Comprehensive model evaluation on held-out test split, error analysis, and metric reporting."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    classes: List[str],
    class_to_idx: Dict[str, int],
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run full evaluation on held-out test split.

    Calculates:
    - Top-1 Accuracy
    - Top-3 Accuracy (if num_classes >= 3)
    - Macro Precision, Recall, and F1
    - Balanced Accuracy
    - Per-class metrics
    - Confusion matrix
    - List of correctly classified and misclassified examples with confidence scores.

    Args:
        model: Trained PyTorch model.
        test_loader: Test DataLoader.
        classes: Ordered list of class names.
        class_to_idx: Mapping from class name to integer index.
        device: Computation device.

    Returns:
        Comprehensive dictionary of test metrics and error records.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []
    all_paths: List[str] = []

    with torch.inference_mode():
        for inputs, targets, paths in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            preds = torch.argmax(probs, dim=1)

            all_targets.extend(targets.tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.append(probs.cpu().numpy())
            all_paths.extend(paths)

    targets_arr = np.array(all_targets)
    preds_arr = np.array(all_preds)
    probs_arr = np.concatenate(all_probs, axis=0)

    # Core global metrics
    acc = accuracy_score(targets_arr, preds_arr)
    balanced_acc = balanced_accuracy_score(targets_arr, preds_arr)
    macro_f1 = f1_score(targets_arr, preds_arr, average="macro", zero_division=0)
    macro_prec = precision_score(targets_arr, preds_arr, average="macro", zero_division=0)
    macro_rec = recall_score(targets_arr, preds_arr, average="macro", zero_division=0)

    num_classes = len(classes)
    top3_acc = None
    if num_classes >= 3:
        try:
            top3_acc = float(top_k_accuracy_score(targets_arr, probs_arr, k=min(3, num_classes), labels=list(range(num_classes))))
        except Exception as exc:
            logger.warning("Top-3 accuracy computation failed: %s", exc)

    # Confusion matrix
    cm = confusion_matrix(targets_arr, preds_arr, labels=list(range(num_classes)))

    # Per-class classification report
    clf_report = classification_report(
        targets_arr,
        preds_arr,
        labels=list(range(num_classes)),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    # Error analysis: isolate correct and misclassified examples
    correct_examples = []
    misclassified_examples = []

    for i in range(len(targets_arr)):
        actual_idx = int(targets_arr[i])
        pred_idx = int(preds_arr[i])
        pred_conf = float(probs_arr[i, pred_idx])
        actual_conf = float(probs_arr[i, actual_idx])
        path = all_paths[i]

        record = {
            "file_path": path,
            "actual_label": actual_idx,
            "actual_class": classes[actual_idx],
            "predicted_label": pred_idx,
            "predicted_class": classes[pred_idx],
            "confidence": pred_conf,
            "true_class_probability": actual_conf,
        }

        if actual_idx == pred_idx:
            correct_examples.append(record)
        else:
            misclassified_examples.append(record)

    # Top confused pairs
    confused_pairs = []
    for r in range(num_classes):
        for c in range(num_classes):
            if r != c and cm[r, c] > 0:
                confused_pairs.append({
                    "actual_class": classes[r],
                    "predicted_class": classes[c],
                    "count": int(cm[r, c]),
                })
    confused_pairs.sort(key=lambda x: x["count"], reverse=True)

    results = {
        "top1_accuracy": float(acc),
        "top3_accuracy": top3_acc,
        "balanced_accuracy": float(balanced_acc),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "total_test_samples": len(targets_arr),
        "correct_count": len(correct_examples),
        "misclassified_count": len(misclassified_examples),
        "confusion_matrix": cm.tolist(),
        "classification_report": clf_report,
        "misclassified_examples": misclassified_examples,
        "correct_examples": correct_examples[:50],  # Keep sample of correct
        "top_confused_pairs": confused_pairs[:10],
    }

    return results


def update_model_comparison_csv(
    comparison_csv_path: Path | str,
    record: Dict[str, Any],
) -> pd.DataFrame:
    """Append or update a model performance record in the comparison CSV.

    Columns:
    - Model Name
    - Architecture
    - Initialization (Pretrained / Scratch)
    - Total Parameters
    - Best Val Macro-F1
    - Test Top-1 Accuracy
    - Test Top-3 Accuracy
    - Test Macro-F1
    - Test Balanced Accuracy
    - Training Duration (s)
    - Median Latency Batch 1 (ms)
    - Throughput (img/s)
    """
    path = Path(comparison_csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.stat().st_size > 0:
        df = pd.read_csv(path)
        # Update existing model row if name matches, else append
        if "Model Name" in df.columns and record.get("Model Name") in df["Model Name"].values:
            idx = df[df["Model Name"] == record["Model Name"]].index[0]
            for col, val in record.items():
                df.at[idx, col] = val
        else:
            df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])

    df.to_csv(path, index=False)
    logger.info("Updated model comparison table at %s", path)
    return df
