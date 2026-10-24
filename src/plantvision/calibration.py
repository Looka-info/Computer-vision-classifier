"""Model calibration metrics, reliability diagrams, and selective classification analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_multiclass_brier_score(targets: np.ndarray, probs: np.ndarray) -> float:
    """Calculate standard multiclass Brier score: (1/N) * sum_i sum_c (p_ic - y_ic)^2.

    Args:
        targets: 1D array of ground-truth integer class labels (N,).
        probs: 2D array of predicted class probabilities (N, C).

    Returns:
        Brier score in [0, 2] (lower is better, 0 indicates perfect probability assignment).
    """
    n_samples, n_classes = probs.shape
    one_hot = np.zeros_like(probs)
    for i, t in enumerate(targets):
        one_hot[i, t] = 1.0

    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    return round(brier, 5)


def compute_calibration_metrics(
    targets: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, Any]:
    """Calculate Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and bin stats.

    Args:
        targets: 1D array of true class labels.
        probs: 2D array of predicted probabilities.
        n_bins: Number of equal-width confidence bins in (0, 1].

    Returns:
        Dictionary containing ECE, MCE, Brier score, bin statistics, and confidence distributions.
    """
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    correct_mask = (preds == targets)

    brier_score = compute_multiclass_brier_score(targets, probs)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []
    ece = 0.0
    mce = 0.0

    total_samples = len(targets)

    for i in range(n_bins):
        lower = bins[i]
        upper = bins[i + 1]
        in_bin = (confs > lower) & (confs <= upper)
        count = int(np.sum(in_bin))
        bin_counts.append(count)

        if count > 0:
            b_acc = float(np.mean(preds[in_bin] == targets[in_bin]))
            b_conf = float(np.mean(confs[in_bin]))
            diff = abs(b_acc - b_conf)
            ece += (count / total_samples) * diff
            mce = max(mce, diff)
            bin_accs.append(b_acc)
            bin_confs.append(b_conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append((lower + upper) / 2.0)

    corr_confs = confs[correct_mask]
    err_confs = confs[~correct_mask]

    confidence_stats = {
        "correct_count": int(np.sum(correct_mask)),
        "error_count": int(np.sum(~correct_mask)),
        "correct_mean": float(np.mean(corr_confs)) if len(corr_confs) > 0 else 0.0,
        "correct_median": float(np.median(corr_confs)) if len(corr_confs) > 0 else 0.0,
        "correct_p25": float(np.percentile(corr_confs, 25)) if len(corr_confs) > 0 else 0.0,
        "correct_p75": float(np.percentile(corr_confs, 75)) if len(corr_confs) > 0 else 0.0,
        "error_mean": float(np.mean(err_confs)) if len(err_confs) > 0 else 0.0,
        "error_median": float(np.median(err_confs)) if len(err_confs) > 0 else 0.0,
        "error_p25": float(np.percentile(err_confs, 25)) if len(err_confs) > 0 else 0.0,
        "error_p75": float(np.percentile(err_confs, 75)) if len(err_confs) > 0 else 0.0,
    }

    return {
        "expected_calibration_error": round(float(ece), 5),
        "maximum_calibration_error": round(float(mce), 5),
        "brier_score": brier_score,
        "n_bins": n_bins,
        "bin_edges": bins.tolist(),
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts": bin_counts,
        "confidence_stats": confidence_stats,
    }


def plot_reliability_diagram(
    calib_metrics: Dict[str, Any],
    output_path: Path | str,
    title: str = "Reliability Diagram (Calibration Curve)",
) -> None:
    """Plot reliability diagram comparing confidence to observed accuracy across bins."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    bins = calib_metrics["bin_edges"]
    bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    accs = calib_metrics["bin_accuracies"]
    confs = calib_metrics["bin_confidences"]
    counts = calib_metrics["bin_counts"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    # Reliability curve
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=1.5)
    valid_mask = [c > 0 for c in counts]
    filtered_confs = [confs[i] for i in range(len(confs)) if valid_mask[i]]
    filtered_accs = [accs[i] for i in range(len(accs)) if valid_mask[i]]

    ax1.plot(filtered_confs, filtered_accs, marker="o", color="#2563eb", linewidth=2, label="Model Confidence vs Accuracy")
    ax1.bar(
        [bin_centers[i] for i in range(len(bin_centers)) if valid_mask[i]],
        filtered_accs,
        width=1.0 / len(counts),
        alpha=0.25,
        color="#3b82f6",
        edgecolor="#1d4ed8",
    )

    ece = calib_metrics["expected_calibration_error"]
    brier = calib_metrics["brier_score"]
    ax1.text(0.05, 0.85, f"ECE: {ece:.4f}\nBrier Score: {brier:.4f}", transform=ax1.transAxes,
             fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1"))

    ax1.set_ylabel("Empirical Accuracy", fontsize=11, fontweight="bold")
    ax1.set_title(title, fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Bin count histogram
    ax2.bar(bin_centers, counts, width=1.0 / len(counts), color="#64748b", edgecolor="#334155")
    ax2.set_xlabel("Confidence Bin", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Sample Count", fontsize=10)
    ax2.set_yscale("log")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved reliability diagram to %s", out_file)


def plot_confidence_distribution(
    targets: np.ndarray,
    probs: np.ndarray,
    output_path: Path | str,
    title: str = "Confidence Distribution: Correct vs Misclassified",
) -> None:
    """Plot distribution of prediction confidence for correct vs misclassified images."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    correct_mask = (preds == targets)

    corr_confs = confs[correct_mask]
    err_confs = confs[~correct_mask]

    plt.figure(figsize=(9, 5))
    bins = np.linspace(0.2, 1.0, 35)

    plt.hist(corr_confs, bins=bins, alpha=0.6, color="#059669", density=True, label=f"Correct (N={len(corr_confs):,}, Median: {np.median(corr_confs):.2%})")
    if len(err_confs) > 0:
        plt.hist(err_confs, bins=bins, alpha=0.7, color="#dc2626", density=True, label=f"Misclassified (N={len(err_confs)}, Median: {np.median(err_confs):.2%})")

    plt.xlabel("Softmax Confidence Score", fontsize=11, fontweight="bold")
    plt.ylabel("Density", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved confidence distribution to %s", out_file)


def evaluate_selective_classification(
    targets: np.ndarray,
    probs: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Evaluate accuracy and coverage across confidence rejection thresholds."""
    if thresholds is None:
        thresholds = np.array([0.0, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99])

    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)

    records = []
    total = len(targets)

    for tau in thresholds:
        accepted_mask = confs >= tau
        accepted_count = int(np.sum(accepted_mask))
        coverage = accepted_count / total
        rejection_rate = 1.0 - coverage

        if accepted_count > 0:
            acc = float(np.mean(preds[accepted_mask] == targets[accepted_mask]))
        else:
            acc = 1.0

        records.append({
            "threshold": round(float(tau), 2),
            "coverage": round(coverage, 4),
            "rejection_rate": round(rejection_rate, 4),
            "accepted_samples": accepted_count,
            "rejected_samples": total - accepted_count,
            "selective_accuracy": round(acc, 4),
        })

    return pd.DataFrame(records)


def select_threshold_from_validation(
    val_targets: np.ndarray,
    val_probs: np.ndarray,
    target_accuracy: float = 0.995,
) -> float:
    """Select rejection threshold strictly using validation data to satisfy a target accuracy."""
    df_val = evaluate_selective_classification(val_targets, val_probs, thresholds=np.linspace(0.5, 0.99, 50))
    # Filter thresholds where validation accuracy >= target_accuracy
    valid_candidates = df_val[df_val["selective_accuracy"] >= target_accuracy]
    if not valid_candidates.empty:
        # Choose the one with maximum coverage (smallest threshold)
        best_row = valid_candidates.sort_values(by="coverage", ascending=False).iloc[0]
        chosen_tau = float(best_row["threshold"])
    else:
        # Fall back to threshold maximizing validation selective accuracy
        best_row = df_val.sort_values(by="selective_accuracy", ascending=False).iloc[0]
        chosen_tau = float(best_row["threshold"])

    logger.info("Selected validation rejection threshold: tau=%.2f (Val Cov: %.2f%%, Val Acc: %.2f%%)",
                chosen_tau, best_row["coverage"] * 100, best_row["selective_accuracy"] * 100)
    return chosen_tau
