"""Tests for model calibration metrics, Brier scores, and selective classification."""

from __future__ import annotations

import numpy as np
import pytest

from plantvision.calibration import (
    compute_calibration_metrics,
    compute_multiclass_brier_score,
    evaluate_selective_classification,
    select_threshold_from_validation,
)


def test_multiclass_brier_score_bounds():
    """Verify Brier score bounds: 0 for perfect predictions, bounded by 2 for worst."""
    targets = np.array([0, 1, 2])
    # Perfect probabilities
    perfect_probs = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    assert compute_multiclass_brier_score(targets, perfect_probs) == 0.0

    # Completely wrong probabilities
    worst_probs = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ])
    assert compute_multiclass_brier_score(targets, worst_probs) == 2.0


def test_compute_calibration_metrics():
    """Verify ECE calculation returns valid structure and bounded metrics."""
    np.random.seed(42)
    n_samples = 200
    n_classes = 5

    targets = np.random.randint(0, n_classes, size=n_samples)
    # Generate random logits and softmax
    raw = np.random.randn(n_samples, n_classes)
    probs = np.exp(raw) / np.sum(np.exp(raw), axis=1, keepdims=True)

    metrics = compute_calibration_metrics(targets, probs, n_bins=10)

    assert "expected_calibration_error" in metrics
    assert "maximum_calibration_error" in metrics
    assert "brier_score" in metrics
    assert "confidence_stats" in metrics

    assert 0.0 <= metrics["expected_calibration_error"] <= 1.0
    assert 0.0 <= metrics["maximum_calibration_error"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 2.0
    assert metrics["confidence_stats"]["correct_count"] + metrics["confidence_stats"]["error_count"] == n_samples


def test_evaluate_selective_classification():
    """Verify selective classification produces monotonically non-increasing coverage."""
    targets = np.array([0, 1, 0, 1])
    probs = np.array([
        [0.95, 0.05],
        [0.20, 0.80],
        [0.40, 0.60],  # error (predicted 1, true 0)
        [0.10, 0.90],
    ])

    df = evaluate_selective_classification(targets, probs, thresholds=np.array([0.0, 0.70, 0.85, 0.92]))

    assert len(df) == 4
    # Coverage must not increase as threshold increases
    coverages = df["coverage"].tolist()
    assert coverages == sorted(coverages, reverse=True)
    assert coverages[0] == 1.0  # Threshold 0.0 accepts everything


def test_select_threshold_from_validation():
    """Verify validation threshold selection satisfies target accuracy."""
    targets = np.array([0, 1, 0, 1, 0, 0])
    probs = np.array([
        [0.98, 0.02],  # correct (true 0, pred 0, conf 0.98)
        [0.05, 0.95],  # correct (true 1, pred 1, conf 0.95)
        [0.45, 0.55],  # wrong (true 0, pred 1, conf 0.55)
        [0.10, 0.90],  # correct (true 1, pred 1, conf 0.90)
        [0.92, 0.08],  # correct (true 0, pred 0, conf 0.92)
        [0.40, 0.60],  # wrong (true 0, pred 1, conf 0.60)
    ])

    tau = select_threshold_from_validation(targets, probs, target_accuracy=1.0)
    assert tau >= 0.60  # Must reject the errors with conf 0.55 and 0.60
    assert isinstance(tau, float)
