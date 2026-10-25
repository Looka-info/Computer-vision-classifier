"""Audit model calibration, reliability diagrams, and selective classification with validation-derived threshold."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from plantvision.calibration import (
    compute_calibration_metrics,
    evaluate_selective_classification,
    plot_confidence_distribution,
    plot_reliability_diagram,
    select_threshold_from_validation,
)
from plantvision.config import load_config
from plantvision.dataset import PlantDataset
from plantvision.models import CustomBaselineCNN, build_transfer_resnet18
from plantvision.transforms import get_eval_transforms

logger = logging.getLogger(__name__)


def extract_predictions(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device):
    """Run model inference on DataLoader and return targets and softmax probability vectors."""
    model.eval()
    all_targets = []
    all_probs = []
    with torch.inference_mode():
        for inputs, targets, _ in loader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_targets.extend(targets.tolist())
    return np.array(all_targets), np.concatenate(all_probs, axis=0)


def run_calibration_audit(config_path: str | None = None) -> dict:
    """Execute complete calibration and reliability audit across validation and test splits."""
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits_dir = Path(config["paths"]["splits_dir"])
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")

    eval_tf = get_eval_transforms(image_size=config["preprocessing"]["image_size"])
    val_loader = torch.utils.data.DataLoader(PlantDataset(val_df, eval_tf), batch_size=64, shuffle=False)
    test_loader = torch.utils.data.DataLoader(PlantDataset(test_df, eval_tf), batch_size=64, shuffle=False)

    # Load ResNet-18 model
    resnet_ckpt = Path("models/resnet18_transfer_best.pt")
    if not resnet_ckpt.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {resnet_ckpt}")

    ckpt = torch.load(resnet_ckpt, map_location="cpu", weights_only=False)
    num_classes = ckpt["metadata"]["num_classes"]
    model = build_transfer_resnet18(num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    print("=" * 70)
    print("PLANTVISION CALIBRATION & RELIABILITY ASSESSMENT")
    print("=" * 70)

    # 1. Validation predictions
    print(f"Extracting validation predictions ({len(val_df):,} images)...")
    val_targets, val_probs = extract_predictions(model, val_loader, device)

    # 2. Test predictions
    print(f"Extracting held-out test predictions ({len(test_df):,} images)...")
    test_targets, test_probs = extract_predictions(model, test_loader, device)

    # 3. Calibration metrics
    val_calib = compute_calibration_metrics(val_targets, val_probs, n_bins=15)
    test_calib = compute_calibration_metrics(test_targets, test_probs, n_bins=15)

    print("\n--- Calibration Metrics ---")
    print(f"Validation ECE (15 bins): {val_calib['expected_calibration_error']:.4f}")
    print(f"Validation MCE:           {val_calib['maximum_calibration_error']:.4f}")
    print(f"Validation Brier Score:   {val_calib['brier_score']:.4f}")
    print(f"Test ECE (15 bins):       {test_calib['expected_calibration_error']:.4f}")
    print(f"Test MCE:                 {test_calib['maximum_calibration_error']:.4f}")
    print(f"Test Brier Score:         {test_calib['brier_score']:.4f}")

    print("\n--- Confidence Distributions ---")
    print(f"Test Correct (N={test_calib['confidence_stats']['correct_count']:,}):  Mean: {test_calib['confidence_stats']['correct_mean']:.4f} | Median: {test_calib['confidence_stats']['correct_median']:.4f} | P25: {test_calib['confidence_stats']['correct_p25']:.4f}")
    print(f"Test Misclassified (N={test_calib['confidence_stats']['error_count']}): Mean: {test_calib['confidence_stats']['error_mean']:.4f} | Median: {test_calib['confidence_stats']['error_median']:.4f} | P25: {test_calib['confidence_stats']['error_p25']:.4f}")

    # 4. Generate plots
    figures_dir = Path(config["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    val_rel_path = figures_dir / "resnet18_validation_reliability_diagram.png"
    plot_reliability_diagram(val_calib, val_rel_path, title="ResNet-18 Validation Set Reliability Diagram")

    test_rel_path = figures_dir / "resnet18_test_reliability_diagram.png"
    plot_reliability_diagram(test_calib, test_rel_path, title="ResNet-18 Held-Out Test Set Reliability Diagram")

    conf_dist_path = figures_dir / "resnet18_confidence_distribution.png"
    plot_confidence_distribution(test_targets, test_probs, conf_dist_path, title="ResNet-18 Test Set Confidence Distribution: Correct vs Misclassified")

    # Selective Classification: Calibration threshold selection
    print("\n--- Selective Classification (Rejection Gating) ---")
    chosen_tau = select_threshold_from_validation(val_targets, val_probs, target_accuracy=0.995)
    print(f"Optimal Rejection Threshold: tau* = {chosen_tau:.2f}")

    # Evaluate sweep on both val and test
    sweep_thresholds = np.array([0.0, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99])
    val_selective = evaluate_selective_classification(val_targets, val_probs, sweep_thresholds)
    test_selective = evaluate_selective_classification(test_targets, test_probs, sweep_thresholds)

    # Extract test performance at validation-selected threshold
    test_preds = np.argmax(test_probs, axis=1)
    test_confs = np.max(test_probs, axis=1)
    accepted_test = test_confs >= chosen_tau
    test_cov_at_tau = float(np.mean(accepted_test))
    test_acc_at_tau = float(np.mean(test_preds[accepted_test] == test_targets[accepted_test]))

    print(f"\nHeld-Out Test Performance at tau* = {chosen_tau:.2f}:")
    print(f"  Test Coverage:          {test_cov_at_tau * 100:.2f}% ({np.sum(accepted_test):,} of {len(test_targets):,} images)")
    print(f"  Test Rejection Rate:    {(1 - test_cov_at_tau) * 100:.2f}% ({np.sum(~accepted_test):,} rejected)")
    print(f"  Test Selective Accuracy:{test_acc_at_tau * 100:.2f}% (Baseline unfiltered: {float(np.mean(test_preds == test_targets)) * 100:.2f}%)")

    # Save selective classification table
    comp_df = pd.DataFrame({
        "Threshold": sweep_thresholds,
        "Val_Coverage": val_selective["coverage"],
        "Val_Selective_Accuracy": val_selective["selective_accuracy"],
        "Test_Coverage": test_selective["coverage"],
        "Test_Selective_Accuracy": test_selective["selective_accuracy"],
    })
    sel_csv_path = Path("reports/selective_classification_comparison.csv")
    comp_df.to_csv(sel_csv_path, index=False)
    print(f"\nSelective classification comparison table saved to: {sel_csv_path}")

    # Save audit report json
    audit_report = {
        "model": "resnet18_transfer",
        "validation_metrics": val_calib,
        "test_metrics": test_calib,
        "validation_selected_threshold": chosen_tau,
        "test_performance_at_selected_threshold": {
            "threshold": chosen_tau,
            "coverage": round(test_cov_at_tau, 4),
            "rejection_rate": round(1 - test_cov_at_tau, 4),
            "accepted_samples": int(np.sum(accepted_test)),
            "rejected_samples": int(np.sum(~accepted_test)),
            "selective_accuracy": round(test_acc_at_tau, 4),
            "unfiltered_test_accuracy": round(float(np.mean(test_preds == test_targets)), 4),
        },
    }
    report_json_path = Path("reports/calibration_audit.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)
    print(f"Full calibration audit JSON saved to: {report_json_path}")

    return audit_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit model calibration and selective classification")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_calibration_audit(config_path=args.config)


if __name__ == "__main__":
    main()
