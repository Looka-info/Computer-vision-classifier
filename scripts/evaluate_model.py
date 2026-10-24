"""Evaluate trained model checkpoints on held-out test split, generate error diagnostics and comparison records."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import torch

from plantvision.benchmark import benchmark_model_inference
from plantvision.config import load_config
from plantvision.dataset import PlantDataset
from plantvision.evaluate import evaluate_test_set, update_model_comparison_csv
from plantvision.models import CustomBaselineCNN, build_transfer_resnet18, count_parameters
from plantvision.transforms import get_eval_transforms
from plantvision.visualize import (
    plot_confusion_matrix_heatmap,
    plot_error_analysis_grid,
    plot_top_confused_pairs,
)

logger = logging.getLogger(__name__)


def evaluate_checkpoint(
    checkpoint_path: Path | str,
    test_csv: Path | str,
    classes_json: Path | str,
    config_path: Optional[str] = None,
) -> dict:
    """Run full evaluation workflow for a single model checkpoint."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    metadata = ckpt["metadata"]
    model_name = metadata.get("model_name", ckpt_path.stem)
    arch = metadata.get("architecture", "ResNet")
    num_classes = metadata.get("num_classes", len(metadata.get("classes", [])))

    with open(classes_json, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    classes = sorted(class_to_idx.keys(), key=lambda k: class_to_idx[k])

    # Reconstruct model
    if "baseline" in arch.lower():
        model = CustomBaselineCNN(num_classes=num_classes)
        init_type = "Scratch"
    else:
        model = build_transfer_resnet18(num_classes=num_classes, pretrained=False)
        init_type = "Pretrained (ImageNet-1k)"

    model.load_state_dict(ckpt["state_dict"])
    param_counts = count_parameters(model)

    # Test DataLoader
    eval_tf = get_eval_transforms(
        image_size=metadata.get("input_dimensions", [3, 224, 224])[1],
        norm_mean=metadata.get("norm_mean", [0.485, 0.456, 0.406]),
        norm_std=metadata.get("norm_std", [0.229, 0.224, 0.225]),
    )
    test_df = pd.read_csv(test_csv)
    test_ds = PlantDataset(test_df, transform=eval_tf)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    print("\n" + "=" * 70)
    print(f"EVALUATING MODEL: {model_name} on Held-Out Test Set")
    print("=" * 70)
    print(f"Architecture:   {arch} ({init_type})")
    print(f"Total Params:   {param_counts['total_parameters']:,}")
    print(f"Test Samples:   {len(test_df):,} images")
    print(f"Evaluation on:  {device}")

    eval_results = evaluate_test_set(
        model=model,
        test_loader=test_loader,
        classes=classes,
        class_to_idx=class_to_idx,
        device=device,
    )

    # Print results summary
    print("\n--- Test Set Metrics ---")
    print(f"Top-1 Accuracy:     {eval_results['top1_accuracy']:.2%}")
    if eval_results.get("top3_accuracy") is not None:
        print(f"Top-3 Accuracy:     {eval_results['top3_accuracy']:.2%}")
    print(f"Balanced Accuracy:  {eval_results['balanced_accuracy']:.2%}")
    print(f"Macro Precision:    {eval_results['macro_precision']:.4f}")
    print(f"Macro Recall:       {eval_results['macro_recall']:.4f}")
    print(f"Macro F1-Score:     {eval_results['macro_f1']:.4f}")
    print(f"Correct Count:      {eval_results['correct_count']} / {eval_results['total_test_samples']}")
    print(f"Misclassified Count:{eval_results['misclassified_count']}")

    # Diagnostic figures
    figures_dir = Path(config["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    cm_path = figures_dir / f"{model_name}_confusion_matrix.png"
    plot_confusion_matrix_heatmap(
        cm=eval_results["confusion_matrix"],
        classes=classes,
        output_path=cm_path,
        title=f"{model_name.replace('_', ' ').title()} - Test Set Confusion Matrix",
    )

    grid_path = figures_dir / f"{model_name}_misclassified_grid.png"
    plot_error_analysis_grid(
        misclassified_records=eval_results["misclassified_examples"],
        output_path=grid_path,
        title=f"{model_name.replace('_', ' ').title()}: Misclassified Test Images",
    )

    conf_path = figures_dir / f"{model_name}_top_confusions.png"
    plot_top_confused_pairs(
        confused_pairs=eval_results["top_confused_pairs"],
        output_path=conf_path,
    )

    # Inference speed benchmark
    print("\n--- Benchmarking Inference Speed ---")
    bench_b1 = benchmark_model_inference(model, device=device, batch_size=1, warmup_iterations=10, num_iterations=40)
    bench_b16 = benchmark_model_inference(model, device=device, batch_size=16, warmup_iterations=10, num_iterations=30)
    print(f"Batch Size 1:  Median: {bench_b1['median_per_image_latency_ms']:.2f} ms | P95: {bench_b1['p95_per_image_latency_ms']:.2f} ms | Throughput: {bench_b1['throughput_images_per_sec']:.1f} FPS")
    print(f"Batch Size 16: Median: {bench_b16['median_per_image_latency_ms']:.2f} ms | P95: {bench_b16['p95_per_image_latency_ms']:.2f} ms | Throughput: {bench_b16['throughput_images_per_sec']:.1f} FPS")

    # Update reports/model_comparison.csv
    comp_record = {
        "Model Name": model_name,
        "Architecture": arch,
        "Initialization": init_type,
        "Total Parameters": param_counts["total_parameters"],
        "Best Val Macro-F1": round(metadata.get("metrics", {}).get("val_macro_f1", 0.0), 4),
        "Test Top-1 Accuracy": round(eval_results["top1_accuracy"], 4),
        "Test Top-3 Accuracy": round(eval_results.get("top3_accuracy", 0.0) or 0.0, 4),
        "Test Macro-F1": round(eval_results["macro_f1"], 4),
        "Test Balanced Accuracy": round(eval_results["balanced_accuracy"], 4),
        "Training Duration (s)": round(metadata.get("training_duration_seconds", 0.0), 1),
        "Inference Device": str(device),
        "Median Latency B1 (ms)": bench_b1["median_per_image_latency_ms"],
        "Throughput B16 (img/s)": bench_b16["throughput_images_per_sec"],
    }

    comp_csv_path = Path(config["paths"]["model_comparison_path"])
    update_model_comparison_csv(comp_csv_path, comp_record)

    # Save detailed JSON report
    report_json_path = Path(config["paths"]["reports_dir"]) / f"{model_name}_test_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        # Exclude large raw lists for cleaner json
        clean_res = {k: v for k, v in eval_results.items() if k not in ("misclassified_examples", "correct_examples")}
        clean_res["comparison_record"] = comp_record
        json.dump(clean_res, f, indent=2)

    return eval_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PlantVision Model on Test Set")
    parser.add_argument("--model-path", type=str, default=None, help="Path to checkpoint .pt")
    parser.add_argument("--all", action="store_true", help="Evaluate all models in models/ directory")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = load_config(args.config)

    splits_dir = Path(config["paths"]["splits_dir"])
    test_csv = splits_dir / "test.csv"
    classes_json = splits_dir / "classes.json"

    if not test_csv.exists() or not classes_json.exists():
        raise FileNotFoundError(f"Missing test split or classes.json in {splits_dir}")

    if args.all:
        models_dir = Path(config["paths"]["models_dir"])
        checkpoints = sorted(list(models_dir.glob("*.pt")))
        if not checkpoints:
            print(f"No checkpoint files found in {models_dir}")
            return
        for ckpt in checkpoints:
            evaluate_checkpoint(ckpt, test_csv, classes_json, args.config)
    else:
        ckpt_path = args.model_path or "models/resnet18_transfer_best.pt"
        evaluate_checkpoint(ckpt_path, test_csv, classes_json, args.config)


if __name__ == "__main__":
    main()
