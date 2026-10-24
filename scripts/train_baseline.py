"""Train scratch baseline CNN on PlantVision training split."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import pandas as pd
import torch

from plantvision.config import load_config
from plantvision.dataset import build_dataloaders, compute_class_weights
from plantvision.models import CustomBaselineCNN, count_parameters
from plantvision.train import train_model
from plantvision.visualize import plot_training_curves

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Baseline Scratch CNN")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = load_config(args.config)

    splits_dir = Path(config["paths"]["splits_dir"])
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    classes_json = splits_dir / "classes.json"

    if not train_csv.exists() or not val_csv.exists() or not classes_json.exists():
        raise FileNotFoundError(
            f"Split manifests not found in {splits_dir}. Run `python scripts/prepare_dataset.py` first."
        )

    with open(classes_json, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    classes = sorted(class_to_idx.keys(), key=lambda k: class_to_idx[k])
    num_classes = len(classes)

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    base_cfg = config["baseline"]
    epochs = args.epochs or base_cfg.get("epochs", 8)
    batch_size = args.batch_size or base_cfg.get("batch_size", 32)
    lr = base_cfg.get("learning_rate", 0.001)
    weight_decay = base_cfg.get("weight_decay", 0.0001)
    optimizer_name = base_cfg.get("optimizer", "adamw")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 70)
    print("PLANTVISION — TRAINING BASELINE SCRATCH CNN")
    print("=" * 70)
    print(f"Device:       {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    print(f"Classes:      {num_classes}")
    print(f"Train size:   {len(train_df):,} images")
    print(f"Val size:     {len(val_df):,} images")
    print(f"Epochs:       {epochs}")
    print(f"Batch Size:   {batch_size}")
    print(f"Learning Rate:{lr}")

    # Build model
    model = CustomBaselineCNN(
        num_classes=num_classes,
        conv_channels=base_cfg.get("conv_channels", [32, 64, 128, 256]),
        fc_hidden=base_cfg.get("fc_hidden", 256),
        dropout=base_cfg.get("dropout", 0.3),
    )
    param_counts = count_parameters(model)
    print(f"Parameters:   {param_counts['total_parameters']:,} (all trainable from scratch)")

    # DataLoaders
    train_loader, val_loader, _ = build_dataloaders(
        train_df=train_df,
        val_df=val_df,
        image_size=config["preprocessing"]["image_size"],
        norm_mean=config["preprocessing"]["norm_mean"],
        norm_std=config["preprocessing"]["norm_std"],
        batch_size=batch_size,
        num_workers=config["training"].get("num_workers", 2),
        color_jitter=config["preprocessing"]["augmentation"].get("color_jitter"),
    )

    class_weights = compute_class_weights(train_df, num_classes=num_classes, device=device)
    checkpoint_path = Path(config["paths"]["models_dir"]) / "baseline_cnn_best.pt"

    # Train
    train_results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        classes=classes,
        class_to_idx=class_to_idx,
        checkpoint_path=checkpoint_path,
        epochs=epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        class_weights=class_weights,
        device=device,
        optimizer_name=optimizer_name,
    )

    # Plot training curves
    curves_path = Path(config["paths"]["figures_dir"]) / "baseline_training_curves.png"
    plot_training_curves(
        history=train_results["history"],
        output_path=curves_path,
        title="Custom Baseline CNN: Training & Validation Progression",
    )

    print("\n" + "=" * 70)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Epoch:           {train_results['best_epoch']}")
    print(f"Best Val Macro-F1:    {train_results['best_metrics'].get('val_macro_f1', 0.0):.4f}")
    print(f"Best Val Accuracy:    {train_results['best_metrics'].get('val_accuracy', 0.0):.4f}")
    print(f"Training Duration:    {train_results['total_duration_sec']:.1f}s")
    print(f"Checkpoint saved to:  {checkpoint_path}")
    print(f"Curves saved to:      {curves_path}")


if __name__ == "__main__":
    main()
