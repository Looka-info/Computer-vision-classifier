"""Train ResNet-18 transfer learning classifier (Stage A classifier head + Stage B fine-tuning)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import pandas as pd
import torch

from plantvision.config import load_config
from plantvision.dataset import build_dataloaders, compute_class_weights
from plantvision.models import build_transfer_resnet18, count_parameters
from plantvision.train import train_transfer_learning
from plantvision.visualize import plot_training_curves

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResNet-18 Transfer Learning")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--stage-a-epochs", type=int, default=None, help="Override Stage A epochs")
    parser.add_argument("--stage-b-epochs", type=int, default=None, help="Override Stage B epochs")
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

    tl_cfg = config["transfer_learning"]
    if args.stage_a_epochs is not None:
        tl_cfg["stage_a_epochs"] = args.stage_a_epochs
    if args.stage_b_epochs is not None:
        tl_cfg["stage_b_epochs"] = args.stage_b_epochs
    batch_size = args.batch_size or tl_cfg.get("batch_size", 32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 70)
    print("PLANTVISION — TRAINING RESNET-18 (TRANSFER LEARNING & FINE-TUNING)")
    print("=" * 70)
    print(f"Device:         {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    print(f"Architecture:   ResNet-18 (Pretrained: {tl_cfg.get('weights', 'IMAGENET1K_V1')})")
    print(f"Classes:        {num_classes}")
    print(f"Train size:     {len(train_df):,} images")
    print(f"Val size:       {len(val_df):,} images")
    print(f"Stage A Epochs: {tl_cfg['stage_a_epochs']} (Head only)")
    print(f"Stage B Epochs: {tl_cfg['stage_b_epochs']} (Layer4 Fine-Tuning)")
    print(f"Batch Size:     {batch_size}")

    # Build model with frozen backbone
    model = build_transfer_resnet18(
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone=True,
        dropout=0.3,
    )
    init_params = count_parameters(model)
    print(f"Total Params:   {init_params['total_parameters']:,}")
    print(f"Stage A Params: {init_params['trainable_parameters']:,} trainable, {init_params['frozen_parameters']:,} frozen")

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
    checkpoint_path = Path(config["paths"]["models_dir"]) / "resnet18_transfer_best.pt"

    # Train two-stage transfer learning
    train_results = train_transfer_learning(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        classes=classes,
        class_to_idx=class_to_idx,
        checkpoint_path=checkpoint_path,
        class_weights=class_weights,
        device=device,
    )

    # Plot training curves
    curves_path = Path(config["paths"]["figures_dir"]) / "resnet18_training_curves.png"
    plot_training_curves(
        history=train_results["history"],
        output_path=curves_path,
        title="ResNet-18 Transfer Learning & Fine-Tuning Progression",
    )

    print("\n" + "=" * 70)
    print("RESNET-18 TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Epoch:           {train_results['best_epoch']}")
    print(f"Best Val Macro-F1:    {train_results['best_metrics'].get('val_macro_f1', 0.0):.4f}")
    print(f"Best Val Accuracy:    {train_results['best_metrics'].get('val_accuracy', 0.0):.4f}")
    print(f"Training Duration:    {train_results['total_duration_sec']:.1f}s")
    print(f"Checkpoint saved to:  {checkpoint_path}")
    print(f"Curves saved to:      {curves_path}")


if __name__ == "__main__":
    main()
