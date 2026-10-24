"""Prepare dataset: validate raw images, check duplicates, generate splits, and output diagnostics."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import pandas as pd

from plantvision.config import load_config
from plantvision.data_validation import (
    create_leak_free_splits,
    plot_class_distribution,
    save_split_manifests,
    scan_dataset,
)

logger = logging.getLogger(__name__)


def prepare_dataset_pipeline(config_path: str | Path | None = None) -> None:
    """Run full validation, integrity audit, duplicate detection, and splitting."""
    config = load_config(config_path)

    raw_dir = Path(config["paths"]["raw_data_dir"])
    splits_dir = Path(config["paths"]["splits_dir"])
    reports_dir = Path(config["paths"]["reports_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PLANTVISION DATASET VALIDATION & PREPARATION")
    print("=" * 70)
    print(f"Scanning raw dataset at: {raw_dir}")

    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        print(f"\n[ERROR] Raw data directory is empty or missing: {raw_dir}")
        print("Please run `python scripts/download_dataset.py` first to acquire the dataset.")
        return

    # 1. Scan and validate
    scan_res = scan_dataset(
        raw_dir=raw_dir,
        supported_extensions=config["dataset"]["supported_extensions"],
    )

    print("\n--- Scan Results ---")
    print(f"Total files scanned:       {scan_res.total_files_scanned:,}")
    print(f"Valid decoded images:      {scan_res.valid_count:,}")
    print(f"Corrupt / unreadable:      {scan_res.corrupt_count}")
    print(f"Discovered classes:        {scan_res.num_classes}")
    print(f"Exact duplicate groups:    {scan_res.duplicate_groups}")
    print(f"Redundant duplicate files: {scan_res.exact_duplicate_count}")

    # Save corrupt image report if any
    if not scan_res.df_corrupt.empty:
        corrupt_path = reports_dir / "corrupt_images.csv"
        scan_res.df_corrupt.to_csv(corrupt_path, index=False)
        print(f"[WARNING] {len(scan_res.df_corrupt)} corrupt images logged to: {corrupt_path}")

    # Save dataset summary CSV
    summary_rows = []
    for cls_name, count in scan_res.class_counts.items():
        summary_rows.append({"class_name": cls_name, "image_count": count})
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(reports_dir / "dataset_summary.csv", index=False)
    print(f"Dataset summary CSV saved to: {reports_dir / 'dataset_summary.csv'}")

    # Plot class distribution chart
    chart_path = figures_dir / "class_distribution.png"
    plot_class_distribution(
        class_counts=scan_res.class_counts,
        output_path=chart_path,
        title=f"PlantVision - Class Distribution ({scan_res.valid_count:,} Images, {scan_res.num_classes} Classes)",
    )

    # 2. Split dataset with zero leakage
    print("\n--- Creating Leak-Free Stratified Splits ---")
    train_r = config["dataset"]["train_ratio"]
    val_r = config["dataset"]["val_ratio"]
    test_r = config["dataset"]["test_ratio"]
    seed = config["project"]["random_seed"]

    train_df, val_df, test_df = create_leak_free_splits(
        df_valid=scan_res.df_valid,
        train_ratio=train_r,
        val_ratio=val_r,
        test_ratio=test_r,
        random_seed=seed,
        deduplicate_by_hash=True,
    )

    print(f"Train samples: {len(train_df):,} ({len(train_df)/len(scan_res.df_valid):.1%})")
    print(f"Val samples:   {len(val_df):,} ({len(val_df)/len(scan_res.df_valid):.1%})")
    print(f"Test samples:  {len(test_df):,} ({len(test_df)/len(scan_res.df_valid):.1%})")

    # 3. Save manifests
    save_split_manifests(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        splits_dir=splits_dir,
    )

    print(f"\nManifests successfully written to: {splits_dir}")
    print("Preparation complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and validate PlantVision dataset")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    prepare_dataset_pipeline(config_path=args.config)


if __name__ == "__main__":
    main()
