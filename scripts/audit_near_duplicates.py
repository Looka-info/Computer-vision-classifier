"""Visual near-duplicate audit using perceptual difference hashing (dHash)."""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from PIL import Image

from plantvision.data_validation import compute_dhash, hamming_distance

logger = logging.getLogger(__name__)


def audit_splits_for_near_duplicates(
    splits_dir: str | Path = "data/splits",
    max_hamming_dist: int = 2,
    max_samples_per_class: int | None = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Scan split manifests and detect perceptual hash collisions and near-matches across splits."""
    s_dir = Path(splits_dir)
    train_df = pd.read_csv(s_dir / "train.csv")
    val_df = pd.read_csv(s_dir / "val.csv")
    test_df = pd.read_csv(s_dir / "test.csv")

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    if max_samples_per_class:
        combined = combined.groupby(["split", "class_name"]).head(max_samples_per_class).reset_index(drop=True)

    print(f"Computing perceptual hashes for {len(combined):,} images across 38 classes...")
    hashes = []
    for idx, p in enumerate(combined["file_path"]):
        try:
            with Image.open(p) as img:
                h = compute_dhash(img)
                hashes.append(h)
        except Exception:
            hashes.append(0)

        if (idx + 1) % 5000 == 0 or idx == len(combined) - 1:
            print(f"Hashed {idx + 1}/{len(combined)} images...", end="\r")

    print("\nHash computation complete. Analyzing cross-split collisions...")
    combined["dhash"] = hashes

    # Find exact perceptual hash collisions
    hash_groups = combined.groupby("dhash")
    exact_cross_split = []

    for dhash_val, grp in hash_groups:
        if dhash_val == 0:
            continue
        splits_in_group = grp["split"].unique()
        if len(splits_in_group) > 1:
            # Group spans across multiple splits
            for i in range(len(grp)):
                for j in range(i + 1, len(grp)):
                    row1 = grp.iloc[i]
                    row2 = grp.iloc[j]
                    if row1["split"] != row2["split"]:
                        exact_cross_split.append({
                            "type": "exact_perceptual_collision",
                            "hamming_distance": 0,
                            "split_a": row1["split"],
                            "path_a": row1["file_path"],
                            "class_a": row1["class_name"],
                            "split_b": row2["split"],
                            "path_b": row2["file_path"],
                            "class_b": row2["class_name"],
                        })

    # Find near-matches (Hamming distance 1 or 2 within same class across splits)
    near_matches = []
    # Index by class to keep O(N^2) tractable within classes
    classes = combined["class_name"].unique()
    for cls in classes:
        cls_df = combined[combined["class_name"] == cls]
        # Partition by split
        tr = cls_df[cls_df["split"] == "train"]
        va = cls_df[cls_df["split"] == "val"]
        te = cls_df[cls_df["split"] == "test"]

        # Check train vs val
        for _, r_tr in tr.iterrows():
            h_tr = r_tr["dhash"]
            # Fast check against val
            for _, r_va in va.iterrows():
                dist = hamming_distance(h_tr, r_va["dhash"])
                if 1 <= dist <= max_hamming_dist:
                    near_matches.append({
                        "type": "near_duplicate",
                        "hamming_distance": dist,
                        "split_a": "train",
                        "path_a": r_tr["file_path"],
                        "class_a": r_tr["class_name"],
                        "split_b": "val",
                        "path_b": r_va["file_path"],
                        "class_b": r_va["class_name"],
                    })

        # Check train vs test
        for _, r_tr in tr.iterrows():
            h_tr = r_tr["dhash"]
            for _, r_te in te.iterrows():
                dist = hamming_distance(h_tr, r_te["dhash"])
                if 1 <= dist <= max_hamming_dist:
                    near_matches.append({
                        "type": "near_duplicate",
                        "hamming_distance": dist,
                        "split_a": "train",
                        "path_a": r_tr["file_path"],
                        "class_a": r_tr["class_name"],
                        "split_b": "test",
                        "path_b": r_te["file_path"],
                        "class_b": r_te["class_name"],
                    })

    df_matches = pd.DataFrame(exact_cross_split + near_matches)
    summary = {
        "total_images_audited": len(combined),
        "exact_perceptual_collisions_cross_split": len(exact_cross_split),
        "near_matches_dist_1_or_2_cross_split": len(near_matches),
    }

    return df_matches, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit split manifests for visual near-duplicates")
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--max-hamming-dist", type=int, default=1)
    parser.add_argument("--sample-per-class", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    print("=" * 70)
    print("PLANTVISION VISUAL NEAR-DUPLICATE AUDIT")
    print("=" * 70)

    matches_df, summary = audit_splits_for_near_duplicates(
        splits_dir=args.splits_dir,
        max_hamming_dist=args.max_hamming_dist,
        max_samples_per_class=args.sample_per_class,
    )

    print("\n--- Audit Summary ---")
    print(f"Total images audited:                  {summary['total_images_audited']:,}")
    print(f"Exact perceptual collisions across splits: {summary['exact_perceptual_collisions_cross_split']}")
    print(f"Near-matches (Hamming <= {args.max_hamming_dist}) across splits:    {summary['near_matches_dist_1_or_2_cross_split']}")

    out_csv = Path("reports/near_duplicates_audit.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    matches_df.to_csv(out_csv, index=False)
    print(f"Audit log written to: {out_csv}")


if __name__ == "__main__":
    main()
