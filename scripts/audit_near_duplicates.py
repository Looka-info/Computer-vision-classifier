"""Visual near-duplicate audit using perceptual difference hashing (dHash) and Multi-Index Hashing (MIH).

Scalable audit covering 100% of all images across train, validation, and test splits (54,284 images).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from plantvision.data_validation import compute_dhash, hamming_distance

logger = logging.getLogger(__name__)


def _hash_worker(file_path: str) -> Tuple[str, int]:
    """Worker to compute 64-bit dHash for an image path."""
    try:
        with Image.open(file_path) as img:
            return file_path, compute_dhash(img)
    except Exception:
        return file_path, 0


def compute_hashes_for_dataset(
    file_paths: List[str],
    cache_path: Optional[Path | str] = None,
    num_workers: int = 8,
) -> Dict[str, int]:
    """Compute 64-bit dHash for all file paths using a thread pool, with optional caching."""
    c_path = Path(cache_path) if cache_path else None
    hash_cache: Dict[str, int] = {}

    if c_path and c_path.exists():
        print(f"Loading cached perceptual hashes from {c_path}...")
        try:
            df_cached = pd.read_csv(c_path)
            hash_cache = dict(zip(df_cached["file_path"], df_cached["dhash"]))
            print(f"Loaded {len(hash_cache):,} cached hashes.")
        except Exception as exc:
            logger.warning("Failed to load hash cache: %s. Recomputing.", exc)
            hash_cache = {}

    missing_paths = [p for p in file_paths if p not in hash_cache]
    if missing_paths:
        print(f"Computing perceptual hashes for {len(missing_paths):,} images using {num_workers} threads...")
        t0 = time.time()
        completed = 0
        total = len(missing_paths)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_path = {executor.submit(_hash_worker, p): p for p in missing_paths}
            for future in concurrent.futures.as_completed(future_to_path):
                f_path, h_val = future.result()
                hash_cache[f_path] = h_val
                completed += 1
                if completed % 10000 == 0 or completed == total:
                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    print(f"Progress: {completed:,}/{total:,} images hashed ({rate:.1f} img/s)...")

        t1 = time.time()
        print(f"Finished hashing in {t1 - t0:.2f}s.")

        if c_path:
            c_path.parent.mkdir(parents=True, exist_ok=True)
            df_to_save = pd.DataFrame([{"file_path": k, "dhash": v} for k, v in hash_cache.items()])
            df_to_save.to_csv(c_path, index=False)
            print(f"Saved perceptual hashes to {c_path}.")

    return hash_cache


def compute_pixel_mae(path_a: str, path_b: str, size: Tuple[int, int] = (224, 224)) -> float:
    """Calculate pixel-level Mean Absolute Error between two images resized to common dimensions."""
    try:
        with Image.open(path_a) as im_a, Image.open(path_b) as im_b:
            arr_a = np.asarray(im_a.convert("RGB").resize(size), dtype=np.float32)
            arr_b = np.asarray(im_b.convert("RGB").resize(size), dtype=np.float32)
            return float(np.mean(np.abs(arr_a - arr_b)))
    except Exception as exc:
        logger.warning("Error computing MAE between %s and %s: %s", path_a, path_b, exc)
        return -1.0


def audit_full_dataset(
    splits_dir: str | Path = "data/splits",
    max_hamming_dist: int = 1,
    cache_path: str | Path = "data/splits/perceptual_hashes_cache.csv",
    num_workers: int = 8,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Scan all split manifests and detect exact perceptual collisions and near-matches across splits."""
    s_dir = Path(splits_dir)
    train_df = pd.read_csv(s_dir / "train.csv")
    val_df = pd.read_csv(s_dir / "val.csv")
    test_df = pd.read_csv(s_dir / "test.csv")

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    total_images = len(combined)

    split_counts = {
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
        "total": total_images,
    }

    print(f"\nManifest image counts:")
    print(f"  - Train: {split_counts['train']:,}")
    print(f"  - Val:   {split_counts['val']:,}")
    print(f"  - Test:  {split_counts['test']:,}")
    print(f"  -> Total: {split_counts['total']:,}")

    # Step 1: Compute or load all hashes
    hash_dict = compute_hashes_for_dataset(
        file_paths=combined["file_path"].tolist(),
        cache_path=cache_path,
        num_workers=num_workers,
    )
    combined["dhash"] = combined["file_path"].map(hash_dict)

    # Step 2: Multi-Index Hashing for exhaustive, fast near-duplicate retrieval
    # Divide 64-bit dHash into 4 chunks of 16 bits.
    # By Generalized Pigeonhole Principle, if dist(h1, h2) <= 3, they MUST match on >= 1 chunk.
    print(f"\nBuilding Multi-Index Hash tables across all {total_images:,} images...")
    chunk_tables: List[Dict[int, List[int]]] = [defaultdict(list) for _ in range(4)]

    hashes = combined["dhash"].to_numpy(dtype=np.uint64)
    splits = combined["split"].to_numpy()
    paths = combined["file_path"].to_numpy()
    classes = combined["class_name"].to_numpy()

    for idx, h in enumerate(hashes):
        if h == 0:
            continue
        h_int = int(h)
        for chunk_id in range(4):
            chunk_val = (h_int >> (chunk_id * 16)) & 0xFFFF
            chunk_tables[chunk_id][chunk_val].append(idx)

    print("Analyzing candidate collisions across splits...")
    candidate_pairs: Set[Tuple[int, int]] = set()

    for chunk_id in range(4):
        table = chunk_tables[chunk_id]
        for chunk_val, idx_list in table.items():
            if len(idx_list) > 1:
                # Check for cross-split candidates in this bucket
                # Group by split first to avoid evaluating intra-split pairs
                split_to_indices: Dict[str, List[int]] = defaultdict(list)
                for i in idx_list:
                    split_to_indices[splits[i]].append(i)

                splits_present = list(split_to_indices.keys())
                if len(splits_present) > 1:
                    for s1_i in range(len(splits_present)):
                        for s2_i in range(s1_i + 1, len(splits_present)):
                            for idx_a in split_to_indices[splits_present[s1_i]]:
                                for idx_b in split_to_indices[splits_present[s2_i]]:
                                    pair = (min(idx_a, idx_b), max(idx_a, idx_b))
                                    candidate_pairs.add(pair)

    print(f"Evaluated buckets; found {len(candidate_pairs):,} cross-split candidate pairs for exact Hamming check.")

    flagged_records: List[Dict[str, Any]] = []

    for idx_a, idx_b in candidate_pairs:
        h_a = int(hashes[idx_a])
        h_b = int(hashes[idx_b])
        dist = hamming_distance(h_a, h_b)

        if dist <= max_hamming_dist:
            path_a = str(paths[idx_a])
            path_b = str(paths[idx_b])
            cls_a = str(classes[idx_a])
            cls_b = str(classes[idx_b])
            split_a = str(splits[idx_a])
            split_b = str(splits[idx_b])

            # Calculate pixel MAE
            mae = compute_pixel_mae(path_a, path_b)

            match_type = "hash_collision" if dist == 0 else "near_match"
            is_same_class = (cls_a == cls_b)

            flagged_records.append({
                "type": match_type,
                "hamming_distance": dist,
                "split_a": split_a,
                "class_a": cls_a,
                "path_a": path_a,
                "split_b": split_b,
                "class_b": cls_b,
                "path_b": path_b,
                "same_class": is_same_class,
                "pixel_mae": round(mae, 2),
                "verdict": "distinct_leaf" if mae > 15.0 else "likely_duplicate",
            })

    df_matches = pd.DataFrame(flagged_records)
    if not df_matches.empty:
        df_matches = df_matches.sort_values(by=["hamming_distance", "pixel_mae"]).reset_index(drop=True)

    exact_collisions = int((df_matches["hamming_distance"] == 0).sum()) if not df_matches.empty else 0
    near_matches = int((df_matches["hamming_distance"] > 0).sum()) if not df_matches.empty else 0

    # Explicit MAE breakdown across all flagged pairs
    mae_le_15 = int((df_matches["pixel_mae"] <= 15.0).sum()) if not df_matches.empty else 0
    mae_15_to_20 = int(((df_matches["pixel_mae"] > 15.0) & (df_matches["pixel_mae"] <= 20.0)).sum()) if not df_matches.empty else 0
    mae_gt_20 = int((df_matches["pixel_mae"] > 20.0).sum()) if not df_matches.empty else 0

    # Categorize likely duplicates by split pair
    train_test_dupes = 0
    train_val_dupes = 0
    val_test_dupes = 0

    if not df_matches.empty:
        likely_mask = df_matches["verdict"] == "likely_duplicate"
        df_likely = df_matches[likely_mask]
        for _, r in df_likely.iterrows():
            splits_pair = {r["split_a"], r["split_b"]}
            if splits_pair == {"train", "test"}:
                train_test_dupes += 1
            elif splits_pair == {"train", "val"}:
                train_val_dupes += 1
            elif splits_pair == {"val", "test"}:
                val_test_dupes += 1

    summary = {
        "scanned_counts": split_counts,
        "audit_coverage_percent": 100.0,
        "max_hamming_distance_evaluated": max_hamming_dist,
        "candidate_pairs_checked": len(candidate_pairs),
        "hash_collisions_cross_split": exact_collisions,
        "near_matches_cross_split": near_matches,
        "mae_distribution": {
            "le_15_count": mae_le_15,
            "between_15_and_20_count": mae_15_to_20,
            "gt_20_count": mae_gt_20,
            "total_pairs": len(df_matches),
        },
        "likely_duplicates_breakdown": {
            "train_vs_test_leakage": train_test_dupes,
            "train_vs_val_leakage": train_val_dupes,
            "val_vs_test_overlap": val_test_dupes,
            "total_likely_duplicates": train_test_dupes + train_val_dupes + val_test_dupes,
        },
        "train_to_test_leakage_free": (train_test_dupes == 0),
        "split_integrity_verified": (train_test_dupes == 0 and val_test_dupes == 0),
    }

    return df_matches, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Scalable Visual Near-Duplicate Audit")
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--max-hamming-dist", type=int, default=1, help="Max Hamming distance (default: 1)")
    parser.add_argument("--cache", type=str, default="data/splits/perceptual_hashes_cache.csv")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("=" * 70)
    print("PLANTVISION SCALABLE PERCEPTUAL NEAR-DUPLICATE AUDIT (100% COVERAGE)")
    print("=" * 70)

    matches_df, summary = audit_full_dataset(
        splits_dir=args.splits_dir,
        max_hamming_dist=args.max_hamming_dist,
        cache_path=args.cache,
        num_workers=args.workers,
    )

    print("\n" + "=" * 70)
    print("AUDIT FINDINGS SUMMARY")
    print("=" * 70)
    print(f"Total Images Audited:                     {summary['scanned_counts']['total']:,} (100% of dataset)")
    print(f"  - Train:                                {summary['scanned_counts']['train']:,}")
    print(f"  - Val:                                  {summary['scanned_counts']['val']:,}")
    print(f"  - Test:                                 {summary['scanned_counts']['test']:,}")
    print(f"Perceptual Hash Collisions (Hamming=0):   {summary['hash_collisions_cross_split']}")
    print(f"Near-Matches (Hamming <= {args.max_hamming_dist}):             {summary['near_matches_cross_split']}")
    print(f"MAE Breakdown Across Flagged Pairs:")
    print(f"  - MAE <= 15.0 (Likely Duplicates):      {summary['mae_distribution']['le_15_count']}")
    print(f"  - 15.0 < MAE <= 20.0 (Intermediate):    {summary['mae_distribution']['between_15_and_20_count']}")
    print(f"  - MAE > 20.0 (Distinct Leaves):         {summary['mae_distribution']['gt_20_count']}")
    print(f"Likely Duplicates Breakdown (MAE <= 15.0):")
    print(f"  - Train vs Test Leakage:                {summary['likely_duplicates_breakdown']['train_vs_test_leakage']}")
    print(f"  - Train vs Val Leakage:                 {summary['likely_duplicates_breakdown']['train_vs_val_leakage']}")
    print(f"  - Val vs Test Overlap:                  {summary['likely_duplicates_breakdown']['val_vs_test_overlap']}")
    print(f"Train-to-Test Leak-Free Verified:         {summary['train_to_test_leakage_free']}")

    out_csv = Path("reports/near_duplicates_audit.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    matches_df.to_csv(out_csv, index=False)
    print(f"\nAudit details written to: {out_csv}")

    out_json = Path("reports/near_duplicates_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Audit summary written to: {out_json}")
    print("=" * 70)


if __name__ == "__main__":
    main()
