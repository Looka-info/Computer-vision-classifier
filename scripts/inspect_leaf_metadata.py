"""Inspect dataset filenames against official PlantVillage leaf-map.json grouping metadata."""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

def main():
    with open("data/splits/leaf_map_raw.json", "r", encoding="utf-8") as f:
        leaf_map = json.load(f)

    tr = pd.read_csv("data/splits/train.csv")
    va = pd.read_csv("data/splits/val.csv")
    te = pd.read_csv("data/splits/test.csv")
    tr["split"] = "train"
    va["split"] = "val"
    te["split"] = "test"
    all_df = pd.concat([tr, va, te], ignore_index=True)

    def extract_key(path_str: str) -> str:
        stem = Path(path_str).stem
        if "___" in stem:
            part = stem.split("___")[-1]
        else:
            part = stem
        return part.lower().strip()

    all_df["clean_key"] = all_df["file_path"].apply(extract_key)

    matched = 0
    leaf_ids = []
    unmatched_rows = []

    for _, row in all_df.iterrows():
        k = row["clean_key"]
        val = leaf_map.get(k)
        if val:
            matched += 1
            leaf_ids.append(val[0])
        else:
            unmatched_rows.append(row)
            leaf_ids.append(f"unmapped:::{row['class_name']}:::{k}")

    all_df["leaf_id"] = leaf_ids

    print("=" * 70)
    print("PLANTVILLAGE LEAF-MAP GROUPING AUDIT")
    print("=" * 70)
    print(f"Total dataset images:               {len(all_df):,}")
    print(f"Matched with official leaf-map.json:{matched:,} ({matched / len(all_df) * 100:.2f}%)")
    print(f"Total unique physical leaf entities:{all_df['leaf_id'].nunique():,}")

    # Check cross-split leaf_id overlaps in current manifests
    train_leaves = set(all_df[all_df["split"] == "train"]["leaf_id"])
    val_leaves = set(all_df[all_df["split"] == "val"]["leaf_id"])
    test_leaves = set(all_df[all_df["split"] == "test"]["leaf_id"])

    train_val_overlap = train_leaves.intersection(val_leaves)
    train_test_overlap = train_leaves.intersection(test_leaves)
    val_test_overlap = val_leaves.intersection(test_leaves)

    print("\n--- Specimen (leaf_id) Cross-Split Overlaps in Current Splits ---")
    print(f"Train vs. Val overlapping leaf IDs:  {len(train_val_overlap):,}")
    print(f"Train vs. Test overlapping leaf IDs: {len(train_test_overlap):,}")
    print(f"Val vs. Test overlapping leaf IDs:   {len(val_test_overlap):,}")

    if train_test_overlap:
        print("\nSample Train vs Test overlapping leaf IDs:")
        for lid in list(train_test_overlap)[:10]:
            sub = all_df[all_df["leaf_id"] == lid]
            print(f"  Leaf {lid}: {len(sub)} images across {sub['split'].unique()}")

    unmatched_df = pd.DataFrame(unmatched_rows)
    if not unmatched_df.empty:
        print("\nUnmatched classes breakdown (missing from leaf-map.json):")
        print(unmatched_df["class_name"].value_counts().head(10))

if __name__ == "__main__":
    main()
