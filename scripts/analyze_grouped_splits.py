"""Analyze grouped split strategies for PlantVision split integrity."""

import json
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    with open("data/splits/leaf_map_raw.json", "r") as f:
        lm = json.load(f)

    tr = pd.read_csv("data/splits/train.csv")
    va = pd.read_csv("data/splits/val.csv")
    te = pd.read_csv("data/splits/test.csv")
    tr["orig_split"] = "train"
    va["orig_split"] = "val"
    te["orig_split"] = "test"
    all_df = pd.concat([tr, va, te], ignore_index=True)

    def extract_key(path_str):
        stem = Path(path_str).stem
        if "___" in stem:
            part = stem.split("___")[-1]
        else:
            part = stem
        return part.lower().strip()

    all_df["clean_key"] = all_df["file_path"].apply(extract_key)

    leaf_ids = []
    mapped_count = 0
    for _, r in all_df.iterrows():
        k = r["clean_key"]
        v_list = lm.get(k)
        matched_val = None
        if v_list:
            for v in v_list:
                cls_prefix = v.split(":::")[0]
                if cls_prefix == r["class_name"]:
                    matched_val = v
                    break
        if matched_val:
            leaf_ids.append(matched_val)
            mapped_count += 1
        else:
            leaf_ids.append(f"unmapped:::{r['class_name']}:::{k}")

    all_df["leaf_id"] = leaf_ids

    # Greedy 2-way partition per class for evaluation set (val + test)
    all_eval = all_df[all_df["orig_split"].isin(["val", "test"])].copy()
    
    best_assignment = {}
    for cname, grp in all_eval.groupby("class_name"):
        leaf_sizes = grp.groupby("leaf_id").size().to_dict()
        leaves = sorted(leaf_sizes.keys(), key=lambda l: leaf_sizes[l], reverse=True)
        
        val_set = set()
        val_count = 0
        test_set = set()
        test_count = 0
        
        for l in leaves:
            sz = leaf_sizes[l]
            if val_count <= test_count:
                val_set.add(l)
                val_count += sz
            else:
                test_set.add(l)
                test_count += sz
                
        for l in val_set:
            best_assignment[l] = "val"
        for l in test_set:
            best_assignment[l] = "test"

    all_eval["new_split"] = all_eval["leaf_id"].map(best_assignment)
    vc = all_eval["new_split"].value_counts()
    print("Evaluation set greedy grouped partition:")
    print(f"  Val count:  {vc.get('val'):,} ({vc.get('val')/len(all_df)*100:.2f}%)")
    print(f"  Test count: {vc.get('test'):,} ({vc.get('test')/len(all_df)*100:.2f}%)")

    v_leaves = set(all_eval[all_eval["new_split"] == "val"]["leaf_id"])
    t_leaves = set(all_eval[all_eval["new_split"] == "test"]["leaf_id"])
    print(f"  Overlapping leaf IDs between Val and Test: {len(v_leaves & t_leaves)}")
    
    c5 = all_eval[all_eval["leaf_id"] == "Cherry_(including_sour)___Powdery_mildew:::5.0"]
    print(f"  Cherry Powdery Mildew leaf 5.0 in eval: {c5['new_split'].value_counts().to_dict()}")

    diffs = []
    for cname, grp in all_eval.groupby("class_name"):
        c_vc = grp["new_split"].value_counts()
        diffs.append(abs(c_vc.get("val", 0) - c_vc.get("test", 0)))
    print(f"  Max class imbalance between val and test: {max(diffs)} images")
    print(f"  Median class imbalance: {np.median(diffs):.1f} images")

if __name__ == "__main__":
    main()
