"""Tests for dataset scanning, integrity checks, duplicate detection, and splitting."""

from __future__ import annotations

from pathlib import Path
import pytest
from plantvision.data_validation import (
    compute_file_hash,
    create_leak_free_splits,
    inspect_image_file,
    scan_dataset,
)


def test_missing_dataset_directory(tmp_path: Path):
    """Test that scanning a non-existent directory raises FileNotFoundError."""
    missing_dir = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        scan_dataset(missing_dir)


def test_empty_class_directory(tmp_path: Path):
    """Test handling of directory with no class subfolders."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(ValueError):
        scan_dataset(empty_dir)


def test_scan_dataset_detects_corrupt_and_duplicates(synthetic_dataset_dir: Path):
    """Verify scanner identifies valid images, catches corrupt files, and detects duplicates."""
    result = scan_dataset(synthetic_dataset_dir)

    assert result.num_classes == 3
    assert result.corrupt_count == 1
    assert result.valid_count == 16  # 3*5 = 15 + 1 duplicate = 16
    assert result.duplicate_groups >= 1
    assert result.exact_duplicate_count >= 1

    # Check corrupt record content
    corrupt_df = result.df_corrupt
    assert len(corrupt_df) == 1
    assert "Decode failure" in corrupt_df.iloc[0]["reason"]


def test_create_leak_free_splits_no_cross_split_duplicates(synthetic_dataset_dir: Path):
    """Verify that split creation guarantees zero duplicate hashes between train, val, and test."""
    result = scan_dataset(synthetic_dataset_dir)
    train_df, val_df, test_df = create_leak_free_splits(
        df_valid=result.df_valid,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        random_seed=42,
        deduplicate_by_hash=True,
    )

    train_hashes = set(train_df["file_hash"])
    val_hashes = set(val_df["file_hash"])
    test_hashes = set(test_df["file_hash"])

    assert len(train_hashes.intersection(val_hashes)) == 0, "Leakage between train and val"
    assert len(train_hashes.intersection(test_hashes)) == 0, "Leakage between train and test"
    assert len(val_hashes.intersection(test_hashes)) == 0, "Leakage between val and test"

    # All classes represented in train
    assert set(train_df["class_name"]) == set(result.classes)


def test_inspect_image_file_valid_and_invalid(tmp_path: Path):
    """Test low-level inspect_image_file helper on valid and corrupted images."""
    from PIL import Image
    import numpy as np

    valid_p = tmp_path / "valid.png"
    Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(valid_p)

    is_valid, err, dims, channels = inspect_image_file(valid_p)
    assert is_valid is True
    assert err is None
    assert dims == (10, 10)
    assert channels == 3

    corrupt_p = tmp_path / "corrupt.png"
    corrupt_p.write_bytes(b"garbage")
    is_valid_c, err_c, _, _ = inspect_image_file(corrupt_p)
    assert is_valid_c is False
    assert err_c is not None


def test_dhash_and_hamming_distance():
    """Verify perceptual difference hash calculation and Hamming distance properties."""
    from PIL import Image
    import numpy as np
    from plantvision.data_validation import compute_dhash, hamming_distance

    # Create two identical images
    arr1 = np.zeros((100, 100, 3), dtype=np.uint8)
    arr1[20:80, 20:80] = 255
    img1 = Image.fromarray(arr1)
    img2 = Image.fromarray(arr1.copy())

    h1 = compute_dhash(img1)
    h2 = compute_dhash(img2)
    assert h1 == h2
    assert hamming_distance(h1, h2) == 0

    # Create distinct image with inverted gradient
    arr3 = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        arr3[:, i] = i * 2
    img3 = Image.fromarray(arr3)
    h3 = compute_dhash(img3)

    assert hamming_distance(h1, h3) > 0
    # Reflexive & symmetric properties
    assert hamming_distance(h1, h1) == 0
    assert hamming_distance(h1, h3) == hamming_distance(h3, h1)


def test_evaluation_splits_zero_specimen_overlap():
    """Verify that current val.csv and test.csv manifests share zero physical leaf specimens."""
    import json
    import pandas as pd

    splits_dir = Path("data/splits")
    leaf_map_p = splits_dir / "leaf_map_raw.json"
    val_p = splits_dir / "val.csv"
    test_p = splits_dir / "test.csv"

    if not (leaf_map_p.exists() and val_p.exists() and test_p.exists()):
        pytest.skip("Split manifests or leaf_map_raw.json not present.")

    with open(leaf_map_p, "r", encoding="utf-8") as f:
        lm = json.load(f)

    def get_leaf_ids(csv_path: Path) -> set:
        df = pd.read_csv(csv_path)
        ids = set()
        for _, r in df.iterrows():
            stem = Path(r["file_path"]).stem
            part = stem.split("___")[-1].lower().strip() if "___" in stem else stem.lower().strip()
            v_list = lm.get(part)
            if v_list:
                for v in v_list:
                    if v.split(":::")[0] == r["class_name"]:
                        ids.add(v)
                        break
        return ids

    val_leaves = get_leaf_ids(val_p)
    test_leaves = get_leaf_ids(test_p)

    overlap = val_leaves.intersection(test_leaves)
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping specimens between val and test: {overlap}"


def test_multi_image_specimen_isolation():
    """Verify that multi-image specimens do not cross validation and test partitions."""
    import json
    import pandas as pd

    splits_dir = Path("data/splits")
    leaf_map_p = splits_dir / "leaf_map_raw.json"
    val_p = splits_dir / "val.csv"
    test_p = splits_dir / "test.csv"

    if not (leaf_map_p.exists() and val_p.exists() and test_p.exists()):
        pytest.skip("Manifests not available.")

    with open(leaf_map_p, "r", encoding="utf-8") as f:
        lm = json.load(f)

    target_leaf = "Cherry_(including_sour)___Powdery_mildew:::5.0"

    def count_leaf(csv_path: Path) -> int:
        df = pd.read_csv(csv_path)
        cnt = 0
        for _, r in df.iterrows():
            if r["class_name"] != "Cherry_(including_sour)___Powdery_mildew":
                continue
            stem = Path(r["file_path"]).stem
            part = stem.split("___")[-1].lower().strip() if "___" in stem else stem.lower().strip()
            v_list = lm.get(part)
            if v_list and target_leaf in v_list:
                cnt += 1
        return cnt

    val_count = count_leaf(val_p)
    test_count = count_leaf(test_p)

    # Must NOT occur in both splits
    assert not (val_count > 0 and test_count > 0), (
        f"Specimen {target_leaf} spans multiple evaluation splits! Val: {val_count}, Test: {test_count}"
    )


def test_near_duplicates_summary_integrity():
    """Verify that near-duplicate summary metrics match record-level audit counts."""
    import json
    import pandas as pd

    summary_p = Path("reports/near_duplicates_summary.json")
    audit_csv_p = Path("reports/near_duplicates_audit.csv")

    if not (summary_p.exists() and audit_csv_p.exists()):
        pytest.skip("Audit artifacts not available.")

    with open(summary_p, "r", encoding="utf-8") as f:
        summary = json.load(f)

    df = pd.read_csv(audit_csv_p)

    total_records = len(df)
    assert total_records == summary["mae_distribution"]["total_pairs"]

    le_15_actual = int((df["pixel_mae"] <= 15.0).sum())
    assert le_15_actual == summary["mae_distribution"]["le_15_count"]

    between_15_20_actual = int(((df["pixel_mae"] > 15.0) & (df["pixel_mae"] <= 20.0)).sum())
    assert between_15_20_actual == summary["mae_distribution"]["between_15_and_20_count"]

    gt_20_actual = int((df["pixel_mae"] > 20.0).sum())
    assert gt_20_actual == summary["mae_distribution"]["gt_20_count"]

    assert summary["likely_duplicates_breakdown"]["train_vs_test_leakage"] == 0
    assert summary["likely_duplicates_breakdown"]["val_vs_test_overlap"] == 0
    assert summary["train_to_test_leakage_free"] is True


