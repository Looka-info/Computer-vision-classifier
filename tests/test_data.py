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
