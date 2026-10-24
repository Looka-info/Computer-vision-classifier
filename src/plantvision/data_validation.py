"""Data validation, integrity checking, duplicate detection, and leak-free splitting."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass
class DatasetScanResult:
    """Encapsulates dataset scan and validation diagnostics."""

    raw_dir: Path
    total_files_scanned: int
    valid_count: int
    corrupt_count: int
    num_classes: int
    classes: List[str]
    df_valid: pd.DataFrame
    df_corrupt: pd.DataFrame
    duplicate_groups: int
    exact_duplicate_count: int
    class_counts: Dict[str, int] = field(default_factory=dict)
    channel_counts: Dict[int, int] = field(default_factory=dict)
    resolution_counts: Dict[Tuple[int, int], int] = field(default_factory=dict)


def compute_file_hash(path: Path | str, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file for exact duplicate detection.

    Args:
        path: Path to the image file.
        chunk_size: Bytes to read per chunk.

    Returns:
        Hexadecimal SHA-256 string.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dhash(img: Image.Image, hash_size: int = 8) -> int:
    """Compute 64-bit difference hash (dHash) for visual near-duplicate detection.

    Args:
        img: PIL Image to hash.
        hash_size: Grid size (default 8 produces 8x8 = 64-bit hash).

    Returns:
        64-bit integer representation of gradient structure.
    """
    resized = img.resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR).convert("L")
    pixels = np.asarray(resized)
    diff = pixels[:, 1:] > pixels[:, :-1]
    val = 0
    for idx, bit in enumerate(diff.flatten()):
        if bit:
            val |= 1 << idx
    return val


def hamming_distance(h1: int, h2: int) -> int:
    """Calculate bitwise Hamming distance between two 64-bit integers."""
    return bin(h1 ^ h2).count("1")


def inspect_image_file(path: Path) -> Tuple[bool, Optional[str], Optional[Tuple[int, int]], Optional[int]]:
    """Attempt to open and verify image file, extracting metadata.

    Args:
        path: Path to image file.

    Returns:
        Tuple of (is_valid, error_reason, (width, height), channel_count).
    """
    try:
        with Image.open(path) as img:
            img.verify()
        # verify() closes or invalidates the image instance, so reopen for dimensions
        with Image.open(path) as img:
            width, height = img.size
            channels = len(img.getbands())
            # Basic sanity check
            if width <= 0 or height <= 0:
                return False, "Invalid dimensions (width or height <= 0)", None, None
            return True, None, (width, height), channels
    except Exception as exc:
        return False, f"Decode failure: {type(exc).__name__}: {str(exc)}", None, None


def scan_dataset(
    raw_dir: Path | str,
    supported_extensions: Optional[List[str]] = None,
) -> DatasetScanResult:
    """Recursively scan class directories in raw_dir and validate all images.

    Args:
        raw_dir: Path to directory containing class subdirectories.
        supported_extensions: Allowed file extensions.

    Returns:
        DatasetScanResult with detailed findings.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Dataset root directory does not exist: {raw_path}")

    if supported_extensions is None:
        supported_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    supported_ext_set = {ext.lower() for ext in supported_extensions}

    # Discover class directories (exclude hidden directories)
    class_dirs = sorted([d for d in raw_path.iterdir() if d.is_dir() and not d.name.startswith(".")])
    if not class_dirs:
        raise ValueError(f"No class subdirectories found in: {raw_path}")

    class_names = [d.name for d in class_dirs]
    valid_records = []
    corrupt_records = []
    hash_map: Dict[str, List[Path]] = {}

    channel_counts: Dict[int, int] = {}
    res_counts: Dict[Tuple[int, int], int] = {}

    total_scanned = 0

    for class_dir in class_dirs:
        class_name = class_dir.name
        # Find all files with supported extensions
        files = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in supported_ext_set
        ]

        for file_path in files:
            total_scanned += 1
            is_valid, err_reason, dims, channels = inspect_image_file(file_path)

            if not is_valid:
                corrupt_records.append({
                    "file_path": str(file_path),
                    "class_name": class_name,
                    "reason": err_reason,
                })
                continue

            file_hash = compute_file_hash(file_path)
            hash_map.setdefault(file_hash, []).append(file_path)

            width, height = dims if dims else (0, 0)
            channel_counts[channels] = channel_counts.get(channels, 0) + 1
            res_counts[(width, height)] = res_counts.get((width, height), 0) + 1

            valid_records.append({
                "file_path": str(file_path),
                "relative_path": str(file_path.relative_to(raw_path)),
                "class_name": class_name,
                "file_hash": file_hash,
                "width": width,
                "height": height,
                "channels": channels,
                "file_size_bytes": file_path.stat().st_size,
            })

    df_valid = pd.DataFrame(valid_records)
    df_corrupt = pd.DataFrame(corrupt_records)

    # Class-level counts
    if not df_valid.empty:
        class_counts = df_valid["class_name"].value_counts().to_dict()
    else:
        class_counts = {}

    # Exact duplicate detection
    duplicate_groups = sum(1 for h, paths in hash_map.items() if len(paths) > 1)
    exact_duplicate_count = sum(len(paths) - 1 for h, paths in hash_map.items() if len(paths) > 1)

    return DatasetScanResult(
        raw_dir=raw_path,
        total_files_scanned=total_scanned,
        valid_count=len(df_valid),
        corrupt_count=len(df_corrupt),
        num_classes=len(class_names),
        classes=class_names,
        df_valid=df_valid,
        df_corrupt=df_corrupt,
        duplicate_groups=duplicate_groups,
        exact_duplicate_count=exact_duplicate_count,
        class_counts=class_counts,
        channel_counts=channel_counts,
        resolution_counts=res_counts,
    )


def create_leak_free_splits(
    df_valid: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    deduplicate_by_hash: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create reproducible train, val, and test splits with zero duplicate leakage.

    Guarantees:
    1. If deduplicate_by_hash is True, exact duplicate files are pruned to a single instance.
    2. No content hash appears in more than one split (provably zero data leakage).
    3. Stratified splitting preserves class distribution.
    4. Deterministic with fixed random_seed.

    Args:
        df_valid: DataFrame of valid records from scan_dataset.
        train_ratio: Fraction for training (e.g. 0.70).
        val_ratio: Fraction for validation (e.g. 0.15).
        test_ratio: Fraction for held-out test (e.g. 0.15).
        random_seed: Reproducibility seed.
        deduplicate_by_hash: Whether to keep only one copy per unique file hash.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    if df_valid.empty:
        raise ValueError("Cannot create splits from an empty valid dataframe.")

    np.testing.assert_almost_equal(train_ratio + val_ratio + test_ratio, 1.0, decimal=3)

    working_df = df_valid.copy()
    if deduplicate_by_hash:
        # Keep first occurrence of each unique file content hash
        working_df = working_df.drop_duplicates(subset=["file_hash"]).reset_index(drop=True)

    # Encode class labels numerically
    classes = sorted(working_df["class_name"].unique())
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    working_df["label"] = working_df["class_name"].map(class_to_idx)

    # Check for small classes that cannot be stratified
    class_counts = working_df["label"].value_counts()
    min_count = class_counts.min()
    if min_count < 3:
        logger.warning(
            "Some classes have fewer than 3 samples (%s). Stratification will fall back to shuffle.",
            min_count,
        )
        stratify_col = None
    else:
        stratify_col = working_df["label"]

    # First split: Train vs Temp (val + test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        working_df,
        test_size=temp_ratio,
        random_state=random_seed,
        stratify=stratify_col,
    )

    # Second split: Val vs Test
    test_rel_ratio = test_ratio / temp_ratio
    temp_stratify = temp_df["label"] if stratify_col is not None else None

    # Handle rare case where a class in temp_df has count 1
    if temp_stratify is not None and temp_stratify.value_counts().min() < 2:
        temp_stratify = None

    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_rel_ratio,
        random_state=random_seed,
        stratify=temp_stratify,
    )

    # Explicit cross-split duplicate verification
    train_hashes = set(train_df["file_hash"])
    val_hashes = set(val_df["file_hash"])
    test_hashes = set(test_df["file_hash"])

    train_val_overlap = train_hashes.intersection(val_hashes)
    train_test_overlap = train_hashes.intersection(test_hashes)
    val_test_overlap = val_hashes.intersection(test_hashes)

    if train_val_overlap or train_test_overlap or val_test_overlap:
        raise RuntimeError(
            f"DATA LEAKAGE DETECTED! Duplicate hashes between splits: "
            f"train/val={len(train_val_overlap)}, "
            f"train/test={len(train_test_overlap)}, "
            f"val/test={len(val_test_overlap)}"
        )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_split_manifests(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    splits_dir: Path | str,
    class_to_idx: Optional[Dict[str, int]] = None,
) -> None:
    """Save split CSV manifests and class mapping JSON to disk."""
    out_dir = Path(splits_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    if class_to_idx is None:
        all_classes = sorted(train_df["class_name"].unique())
        class_to_idx = {name: idx for idx, name in enumerate(all_classes)}

    with open(out_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)

    logger.info("Saved split manifests and class mapping to %s", out_dir)


def plot_class_distribution(
    class_counts: Dict[str, int],
    output_path: Path | str,
    title: str = "PlantVision - Dataset Class Distribution",
) -> None:
    """Render and save a readable class distribution bar chart."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    sorted_items = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [k.replace("___", "\n") for k, _ in sorted_items]
    counts = [v for _, v in sorted_items]

    plt.figure(figsize=(14, max(6, len(labels) * 0.35)))
    y_pos = np.arange(len(labels))
    plt.barh(y_pos, counts, color="#2563eb", edgecolor="#1d4ed8", alpha=0.85)
    plt.yticks(y_pos, labels, fontsize=8)
    plt.xlabel("Number of Images", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.gca().invert_yaxis()
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    for i, v in enumerate(counts):
        plt.text(v + max(counts) * 0.01, i, f" {v}", va="center", fontsize=8, color="#1e293b")

    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()
    logger.info("Saved class distribution chart to %s", out_file)
