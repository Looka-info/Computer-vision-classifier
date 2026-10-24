"""Pytest configuration and shared test fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch


@pytest.fixture
def synthetic_dataset_dir(tmp_path: Path) -> Path:
    """Create a temporary synthetic dataset with 3 classes, valid and duplicate images."""
    data_dir = tmp_path / "synthetic_raw"
    data_dir.mkdir()

    classes = ["Apple___healthy", "Apple___scab", "Tomato___healthy"]
    for cls in classes:
        cls_dir = data_dir / cls
        cls_dir.mkdir()
        # Create 5 distinct images per class
        for i in range(5):
            arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            img.save(cls_dir / f"img_{i}.jpg")

    # Add an exact duplicate in Apple___healthy
    duplicate_source = data_dir / "Apple___healthy" / "img_0.jpg"
    duplicate_target = data_dir / "Apple___healthy" / "img_0_copy.jpg"
    with open(duplicate_source, "rb") as src, open(duplicate_target, "wb") as dst:
        dst.write(src.read())

    # Add an intentionally unreadable/corrupt image
    corrupt_file = data_dir / "Tomato___healthy" / "corrupt_sample.jpg"
    with open(corrupt_file, "wb") as f:
        f.write(b"NOT_A_VALID_JPEG_HEADER_CORRUPTED_BYTES")

    return data_dir


@pytest.fixture
def synthetic_manifests(tmp_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Create synthetic train/val/test manifests with tiny images for fast testing."""
    img_dir = tmp_path / "test_imgs"
    img_dir.mkdir()

    classes = ["Apple___healthy", "Apple___scab", "Tomato___healthy"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    records = []
    for cls_name, cls_idx in class_to_idx.items():
        for i in range(6):
            p = img_dir / f"{cls_name}_{i}.jpg"
            arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(arr).save(p)
            records.append({
                "file_path": str(p),
                "class_name": cls_name,
                "label": cls_idx,
                "file_hash": f"hash_{cls_name}_{i}",
                "width": 32,
                "height": 32,
                "channels": 3,
            })

    df = pd.DataFrame(records)
    # 4 train, 1 val, 1 test per class
    train_df = df.groupby("label").head(4).reset_index(drop=True)
    temp_df = df.groupby("label").tail(2).reset_index(drop=True)
    val_df = temp_df.groupby("label").head(1).reset_index(drop=True)
    test_df = temp_df.groupby("label").tail(1).reset_index(drop=True)

    return train_df, val_df, test_df, class_to_idx
