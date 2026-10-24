"""Dataset classes, DataLoaders, and batch builders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from plantvision.transforms import get_eval_transforms, get_train_transforms


class PlantDataset(Dataset):
    """PyTorch Dataset for PlantVision leaf images.

    Loads images according to a manifest DataFrame or CSV path, ensuring RGB conversion.
    """

    def __init__(
        self,
        manifest: Union[pd.DataFrame, str, Path],
        transform: Optional[Callable] = None,
        raw_root: Optional[Union[str, Path]] = None,
    ):
        """Initialize PlantDataset.

        Args:
            manifest: DataFrame or path to CSV split manifest containing 'file_path' and 'label'.
            transform: Optional torchvision transform to apply to PIL Image.
            raw_root: Optional root directory if 'file_path' is relative.
        """
        if isinstance(manifest, (str, Path)):
            self.df = pd.read_csv(manifest)
        else:
            self.df = manifest.copy().reset_index(drop=True)

        self.transform = transform
        self.raw_root = Path(raw_root) if raw_root else None

        if "file_path" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("Manifest must contain 'file_path' and 'label' columns.")

        self.file_paths = self.df["file_path"].tolist()
        self.labels = self.df["label"].astype(int).tolist()
        self.class_names = (
            self.df["class_name"].tolist() if "class_name" in self.df.columns else None
        )

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path_str = self.file_paths[idx]
        img_path = Path(path_str)
        if self.raw_root and not img_path.is_absolute():
            img_path = self.raw_root / img_path

        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Error loading image at {img_path}: {exc}") from exc

        if self.transform is not None:
            tensor = self.transform(img_rgb)
        else:
            # Fallback to basic tensor conversion
            tensor = transforms.ToTensor()(img_rgb)

        label = self.labels[idx]
        return tensor, label, str(img_path)


def compute_class_weights(
    train_df: pd.DataFrame,
    num_classes: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Calculate balanced inverse-frequency class weights for cross-entropy loss.

    Formula: N / (num_classes * count_i)

    Args:
        train_df: Manifest of the training split.
        num_classes: Total number of classes.
        device: PyTorch device.

    Returns:
        torch.Tensor of class weights.
    """
    counts = train_df["label"].value_counts().to_dict()
    total_samples = len(train_df)
    weights = []

    for idx in range(num_classes):
        c = counts.get(idx, 1)
        w = total_samples / (num_classes * c)
        weights.append(w)

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    # Normalize weights so mean is 1.0
    weights_tensor = weights_tensor / weights_tensor.mean()

    if device is not None:
        weights_tensor = weights_tensor.to(device)

    return weights_tensor


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    image_size: int = 224,
    norm_mean: Optional[List[float]] = None,
    norm_std: Optional[List[float]] = None,
    batch_size: int = 32,
    num_workers: int = 2,
    color_jitter: Optional[Dict[str, float]] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Build training, validation, and optional test DataLoaders.

    Args:
        train_df: Manifest for training.
        val_df: Manifest for validation.
        test_df: Optional manifest for testing.
        image_size: Image resolution.
        norm_mean: Normalization mean.
        norm_std: Normalization std.
        batch_size: Batch size.
        num_workers: Worker processes.
        color_jitter: Training color jitter settings.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_tf = get_train_transforms(
        image_size=image_size,
        norm_mean=norm_mean,
        norm_std=norm_std,
        color_jitter=color_jitter,
    )
    eval_tf = get_eval_transforms(
        image_size=image_size,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )

    train_ds = PlantDataset(train_df, transform=train_tf)
    val_ds = PlantDataset(val_df, transform=eval_tf)
    test_ds = PlantDataset(test_df, transform=eval_tf) if test_df is not None else None

    # On Windows, persistent workers with pin_memory=True and num_workers >= 0
    pin_mem = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False,
    )

    test_loader = (
        DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=False,
        )
        if test_ds is not None
        else None
    )

    return train_loader, val_loader, test_loader
