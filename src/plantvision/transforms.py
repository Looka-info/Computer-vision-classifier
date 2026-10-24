"""Image preprocessing and augmentation pipelines for training and evaluation."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from torchvision import transforms


def get_train_transforms(
    image_size: int = 224,
    norm_mean: Optional[List[float]] = None,
    norm_std: Optional[List[float]] = None,
    hflip_prob: float = 0.5,
    rotation_degrees: int = 15,
    color_jitter: Optional[Dict[str, float]] = None,
) -> transforms.Compose:
    """Construct data augmentation pipeline for training split.

    Conservative augmentations ensure disease-specific color and lesion geometry
    remain intact while preventing overfitting.

    Args:
        image_size: Target square image dimension (e.g. 224).
        norm_mean: RGB normalization means (default: ImageNet [0.485, 0.456, 0.406]).
        norm_std: RGB normalization stds (default: ImageNet [0.229, 0.224, 0.225]).
        hflip_prob: Probability of horizontal flip.
        rotation_degrees: Maximum degrees for random rotation.
        color_jitter: Dict with brightness, contrast, saturation, hue deltas.

    Returns:
        torchvision transforms.Compose pipeline.
    """
    if norm_mean is None:
        norm_mean = [0.485, 0.456, 0.406]
    if norm_std is None:
        norm_std = [0.229, 0.224, 0.225]

    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=hflip_prob),
    ]

    if rotation_degrees > 0:
        transform_list.append(transforms.RandomRotation(degrees=rotation_degrees))

    if color_jitter:
        transform_list.append(
            transforms.ColorJitter(
                brightness=color_jitter.get("brightness", 0.0),
                contrast=color_jitter.get("contrast", 0.0),
                saturation=color_jitter.get("saturation", 0.0),
                hue=color_jitter.get("hue", 0.0),
            )
        )

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    return transforms.Compose(transform_list)


def get_eval_transforms(
    image_size: int = 224,
    norm_mean: Optional[List[float]] = None,
    norm_std: Optional[List[float]] = None,
) -> transforms.Compose:
    """Construct deterministic preprocessing pipeline for validation, testing, and inference.

    No stochastic augmentation is performed.

    Args:
        image_size: Target square image dimension.
        norm_mean: RGB normalization means.
        norm_std: RGB normalization stds.

    Returns:
        torchvision transforms.Compose pipeline.
    """
    if norm_mean is None:
        norm_mean = [0.485, 0.456, 0.406]
    if norm_std is None:
        norm_std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])
