"""Tests for image transformation and augmentation pipelines."""

from __future__ import annotations

import numpy as np
from PIL import Image
import torch
from plantvision.transforms import get_eval_transforms, get_train_transforms


def test_eval_transform_shape_and_type():
    """Verify eval transform produces correct tensor shape (3, H, W) and normalized float32 type."""
    img = Image.new("RGB", (300, 400), color=(100, 150, 200))
    tf = get_eval_transforms(image_size=224)

    tensor = tf(img)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_train_transform_shape_and_type():
    """Verify train transform with augmentation produces consistent output dimensions."""
    img = Image.new("RGB", (128, 128), color=(50, 100, 150))
    tf = get_train_transforms(
        image_size=224,
        hflip_prob=0.5,
        rotation_degrees=15,
        color_jitter={"brightness": 0.1, "contrast": 0.1, "saturation": 0.1, "hue": 0.02},
    )

    tensor = tf(img)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_eval_transform_deterministic():
    """Verify eval transform is strictly deterministic across repeated runs."""
    img = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
    tf = get_eval_transforms(image_size=112)

    t1 = tf(img)
    t2 = tf(img)
    assert torch.allclose(t1, t2)
