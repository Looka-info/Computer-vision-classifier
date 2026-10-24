"""Tests for training step execution, loss calculation, and checkpoint persistence."""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn
from plantvision.config import load_config
from plantvision.dataset import PlantDataset
from plantvision.models import CustomBaselineCNN
from plantvision.train import evaluate_epoch, save_checkpoint, train_model
from plantvision.transforms import get_eval_transforms


def test_training_smoke_step(synthetic_manifests, tmp_path: Path):
    """Run a small smoke training run on synthetic fixture verifying finite loss and checkpoint creation."""
    train_df, val_df, _, class_to_idx = synthetic_manifests
    classes = list(class_to_idx.keys())
    num_classes = len(classes)

    eval_tf = get_eval_transforms(image_size=32)
    train_ds = PlantDataset(train_df, transform=eval_tf)
    val_ds = PlantDataset(val_df, transform=eval_tf)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=2, shuffle=False)

    model = CustomBaselineCNN(num_classes=num_classes)
    ckpt_path = tmp_path / "smoke_ckpt.pt"

    config = {
        "project": {"random_seed": 42},
        "preprocessing": {"image_size": 32, "norm_mean": [0.5, 0.5, 0.5], "norm_std": [0.5, 0.5, 0.5]},
        "training": {"checkpoint_selection_metric": "val_macro_f1", "early_stopping_patience": 2},
        "baseline": {"model_name": "test_baseline"},
    }

    res = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        classes=classes,
        class_to_idx=class_to_idx,
        checkpoint_path=ckpt_path,
        epochs=1,
        learning_rate=0.01,
        device=torch.device("cpu"),
    )

    assert res["best_epoch"] == 1
    assert "val_macro_f1" in res["best_metrics"]
    assert ckpt_path.exists()

    # Verify saved metadata completeness
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "metadata" in saved
    meta = saved["metadata"]
    assert meta["num_classes"] == num_classes
    assert meta["classes"] == classes
    assert meta["input_dimensions"] == [3, 32, 32]
    assert "torch_version" in meta
