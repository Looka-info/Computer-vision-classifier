"""Model training engine with metric tracking, early stopping, and checkpoint metadata."""

from __future__ import annotations

import datetime
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler
from torch.utils.data import DataLoader

from plantvision.models import count_parameters, unfreeze_resnet_layers

logger = logging.getLogger(__name__)


def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Run validation loop over dataloader and compute classification metrics.

    Args:
        model: PyTorch model.
        dataloader: Validation DataLoader.
        criterion: Loss function.
        device: Computation device.

    Returns:
        Dict containing val_loss, val_accuracy, val_macro_f1, val_macro_precision, val_macro_recall.
    """
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.inference_mode():
        for inputs, targets, _ in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    dataset_size = len(dataloader.dataset)
    avg_loss = total_loss / max(dataset_size, 1)

    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    macro_prec = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    macro_rec = recall_score(all_targets, all_preds, average="macro", zero_division=0)

    return {
        "val_loss": float(avg_loss),
        "val_accuracy": float(acc),
        "val_macro_f1": float(macro_f1),
        "val_macro_precision": float(macro_prec),
        "val_macro_recall": float(macro_rec),
    }


def save_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    classes: List[str],
    class_to_idx: Dict[str, int],
    config: Dict[str, Any],
    model_name: str,
    training_duration_sec: float,
) -> None:
    """Save model checkpoint with full self-contained deployment metadata."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model_name": model_name,
        "architecture": model.__class__.__name__,
        "epoch": epoch,
        "classes": classes,
        "class_to_idx": class_to_idx,
        "num_classes": len(classes),
        "metrics": metrics,
        "input_dimensions": [3, config["preprocessing"]["image_size"], config["preprocessing"]["image_size"]],
        "norm_mean": config["preprocessing"]["norm_mean"],
        "norm_std": config["preprocessing"]["norm_std"],
        "random_seed": config["project"]["random_seed"],
        "selected_checkpoint_rule": config["training"]["checkpoint_selection_metric"],
        "training_duration_seconds": round(training_duration_sec, 2),
        "date_trained": datetime.datetime.now().isoformat(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "parameters": count_parameters(model),
    }

    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metadata": metadata,
    }

    torch.save(checkpoint, out_path)
    logger.info("Saved best checkpoint to %s (Val Macro-F1: %.4f)", out_path, metrics.get("val_macro_f1", 0.0))


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    classes: List[str],
    class_to_idx: Dict[str, int],
    checkpoint_path: Path | str,
    epochs: int = 10,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    class_weights: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    optimizer_name: str = "adamw",
) -> Dict[str, Any]:
    """Train a single-stage model (e.g. scratch baseline) and save best validation checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    if optimizer_name.lower() == "adam":
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.05)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "val_macro_precision": [],
        "val_macro_recall": [],
        "learning_rates": [],
    }

    best_metric_val = -1.0
    best_epoch = -1
    best_metrics: Dict[str, float] = {}
    patience = config["training"].get("early_stopping_patience", 4)
    patience_counter = 0

    start_time = time.time()
    logger.info("Beginning training on %s for %d epochs...", device, epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_samples = 0

        for inputs, targets, _ in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == targets).sum().item()
            total_samples += inputs.size(0)

        scheduler.step()

        epoch_train_loss = train_loss / max(total_samples, 1)
        epoch_train_acc = train_correct / max(total_samples, 1)
        val_metrics = evaluate_epoch(model, val_loader, criterion, device)

        curr_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(epoch_train_loss)
        history["train_accuracy"].append(epoch_train_acc)
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_accuracy"].append(val_metrics["val_accuracy"])
        history["val_macro_f1"].append(val_metrics["val_macro_f1"])
        history["val_macro_precision"].append(val_metrics["val_macro_precision"])
        history["val_macro_recall"].append(val_metrics["val_macro_recall"])
        history["learning_rates"].append(curr_lr)

        logger.info(
            "Epoch %d/%d | Train Loss: %.4f | Train Acc: %.4f | Val Loss: %.4f | Val Acc: %.4f | Val F1: %.4f | LR: %.6f",
            epoch, epochs, epoch_train_loss, epoch_train_acc,
            val_metrics["val_loss"], val_metrics["val_accuracy"], val_metrics["val_macro_f1"], curr_lr,
        )

        metric_to_check = val_metrics["val_macro_f1"]
        if metric_to_check > best_metric_val:
            best_metric_val = metric_to_check
            best_epoch = epoch
            best_metrics = val_metrics
            patience_counter = 0
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                classes=classes,
                class_to_idx=class_to_idx,
                config=config,
                model_name=config["baseline"].get("model_name", "baseline_cnn"),
                training_duration_sec=time.time() - start_time,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered after %d epochs without improvement.", patience)
                break

    total_duration = time.time() - start_time
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "total_duration_sec": total_duration,
    }


def train_transfer_learning(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    classes: List[str],
    class_to_idx: Dict[str, int],
    checkpoint_path: Path | str,
    class_weights: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Execute two-stage transfer learning: Stage A (frozen backbone) + Stage B (fine-tuning)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    tl_cfg = config["transfer_learning"]

    stage_a_epochs = tl_cfg.get("stage_a_epochs", 4)
    stage_b_epochs = tl_cfg.get("stage_b_epochs", 6)
    total_epochs = stage_a_epochs + stage_b_epochs

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "val_macro_precision": [],
        "val_macro_recall": [],
        "learning_rates": [],
    }

    best_metric_val = -1.0
    best_epoch = -1
    best_metrics: Dict[str, float] = {}
    start_time = time.time()

    # --- STAGE A: Train Classifier Head Only ---
    logger.info("=== Starting Stage A: Classifier Head Training (%d epochs) ===", stage_a_epochs)
    optimizer_a = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=tl_cfg.get("stage_a_lr", 0.001),
        weight_decay=tl_cfg.get("weight_decay", 0.0001),
    )
    scheduler_a = CosineAnnealingLR(optimizer_a, T_max=stage_a_epochs)

    current_epoch = 0
    for epoch in range(1, stage_a_epochs + 1):
        current_epoch += 1
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_samples = 0

        for inputs, targets, _ in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer_a.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_a.step()

            train_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == targets).sum().item()
            total_samples += inputs.size(0)

        scheduler_a.step()
        epoch_loss = train_loss / max(total_samples, 1)
        epoch_acc = train_correct / max(total_samples, 1)
        val_metrics = evaluate_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(epoch_loss)
        history["train_accuracy"].append(epoch_acc)
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_accuracy"].append(val_metrics["val_accuracy"])
        history["val_macro_f1"].append(val_metrics["val_macro_f1"])
        history["val_macro_precision"].append(val_metrics["val_macro_precision"])
        history["val_macro_recall"].append(val_metrics["val_macro_recall"])
        history["learning_rates"].append(optimizer_a.param_groups[0]["lr"])

        logger.info(
            "Stage A - Epoch %d/%d | Train Loss: %.4f | Train Acc: %.4f | Val Acc: %.4f | Val F1: %.4f",
            epoch, stage_a_epochs, epoch_loss, epoch_acc, val_metrics["val_accuracy"], val_metrics["val_macro_f1"],
        )

        if val_metrics["val_macro_f1"] > best_metric_val:
            best_metric_val = val_metrics["val_macro_f1"]
            best_epoch = current_epoch
            best_metrics = val_metrics
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer_a,
                epoch=current_epoch,
                metrics=val_metrics,
                classes=classes,
                class_to_idx=class_to_idx,
                config=config,
                model_name=tl_cfg.get("model_name", "resnet18_transfer"),
                training_duration_sec=time.time() - start_time,
            )

    # --- STAGE B: Unfreeze High-Level Layers and Fine-Tune with Differential LR ---
    logger.info("=== Starting Stage B: Fine-Tuning Residual Block 4 (%d epochs) ===", stage_b_epochs)
    unfreeze_resnet_layers(model, unfreeze_layers=tl_cfg.get("unfreeze_layers", ["layer4", "fc"]))

    # Differential learning rate: smaller for backbone, slightly larger for fc
    backbone_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "fc" in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    optimizer_b = AdamW(
        [
            {"params": backbone_params, "lr": tl_cfg.get("stage_b_backbone_lr", 0.0001)},
            {"params": classifier_params, "lr": tl_cfg.get("stage_b_classifier_lr", 0.0005)},
        ],
        weight_decay=tl_cfg.get("weight_decay", 0.0001),
    )
    scheduler_b = CosineAnnealingLR(optimizer_b, T_max=stage_b_epochs)

    for epoch in range(1, stage_b_epochs + 1):
        current_epoch += 1
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_samples = 0

        for inputs, targets, _ in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer_b.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_b.step()

            train_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == targets).sum().item()
            total_samples += inputs.size(0)

        scheduler_b.step()
        epoch_loss = train_loss / max(total_samples, 1)
        epoch_acc = train_correct / max(total_samples, 1)
        val_metrics = evaluate_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(epoch_loss)
        history["train_accuracy"].append(epoch_acc)
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_accuracy"].append(val_metrics["val_accuracy"])
        history["val_macro_f1"].append(val_metrics["val_macro_f1"])
        history["val_macro_precision"].append(val_metrics["val_macro_precision"])
        history["val_macro_recall"].append(val_metrics["val_macro_recall"])
        history["learning_rates"].append(optimizer_b.param_groups[0]["lr"])

        logger.info(
            "Stage B - Epoch %d/%d (Total: %d) | Train Loss: %.4f | Val Acc: %.4f | Val F1: %.4f",
            epoch, stage_b_epochs, current_epoch, epoch_loss, val_metrics["val_accuracy"], val_metrics["val_macro_f1"],
        )

        if val_metrics["val_macro_f1"] > best_metric_val:
            best_metric_val = val_metrics["val_macro_f1"]
            best_epoch = current_epoch
            best_metrics = val_metrics
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer_b,
                epoch=current_epoch,
                metrics=val_metrics,
                classes=classes,
                class_to_idx=class_to_idx,
                config=config,
                model_name=tl_cfg.get("model_name", "resnet18_transfer"),
                training_duration_sec=time.time() - start_time,
            )

    total_duration = time.time() - start_time
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "total_duration_sec": total_duration,
    }
