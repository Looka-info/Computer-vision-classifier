"""Model architectures: Baseline scratch CNN and Transfer Learning ResNet-18."""

from __future__ import annotations

from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torchvision import models


class CustomBaselineCNN(nn.Module):
    """Compact Convolutional Neural Network trained from scratch as an empirical baseline.

    Features:
    - 4 convolutional blocks with batch normalization and ReLU activations.
    - Progressive channel scaling (32 -> 64 -> 128 -> 256).
    - Spatial reduction via 2x2 max-pooling and global adaptive average pooling.
    - Regularization via dropout to mitigate overfitting.
    - Classification head tailored to target class count.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        conv_channels: Optional[List[int]] = None,
        fc_hidden: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64, 128, 256]

        self.num_classes = num_classes

        # Feature extractor
        layers: List[nn.Module] = []
        c_in = in_channels
        for idx, c_out in enumerate(conv_channels):
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.ReLU(inplace=True))
            if idx < len(conv_channels) - 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            c_in = c_out

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*layers)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(conv_channels[-1], fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        logits = self.classifier(feats)
        return logits


def build_transfer_resnet18(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """Build ResNet-18 model with custom classification head for transfer learning.

    Stage A: Feature extractor frozen, only classification head trained.
    Stage B: Fine-tuning activated via unfreeze_resnet_layers().

    Args:
        num_classes: Number of leaf disease classes.
        pretrained: If True, load torchvision IMAGENET1K_V1 official weights.
        freeze_backbone: If True, set requires_grad=False on all backbone parameters.
        dropout: Dropout rate before final linear layer.

    Returns:
        PyTorch ResNet-18 model instance.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    # Freeze backbone parameters if requested (Stage A)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final linear layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )

    # Ensure classifier head parameters are trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def unfreeze_resnet_layers(
    model: nn.Module,
    unfreeze_layers: Optional[List[str]] = None,
) -> None:
    """Unfreeze selected high-level residual blocks for Stage B fine-tuning.

    Args:
        model: ResNet model instance.
        unfreeze_layers: List of layer names to unfreeze (e.g. ['layer4', 'fc']).
    """
    if unfreeze_layers is None:
        unfreeze_layers = ["layer4", "fc"]

    for name, child in model.named_children():
        if name in unfreeze_layers:
            for param in child.parameters():
                param.requires_grad = True


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Calculate total, trainable, and frozen parameters of a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
    }


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    **kwargs,
) -> nn.Module:
    """Factory function to build baseline or transfer learning architectures."""
    normalized_name = model_name.lower()
    if "baseline" in normalized_name:
        return CustomBaselineCNN(
            num_classes=num_classes,
            conv_channels=kwargs.get("conv_channels", [32, 64, 128, 256]),
            fc_hidden=kwargs.get("fc_hidden", 256),
            dropout=kwargs.get("dropout", 0.3),
        )
    elif "resnet" in normalized_name:
        return build_transfer_resnet18(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=kwargs.get("dropout", 0.3),
        )
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")
