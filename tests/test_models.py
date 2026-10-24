"""Tests for model architectures, forward pass shapes, parameter counts, and freezing."""

from __future__ import annotations

import torch
from plantvision.models import (
    CustomBaselineCNN,
    build_model,
    build_transfer_resnet18,
    count_parameters,
    unfreeze_resnet_layers,
)


def test_baseline_cnn_forward_shape():
    """Verify CustomBaselineCNN handles (B, 3, 224, 224) and outputs (B, num_classes)."""
    num_classes = 5
    model = CustomBaselineCNN(num_classes=num_classes)
    dummy_input = torch.randn(2, 3, 224, 224)

    output = model(dummy_input)
    assert output.shape == (2, num_classes)
    assert torch.isfinite(output).all()


def test_transfer_resnet18_frozen_backbone():
    """Verify ResNet-18 has backbone parameters frozen in Stage A and classifier trainable."""
    num_classes = 10
    model = build_transfer_resnet18(num_classes=num_classes, pretrained=False, freeze_backbone=True)

    params = count_parameters(model)
    assert params["frozen_parameters"] > 0
    assert params["trainable_parameters"] > 0
    # Classifier linear layer weights must be trainable
    assert model.fc[1].weight.requires_grad is True
    # Initial conv layer must be frozen
    assert model.conv1.weight.requires_grad is False


def test_transfer_resnet18_unfreeze_stage_b():
    """Verify Stage B unfreezes layer4 parameters."""
    num_classes = 10
    model = build_transfer_resnet18(num_classes=num_classes, pretrained=False, freeze_backbone=True)

    unfreeze_resnet_layers(model, unfreeze_layers=["layer4", "fc"])
    for param in model.layer4.parameters():
        assert param.requires_grad is True
    # Earlier layers (e.g. layer1) remain frozen
    for param in model.layer1.parameters():
        assert param.requires_grad is False


def test_model_factory_dispatch():
    """Verify build_model dispatches correctly based on architecture string."""
    m_base = build_model("baseline", num_classes=4)
    assert isinstance(m_base, CustomBaselineCNN)

    m_tl = build_model("resnet18", num_classes=4, pretrained=False)
    assert hasattr(m_tl, "fc")
