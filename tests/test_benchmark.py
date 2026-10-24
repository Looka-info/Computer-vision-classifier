"""Tests for inference benchmarking routines and hardware profiling."""

from __future__ import annotations

import torch
from plantvision.benchmark import (
    benchmark_model_inference,
    benchmark_preprocessing_latency,
    get_hardware_info,
)
from plantvision.models import CustomBaselineCNN


def test_hardware_info_structure():
    """Verify hardware info contains required system keys."""
    dev = torch.device("cpu")
    info = get_hardware_info(dev)
    assert "device" in info
    assert "platform" in info
    assert "torch_version" in info


def test_benchmark_preprocessing():
    """Verify preprocessing benchmark runs and returns positive latency."""
    res = benchmark_preprocessing_latency(image_size=64, num_iterations=10)
    assert "median_preprocessing_ms" in res
    assert res["median_preprocessing_ms"] > 0


def test_benchmark_model_inference_output():
    """Verify inference benchmarking returns latency and throughput metrics."""
    model = CustomBaselineCNN(num_classes=3)
    dev = torch.device("cpu")

    res = benchmark_model_inference(
        model=model,
        device=dev,
        image_size=64,
        batch_size=2,
        warmup_iterations=2,
        num_iterations=5,
    )

    assert res["batch_size"] == 2
    assert res["median_per_image_latency_ms"] > 0
    assert res["p95_per_image_latency_ms"] >= res["median_per_image_latency_ms"] * 0.5
    assert res["throughput_images_per_sec"] > 0
