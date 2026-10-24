"""Inference benchmarking suite measuring empirical latency, throughput, and hardware profile."""

from __future__ import annotations

import argparse
import logging
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from plantvision.config import load_config
from plantvision.models import CustomBaselineCNN, build_transfer_resnet18, count_parameters
from plantvision.transforms import get_eval_transforms

logger = logging.getLogger(__name__)


def get_hardware_info(device: torch.device) -> Dict[str, Any]:
    """Capture precise hardware and software platform details."""
    info = {
        "device": str(device),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(device)
        info["gpu_capability"] = torch.cuda.get_device_capability(device)
        info["cuda_version"] = torch.version.cuda
    else:
        info["gpu_name"] = "CPU Execution"
    return info


def benchmark_model_inference(
    model: nn.Module,
    device: torch.device,
    image_size: int = 224,
    batch_size: int = 1,
    warmup_iterations: int = 15,
    num_iterations: int = 50,
) -> Dict[str, Any]:
    """Measure steady-state inference latency and throughput.

    Uses torch.cuda.synchronize() for accurate GPU kernel measurement.

    Args:
        model: PyTorch model in eval mode.
        device: CPU or CUDA device.
        image_size: Input resolution.
        batch_size: Number of images per forward pass.
        warmup_iterations: Passes discarded before measurement.
        num_iterations: Number of timed benchmark iterations.

    Returns:
        Dict with median, p95 latency (ms) per image and throughput (img/s).
    """
    model.eval()
    model.to(device)

    dummy_input = torch.randn(batch_size, 3, image_size, image_size, device=device)

    # Warmup
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    latencies_ms: List[float] = []

    with torch.inference_mode():
        for _ in range(num_iterations):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_start = time.perf_counter()

            _ = model(dummy_input)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_end = time.perf_counter()

            batch_latency_ms = (t_end - t_start) * 1000.0
            latencies_ms.append(batch_latency_ms)

    latencies_arr = np.array(latencies_ms)
    median_batch_ms = float(np.median(latencies_arr))
    p95_batch_ms = float(np.percentile(latencies_arr, 95))

    # Per image calculations
    median_per_image_ms = median_batch_ms / batch_size
    p95_per_image_ms = p95_batch_ms / batch_size
    throughput_fps = (batch_size * 1000.0) / median_batch_ms if median_batch_ms > 0 else 0.0

    return {
        "batch_size": batch_size,
        "num_iterations": num_iterations,
        "median_batch_latency_ms": round(median_batch_ms, 3),
        "p95_batch_latency_ms": round(p95_batch_ms, 3),
        "median_per_image_latency_ms": round(median_per_image_ms, 3),
        "p95_per_image_latency_ms": round(p95_per_image_ms, 3),
        "throughput_images_per_sec": round(throughput_fps, 2),
    }


def benchmark_preprocessing_latency(
    image_size: int = 224,
    num_iterations: int = 50,
) -> Dict[str, float]:
    """Measure CPU preprocessing transformation latency on sample PIL images."""
    sample_img = Image.new("RGB", (256, 256), color=(120, 180, 70))
    eval_tf = get_eval_transforms(image_size=image_size)

    # Warmup
    for _ in range(10):
        _ = eval_tf(sample_img)

    times = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        _ = eval_tf(sample_img)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    arr = np.array(times)
    return {
        "median_preprocessing_ms": round(float(np.median(arr)), 3),
        "p95_preprocessing_ms": round(float(np.percentile(arr, 95)), 3),
    }


def run_benchmark_suite(
    model_path: Optional[str] = None,
    architecture: str = "resnet18",
    num_classes: int = 38,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute complete benchmarking suite for specified model or checkpoint."""
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = config["preprocessing"]["image_size"]
    hw_info = get_hardware_info(device)

    # Load model
    if model_path and Path(model_path).exists():
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        meta = ckpt.get("metadata", {})
        arch_name = meta.get("architecture", architecture)
        num_classes = meta.get("num_classes", num_classes)
        if "baseline" in arch_name.lower():
            model = CustomBaselineCNN(num_classes=num_classes)
        else:
            model = build_transfer_resnet18(num_classes=num_classes, pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
    else:
        if "baseline" in architecture.lower():
            model = CustomBaselineCNN(num_classes=num_classes)
        else:
            model = build_transfer_resnet18(num_classes=num_classes, pretrained=False)

    params = count_parameters(model)
    prep_bench = benchmark_preprocessing_latency(image_size=image_size)

    results = {
        "hardware": hw_info,
        "parameters": params,
        "preprocessing": prep_bench,
        "benchmarks": {},
    }

    batch_sizes = config["benchmark"].get("batch_sizes", [1, 16])
    warmup = config["benchmark"].get("warmup_iterations", 15)
    iters = config["benchmark"].get("num_iterations", 50)

    for bsz in batch_sizes:
        bench = benchmark_model_inference(
            model=model,
            device=device,
            image_size=image_size,
            batch_size=bsz,
            warmup_iterations=warmup,
            num_iterations=iters,
        )
        results["benchmarks"][f"batch_{bsz}"] = bench

    return results


def main() -> None:
    """CLI entrypoint for inference benchmarking."""
    parser = argparse.ArgumentParser(description="PlantVision Inference Benchmarking Suite")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model checkpoint .pt")
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "baseline"], help="Model architecture")
    parser.add_argument("--classes", type=int, default=38, help="Number of classes if checkpoint not provided")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("\n--- PlantVision Inference Benchmark ---")
    res = run_benchmark_suite(model_path=args.model_path, architecture=args.arch, num_classes=args.classes)

    print(f"Device: {res['hardware']['device']} ({res['hardware']['gpu_name']})")
    print(f"Total Parameters: {res['parameters']['total_parameters']:,}")
    print(f"Preprocessing Median Latency: {res['preprocessing']['median_preprocessing_ms']} ms")
    for name, b in res["benchmarks"].items():
        print(f"\n{name.upper()}:")
        print(f"  Median Per-Image Latency: {b['median_per_image_latency_ms']} ms")
        print(f"  P95 Per-Image Latency:    {b['p95_per_image_latency_ms']} ms")
        print(f"  Throughput:               {b['throughput_images_per_sec']} img/s")


if __name__ == "__main__":
    main()
