"""Inference benchmarking suite measuring empirical latency, throughput, and hardware profile."""

from __future__ import annotations

import argparse
import json
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


def measure_detailed_latency_breakdown(
    checkpoint_path: str | Path,
    image_path: str | Path,
    device: Optional[torch.device] = None,
    num_warm_iterations: int = 50,
) -> Dict[str, Any]:
    """Reconcile and measure granular latency across every stage of inference.

    Measures:
    1. Checkpoint loading and model initialization (cold disk to VRAM).
    2. Image decoding and preprocessing (file to PIL to Tensor).
    3. Cold model-only forward pass (first CUDA kernel execution with lazy init).
    4. Steady-state warm forward pass (median, p95, min, max across iterations with synchronization).
    5. Postprocessing (Softmax and Top-k selection).
    6. End-to-end latency (both cold and warm).
    """
    ckpt_file = Path(checkpoint_path)
    img_file = Path(image_path)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hw_info = get_hardware_info(device)

    # 1. Checkpoint load & init
    t0 = time.perf_counter()
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    num_classes = ckpt["metadata"]["num_classes"]
    arch = ckpt["metadata"].get("architecture", "ResNet")

    if "baseline" in arch.lower():
        model = CustomBaselineCNN(num_classes=num_classes)
    else:
        model = build_transfer_resnet18(num_classes=num_classes, pretrained=False)

    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    ckpt_init_ms = (t1 - t0) * 1000.0

    # 2. Image decoding & preprocessing
    t2 = time.perf_counter()
    with Image.open(img_file) as im:
        im_rgb = im.convert("RGB")
    tf = get_eval_transforms(image_size=ckpt["metadata"].get("input_dimensions", [3, 224, 224])[1])
    tensor = tf(im_rgb).unsqueeze(0).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t3 = time.perf_counter()
    decode_prep_ms = (t3 - t2) * 1000.0

    # 3. Cold first forward pass (lazy CUDA init)
    t4 = time.perf_counter()
    with torch.inference_mode():
        logits_cold = model(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t5 = time.perf_counter()
    cold_forward_ms = (t5 - t4) * 1000.0

    # 4. Steady-state warm forward passes
    with torch.inference_mode():
        for _ in range(15):
            _ = model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    warm_times = []
    with torch.inference_mode():
        for _ in range(num_warm_iterations):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            ts = time.perf_counter()
            _ = model(tensor)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            warm_times.append((time.perf_counter() - ts) * 1000.0)

    warm_arr = np.array(warm_times)
    warm_median_ms = float(np.median(warm_arr))
    warm_p95_ms = float(np.percentile(warm_arr, 95))
    warm_min_ms = float(np.min(warm_arr))
    warm_max_ms = float(np.max(warm_arr))

    # 5. Postprocessing
    t6 = time.perf_counter()
    probs = torch.softmax(logits_cold, dim=1).squeeze(0).cpu().numpy()
    top_indices = np.argsort(probs)[::-1][:3]
    t7 = time.perf_counter()
    postproc_ms = (t7 - t6) * 1000.0

    # End-to-end calculations
    end_to_end_warm_ms = decode_prep_ms + warm_median_ms + postproc_ms
    end_to_end_cold_ms = ckpt_init_ms + decode_prep_ms + cold_forward_ms + postproc_ms

    return {
        "model_name": ckpt["metadata"].get("model_name", ckpt_file.stem),
        "architecture": arch,
        "checkpoint_file": str(ckpt_file),
        "image_file": str(img_file),
        "device": str(device),
        "hardware": hw_info,
        "measurements_ms": {
            "1_checkpoint_load_and_init": round(ckpt_init_ms, 2),
            "2_image_decode_and_preprocess": round(decode_prep_ms, 2),
            "3_cold_first_forward_pass": round(cold_forward_ms, 2),
            "4_warm_steady_state_forward_median": round(warm_median_ms, 2),
            "4_warm_steady_state_forward_p95": round(warm_p95_ms, 2),
            "4_warm_steady_state_forward_min": round(warm_min_ms, 2),
            "4_warm_steady_state_forward_max": round(warm_max_ms, 2),
            "5_postprocessing_softmax_and_topk": round(postproc_ms, 2),
            "end_to_end_pipeline_warm_single_image": round(end_to_end_warm_ms, 2),
            "end_to_end_pipeline_cold_including_model_load": round(end_to_end_cold_ms, 2),
        },
        "throughput_img_per_sec": round(1000.0 / warm_median_ms, 2) if warm_median_ms > 0 else 0.0,
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
    parser.add_argument("--breakdown", action="store_true", help="Execute detailed 5-stage latency breakdown")
    parser.add_argument("--sample-image", type=str, default="data/sample_images/tomato_late_blight.jpg")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("\n" + "=" * 70)
    print("PLANTVISION INFERENCE BENCHMARKING SUITE")
    print("=" * 70)

    if args.breakdown:
        checkpoints_to_run = []
        if args.model_path:
            checkpoints_to_run.append(args.model_path)
        else:
            default_ckpts = ["models/resnet18_transfer_best.pt", "models/baseline_cnn_best.pt"]
            for cp in default_ckpts:
                if Path(cp).exists():
                    checkpoints_to_run.append(cp)

        combined_breakdown: Dict[str, Any] = {}
        for ckpt_path in checkpoints_to_run:
            print(f"\n--- 5-Stage Latency Breakdown for: {ckpt_path} ---")
            breakdown = measure_detailed_latency_breakdown(ckpt_path, args.sample_image)
            m = breakdown["measurements_ms"]
            print(f"Model: {breakdown['model_name']} ({breakdown['architecture']}) on {breakdown['device']} ({breakdown['hardware'].get('gpu_name', 'CPU')})")
            print(f"1. Checkpoint Loading & Init:        {m['1_checkpoint_load_and_init']} ms")
            print(f"2. Image Decode & Preprocessing:     {m['2_image_decode_and_preprocess']} ms")
            print(f"3. Cold Forward Pass (First Kernel): {m['3_cold_first_forward_pass']} ms")
            print(f"4. Steady-State Warm Forward Pass:   {m['4_warm_steady_state_forward_median']} ms (p95: {m['4_warm_steady_state_forward_p95']} ms)")
            print(f"5. Postprocessing (Softmax & Top-k): {m['5_postprocessing_softmax_and_topk']} ms")
            print(f"-> End-to-End Warm Pipeline Latency:  {m['end_to_end_pipeline_warm_single_image']} ms ({breakdown['throughput_img_per_sec']} FPS)")
            print(f"-> End-to-End Cold Pipeline Latency:  {m['end_to_end_pipeline_cold_including_model_load']} ms")

            combined_breakdown[breakdown["model_name"]] = breakdown

        out_json = Path("reports/latency_breakdown.json")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            if len(combined_breakdown) == 1:
                json.dump(list(combined_breakdown.values())[0], f, indent=2)
            else:
                json.dump(combined_breakdown, f, indent=2)
        print(f"\nSaved latency breakdown report to: {out_json}")
    else:
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
