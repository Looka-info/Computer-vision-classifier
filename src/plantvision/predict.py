"""Production-ready inference pipeline for single-image and batch predictions."""

from __future__ import annotations

import argparse
import io
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from plantvision.models import CustomBaselineCNN, build_transfer_resnet18
from plantvision.transforms import get_eval_transforms

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Encapsulates output of a single prediction."""

    top_class: str
    top_score: float
    top_k: List[Tuple[str, float]]
    all_scores: Dict[str, float]
    inference_latency_ms: float
    model_name: str
    device: str
    explanation: str
    timing_breakdown: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "predicted_class": self.top_class,
            "confidence_score": round(self.top_score, 4),
            "top_predictions": [(cls, round(score, 4)) for cls, score in self.top_k],
            "latency_ms": round(self.inference_latency_ms, 2),
            "model": self.model_name,
            "device": self.device,
            "note": self.explanation,
        }
        if self.timing_breakdown:
            d["timing_breakdown_ms"] = {k: round(v, 2) for k, v in self.timing_breakdown.items()}
        return d


class PlantPredictor:
    """Self-contained prediction pipeline loaded from a trained model checkpoint."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        warmup: bool = False,
    ):
        """Initialize predictor from saved checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint containing state_dict and metadata.
            device: Optional torch.device or 'cuda'/'cpu'. Auto-detects if None.
            warmup: If True, executes a single dummy forward pass to initialize CUDA context.
        """
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {self.checkpoint_path}")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.init_time_ms: float = 0.0
        self._load_model()
        if warmup:
            self.warmup()

    def _load_model(self) -> None:
        """Load architecture, weights, and preprocessing configuration."""
        t_start = time.perf_counter()
        try:
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to load checkpoint at {self.checkpoint_path}: {exc}") from exc

        if "metadata" not in ckpt or "state_dict" not in ckpt:
            raise ValueError(
                f"Checkpoint at {self.checkpoint_path} is missing required 'metadata' or 'state_dict' keys."
            )

        self.metadata = ckpt["metadata"]
        self.model_name = self.metadata.get("model_name", "plantvision_model")
        self.architecture = self.metadata.get("architecture", "ResNet")
        self.classes: List[str] = self.metadata.get("classes", [])
        self.class_to_idx: Dict[str, int] = self.metadata.get("class_to_idx", {})
        self.num_classes = len(self.classes)

        if not self.classes:
            raise ValueError("Checkpoint metadata contains no class labels.")

        # Recreate model architecture
        if "baseline" in self.architecture.lower():
            self.model = CustomBaselineCNN(num_classes=self.num_classes)
        else:
            self.model = build_transfer_resnet18(num_classes=self.num_classes, pretrained=False)

        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Build eval preprocessing
        norm_mean = self.metadata.get("norm_mean", [0.485, 0.456, 0.406])
        norm_std = self.metadata.get("norm_std", [0.229, 0.224, 0.225])
        dims = self.metadata.get("input_dimensions", [3, 224, 224])
        self.image_size = dims[1] if len(dims) >= 2 else 224

        self.transform = get_eval_transforms(
            image_size=self.image_size,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.init_time_ms = (time.perf_counter() - t_start) * 1000.0

        logger.info(
            "Loaded %s (%s) with %d classes on %s (init took %.2f ms)",
            self.model_name, self.architecture, self.num_classes, self.device, self.init_time_ms,
        )

    def warmup(self) -> None:
        """Execute a dummy forward pass to warm up CUDA context and kernel caches."""
        dummy_tensor = torch.zeros((1, 3, self.image_size, self.image_size), device=self.device)
        with torch.inference_mode():
            _ = self.model(dummy_tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, bytes],
        top_k: int = 3,
    ) -> PredictionResult:
        """Run inference on a single image.

        Args:
            image_input: File path, PIL Image, or raw image bytes.
            top_k: Number of highest scoring classes to return.

        Returns:
            PredictionResult with top prediction, scores, and granular latency.
        """
        # Preprocess input image
        t_prep_start = time.perf_counter()
        pil_img: Image.Image
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.exists():
                raise FileNotFoundError(f"Input image not found: {img_path}")
            try:
                pil_img = Image.open(img_path)
                pil_img.load()
            except Exception as exc:
                raise ValueError(f"Corrupt or unreadable image file ({img_path}): {exc}") from exc
        elif isinstance(image_input, bytes):
            try:
                pil_img = Image.open(io.BytesIO(image_input))
                pil_img.load()
            except Exception as exc:
                raise ValueError(f"Failed to decode image from bytes: {exc}") from exc
        elif isinstance(image_input, Image.Image):
            pil_img = image_input
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t_prep_end = time.perf_counter()
        prep_ms = (t_prep_end - t_prep_start) * 1000.0

        # Model forward pass
        t_infer_start = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t_infer_end = time.perf_counter()
        infer_ms = (t_infer_end - t_infer_start) * 1000.0

        # Postprocessing: Softmax and top-k selection
        t_post_start = time.perf_counter()
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        k = min(top_k, self.num_classes)
        top_indices = np.argsort(probs)[::-1][:k]

        top_predictions = [(self.classes[i], float(probs[i])) for i in top_indices]
        top_class, top_score = top_predictions[0]

        all_scores = {self.classes[i]: float(probs[i]) for i in range(self.num_classes)}
        explanation = f"{top_class.replace('___', ' ')} ({top_score:.2%})"
        t_post_end = time.perf_counter()
        post_ms = (t_post_end - t_post_start) * 1000.0

        total_ms = prep_ms + infer_ms + post_ms

        timing_breakdown = {
            "checkpoint_init_ms": self.init_time_ms,
            "preprocessing_ms": prep_ms,
            "model_inference_ms": infer_ms,
            "postprocessing_ms": post_ms,
            "end_to_end_pipeline_ms": total_ms,
        }

        return PredictionResult(
            top_class=top_class,
            top_score=top_score,
            top_k=top_predictions,
            all_scores=all_scores,
            inference_latency_ms=infer_ms,
            model_name=self.model_name,
            device=str(self.device),
            explanation=explanation,
            timing_breakdown=timing_breakdown,
        )


def main() -> None:
    """CLI prediction command."""
    parser = argparse.ArgumentParser(description="PlantVision Single Image Prediction")
    parser.add_argument("--image", type=str, required=True, help="Path to leaf image")
    parser.add_argument("--model", type=str, default="models/resnet18_transfer_best.pt", help="Path to checkpoint")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top classes")
    parser.add_argument("--warmup", action="store_true", help="Perform warm-up forward pass before timing")
    args = parser.parse_args()

    predictor = PlantPredictor(checkpoint_path=args.model, warmup=args.warmup)
    result = predictor.predict(args.image, top_k=args.top_k)

    print("\n" + "=" * 60)
    print("PLANTVISION PREDICTION RESULT")
    print("=" * 60)
    print(f"File:            {args.image}")
    print(f"Model:           {result.model_name} ({result.device})")
    print(f"Predicted Class: {result.top_class}")
    print(f"Confidence:      {result.top_score:.2%}")
    print("\nTop Candidates:")
    for cls_name, score in result.top_k:
        print(f"  - {cls_name:40s}: {score:.2%}")

    if result.timing_breakdown:
        tb = result.timing_breakdown
        print("\nLatency Breakdown:")
        print(f"  - Checkpoint Load & Init:  {tb['checkpoint_init_ms']:.2f} ms (one-time)")
        print(f"  - Image Decode & Prep:     {tb['preprocessing_ms']:.2f} ms")
        print(f"  - Model Inference:         {tb['model_inference_ms']:.2f} ms {'(warmed)' if args.warmup else '(cold first pass)'}")
        print(f"  - Postprocessing:          {tb['postprocessing_ms']:.2f} ms")
        print(f"  -> End-to-End Pipeline:    {tb['end_to_end_pipeline_ms']:.2f} ms")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
