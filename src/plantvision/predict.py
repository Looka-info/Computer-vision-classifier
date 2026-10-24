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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_class": self.top_class,
            "confidence_score": round(self.top_score, 4),
            "top_predictions": [(cls, round(score, 4)) for cls, score in self.top_k],
            "latency_ms": round(self.inference_latency_ms, 2),
            "model": self.model_name,
            "device": self.device,
            "note": self.explanation,
        }


class PlantPredictor:
    """Self-contained prediction pipeline loaded from a trained model checkpoint."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
    ):
        """Initialize predictor from saved checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint containing state_dict and metadata.
            device: Optional torch.device or 'cuda'/'cpu'. Auto-detects if None.
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

        self._load_model()

    def _load_model(self) -> None:
        """Load architecture, weights, and preprocessing configuration."""
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

        logger.info(
            "Loaded %s (%s) with %d classes on %s",
            self.model_name, self.architecture, self.num_classes, self.device,
        )

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
            PredictionResult with top prediction, scores, and latency.
        """
        # Load and validate image
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

        # Convert to RGB (handles RGBA, grayscale, CMYK)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        # Preprocess
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # Inference
        t_start = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        # Class scores
        k = min(top_k, self.num_classes)
        top_indices = np.argsort(probs)[::-1][:k]

        top_predictions = [(self.classes[i], float(probs[i])) for i in top_indices]
        top_class, top_score = top_predictions[0]

        all_scores = {self.classes[i]: float(probs[i]) for i in range(self.num_classes)}

        explanation = (
            f"Predicted category: {top_class.replace('___', ' ')} with model score {top_score:.2%}. "
            f"Note: Softmax scores represent relative model confidence on known training distributions, "
            f"not calibrated epidemiological diagnosis probabilities."
        )

        return PredictionResult(
            top_class=top_class,
            top_score=top_score,
            top_k=top_predictions,
            all_scores=all_scores,
            inference_latency_ms=latency_ms,
            model_name=self.model_name,
            device=str(self.device),
            explanation=explanation,
        )


def main() -> None:
    """CLI prediction command."""
    parser = argparse.ArgumentParser(description="PlantVision Single Image Prediction")
    parser.add_argument("--image", type=str, required=True, help="Path to leaf image")
    parser.add_argument("--model", type=str, default="models/resnet18_transfer_best.pt", help="Path to checkpoint")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top classes")
    args = parser.parse_args()

    predictor = PlantPredictor(checkpoint_path=args.model)
    result = predictor.predict(args.image, top_k=args.top_k)

    print("\n--- PlantVision Prediction Result ---")
    print(f"File:        {args.image}")
    print(f"Prediction:  {result.top_class}")
    print(f"Confidence:  {result.top_score:.2%}")
    print(f"Latency:     {result.inference_latency_ms:.2f} ms ({result.device})")
    print("\nTop Candidates:")
    for cls_name, score in result.top_k:
        print(f"  - {cls_name:40s}: {score:.2%}")


if __name__ == "__main__":
    main()
