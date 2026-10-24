"""Tests for prediction pipeline, score calibration, top-k ranking, and error handling."""

from __future__ import annotations

import io
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from plantvision.models import CustomBaselineCNN
from plantvision.predict import PlantPredictor, PredictionResult


@pytest.fixture
def dummy_checkpoint(tmp_path: Path) -> Path:
    """Create a valid temporary model checkpoint file with metadata for testing."""
    classes = ["Apple___healthy", "Apple___scab", "Tomato___healthy"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    model = CustomBaselineCNN(num_classes=3)
    ckpt_path = tmp_path / "test_model.pt"

    metadata = {
        "model_name": "test_cnn",
        "architecture": "CustomBaselineCNN",
        "classes": classes,
        "class_to_idx": class_to_idx,
        "num_classes": 3,
        "input_dimensions": [3, 64, 64],
        "norm_mean": [0.485, 0.456, 0.406],
        "norm_std": [0.229, 0.224, 0.225],
    }

    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, ckpt_path)
    return ckpt_path


def test_missing_checkpoint_raises_error(tmp_path: Path):
    """Predictor must raise FileNotFoundError for missing checkpoint."""
    with pytest.raises(FileNotFoundError):
        PlantPredictor(checkpoint_path=tmp_path / "missing.pt")


def test_prediction_valid_image(dummy_checkpoint: Path, tmp_path: Path):
    """Verify inference produces valid PredictionResult with softmax summing close to 1."""
    predictor = PlantPredictor(checkpoint_path=dummy_checkpoint, device="cpu")

    img_path = tmp_path / "test_leaf.png"
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(arr).save(img_path)

    result = predictor.predict(img_path, top_k=3)
    assert isinstance(result, PredictionResult)
    assert result.top_class in predictor.classes
    assert 0.0 <= result.top_score <= 1.0

    # Probabilities sum to 1.0
    total_prob = sum(result.all_scores.values())
    assert abs(total_prob - 1.0) < 1e-4

    # Top-k count
    assert len(result.top_k) == 3
    # Top score matches first in top_k
    assert result.top_class == result.top_k[0][0]
    assert abs(result.top_score - result.top_k[0][1]) < 1e-5


def test_prediction_corrupt_image(dummy_checkpoint: Path, tmp_path: Path):
    """Predictor must raise ValueError with diagnostic message for corrupted image."""
    predictor = PlantPredictor(checkpoint_path=dummy_checkpoint, device="cpu")

    corrupt_path = tmp_path / "broken.jpg"
    corrupt_path.write_bytes(b"invalid image bytes")

    with pytest.raises(ValueError, match="Corrupt or unreadable"):
        predictor.predict(corrupt_path)


def test_prediction_from_pil_and_bytes(dummy_checkpoint: Path):
    """Predictor should accept PIL Image and raw bytes directly."""
    predictor = PlantPredictor(checkpoint_path=dummy_checkpoint, device="cpu")

    img = Image.new("RGB", (64, 64), color=(30, 150, 40))
    res_pil = predictor.predict(img, top_k=2)
    assert res_pil.top_class in predictor.classes

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    res_bytes = predictor.predict(buf.getvalue(), top_k=2)
    assert res_bytes.top_class in predictor.classes
