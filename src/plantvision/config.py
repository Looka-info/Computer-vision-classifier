"""Configuration loading and validation module."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict
import yaml

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file. Defaults to configs/config.yaml.

    Returns:
        Dict containing project configuration settings.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not path.exists():
        # Fall back to searching parent directories if running from subdirectory
        alt_path = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Optional environment overrides
    if "DATA_DIR" in os.environ:
        config["paths"]["raw_data_dir"] = os.environ["DATA_DIR"]
    if "SPLITS_DIR" in os.environ:
        config["paths"]["splits_dir"] = os.environ["SPLITS_DIR"]
    if "MODELS_DIR" in os.environ:
        config["paths"]["models_dir"] = os.environ["MODELS_DIR"]
    if "RANDOM_SEED" in os.environ:
        config["project"]["random_seed"] = int(os.environ["RANDOM_SEED"])

    return config


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parents[2]
