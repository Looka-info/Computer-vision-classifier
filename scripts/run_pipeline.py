"""End-to-End PlantVision Pipeline: Preparation, Baseline, Transfer Learning, and Evaluation."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from plantvision.config import load_config

logger = logging.getLogger(__name__)


def run_command(cmd: list[str], desc: str) -> None:
    """Execute a pipeline command and stream output."""
    print("\n" + "=" * 70)
    print(f"PIPELINE STEP: {desc}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 70)

    res = subprocess.run(cmd, check=True)
    if res.returncode != 0:
        raise RuntimeError(f"Step '{desc}' failed with exit code {res.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete PlantVision ML Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download step")
    parser.add_argument("--quick", action="store_true", help="Quick mode for rapid smoke testing")
    args = parser.parse_args()

    python_exe = sys.executable

    # Acquire dataset if missing
    raw_dir = Path("data/raw/PlantVillage")
    if not args.skip_download:
        if not raw_dir.exists() or len(list(raw_dir.glob("*"))) < 3:
            run_command([python_exe, "scripts/download_dataset.py"], "Acquiring PlantVillage Dataset")

    # Validate data and generate stratified splits
    run_command([python_exe, "scripts/prepare_dataset.py"], "Validating and Preparing Dataset Splits")

    # Train scratch baseline CNN
    base_cmd = [python_exe, "scripts/train_baseline.py"]
    if args.quick:
        base_cmd.extend(["--epochs", "2", "--batch-size", "16"])
    run_command(base_cmd, "Training Scratch Baseline CNN")

    # Train transfer learning ResNet-18
    tl_cmd = [python_exe, "scripts/train_transfer_learning.py"]
    if args.quick:
        tl_cmd.extend(["--stage-a-epochs", "1", "--stage-b-epochs", "1", "--batch-size", "16"])
    run_command(tl_cmd, "Training ResNet-18 Transfer Learning")

    # Evaluate models on held-out test split
    run_command([python_exe, "scripts/evaluate_model.py", "--all"], "Evaluating Models on Held-Out Test Set")

    print("\n" + "=" * 70)
    print("PLANTVISION PIPELINE EXECUTION COMPLETE!")
    print("Reports and figures are available in reports/ and reports/figures/")
    print("Launch the interactive demo using: streamlit run app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
