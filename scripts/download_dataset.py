"""PlantVillage dataset acquisition utility supporting direct download and verification."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Optional
import urllib.request

logger = logging.getLogger(__name__)

# Official open-access mirrors
HF_DATA_ZIP_URL = "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/data.zip"


class DownloadProgressBar:
    """Console progress reporter for urllib downloads."""

    def __init__(self):
        self.last_percent = -1

    def __call__(self, block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        percent = min(100, int(downloaded * 100 / total_size))
        if percent != self.last_percent and percent % 5 == 0:
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"Downloading: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)...", end="\r")
            self.last_percent = percent
        if downloaded >= total_size:
            print("\nDownload complete.")


def download_plantvillage_dataset(
    output_dir: Path | str = "data/raw/PlantVillage",
    zip_url: str = HF_DATA_ZIP_URL,
    keep_zip: bool = False,
    subsets: Optional[list[str]] = None,
) -> Path:
    """Download and extract the PlantVillage dataset from the verified open-access mirror.

    Args:
        output_dir: Directory where raw class subdirectories will be extracted.
        zip_url: URL to download data.zip.
        keep_zip: Whether to preserve downloaded data.zip archive after extraction.
        subsets: Optional filter for specific crops (e.g. ['Apple', 'Tomato', 'Potato']).

    Returns:
        Path to prepared dataset directory.
    """
    target_path = Path(output_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    # Check if dataset is already extracted and populated
    existing_dirs = [d for d in target_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if existing_dirs and len(existing_dirs) >= 3:
        logger.info("Found %d existing class directories in %s. Skipping download.", len(existing_dirs), target_path)
        return target_path

    downloads_dir = Path("data/downloads")
    downloads_dir.mkdir(parents=True, exist_ok=True)
    zip_path = downloads_dir / "plantvillage_data.zip"

    if not zip_path.exists() or zip_path.stat().st_size < 100_000_000:
        print(f"Initiating download of PlantVillage dataset archive from:\n{zip_url}")
        print("This is a ~2 GB archive containing color, grayscale, and segmented leaf images.")
        progress = DownloadProgressBar()
        try:
            opener = urllib.request.build_opener()
            opener.addheaders = [("User-Agent", "Mozilla/5.0 (PlantVision ML Pipeline)")]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(zip_url, str(zip_path), reporthook=progress)
        except Exception as exc:
            logger.error("Download failed: %s", exc)
            print("\n" + "=" * 70)
            print("MANUAL DOWNLOAD INSTRUCTIONS:")
            print("If automated download is blocked or fails:")
            print("1. Download PlantVillage from Kaggle: https://www.kaggle.com/datasets/emmarex/plantdisease")
            print("   or GitHub: https://github.com/spMohanty/PlantVillage-Dataset")
            print(f"2. Extract the color images into: {target_path.resolve()}")
            print("   Each class should be in its own directory, e.g.:")
            print(f"   {target_path / 'Apple___healthy'}")
            print(f"   {target_path / 'Tomato___Late_blight'}")
            print("=" * 70)
            raise

    print(f"Extracting color images from {zip_path} into {target_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # We only want color images (under raw/color or color/)
        color_members = [m for m in zf.namelist() if "/color/" in m.lower() or m.lower().startswith("color/")]
        if not color_members:
            # Fallback to all members with image extensions
            color_members = [
                m for m in zf.namelist()
                if m.lower().endswith((".jpg", ".jpeg", ".png")) and not m.startswith("__MACOSX")
            ]

        # Filter by requested crop subsets if provided
        if subsets:
            allowed_prefixes = tuple(s.lower() for s in subsets)
            color_members = [
                m for m in color_members
                if any(part.lower().startswith(allowed_prefixes) for part in Path(m).parts)
            ]

        total_members = len(color_members)
        print(f"Found {total_members} color images to extract.")

        for idx, member in enumerate(color_members):
            if member.endswith("/"):
                continue
            parts = Path(member).parts
            # Look for the class folder name (typically preceding the filename)
            # e.g., raw/color/Apple___healthy/image1.jpg -> class is Apple___healthy
            try:
                if "color" in [p.lower() for p in parts]:
                    color_idx = [p.lower() for p in parts].index("color")
                    class_name = parts[color_idx + 1]
                    filename = parts[-1]
                else:
                    class_name = parts[-2]
                    filename = parts[-1]

                dest_dir = target_path / class_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_file = dest_dir / filename

                if not dest_file.exists():
                    with zf.open(member) as src, open(dest_file, "wb") as dst:
                        shutil.copyfileobj(src, dst)
            except Exception as e:
                logger.debug("Skipping entry %s: %s", member, e)

            if (idx + 1) % 1000 == 0 or idx == total_members - 1:
                print(f"Extracted {idx + 1}/{total_members} images...", end="\r")

    print("\nExtraction completed successfully.")

    if not keep_zip and zip_path.exists():
        try:
            zip_path.unlink()
            print("Removed temporary zip archive to conserve disk space.")
        except Exception:
            pass

    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify PlantVillage dataset")
    parser.add_argument("--output-dir", type=str, default="data/raw/PlantVillage", help="Destination directory")
    parser.add_argument("--subsets", nargs="+", default=None, help="Optional crop filters, e.g. Apple Tomato Potato")
    parser.add_argument("--keep-zip", action="store_true", help="Preserve zip file after extraction")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    download_plantvillage_dataset(
        output_dir=args.output_dir,
        keep_zip=args.keep_zip,
        subsets=args.subsets,
    )


if __name__ == "__main__":
    main()
