"""
Utility to generate a CSV file that lists spectrogram images and their labels
for deep-fake detection experiments.

The script preserves the original logic:
    1.  Collect all *.png* files in the *original* and *deepfake* folders.
    2.  Write relative paths (e.g., original/cafe_img/<file>.png) to a CSV file.
    3.  Shuffle the rows to randomise ordering.

Author : Atharva Pore  
Date   : 21 May 2025
"""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path


def create_csv(
    original_dir: str | Path,
    deepfake_dir: str | Path,
    output_csv: str | Path,
) -> None:
    """
    Build a CSV mapping image paths to ground-truth labels.

    Parameters
    ----------
    original_dir : str | Path
        Directory containing PNG spectrograms of *original* audio.
    deepfake_dir : str | Path
        Directory containing PNG spectrograms of *deepfake* audio.
    output_csv : str | Path
        Destination CSV file to be created.

    The resulting file has three columns:

    | img_path                            | actual_output | expected_output |
    |-------------------------------------|---------------|-----------------|
    | original/cafe_img/<file>.png        | Original      |                 |
    | deepfake/cafe_img/<file>.png        | Deepfake      |                 |
    """
    original_dir = Path(original_dir)
    deepfake_dir = Path(deepfake_dir)
    output_csv = Path(output_csv)

    rows: list[list[str]] = []

    # Traverse original images
    for file in original_dir.iterdir():
        if file.suffix.lower() == ".png":
            img_path = Path("original") / "cafe_img" / file.name
            rows.append([str(img_path), "Original", ""])

    # Traverse deepfake images
    for file in deepfake_dir.iterdir():
        if file.suffix.lower() == ".png":
            img_path = Path("deepfake") / "cafe_img" / file.name
            rows.append([str(img_path), "Deepfake", ""])

    # Shuffle the data
    random.shuffle(rows)

    # Ensure output directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Write to CSV
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["img_path", "actual_output", "expected_output"])
        writer.writerows(rows)


if __name__ == "__main__":
    ORIGINAL_FOLDER = Path(r"path\to\original_img")
    DEEPFAKE_FOLDER = Path(r"path\to\deepfake_img")
    OUTPUT_CSV_PATH = Path(r"path\to\test.csv")

    create_csv(ORIGINAL_FOLDER, DEEPFAKE_FOLDER, OUTPUT_CSV_PATH)
    print(f"CSV file created: {OUTPUT_CSV_PATH.resolve()}")
