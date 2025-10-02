"""
spectrogram_generator.py
========================
Create RGB spectrogram images (PNG) from a directory of WAV files.
The script preserves the original logic:

1. Walk through every *.wav* file in the *source_folder*.
2. Compute a log‐scale STFT spectrogram (n_fft=1024, hop_length=512).
3. Save each spectrogram as a *.png* with the pattern:
       <original_stem>_<source_folder_name>.png
4. Skip generation if the target file already exists or the path contains
   the substring '._' (to avoid macOS resource-fork artefacts).

Author : Atharva Pore  
Date   : 21 May 2025
"""

from __future__ import annotations

import os
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def plot_spectrogram(
    audio_path: str | Path,
    save_path: str | Path,
    dpi: int = 300,
) -> None:
    """Compute and store an RGB spectrogram for a single WAV file."""
    # Load the audio file (keep original sampling rate)
    samples, sr = librosa.load(audio_path, sr=None)

    # Short-Time Fourier Transform (STFT)
    magnitude = np.abs(librosa.stft(samples, n_fft=1024, hop_length=512))

    # Convert amplitude to dB
    spec_db = librosa.amplitude_to_db(magnitude, ref=np.max)

    # Plot and save the spectrogram
    plt.figure(figsize=(10, 5), dpi=dpi)
    librosa.display.specshow(
        spec_db,
        sr=sr,
        hop_length=512,
        x_axis="time",
        y_axis="log",
        cmap="magma",
    )
    plt.axis("off")  # Remove axes for a clean image
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0, format="png")
    plt.close()


def process_wav_files(
    source_folder: str | Path,
    destination_folder: str | Path,
    dpi: int = 300,
) -> None:
    """
    Convert every WAV in *source_folder* to a PNG spectrogram in *destination_folder*.
    Filenames are suffixed with the source folder name to ensure uniqueness.
    """
    source_folder = Path(source_folder)
    destination_folder = Path(destination_folder)
    destination_folder.mkdir(parents=True, exist_ok=True)

    folder_suffix = source_folder.name  # e.g. 'street', 'cafe'

    for wav_file in source_folder.iterdir():
        if wav_file.suffix.lower() == ".wav":
            target_name = f"{wav_file.stem}_{folder_suffix}.png"
            save_path = destination_folder / target_name

            # Skip if already generated or macOS resource fork
            if save_path.exists() or "._" in save_path.name:
                continue

            plot_spectrogram(wav_file, save_path, dpi=dpi)
            print(f"Saved spectrogram for {wav_file.name} → {target_name}")


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # User-defined paths (replace placeholders before running)
    # ------------------------------------------------------------------
    SOURCE_FOLDER = Path(r"path\to\wav_files")
    DESTINATION_FOLDER = Path(r"path\to\output_images")

    process_wav_files(SOURCE_FOLDER, DESTINATION_FOLDER)
