"""
laundering_batch_processor.py
=============================

Apply a predefined set of *laundering attacks* to every WAV file under
*input_dir* and write the processed output to *output_dir*.

The script preserves the original logic:

1.  Walk `input_dir` recursively and locate all `.wav` files.
2.  For each file, apply every attack listed in `laundering_attack_types`.
3.  Optionally save results into dedicated sub-folders (e.g. `reverberation/`,
    `resampling/`, …) as controlled by `save_in_subfolders`.
4.  Log progress to the console; suppress mundane library output.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import pyroomacoustics as pra  # noqa: F401  (import kept for side-effects)
import soundfile as sf

from laundering import (  # noqa: F401
    room_reverb,
    noise_add,
    recompression,
    filtering,
    resampling,
)
import config  # noqa: F401

# ---------------------------------------------------------------------
# 1️ USER CONFIGURATION
# ---------------------------------------------------------------------
input_dir = Path(r"path\to\input_wav")         # <— replace with source folder
output_dir = Path(r"path\to\output_wav")       # <— replace with destination folder

save_in_subfolders: bool = True  # True → attack-specific sub-directories

config.out_dir = str(output_dir)  # propagate to global `config`

# ---------------------------------------------------------------------
# 2️ ATTACK DEFINITIONS
# ---------------------------------------------------------------------
laundering_attack_types = [
    "rt_0.3", "rt_0.6", "rt_0.9",
    "resample_22050", "resample_44100", "resample_8000", "resample_11025",
    "recompression_128k", "recompression_64k", "recompression_196k",
    "recompression_16k", "recompression_256k", "recompression_320k",
    "babble_0", "babble_10", "babble_20",
    "volvo_0", "volvo_10", "volvo_20",
    "white_0", "white_10", "white_20",
    "street_0", "street_10", "street_20",
    "cafe_0", "cafe_10", "cafe_20",
    "lpf_7000",
]

attack_folder_map = {
    "rt": "reverberation",
    "resample": "resampling",
    "recompression": "recompression",
    "babble": "babble",
    "volvo": "volvo",
    "white": "white",
    "street": "street",
    "cafe": "cafe",
    "lpf": "low_pass_filter",
}

# ---------------------------------------------------------------------
# 3️ UTILITY: SILENCE VERBOSE OUTPUT
# ---------------------------------------------------------------------
def block_printing(func):
    """Decorator to suppress stdout within noisy library calls."""
    def wrapper(*args, **kwargs):
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        value = func(*args, **kwargs)
        sys.stdout.close()
        sys.stdout = original_stdout
        return value
    return wrapper

# ---------------------------------------------------------------------
# 4️ AUDIO LAUNDERING
# ---------------------------------------------------------------------
@block_printing
def audio_laundering(audio_path: str | Path, attack_type: str,
                     category_folder: str | None = None) -> str | None:
    """
    Apply *attack_type* to *audio_path* and return the saved file path.
    The output filename format is unchanged from the original implementation.
    """
    file_name_with_ext = Path(audio_path).name
    file_stem, file_ext = os.path.splitext(file_name_with_ext)
    audio_data, sr = librosa.load(audio_path, sr=None)

    attack, parameter = attack_type.split("_")
    out_dir: Path

    if save_in_subfolders and category_folder:
        out_dir = output_dir / category_folder
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = output_dir

    config.out_dir = str(out_dir)  # keep global config in sync
    wav_file: Path | None = None

    # ---- Individual attacks ------------------------------------------------
    if attack == "rt":
        rvb = room_reverb(audio_data, sr, float(parameter))
        param_main, _, param_dec = parameter.partition(".")
        wav_file = out_dir / f"{file_stem}_RT{param_main}{param_dec or '0'}.wav"
        rvb.to_wav(str(wav_file), norm=True, bitdepth=np.int16)

    elif attack in {"babble", "volvo", "white", "cafe", "street"}:
        noise_path = Path(r"path\to\noise_files") / f"{attack}.wav"  # placeholder
        noise = noise_add(
            str(noise_path),
            float(parameter),
            float(parameter) + 0.5,
            audio_data,
            sr,
        )
        wav_file = out_dir / f"{file_stem}_{attack}{parameter}.wav"
        sf.write(str(wav_file), noise, sr, subtype="PCM_16")

    elif attack == "recompression":
        recompression(str(audio_path), str(out_dir), str(out_dir), parameter)
        wav_file = out_dir / f"{file_stem}_recompression{parameter}.wav"

    elif attack == "lpf":
        filtered = filtering(audio_data, sr)
        wav_file = out_dir / f"{file_stem}_lpf{parameter}.wav"
        sf.write(str(wav_file), filtered, sr, subtype="PCM_16")

    elif attack == "resample":
        new_sr = int(parameter)
        resampled_audio = resampling(str(audio_path), new_sr)
        wav_file = out_dir / f"{file_stem}_resample{parameter}.wav"
        sf.write(str(wav_file), resampled_audio, new_sr, subtype="PCM_16")

    elif attack == "copy":
        wav_file = out_dir / f"{file_stem}.wav"
        sf.write(str(wav_file), audio_data, sr, subtype="PCM_16")

    return str(wav_file) if wav_file else None

# ---------------------------------------------------------------------
# 5️ MAIN LOOP
# ---------------------------------------------------------------------
def main() -> None:
    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        root_path = Path(root)
        print(f"Found {len(files)} files in: {root_path}")
        for attack in laundering_attack_types:
            attack_key = attack.split("_")[0]
            folder_name = attack_folder_map.get(attack_key)
            for idx, file in enumerate(files):
                if file.endswith(".wav") and "._" not in file:
                    input_file = root_path / file
                    print(f"[{idx}] Processing: {input_file}")
                    print(f"Attack Type: {attack}")
                    output_file = audio_laundering(input_file, attack, folder_name)
                    if output_file and Path(output_file).exists():
                        print(f"Saved laundered file to: {output_file}")
                    else:
                        print(f"⚠️ Failed to save: {input_file}")
                    print("-" * 100)

    print("✅ All files processed successfully!")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
