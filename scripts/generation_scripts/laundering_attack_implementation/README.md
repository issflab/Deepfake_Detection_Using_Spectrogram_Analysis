# laundering\_batch\_processor.py

## End-to-End Audio Laundering Pipeline

`laundering_batch_processor.py` is a command-line utility that **generates an
augmented data set** by applying multiple *laundering attacks*—reverberation,
bit-rate recompression, resampling, additive noise, low-pass filtering—to every
WAV file found in a source tree.
The script is designed for large-scale, fully automated processing and retains
the exact filename conventions used throughout the research code-base.

---

## 1.  Key Features

| Capability                                  | Detail                                                                |
| ------------------------------------------- | --------------------------------------------------------------------- |
| Recursive crawl                             | Walks the entire `input_dir`, regardless of folder depth.             |
| 29 distinct attacks                         | Defined in `laundering_attack_types`; easy to extend.                 |
| Optional attack-specific sub-folders        | Controlled by the `save_in_subfolders` flag (`True` by default).      |
| Noise suppression for third-party libraries | Decorator `block_printing` hides verbose output from `ffmpeg`, etc.   |
| Consistent naming                           | Output filenames match the patterns expected by downstream pipelines. |
| Fault-tolerant copying/linking              | Skips files that already exist and reports failures per file.         |

---

## 2.  Repository Layout

```
project_root/
│
├─ laundering_batch_processor.py      # this script
├─ laundering.py                      # core attack primitives (imported)
├─ config.py                          # shared configuration object
├─ noises/                            # *.wav noise files for additive-noise attacks
│    ├─ babble.wav
│    ├─ white.wav
│    └─ …
└─ input_wav/                         # your raw WAV hierarchy (set in script)
```

---

## 3.  Quick Configuration

Open the header section **“1  USER CONFIGURATION”** and edit three items:

```python
input_dir  = Path(r"path\to\input_wav")   # source tree
output_dir = Path(r"path\to\output_wav")  # destination tree
save_in_subfolders = True                 # attack → sub-folder mapping
```

> The variable `config.out_dir` is updated automatically so that all functions
> from **`laundering.py`** inherit the same destination path.

---

## 4.  Attack Catalogue

| Category        | Token Examples                    | Output Folder (if enabled) |
| --------------- | --------------------------------- | -------------------------- |
| Reverberation   | `rt_0.3`, `rt_0.6`, `rt_0.9`      | `reverberation/`           |
| Resampling      | `resample_8000`, `resample_44100` | `resampling/`              |
| Recompression   | `recompression_128k`, … `_320k`   | `recompression/`           |
| Additive noise  | `babble_10`, `street_20`, …       | `babble/`, `street/`, …    |
| Low-pass filter | `lpf_7000`                        | `low_pass_filter/`         |

Attacks are listed in `laundering_attack_types`.
To add or remove items, simply edit that list—the main loop adapts
automatically.

---

## 5.  Running the Script

```bash
python laundering_batch_processor.py
```

Typical console output:

```
Found 184 files in: input_wav\session01
[0] Processing: …\input.wav
Attack Type: rt_0.3
Saved laundered file to: …\reverberation\input_RT03.wav
----------------------------------------------------------------------------------------------------
…
✅ All files processed successfully!
```

Each source file is processed **once per attack**; expect
`len(wav_files) × len(laundering_attack_types)` outputs.

---

## 6.  Output Structure

With `save_in_subfolders = True`:

```
output_wav/
├─ reverberation/
│   ├─ fileA_RT03.wav
│   ├─ fileA_RT06.wav
│   └─ …
├─ resampling/
│   ├─ fileA_resample8000.wav
│   └─ …
├─ recompression/
│   └─ fileA_recompression128k.wav
├─ babble/
│   └─ fileA_babble10.wav
└─ low_pass_filter/
    └─ fileA_lpf7000.wav
```

If the flag is `False`, all files are written directly to `output_dir`.

---

## 7.  Extending or Customising

| Task                           | How-to                                                              |
| ------------------------------ | ------------------------------------------------------------------- |
| New attack primitive           | Implement in `laundering.py`, import it, add token to list.         |
| Different noise files          | Place WAV in `noises/` and update the path in `audio_laundering()`. |
| Change filename pattern        | Edit the relevant branch inside `audio_laundering()`.               |
| Deterministic processing order | Set `random.seed(<int>)` at the top of the script.                  |
| Disable hard-coded warnings    | Comment or adjust `print()` lines in the main loop.                 |

---

## 8.  Dependencies

| Package           | Purpose                                                   |
| ----------------- | --------------------------------------------------------- |
| `numpy`           | Numeric processing                                        |
| `librosa`         | Audio I/O and STFT utilities                              |
| `soundfile`       | WAV/FLAC read-write                                       |
| `pyroomacoustics` | Fast convolution reverb                                   |
| `ffmpeg` (CLI)    | Required indirectly for recompression via `laundering.py` |

Install all Python requirements:

```bash
pip install numpy librosa soundfile pyroomacoustics pandas
```

Ensure `ffmpeg` is on your system PATH for recompression attacks.

---

## 9.  Troubleshooting

| Symptom                          | Explanation / Fix                                                           |
| -------------------------------- | --------------------------------------------------------------------------- |
| *Input directory does not exist* | Check the `input_dir` path string—must be absolute or valid relative path.  |
| “Failed to save” warning         | Output folder not writable, unsupported WAV subtype, or missing noise file. |
| `FileNotFoundError` (noise WAV)  | Verify the path set in `noise_path` lines.                                  |
| `ffmpeg` errors                  | Install FFmpeg ≥ 4.0 and ensure it is discoverable in `PATH`.               |
| High processing time             | Disable sub-folders or remove heavyweight attacks (e.g., recompression).    |

---

For questions, feature requests, or bug reports, please contact the maintainer.
