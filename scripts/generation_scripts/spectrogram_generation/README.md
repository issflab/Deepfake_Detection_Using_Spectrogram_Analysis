# spectrogram\_generator.py

## Batch WAV ➔ PNG Spectrogram Converter

`spectrogram_generator.py` is a lightweight utility that converts every WAV
file in a source directory into an RGB **log-scaled STFT spectrogram** (PNG).
The filenames of the generated images follow the pattern

```
<original_stem>_<source_folder_name>.png
```

which guarantees uniqueness when multiple folders feed the same
image-classification pipeline.

---

## 1.  Features

| Capability                    | Detail                                                    |
| ----------------------------- | --------------------------------------------------------- |
| Fully automated batch mode    | Processes an entire folder in one command.                |
| Consistent visual parameters  | `n_fft=1024`, `hop_length=512`, *magma* colormap, 300 dpi |
| Safe re-runs                  | Skips images that already exist to avoid duplication.     |
| macOS artefact avoidance      | Ignores files containing `'._'` (resource-fork remnants). |
| Zero third-party dependencies | Uses only `librosa`, `matplotlib`, `numpy`, `pathlib`.    |

---

## 2.  Directory Layout Example

```
my_project/
├─ audio/
│   ├─ street/          (all WAVs from street microphone)
│   └─ cafe/            (all WAVs from café ambience)
└─ images/
    ├─ street_img/      (spectrograms will be written here)
    └─ cafe_img/
```

For each folder in `audio/` you would invoke the script once, pointing
`SOURCE_FOLDER` to `audio/street` and `DESTINATION_FOLDER` to
`images/street_img`, then repeat for the café set.

---

## 3.  Requirements

```bash
pip install librosa matplotlib numpy
```

---

## 4.  Usage

1. **Edit the two path constants** at the bottom of the file:

   ```python
   SOURCE_FOLDER      = Path(r"path\to\wav_files")
   DESTINATION_FOLDER = Path(r"path\to\output_images")
   ```

2. **Run the script**

   ```bash
   python spectrogram_generator.py
   ```

   Typical console output:

   ```
   Saved spectrogram for ambience_01.wav → ambience_01_street.png
   Saved spectrogram for ambience_02.wav → ambience_02_street.png
   …
   ```

---

## 5.  How It Works

| Step | Description                                                                |
| ---- | -------------------------------------------------------------------------- |
| 1    | Load the audio with `librosa.load` (original sampling rate preserved).     |
| 2    | Compute magnitude STFT (`n_fft=1024`, `hop_length=512`).                   |
| 3    | Convert to decibel scale via `librosa.amplitude_to_db`.                    |
| 4    | Render the spectrogram using `librosa.display.specshow`, colormap *magma*. |
| 5    | Save a 300 dpi, axis-free PNG to `DESTINATION_FOLDER`.                     |

---

## 6.  Customisation

| Need                                  | Change                                                                   |
| ------------------------------------- | ------------------------------------------------------------------------ |
| Different FFT size or hop length      | Edit `librosa.stft` arguments in `plot_spectrogram()`.                   |
| Alternative colour scheme             | Replace `cmap="magma"` with any Matplotlib colormap name.                |
| Produce grayscale images              | Add `cmap="gray"` and remove the RGB default.                            |
| Lower resolution for web              | Reduce `dpi` parameter (default 300) when calling `process_wav_files()`. |
| Additional audio formats (`.flac`, …) | Adjust the suffix check in `process_wav_files()`.                        |

---

## 7.  Troubleshooting

| Issue                             | Explanation / Fix                                       |
| --------------------------------- | ------------------------------------------------------- |
| *Nothing happens*                 | Check that `SOURCE_FOLDER` contains `.wav` files.       |
| `UserWarning: PySoundFile failed` | Install the **libsndfile** system library or use conda. |
| Images look “blank”               | Ensure audio is not silent; verify STFT parameters.     |
| Out-of-memory on large sets       | Process folders in smaller batches or lower `dpi`.      |

---

For questions or improvements, please open an issue or pull request.
