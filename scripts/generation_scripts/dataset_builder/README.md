# balanced\_dataset\_builder.py – 30 000-Image Deep-Fake Benchmark Creator

**Author:** Atharva Pore
**Purpose:** Construct a **balanced** image dataset for deep-fake detection research:

| Class     | Base Images |         Variants per Base | Total Images |
| --------- | ----------: | ------------------------: | -----------: |
| Original  |         500 | 30 (1 clean + 29 attacks) |       15 000 |
| Deep-fake |         500 | 30 (1 clean + 29 attacks) |       15 000 |

Final structure (15 000 × 2 classes = 30 000 PNG files):

```
out_root/
├─ original/
│  ├─ clean/           (500 images)
│  └─ attacked/        (~14 500 attack images)
└─ deepfake/
   ├─ clean/           (500 images)
   └─ attacked/        (~14 500 attack images)
```

---

## 1. How the Script Works

1. **Scan source trees**
   *ORIG\_DIR* and *DF\_DIR* are traversed recursively to collect:

   * *clean originals* (`*_Original.png`)
   * *clean deep-fakes* (filenames that include one of **DEEPFAKE\_ENGINES** but none of the attack tokens)

2. **Sample 500 + 500 bases**

   * 500 originals are drawn uniformly at random.
   * 500 deep-fakes are drawn **stratified by engine** (4 engines contribute 63 images each; the remaining 4 contribute 62).

3. **Generate attack permutations**
   For every base image the script attempts to locate 29 attack versions listed in **ATTACK\_NAMES** (various reverberation, resampling, bitrate recompression, additive noise, low-pass filter).
   Filename patterns are normalised by:

   * handling ID padding (`00001` → `1`)
   * accepting alternative attack spellings (`RT03` vs `rt_0.3`, `recompression128k` vs `recompression_128k`, etc.).

4. **Copy or hard-link** each file into *OUT\_DIR*
   Hard links are used when source and destination share the same filesystem; otherwise files are copied byte-for-byte.

5. **Integrity checks & summary**

   * Reports any missing attack variants per ID.
   * Prints global counts for missing attacks.
   * Confirms the final total.

---

## 2. Configuration

Edit the paths at the top of the script:

```python
from pathlib import Path

ORIG_DIR = Path(r"path\to\original_images")       # source of original PNG images
DF_DIR   = Path(r"path\to\deepfake_images")       # source of deep-fake PNG images
OUT_DIR  = Path(r"path\to\balanced_dataset_out")  # destination root
```

Optional adjustments:

* **DEEPFAKE\_ENGINES** – list of engine identifiers used in filenames.
* **ATTACK\_NAMES** – list of 29 attacks to search for per base.
* **RANDOM\_SEED** – change to reproduce a different 500 + 500 sample.

No command-line arguments are required; run the script directly after editing.

---

## 3. Requirements

The script relies only on Python 3 standard-library modules:

* `os`, `shutil`, `random`, `re`, `collections`, `pathlib`

No third-party packages are imported; however the destination filesystem must support hard links to benefit from the faster copy mode.

---

## 4. Running the Script

```bash
python balanced_dataset_builder.py
```

Sample console output (abridged):

```
Scanning source folders …
Found  8120 clean originals
Found  9420 clean deepfakes
  ElevenLabs : 1300
  Maskgct    : 1175
  …

Sampled deep-fake distribution:
  ElevenLabs : 63
  Maskgct    : 62
  …

Copying ORIGINAL files …
[WARN] original ID=00017 missing 2/29 → ['babble20', 'lpf7000']
…

Copying DEEPFAKE files …
[DEEPFAKE] GLOBAL missing attacks:
   lpf7000                23
…

✅ Dataset build complete — 30,000 files in D:\…\processed_small_dataset
Structure:
  D:\…\original\clean
  D:\…\original\attacked
  D:\…\deepfake\clean
  D:\…\deepfake\attacked
```

Warnings highlight any missing attack variants so you can verify your source repository.

---

## 5. Customisation Tips

| Objective                                | Change                                                                               |
| ---------------------------------------- | ------------------------------------------------------------------------------------ |
| Use 1 000 bases per class                | Adjust `orig_sample = random.sample(orig_clean, 1000)` and deep-fake sampling logic. |
| Add/remove attack types                  | Edit `ATTACK_NAMES` and extend `attack_variants()`.                                  |
| Different filename conventions           | Adapt `id_from_name()`, `engine_from_name()`, and `build_attack_filename()`.         |
| Force file **copy** instead of hard link | Replace `os.link()` call in `copy_file()` with `shutil.copy2()`.                     |
| Disable warning spam                     | Comment out `print()` lines inside `process_group()`.                                |

---

## 6. Troubleshooting

| Symptom                               | Explanation / Resolution                                                                        |
| ------------------------------------- | ----------------------------------------------------------------------------------------------- |
| “Expected 14 500 attack files copied” | Some attack variants are missing in source folders; check warnings for details.                 |
| `ValueError: No ID in …`              | The filename does not contain an ID pattern recognised by `id_from_name()`.                     |
| `OSError: [WinError 17]` when linking | Source and destination are on different drives; the script automatically falls back to `copy2`. |
| Output count not 30 000               | Verify that all 29 attack variants exist for every base image.                                  |

---

For further questions or contributions, please contact the author.
