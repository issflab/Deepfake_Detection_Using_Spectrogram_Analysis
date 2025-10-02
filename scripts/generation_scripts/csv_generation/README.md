# gen\_csv.py – Image → Label Manifest Generator

**Purpose**
`gen_csv.py` builds a manifest (`.csv`) that lists every spectrogram image in two
folders—**original** and **deep-fake**—and assigns the correct ground-truth
label. The CSV can be fed directly to training or evaluation pipelines that
expect the columns:

\| img\_path | actual\_output | expected\_output |

---

## 1.  What the Script Does

1. Scans one directory containing *original* PNG images and another containing
   *deep-fake* PNG images.
2. Constructs a row for every file:

   ```
   original/cafe_img/<filename>.png , Original , 
   deepfake/cafe_img/<filename>.png , Deepfake , 
   ```
3. Shuffles all rows to avoid any ordering bias.
4. Writes the resulting table to a user-specified CSV file.
5. Creates the output directory automatically if it does not exist.

---

## 2.  Requirements

* Python 3.9 or newer (standard library only; no third-party packages).

---

## 3.  Directory Assumptions

The default code expects images to reside in sub-folders named exactly
`original/cafe_img/…` and `deepfake/cafe_img/…`.
If your project uses a different naming convention, edit the two lines inside
`create_csv()` that build `img_path`:

```python
img_path = Path("original") / "cafe_img" / file.name
img_path = Path("deepfake") / "cafe_img" / file.name
```

---

## 4.  How to Use

1. **Place images**

   ```
   data/
   ├── original_img/        #  all Original *.png
   └── deepfake_img/        #  all Deep-fake *.png
   ```

2. **Adjust paths** (near the bottom of *gen\_csv.py*):

   ```python
   ORIGINAL_FOLDER = Path(r"path\to\original_img")
   DEEPFAKE_FOLDER = Path(r"path\to\deepfake_img")
   OUTPUT_CSV_PATH = Path(r"path\to\test.csv")
   ```

3. **Run the script**

   ```bash
   python gen_csv.py
   ```

   Example console output:

   ```
   CSV file created: C:\...\test.csv
   ```

---

## 5.  Output Format

The generated CSV contains three columns:

| Column name       | Description                                       |
| ----------------- | ------------------------------------------------- |
| `img_path`        | **Relative** path used later by your data loader. |
| `actual_output`   | Ground-truth label (`Original` or `Deepfake`).    |
| `expected_output` | Reserved for model predictions (left blank).      |

A fragment of the file might look like:

```
img_path,actual_output,expected_output
original/cafe_img/DT_00001.png,Original,
deepfake/cafe_img/DT_04217.png,Deepfake,
…
```

---

## 6.  Customisation Tips

* **Additional labels** If you add more classes, update
  `rows.append([...])` inside `create_csv()`.
* **Different file types** Change the extension check from
  `".png"` to `(".png", ".jpg")`, etc.
* **Deterministic shuffling** Call `random.seed(<number>)` before
  `random.shuffle(rows)`.

---

## 7.  Troubleshooting

| Issue                           | Explanation / Fix                                                |
| ------------------------------- | ---------------------------------------------------------------- |
| *CSV is empty*                  | Verify the two source folders contain `.png` files.              |
| *Paths incorrect at training*   | Ensure your training code prepends the correct base directory.   |
| *UnicodeEncodeError on Windows* | Make sure the script is saved with UTF-8 and run in UTF-8 shell. |

