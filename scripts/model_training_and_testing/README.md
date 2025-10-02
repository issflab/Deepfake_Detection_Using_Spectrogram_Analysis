# Deepfake Audio Detection using Spectrogram Images and ResNet-18

**Author**: Atharva Pore
**Date**: May 21, 2025

This project provides a complete pipeline to detect deepfake audio using image-based spectrogram classification. It involves two scripts:

1. **Training Script** (`deepfake_detection_cnn_resnet_18.py`)
   Trains a binary classification model (original vs. deepfake) using spectrogram images and ResNet-18 with transfer learning.

2. **Evaluation Script**
   Loads the trained model, runs inference on test image paths listed in a CSV file, and optionally computes evaluation metrics (accuracy, confusion matrix, classification report).

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Environment Setup](#environment-setup)
3. [Directory Preparation](#directory-preparation)
4. [Running the Training Script](#running-the-training-script)
5. [Running the Evaluation Script](#running-the-evaluation-script)
6. [Output Files](#output-files)
7. [Troubleshooting](#troubleshooting)

---

## Project Structure

```
project_root/
│
├── train_deepfake_detector.py             # Training script
├── evaluate_deepfake_model.py             # Evaluation script (filename is assumed)
├── requirements.txt
│
├── data/
│   ├── deepfake/combined/                 # Spectrogram images labeled as deepfake
│   └── original/combined/                 # Spectrogram images labeled as original
│
├── models/experiments/
│   ├── best_laundered_processed.pth       # Best checkpoint saved during training
│   └── final_laundered_processed.pth      # Final model after training
│
├── test_data/
│   ├── test.csv                           # CSV listing image paths and actual_output labels (optional)
│   ├── test_report.txt                    # Generated classification report
│   └── test.png                           # Confusion matrix
```

---

## Environment Setup

Use Python 3.9 or above. The recommended way to install all dependencies is using the provided `requirements.txt`.

### Create virtual environment (optional but recommended):

```bash
python -m venv venv
venv\Scripts\activate
```

### Install dependencies:

```bash
pip install -r requirements.txt
```

---

### requirements.txt

```txt
torch==2.5.1+cu118
torchvision==0.20.1+cu118
torchaudio==2.5.1+cu118
scikit-learn==1.5.1
pandas==2.2.2
matplotlib==3.9.2
seaborn==0.13.2
tqdm==4.66.5
pillow==10.4.0
```

> If you do not have a GPU, remove `+cu118` from the `torch`, `torchvision`, and `torchaudio` versions.

---

## Directory Preparation

1. **Spectrogram Images**
   Place your spectrogram images in two directories:

   ```
   data/
   ├── deepfake/combined/
   └── original/combined/
   ```

2. **CSV for Evaluation**
   Prepare a CSV file like this:

   ```
   img_path,actual_output
   deepfake/combined/sample1.png,Deepfake
   original/combined/sample2.png,Original
   ...
   ```

---

## Running the Training Script

Open the terminal in the project directory and run:

```bash
python train_resnet18_deepfake.py
```

This will:

* Load and augment spectrogram data.
* Train a ResNet-18 model with transfer learning.
* Save the best model during training to:

  ```
  models/experiments/best_laundered_processed.pth
  ```
* Also save the final model after all epochs to:

  ```
  models/experiments/final_laundered_processed.pth
  ```

---

## Running the Evaluation Script

Edit the configuration section of the evaluation script to set:

```python
MODEL_PATH = Path("models/experiments/best_laundered_processed.pth")
IMG_BASE_PATH = Path("data")  # Base folder for relative img_path entries in CSV
CSV_FILES = ["test_data/test.csv"]
```

Then run:

```bash
python evaluate_deepfake_model.py
```

This will:

* Load the model from `MODEL_PATH`
* Read each `img_path` in `test.csv` (relative to `IMG_BASE_PATH`)
* Predict whether the spectrogram represents "Original" or "Deepfake"
* Write predictions back into the same CSV
* If `actual_output` column is present:

  * Save a classification report as `test_report.txt`
  * Save a confusion matrix plot as `test.png`
  * Save a JSON summary as `aggregate_test_reports.json`

---

## Output Files

After evaluation:

* `test.csv` will be updated with a new `predicted_output` column.
* A classification report will be saved as `test_report.txt`
* Confusion matrix image will be saved as `test.png`
* A summary JSON will be saved as `aggregate_test_reports.json` (optional for batch use)

---

## Troubleshooting

| Problem                      | Possible Cause or Fix                             |
| ---------------------------- | ------------------------------------------------- |
| CUDA not available           | Script falls back to CPU automatically            |
| FileNotFoundError            | Ensure paths to images and models are correct     |
| Out-of-Memory on GPU         | Lower batch size in training script               |
| Wrong predictions            | Check if your spectrogram format matches training |
| Model doesn’t train properly | Ensure class balance and data preprocessing       |

---

For questions, suggestions, or help extending this pipeline, feel free to reach out.
