# 🎙️ Deepfake Detection Using Spectrogram Analysis

## 📖 Overview
This repository contains all core scripts and utilities for **deepfake detection research** using **spectrogram analysis** and **CNN-based models**.  
It provides a **modular and reproducible pipeline** to process audio data, generate spectrograms, simulate laundering attacks, and train/evaluate deep learning models.

The goal of this codebase is to create a **scalable, attack-resilient workflow** that enables reproducible experiments on benchmark datasets.

---

## 📂 Folder Structure

scripts/
│
├── generation_scripts/                # Dataset generation & preprocessing utilities
│   ├── csv_generation/
│   │   ├── gen_csv.py                 # Generate CSV files with spectrogram paths + labels
│   │   └── README.md
│   │
│   ├── dataset_builder/
│   │   ├── balanced_dataset_builder.py# Build balanced dataset (original vs fake samples)
│   │   └── README.md
│   │
│   ├── laundering_attack_implementation/
│   │   ├── laundering.py              # Implements laundering attacks (noise, reverb, resample, etc.)
│   │   ├── stochastic_attack.py       # Automates attack pipelines across datasets
│   │   ├── noises/                    # Background noise samples
│   │   │   ├── babble.wav
│   │   │   ├── cafe.wav
│   │   │   ├── street.wav
│   │   │   ├── volvo.wav
│   │   │   └── white.wav
│   │   └── README.md
│   │
│   └── spectrogram_generation/
│       ├── gen_spectra.py             # Convert audio to spectrogram images
│       └── README.md
│
└── model_training_and_testing/
├── deepfake_detection_cnn_resnet_18.py   # Train ResNet-18 CNN on spectrogram dataset
├── evaluate_resnet18_deepfake.py         # Evaluate trained model (metrics, confusion matrix)
└── README.md

---

## ⚡ Workflow Pipeline

### 1️⃣ Data Preparation
- Collect raw audio files (`.wav`) from dataset(s).  
- Build a balanced dataset:
```bash```
python scripts/generation_scripts/dataset_builder/balanced_dataset_builder.py --input ./audio --output ./balanced_data

	•	Generate CSV metadata:

python scripts/generation_scripts/csv_generation/gen_csv.py --input ./spectrograms --output metadata.csv


⸻

2️⃣ Laundering Attack Simulation (optional, for robustness testing)
	•	Apply predefined attacks (e.g., reverberation, resampling, compression, noise):

python scripts/generation_scripts/laundering_attack_implementation/laundering.py --input ./audio --attack reverberation

	•	Run stochastic (randomized) attacks:

python scripts/generation_scripts/laundering_attack_implementation/stochastic_attack.py --input ./audio --output ./attacked_data


⸻

3️⃣ Spectrogram Generation

Convert .wav audio to spectrogram PNGs:

python scripts/generation_scripts/spectrogram_generation/gen_spectra.py --input ./balanced_data --output ./spectrograms


⸻

4️⃣ Model Training

Train a ResNet-18 CNN:

python scripts/model_training_and_testing/deepfake_detection_cnn_resnet_18.py \
    --data ./spectrograms \
    --epochs 30 \
    --batch_size 32 \
    --lr 0.001

Includes:
	•	Train/Validation/Test split (70/15/15).
	•	Data augmentation (random crops, flips).
	•	Early stopping & checkpoint saving.

⸻

5️⃣ Model Evaluation

Evaluate trained model:

python scripts/model_training_and_testing/evaluate_resnet18_deepfake.py \
    --model ./checkpoints/best_model.pth \
    --data ./spectrograms/test


⸻

Requirements

Create a requirements.txt:

torch
torchvision
librosa
soundfile
audiomentations
pyroomacoustics
scipy
numpy
pandas
matplotlib
scikit-learn
tqdm
seaborn
ffmpeg-python

Install dependencies:

pip install -r requirements.txt


⸻

Example Experiment Workflow
	1.	Prepare balanced dataset:

python balanced_dataset_builder.py --input ./raw_audio --output ./balanced_data

	2.	Apply laundering attacks:

python laundering.py --input ./balanced_data --attack resample_22050

	3.	Generate spectrograms:

python gen_spectra.py --input ./balanced_data --output ./spectrograms

	4.	Train ResNet-18:

python deepfake_detection_cnn_resnet_18.py --data ./spectrograms

	5.	Evaluate:

python evaluate_resnet18_deepfake.py --model ./checkpoints/best_model.pth


⸻

Outputs
	•	Spectrogram PNGs (preprocessed features).
	•	CSV metadata with file paths + labels.
	•	Augmented datasets (with attacks applied).
	•	Trained CNN models (.pth checkpoints).
	•	Evaluation reports (confusion matrix, classification report).

⸻

Key Features
	•	Modular pipeline for reproducible experiments.
	•	Implements laundering attacks (noise, reverb, resampling, compression).
	•	Spectrogram generation with configurable parameters.
	•	Deep learning training using CNN (ResNet-18) with transfer learning.
	•	Evaluation metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix.

⸻

Contribution Guidelines
	•	Follow naming conventions (snake_case for Python scripts).
	•	Add new attacks or models under relevant subfolders.
	•	Use README.md in subfolders for script-specific instructions.

⸻

Citation

If you use this codebase for your research, please cite:

@thesis{2025deepfake_detection_spectrogram_analysis,
  title={Efficient Audio Deepfake Detection Using Spectrogram Filtering and Thresholding},
  author={Atharva Pore and Aishwarya Dekhane},
  year={2025},
  institution={University of Michigan-Dearborn}
}

---
