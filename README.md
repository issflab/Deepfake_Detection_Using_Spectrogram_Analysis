# 🎙️ Deepfake Detection Using Spectrogram Analysis

## Overview
This repository contains all core scripts and utilities for **deepfake detection research** using **spectrogram analysis** and **CNN-based models**.  
It provides a **modular and reproducible pipeline** to process audio data, generate spectrograms, simulate laundering attacks, and train/evaluate deep learning models.

The goal is a **scalable, attack-resilient workflow** for reproducible experiments on benchmark datasets.

---

## Folder Structure

scripts/
│
├── generation_scripts/
│   ├── csv_generation/
│   │   ├── gen_csv.py                 # Generate CSV files with spectrogram paths + labels
│   │   └── README.md
│   ├── dataset_builder/
│   │   ├── balanced_dataset_builder.py # Build balanced dataset (original vs fake samples)
│   │   └── README.md
│   ├── laundering_attack_implementation/
│   │   ├── laundering.py              # Implements laundering attacks (noise, reverb, resample, etc.)
│   │   ├── stochastic_attack.py       # Automates attack pipelines across datasets
│   │   ├── noises/
│   │   │   ├── babble.wav
│   │   │   ├── cafe.wav
│   │   │   ├── street.wav
│   │   │   ├── volvo.wav
│   │   │   └── white.wav
│   │   └── README.md
│   └── spectrogram_generation/
│       ├── gen_spectra.py             # Convert audio to spectrogram images
│       └── README.md
│
└── model_training_and_testing/
    ├── deepfake_detection_cnn_resnet_18.py   # Train ResNet-18 CNN on spectrogram dataset
    ├── evaluate_resnet18_deepfake.py         # Evaluate trained model (metrics, confusion matrix)
    └── README.md


⸻

Workflow Pipeline

1) Data Preparation

Collect raw audio files (.wav) and prepare the dataset.

# Build a balanced dataset
python scripts/generation_scripts/dataset_builder/balanced_dataset_builder.py \
  --input ./audio \
  --output ./balanced_data

# Generate CSV metadata
python scripts/generation_scripts/csv_generation/gen_csv.py \
  --input ./spectrograms \
  --output metadata.csv


⸻

2) Laundering Attack Simulation (optional)

Apply predefined or randomized laundering attacks for robustness testing.

# Apply reverberation attack
python scripts/generation_scripts/laundering_attack_implementation/laundering.py \
  --input ./audio \
  --attack reverberation

# Run stochastic (randomized) attacks
python scripts/generation_scripts/laundering_attack_implementation/stochastic_attack.py \
  --input ./audio \
  --output ./attacked_data


⸻

3) Spectrogram Generation

Convert .wav audio files into spectrogram .png images.

python scripts/generation_scripts/spectrogram_generation/gen_spectra.py \
  --input ./balanced_data \
  --output ./spectrograms


⸻

4) Model Training

Train a ResNet-18 CNN on spectrograms.

python scripts/model_training_and_testing/deepfake_detection_cnn_resnet_18.py \
  --data ./spectrograms \
  --epochs 30 \
  --batch_size 32 \
  --lr 0.001

Includes:
	•	Train/Validation/Test split (70/15/15)
	•	Data augmentation (random crops, flips)
	•	Early stopping & checkpoint saving

⸻

5) Model Evaluation

Evaluate the trained model on the test dataset.

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

# 1) Prepare balanced dataset
python scripts/generation_scripts/dataset_builder/balanced_dataset_builder.py \
  --input ./raw_audio \
  --output ./balanced_data

# 2) Apply laundering attacks
python scripts/generation_scripts/laundering_attack_implementation/laundering.py \
  --input ./balanced_data \
  --attack resample_22050

# 3) Generate spectrograms
python scripts/generation_scripts/spectrogram_generation/gen_spectra.py \
  --input ./balanced_data \
  --output ./spectrograms

# 4) Train ResNet-18
python scripts/model_training_and_testing/deepfake_detection_cnn_resnet_18.py \
  --data ./spectrograms

# 5) Evaluate model
python scripts/model_training_and_testing/evaluate_resnet18_deepfake.py \
  --model ./checkpoints/best_model.pth


⸻

Outputs
	•	Spectrogram PNGs (preprocessed features)
	•	CSV metadata with file paths and labels
	•	Augmented datasets (with attacks applied)
	•	Trained CNN models (.pth checkpoints)
	•	Evaluation reports (confusion matrix, classification report)

⸻

Key Features
	•	Modular pipeline for reproducible experiments
	•	Laundering attack implementations (noise, reverb, resampling, compression)
	•	Configurable spectrogram generation
	•	CNN model training with transfer learning
	•	Evaluation metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix

⸻

Contribution Guidelines
	•	Use snake_case for Python scripts
	•	Place new attacks or models in relevant subfolders
	•	Add README.md files in subfolders for script-specific instructions

⸻

Citation

If you use this codebase for your research, please cite:

@thesis{2025deepfake_detection_spectrogram_analysis,
  title        = {Efficient Audio Deepfake Detection Using Spectrogram Filtering and Thresholding},
  author       = {Atharva Pore and Aishwarya Dekhane},
  year         = {2025},
  institution  = {University of Michigan-Dearborn}
}
