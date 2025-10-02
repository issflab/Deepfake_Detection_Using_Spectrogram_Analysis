# Author: Atharva Pore
# Date: 21st May, 2025
# ----------------------------------------------------------------------

import torch, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import os, json

# ----------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------
MODEL_PATH = Path(r"path\to\model.pth")
IMG_BASE_PATH = Path(r"path\to\image_base")

CSV_FILES = [
    r"path\to\test.csv",
]

IMN_MEAN, IMN_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # same as training

# ----------------------------------------------------------------------
# 2. MODEL & TRANSFORM
# ----------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)          # ← use weights arg instead of deprecated pretrained=
model.fc = torch.nn.Sequential(                 # ← SAME head as in training
    torch.nn.Dropout(0.3),
    torch.nn.Linear(model.fc.in_features, 2)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMN_MEAN, IMN_STD),
])

def predict_image(img_path: Path) -> str:
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        _, pred = torch.max(model(tensor), 1)
    return "Original" if pred.item() == 1 else "Deepfake"

# ----------------------------------------------------------------------
# 3. EVALUATION LOOP
# ----------------------------------------------------------------------
def process_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    print(f"\n🔍 Processing {csv_path.name}  ({len(df)} rows)")

    # Predict
    preds = []
    for p in df["img_path"]:
        full_path = IMG_BASE_PATH / p
        preds.append(predict_image(full_path))
    df["predicted_output"] = preds

    # Save updated CSV
    df.to_csv(csv_path, index=False)
    print(f"✅ Predictions written back to {csv_path.name}")

    # Metrics if ground-truth present
    if "actual_output" in df.columns:
        y_true = df["actual_output"].astype(str)
        y_pred = df["predicted_output"].astype(str)

        # Classification report
        report_txt = classification_report(y_true, y_pred, digits=2)
        rep_path = csv_path.with_name(csv_path.stem + "_report.txt")
        rep_path.write_text("Classification Report\n" + report_txt)
        print(f"📝 Report saved to {rep_path.name}")

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=["Original", "Deepfake"])
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Original", "Deepfake"],
                    yticklabels=["Original", "Deepfake"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix – {csv_path.stem}")
        plt.tight_layout()
        img_path = csv_path.with_name(csv_path.stem + ".png")
        plt.savefig(img_path)
        plt.close()
        print(f"📊 Confusion matrix saved to {img_path.name}")

        return {
            "file": csv_path.name,
            "report": report_txt
        }
    else:
        print(f"⚠️  {csv_path.name} lacks 'actual_output'; metrics skipped.")
        return None

def main():
    summaries = []
    for csv_str in CSV_FILES:
        summary = process_csv(Path(csv_str))
        if summary:
            summaries.append(summary)

    # Optional: aggregate all reports in one JSON for quick reference
    if summaries:
        agg_path = MODEL_PATH.with_name("aggregate_test_reports.json")
        agg_path.write_text(json.dumps(summaries, indent=2))
        print(f"\n📚 Aggregate reports saved to {agg_path}")

if __name__ == "__main__":
    main()
