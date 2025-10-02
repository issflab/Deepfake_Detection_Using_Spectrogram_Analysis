"""
USAGE EXAMPLES
--------------

# 1) Using a CSV (with columns: path,label)
python <filename>.py \
  --csv data/index.csv \
  --image-col path \
  --label-col label \
  --out-dir runs/exp_csv \
  --best-name best.pth \
  --final-name last.pth \
  --batch-size 32 \
  --epochs 15 \
  --num-workers 6

# 2) Using explicit train / val / test directories
python <filename>.py \
  --dirs /data/train /data/val /data/test \
  --out-dir runs/exp_dirs \
  --best-name best_resnet18.pth \
  --final-name final_resnet18.pth \
  --batch-size 32 \
  --epochs 20 \
  --num-workers 8
"""

import argparse
import os
from pathlib import Path
import random
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train a ResNet-18 deepfake detector (no hard-coded paths)")

    # Mutually exclusive data sources
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--csv",
        type=str,
        help="CSV file with image paths and labels. See --image-col and --label-col.",
    )
    src.add_argument(
        "--dirs",
        nargs=3,
        metavar=("TRAIN_DIR", "VAL_DIR", "TEST_DIR"),
        help="Three directories for train/val/test, with subfolders or filename rules for labels.",
    )

    # CSV options
    p.add_argument("--image-col", type=str, default="path", help="CSV image path column (default: path)")
    p.add_argument("--label-col", type=str, default="label", help="CSV label column (default: label)")

    # Output / checkpoints
    p.add_argument("--out-dir", type=str, default="outputs", help="Directory to write checkpoints & logs (default: outputs)")
    p.add_argument("--best-name", type=str, default="best_multi_language.pth", help="Best checkpoint filename")
    p.add_argument("--final-name", type=str, default="final_multi_language.pth", help="Final checkpoint filename")

    # Training knobs
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--unfreeze-epoch", type=int, default=3, help="Epoch number (1-based) to unfreeze backbone")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic-ish; can slow down
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_outdir_and_paths(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = out_dir / args.best_name
    final_ckpt_path = out_dir / args.final_name
    return out_dir, best_ckpt_path, final_ckpt_path


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class ImageDataset(Dataset):
    """
    Simple image dataset given parallel lists of image paths and integer labels.
    """
    def __init__(self, paths: List[str], labels: List[int], train: bool = True):
        self.paths = paths
        self.labels = labels
        self.train = train

        # Augmentations (tweak as needed)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if train:
            self.tx = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.tx = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        y = int(self.labels[idx])
        img = Image.open(p).convert("RGB")
        img = self.tx(img)
        return img, y


def scan_dir_for_paths_and_labels(root: Path) -> Tuple[List[str], List[int]]:
    """
    Example directory scanner:
    - Assumes a structure like:
        root/
          fake/      *.png|*.jpg
          bonafide/  *.png|*.jpg
      OR filenames containing 'fake'/'deepfake' vs 'real'/'bonafide'/'original'.

    Adjust this to match your dataset organization if different.
    """
    paths, labels = [], []

    if (root / "fake").exists() or (root / "bonafide").exists():
        # Class-subfolder convention
        for cls_name, cls_idx in [("fake", 0), ("deepfake", 0), ("bonafide", 1), ("original", 1), ("real", 1)]:
            cls_dir = root / cls_name
            if not cls_dir.exists():
                continue
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
                for fp in cls_dir.rglob(ext):
                    paths.append(str(fp))
                    labels.append(cls_idx)
    else:
        # Fallback: infer from filename
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
            for fp in root.rglob(ext):
                low = fp.name.lower()
                if ("fake" in low) or ("deepfake" in low):
                    labels.append(0)
                elif any(t in low for t in ("bonafide", "original", "real")):
                    labels.append(1)
                else:
                    # If unsure, skip; or default to a class
                    continue
                paths.append(str(fp))

    if len(paths) == 0:
        raise RuntimeError(f"No images found under: {root}")

    return paths, labels


# ──────────────────────────────────────────────────────────────────────────────
# Model / Train / Eval
# ──────────────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = 2) -> nn.Module:
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return m


def freeze_backbone(m: nn.Module, freeze: bool = True):
    for name, p in m.named_parameters():
        if name.startswith("fc"):
            # keep head trainable regardless
            p.requires_grad = True
        else:
            p.requires_grad = not (not freeze)  # freeze=True -> requires_grad=False
            if freeze:
                p.requires_grad = False


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        count += x.size(0)
    return loss_sum / max(count, 1), correct / max(count, 1)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_sum, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        count += x.size(0)
    return loss_sum / max(count, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir, best_ckpt_path, final_ckpt_path = ensure_outdir_and_paths(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Build datasets ──────────────────────────────────────────────────────────
    if args.csv:
        import pandas as pd
        from sklearn.model_selection import train_test_split

        print(f"[INFO] Loading image list from CSV: {args.csv}")
        df = pd.read_csv(args.csv)
        if args.image_col not in df.columns or args.label_col not in df.columns:
            raise ValueError(f"CSV must contain columns `{args.image_col}` and `{args.label_col}`.")

        img_paths = df[args.image_col].astype(str).tolist()
        labels_raw = df[args.label_col].astype(str).tolist()

        def norm_label(s: str) -> int:
            s = s.strip().lower()
            if s in ("fake", "deepfake"):
                return 0
            if s in ("bonafide", "original", "real", "genuine"):
                return 1
            raise ValueError(f"Unrecognized label value: {s}")

        labels = [norm_label(x) for x in labels_raw]

        X_train, X_tmp, y_train, y_tmp = train_test_split(
            img_paths, labels, test_size=0.30, random_state=args.seed, stratify=labels
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=0.50, random_state=args.seed, stratify=y_tmp
        )

        train_ds = ImageDataset(X_train, y_train, train=True)
        val_ds   = ImageDataset(X_val,   y_val,   train=False)
        test_ds  = ImageDataset(X_test,  y_test,  train=False)

    else:
        # Explicit directories
        train_dir, val_dir, test_dir = map(Path, args.dirs)
        for d in (train_dir, val_dir, test_dir):
            if not d.exists():
                raise FileNotFoundError(f"Directory not found: {d}")

        X_train, y_train = scan_dir_for_paths_and_labels(train_dir)
        X_val,   y_val   = scan_dir_for_paths_and_labels(val_dir)
        X_test,  y_test  = scan_dir_for_paths_and_labels(test_dir)

        train_ds = ImageDataset(X_train, y_train, train=True)
        val_ds   = ImageDataset(X_val,   y_val,   train=False)
        test_ds  = ImageDataset(X_test,  y_test,  train=False)

    # ── DataLoaders ─────────────────────────────────────────────────────────────
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ── Model / Optim / Sched ──────────────────────────────────────────────────
    model = build_model(num_classes=2).to(device)
    freeze_backbone(model, freeze=True)  # freeze backbone initially

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    patience = 5

    print(f"[INFO] Starting training for {args.epochs} epochs on {device} …")
    for epoch in range(1, args.epochs + 1):
        # Unfreeze backbone at a chosen epoch (default: 3)
        if epoch == args.unfreeze_epoch:
            freeze_backbone(model, freeze=False)
            # Rebuild optimizer to include newly trainable params
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr, weight_decay=args.weight_decay
            )
            print(f"[INFO] Unfroze backbone at epoch {epoch}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | val acc {val_acc:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, best_ckpt_path)
            epochs_no_improve = 0
            print(f"  ↳ [BEST] Saved: {best_ckpt_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[INFO] Early stopping at epoch {epoch}")
                break

    # Save final (last) state
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, final_ckpt_path)
    print(f"[INFO] Saved final checkpoint: {final_ckpt_path}")

    # ── Test ───────────────────────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"[RESULT] Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
