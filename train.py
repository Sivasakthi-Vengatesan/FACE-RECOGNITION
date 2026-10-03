import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FaceDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        try:
            image = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"Cannot read image: {path}") from exc
        if self.transform:
            image = self.transform(image)
        return image, label


def collect_dataset(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {root}")

    class_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if len(class_dirs) < 2:
        raise ValueError("Dataset must contain at least two identity folders.")

    samples = []
    class_names = []
    for idx, folder in enumerate(class_dirs):
        files = sorted(p for p in folder.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if len(files) < 2:
            raise ValueError(
                f"Identity '{folder.name}' has only {len(files)} image(s). "
                "At least 2 are required."
            )
        class_names.append(folder.name)
        samples.extend((str(p), idx) for p in files)

    return samples, class_names


class FaceIDNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT
        self.backbone = models.resnet50(weights=weights)
        features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.BatchNorm1d(features),
            nn.Dropout(0.35),
            nn.Linear(features, n_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def transforms_for_training():
    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.80, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, val_tf


def evaluate(model, loader, device, class_names):
    model.eval()
    truth, pred = [], []
    with torch.inference_mode():
        for x, y in loader:
            logits = model(x.to(device))
            pred.extend(logits.argmax(1).cpu().tolist())
            truth.extend(y.tolist())
    acc = accuracy_score(truth, pred)
    report = classification_report(
        truth, pred, target_names=class_names, zero_division=0
    )
    matrix = confusion_matrix(truth, pred)
    return float(acc), report, matrix


def main():
    parser = argparse.ArgumentParser(description="Train FaceID Pro on an identity-folder dataset.")
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val", type=float, default=0.20)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()

    if not 0.10 <= args.val <= 0.40:
        raise ValueError("--val must be between 0.10 and 0.40.")
    seed_everything()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples, classes = collect_dataset(args.data)
    labels = np.array([label for _, label in samples])
    n = len(samples)
    n_classes = len(classes)
    val_count = max(n_classes, int(np.ceil(n * args.val)))

    if n - val_count < n_classes:
        raise ValueError("Not enough samples for a stratified split. Add more images per identity.")

    indices = np.arange(n)
    train_idx, val_idx = train_test_split(
        indices, test_size=val_count, random_state=SEED, stratify=labels
    )

    train_tf, val_tf = transforms_for_training()
    train_ds = FaceDataset([samples[i] for i in train_idx], train_tf)
    val_ds = FaceDataset([samples[i] for i in val_idx], val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        pin_memory=device.type == "cuda"
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda"
    )

    model = FaceIDNet(n_classes).to(device)

    # Warm up classifier, then fine-tune the pretrained backbone gently.
    for p in model.backbone.parameters():
        p.requires_grad = False

    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "best_model.pt"
    best_acc = -1.0
    patience, bad = 5, 0
    warmup_epochs = min(3, max(1, args.epochs // 4))

    for epoch in range(args.epochs):
        if epoch == warmup_epochs:
            for p in model.backbone.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW([
                {"params": model.backbone.parameters(), "lr": args.lr / 10},
                {"params": model.head.parameters(), "lr": args.lr},
            ], weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs - epoch)
            )

        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        val_acc, report, matrix = evaluate(model, val_loader, device, classes)
        avg_loss = running_loss / max(1, len(train_loader))
        print(f"Epoch {epoch+1:02d}/{args.epochs} | loss={avg_loss:.4f} | val_accuracy={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc, bad = val_acc, 0
            torch.save(model.state_dict(), best_path)
            np.save(output / "confusion_matrix.npy", matrix)
            (output / "classification_report.txt").write_text(report, encoding="utf-8")
            (output / "metadata.json").write_text(json.dumps({
                "classes": classes,
                "best_val_accuracy": best_acc,
                "image_size": 224,
                "mean": MEAN,
                "std": STD,
                "architecture": "ResNet-50 transfer learning",
                "dataset": "Labeled Faces in the Wild (LFW) prepared subset",
                "seed": SEED,
            }, indent=2), encoding="utf-8")
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    final_acc, report, _ = evaluate(model, val_loader, device, classes)
    print(f"\nBest validation accuracy: {final_acc:.4f}")
    print(report)
    print(f"Artifacts: {output.resolve()}")


if __name__ == "__main__":
    main()
