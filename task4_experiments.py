"""Task 4 controlled leaf classifier experiments based on notebook Tasks 2 and 3."""
import csv
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parent
GROUP = ROOT / "task4_data" / "leaves"
HIST = ROOT / "task4_data" / "historical" / "leaves"
RESULTS = ROOT / "task4_results"
CAMERA = RESULTS / "camera_like"
WEIGHTS = ROOT / "mobilenet_v3_small-047dcff4.pth"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SIZE = 96
BATCH = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LeafNetBN(nn.Module):
    """Task 3 network; optional dropout/GAP are the isolated architecture variants."""
    def __init__(self, dropout=False, global_average_pooling=False):
        super().__init__()
        self.global_average_pooling = global_average_pooling
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        if global_average_pooling:
            self.fc2 = nn.Linear(64, 4)
        else:
            self.fc1 = nn.Linear(64 * 6 * 6, 64)
            self.dropout = nn.Dropout(0.3) if dropout else nn.Identity()
            self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        if self.global_average_pooling:
            x = x.mean(dim=(2, 3))
            return self.fc2(x)
        x = x.flatten(1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_transform(color=False):
    items = [transforms.RandomResizedCrop(SIZE, scale=(0.6, 1.0)),
             transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
             transforms.RandomRotation(20)]
    if color:
        items.append(transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.04))
    items += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(items)


def make_camera_set():
    CAMERA.mkdir(parents=True, exist_ok=True)
    marker = CAMERA / "seed.txt"
    if marker.exists() and marker.read_text().strip() == "240517":
        return datasets.ImageFolder(CAMERA, transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(MEAN, STD)]))
    if CAMERA.exists():
        shutil.rmtree(CAMERA)
    rng = random.Random(240517)
    for class_name in sorted(p.name for p in (GROUP / "val").iterdir() if p.is_dir()):
        out = CAMERA / class_name
        out.mkdir(parents=True, exist_ok=True)
        for path in sorted((GROUP / "val" / class_name).glob("*")):
            with Image.open(path) as im:
                image = im.convert("RGB")
            w, h = image.size
            fraction = rng.uniform(0.68, 0.95)
            cw, ch = int(w * fraction), int(h * fraction)
            left, top = rng.randint(0, w - cw), rng.randint(0, h - ch)
            image = image.crop((left, top, left + cw, top + ch)).resize((SIZE, SIZE), Image.Resampling.BILINEAR)
            image = image.rotate(rng.uniform(-20, 20), resample=Image.Resampling.BILINEAR)
            image = ImageEnhance.Color(image).enhance(rng.uniform(0.65, 1.35))
            image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.8, 1.2))
            image.save(out / path.name, quality=94)
    marker.write_text("240517\n")
    return datasets.ImageFolder(CAMERA, transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)]))


def accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            correct += (model(images).argmax(1) == labels).sum().item()
            total += len(labels)
    return correct / total if total else float("nan")


def create_model(kind, dropout=False, gap=False, frozen=False):
    if kind == "leafnet":
        return LeafNetBN(dropout=dropout, global_average_pooling=gap).to(DEVICE)
    model = models.mobilenet_v3_small(weights=None)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
    if frozen:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    return model.to(DEVICE)


def run_one(name, seed, train_data, color=False, kind="leafnet", dropout=False,
            gap=False, frozen=False, optimizer_name="adam", weight_decay=0.0,
            label_smoothing=0.0, epochs=15, cosine=False, historical_eval=False):
    set_seed(seed)
    train_ds = train_data if not isinstance(train_data, Path) else datasets.ImageFolder(train_data, train_transform(color))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, generator=generator)
    val_ds = datasets.ImageFolder(GROUP / "val", transforms.Compose([
        transforms.Resize((SIZE, SIZE)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]))
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)
    camera_ds = make_camera_set()
    camera_loader = DataLoader(camera_ds, batch_size=BATCH, shuffle=False)
    hist_loader = None
    if historical_eval:
        hist_ds = datasets.ImageFolder(HIST / "val", transforms.Compose([
            transforms.Resize((SIZE, SIZE)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]))
        hist_loader = DataLoader(hist_ds, batch_size=BATCH, shuffle=False)

    model = create_model(kind, dropout, gap, frozen)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(parameters, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if cosine else None
    best_regular, best_epoch, best_state = -1.0, 0, None
    final_regular = final_camera = 0.0
    start = time.time()
    for epoch in range(epochs):
        model.train()
        if frozen:
            model.features.eval()
        loss_sum = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        if scheduler:
            scheduler.step()
        final_regular = accuracy(model, val_loader)
        if final_regular > best_regular:
            best_regular = final_regular
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"{name} seed={seed} epoch={epoch + 1}/{epochs} loss={loss_sum / len(train_loader):.4f} group_val={final_regular:.4f}", flush=True)
    final_hist = accuracy(model, hist_loader) if hist_loader else ""
    final_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    final_camera = accuracy(model, camera_loader)
    model.load_state_dict(best_state)
    best_camera = accuracy(model, camera_loader)
    best_hist = accuracy(model, hist_loader) if hist_loader else ""
    folder = RESULTS / "checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, folder / f"{name}_seed{seed}_best.pt")
    torch.save(final_state, folder / f"{name}_seed{seed}_final.pt")
    result = {
        "variant": name, "seed": seed, "model": "MobileNetV3Small" if kind == "mobilenet" else "Task3 LeafNetBN",
        "train_data": "group+historical" if isinstance(train_data, ConcatDataset) else "group",
        "augmentation": "RandomResizedCrop(96, scale=0.6–1.0) + horizontal/vertical flip + RandomRotation(20)" +
                        (" + ColorJitter(0.25, 0.25, 0.25, hue=0.04)" if color else ""),
        "optimizer": f"{optimizer_name.upper()} lr=0.001 wd={weight_decay}",
        "learning_rate": 0.001, "batch_size": BATCH, "input_size": SIZE,
        "pretrained_weights": "torchvision MobileNetV3_Small_Weights.DEFAULT" if kind == "mobilenet" else "random initialization",
        "training_images": len(train_ds),
        "scheduler": "CosineAnnealingLR" if cosine else "none", "epochs": epochs,
        "checkpoint": f"best epoch {best_epoch} (selection by group regular validation)",
        "best_epoch": best_epoch, "best_group_regular": best_regular, "best_group_camera": best_camera,
        "final_group_regular": final_regular, "final_group_camera": final_camera,
        "best_historical_regular": best_hist, "final_historical_regular": final_hist,
        "elapsed_seconds": round(time.time() - start, 1), "label_smoothing": label_smoothing,
        "dropout": 0.3 if dropout else 0, "global_average_pooling": gap, "frozen_features": frozen,
    }
    return result, model


def main():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if not GROUP.exists() or not HIST.exists():
        raise SystemExit("Expected task4_data/leaves and task4_data/historical/leaves; extract supplied group and historical datasets first.")
    RESULTS.mkdir(exist_ok=True)
    camera = make_camera_set()
    group_transform = train_transform()
    group_ds = datasets.ImageFolder(GROUP / "train", group_transform)
    hist_ds = datasets.ImageFolder(HIST / "train", train_transform())
    combined = ConcatDataset([group_ds, hist_ds])
    # Baseline, per-change ablations, and runs repeated to estimate seed variation.
    matrix = [
        ("baseline", "leafnet", {}, [0, 1, 2], 15, False, "adam", 0, 0, False, False, False),
        ("color_jitter", "leafnet", {"color": True}, [0], 15, False, "adam", 0, 0, False, False, False),
        ("dropout", "leafnet", {"dropout": True}, [0], 15, False, "adam", 0, 0, False, False, False),
        ("global_avg_pool", "leafnet", {"gap": True}, [0], 15, False, "adam", 0, 0, False, False, False),
        ("adamw_wd1e4", "leafnet", {}, [0], 15, False, "adamw", 1e-4, 0, False, False, False),
        ("label_smoothing_01", "leafnet", {}, [0], 15, False, "adam", 0, 0.1, False, False, False),
        ("mobilenet_frozen", "mobilenet", {"frozen": True}, [0, 1, 2], 15, False, "adam", 0, 0, False, True, False),
        ("mobilenet_finetune", "mobilenet", {}, [0, 1, 2], 15, False, "adam", 0, 0, False, False, False),
        ("mobilenet_aug_cosine30", "mobilenet", {"color": True}, [0, 1, 2], 30, True, "adam", 0, 0, False, False, False),
        ("group_plus_historical", "leafnet", {}, [0, 1, 2], 15, False, "adam", 0, 0, False, False, True),
    ]
    all_results = []
    for name, kind, opts, seeds, epochs, cosine, opt, wd, smooth, dropout, frozen, use_hist in matrix:
        training = ConcatDataset([datasets.ImageFolder(GROUP / "train", train_transform(opts.get("color", False))),
                                  datasets.ImageFolder(HIST / "train", train_transform(opts.get("color", False)))]) if use_hist else GROUP / "train"
        for seed in seeds:
            result, _ = run_one(name, seed, training, color=opts.get("color", False), kind=kind,
                                dropout=opts.get("dropout", False), gap=opts.get("gap", False),
                                frozen=opts.get("frozen", False), optimizer_name=opt, weight_decay=wd,
                                label_smoothing=smooth, epochs=epochs, cosine=cosine,
                                historical_eval=(use_hist or name == "baseline"))
            all_results.append(result)
            with (RESULTS / "results.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")
            with (RESULTS / "results.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(all_results[0]))
                writer.writeheader(); writer.writerows(all_results)
    # After individual runs, test the combination chosen by regular validation.
    grouped = {}
    for row in all_results:
        if row["variant"] not in ("group_plus_historical", "baseline"):
            grouped.setdefault(row["variant"], []).append(row["best_group_regular"])
    ranked = sorted(((float(np.mean(scores)), name) for name, scores in grouped.items()), reverse=True)
    # Keep combined changes interpretable: color augmentation + AdamW + label smoothing,
    # and the best individual model family candidate with checkpoint selection.
    for name, color, use_mobile in [("combo_color_adamw_smoothing", True, False),
                                    ("combo_best_pretrained", True, True)]:
        for seed in [0, 1, 2]:
            result, _ = run_one(name, seed, GROUP / "train", color=color,
                                kind="mobilenet" if use_mobile else "leafnet",
                                optimizer_name="adamw", weight_decay=1e-4,
                                label_smoothing=0.1, epochs=30 if use_mobile else 15,
                                cosine=use_mobile, historical_eval=True)
            all_results.append(result)
            with (RESULTS / "results.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")
            with (RESULTS / "results.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(all_results[0]))
                writer.writeheader(); writer.writerows(all_results)
    summary = {name: {"n": len(rows), "mean": round(float(np.mean(rows)), 4),
                      "min": round(float(np.min(rows)), 4), "max": round(float(np.max(rows)), 4)}
               for name, rows in grouped.items()}
    (RESULTS / "individual_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Finished. Individual ranking:", ranked, "combination seeds:", [0, 1, 2], flush=True)


if __name__ == "__main__":
    main()
