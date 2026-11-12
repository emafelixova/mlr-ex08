"""Add explicit run-condition columns to the saved experimental table."""
import csv
from pathlib import Path

path = Path(__file__).resolve().parent / "task4_results" / "results.csv"
rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
for row in rows:
    row["learning_rate"] = "0.001"
    row["batch_size"] = "32"
    row["input_size"] = "96"
    row["training_images"] = "944" if row["train_data"] == "group+historical" else "244"
    row["pretrained_weights"] = "torchvision MobileNetV3_Small_Weights.DEFAULT" if "MobileNet" in row["model"] else "random initialization"
    row["augmentation"] = "RandomResizedCrop(96,scale=0.6–1.0)+HFlip+VFlip+Rotation(20)" + ("+ColorJitter(0.25,0.25,0.25,hue=0.04)" if "ColorJitter" in row["augmentation"] else "")
with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
