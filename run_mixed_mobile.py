"""Evaluate the selected mobile combination with both group and historical training photos."""
import csv
import json
import torch
from torch.utils.data import ConcatDataset
import task4_experiments as exp


def main():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    path = exp.RESULTS / "results.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    train = ConcatDataset([
        exp.datasets.ImageFolder(exp.GROUP / "train", exp.train_transform(color=True)),
        exp.datasets.ImageFolder(exp.HIST / "train", exp.train_transform(color=True)),
    ])
    for seed in [0, 1, 2]:
        name = "combo_frozen_color_adamw_smoothing_group_historical"
        if any(r["variant"] == name and int(r["seed"]) == seed for r in rows):
            continue
        result, _ = exp.run_one(name, seed, train, color=True, kind="mobilenet", frozen=True,
                                optimizer_name="adamw", weight_decay=1e-4,
                                label_smoothing=0.1, epochs=15, cosine=False,
                                historical_eval=True)
        rows.append(result)
        with (exp.RESULTS / "results.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result) + "\n")
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("Mixed-data combination complete.", flush=True)


if __name__ == "__main__":
    main()
