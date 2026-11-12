"""Repeat single-change CNN ablations with seeds 1 and 2."""
import csv
import json
import task4_experiments as exp
import torch


def main():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    path = exp.RESULTS / "results.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    runs = [
        ("color_jitter", dict(color=True)),
        ("dropout", dict(dropout=True)),
        ("global_avg_pool", dict(gap=True)),
        ("adamw_wd1e4", dict(optimizer_name="adamw", weight_decay=1e-4)),
        ("label_smoothing_01", dict(label_smoothing=0.1)),
    ]
    for name, options in runs:
        for seed in [1, 2]:
            if any(r["variant"] == name and int(r["seed"]) == seed for r in rows):
                continue
            result, _ = exp.run_one(name, seed, exp.GROUP / "train", historical_eval=False, **options)
            rows.append(result)
            with (exp.RESULTS / "results.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result) + "\n")
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    print("Ablation repeats complete.", flush=True)


if __name__ == "__main__":
    main()
