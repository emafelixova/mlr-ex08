"""Follow-up combinations around the strongest isolated transfer-learning result."""
import csv
import json
import torch
import task4_experiments as exp


def main():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    exp.RESULTS.mkdir(exist_ok=True)
    rows = list(csv.DictReader((exp.RESULTS / "results.csv").open(encoding="utf-8-sig")))
    combinations = [
        ("combo_frozen_color_cosine30", dict(color=True, kind="mobilenet", frozen=True,
           optimizer_name="adam", weight_decay=0, label_smoothing=0, epochs=30, cosine=True)),
        ("combo_frozen_color_adamw_smoothing", dict(color=True, kind="mobilenet", frozen=True,
           optimizer_name="adamw", weight_decay=1e-4, label_smoothing=0.1, epochs=15, cosine=False)),
    ]
    for name, settings in combinations:
        for seed in [0, 1, 2]:
            result, _ = exp.run_one(name, seed, exp.GROUP / "train", historical_eval=True, **settings)
            rows.append(result)
            with (exp.RESULTS / "results.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result) + "\n")
            with (exp.RESULTS / "results.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    print("Combination runs complete.", flush=True)


if __name__ == "__main__":
    main()
