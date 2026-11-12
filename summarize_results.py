import csv
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
rows = list(csv.DictReader((root / "task4_results" / "results.csv").open(encoding="utf-8-sig")))
by_variant = {}
for row in rows:
    by_variant.setdefault(row["variant"], []).append(row)
summary = {}
for name, runs in by_variant.items():
    metrics = {}
    for key in ("best_group_regular", "best_group_camera", "final_group_regular", "final_group_camera", "best_historical_regular", "final_historical_regular"):
        values = [float(row[key]) for row in runs if row[key] != ""]
        if values:
            metrics[key] = {
                "mean": round(statistics.mean(values), 6),
                "sample_sd": round(statistics.stdev(values), 6) if len(values) > 1 else None,
                "min": round(min(values), 6), "max": round(max(values), 6), "n": len(values),
                "individual": [round(v, 6) for v in values],
            }
    summary[name] = metrics
(root / "task4_results" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
for name, metrics in summary.items():
    def pct(key):
        item = metrics.get(key)
        if not item:
            return "-"
        sd = "" if item["sample_sd"] is None else f" ± {item['sample_sd'] * 100:.1f}"
        return f"{item['mean'] * 100:.1f}{sd} [{item['min'] * 100:.1f}–{item['max'] * 100:.1f}]"
    print(f"{name:38} n={metrics['best_group_regular']['n']} group {pct('best_group_regular')} camera {pct('best_group_camera')} historical {pct('best_historical_regular')}")
