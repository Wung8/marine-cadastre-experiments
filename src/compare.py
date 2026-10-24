"""Print a comparison table of all finished runs (2023 test set) next to the baselines."""
import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
GROUPS = ["cargo", "tanker", "passenger", "service", "military_law"]


def main(markdown=False):
    rows = []
    b = json.loads((RESULTS / "baselines.json").read_text())
    for name, key in (("dead reckoning", "dead_reckoning"), ("const velocity", "const_velocity")):
        t10, t60 = b["10min"][key], b["60min"][key]
        rows.append([name, "-", "-", t10["mean_km"], t10["median_km"], t60["mean_km"], t60["median_km"], None, None,
                     *[t60[f"{g}_mean_km"] for g in GROUPS]])
    for f in sorted(RESULTS.glob("*_h*.json")):
        r = json.loads(f.read_text())
        t = r["test"]
        t10, t60, bon = t["10min"], t["rollout_mean"]["at60"], t["rollout_best_of_16"]["at60"]
        rows.append([r["name"], r["posenc"], f"{r['hist'] * 10} min", t10["mean_km"], t10["median_km"],
                     t60["mean_km"], t60["median_km"], bon["mean_km"], r["train_time_s"] / 60,
                     *[t60[f"{g}_mean_km"] for g in GROUPS]])
    head = ["model", "pos enc", "history", "10m mean", "10m med", "60m mean", "60m med", "60m best16", "train min",
            *[f"60m {g}" for g in GROUPS]]
    fmt = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    if markdown:
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
        for row in rows:
            print("| " + " | ".join(fmt(v) for v in row) + " |")
    else:
        w = [max(len(h), 8) for h in head]
        print("  ".join(h.ljust(x) for h, x in zip(head, w)))
        for row in rows:
            print("  ".join(fmt(v).ljust(x) for v, x in zip(row, w)))


if __name__ == "__main__":
    main(markdown="--md" in sys.argv)
