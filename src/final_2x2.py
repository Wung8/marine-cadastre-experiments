"""Pick the best drift-embedding method on validation (with cleaning), then print the 2x2
(noise handling x drift embedding) on May 8-31 2023 and copy its models to models/.

  --choose   only print the chosen method (used by the run script to decide what to train next)
"""
import json
import shutil
import sys

import numpy as np

from drift import CKPT, MODELS, OUT

# candidates trained with cleaning: (results dir, run name, checkpoint prefix, matching no-cleaning run)
CANDIDATES = {
    "week32": ("factorial", "drift32_clean", "factorial_drift32_clean", ("drift", "drift32", "drift_drift32")),
    "year32_sd": ("real2", "year32_sd", "real2_year32_sd", ("real2", "year32_sd_noclean", "real2_year32_sd_noclean")),
}


def rows(phase, name):
    return [json.loads(f.read_text()) for f in sorted((OUT / phase).glob(f"{name}_s*.json"))]


def choose():
    best = None
    for method, (phase, name, _, _) in CANDIDATES.items():
        rs = rows(phase, name)
        if not rs:
            continue
        for enc in rs[0]["val"]:
            v = float(np.mean([r["val"][enc]["60min_mean"] for r in rs]))
            print(f"  {method} {enc}: val 60-min {v:.4f} ({len(rs)} seeds)", file=sys.stderr)
            if best is None or v < best[0]:
                best = (v, method, enc)
    return best[1], best[2]


def main():
    method, enc = choose()
    if "--choose" in sys.argv:
        print(method)
        return
    phase, name, ck, (nphase, nname, nck) = CANDIDATES[method]
    # the week runs name their last-week vector "a_last_week"; the year runs call it "a_last"
    cells = {
        "neither": ("noise", "none", "noise_none", "plain"),
        "noise_only": ("noise", "clean", "noise_clean", "plain"),
        "drift_only": (nphase, nname, nck, enc),
        "noise+drift": (phase, name, ck, enc),
    }
    summary = {"drift_method": method, "test_encoding": enc, "noise": "clean", "cells": {}}
    MODELS.mkdir(exist_ok=True)
    for cell, (ph, nm, ckp, key) in cells.items():
        rs = rows(ph, nm)
        get = lambda k: [r["test_score"][key][k] for r in rs]
        summary["cells"][cell] = {
            "run": f"{ph}/{nm}", "encoding": key, "seeds": len(rs),
            **{k: float(np.mean(get(k))) for k in ("10min_mean", "10min_median", "60min_mean", "60min_median", "60min_best16")},
            "60min_mean_std": float(np.std(get("60min_mean"))),
        }
        for r in rs:
            shutil.copy(CKPT / f"{ckp}_s{r['seed']}.pt", MODELS / f"{cell}_s{r['seed']}.pt")
    (OUT / "final_2x2.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
