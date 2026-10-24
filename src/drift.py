"""Noise handling, data age (drift check) and drift embeddings for the grid / 2 h history model.

Phases (each run writes results_drift/<phase>/<name>_s<seed>.json and is skipped if it exists):
  noise      4 configs (none / clean / trunc / clean+trunc) x seeds, no drift embedding
  age        one model per May 2009-2022 (equal-size subsample), tested on May 2023
  drift      drift embedding (8-d and 32-d), no noise handling
  factorial  drift embedding + best noise handling; then the 2x2 summary + saved models

Splits: train = May 2009-2021 + May 1-21 2022; val = May 22-31 2022 (val_fit 22-25, val_score 26-31);
test = May 2023 (test_fit = May 1-7, test_score = May 8-31).

python src/drift.py --phase noise --seeds 0 1 2
"""
import argparse
import json
import math
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from aismodel import KN_TO_KM_PER_STEP, ROOT, Net, Tracks, bin_centers, disp_km, soft_labels
from train import batch_inputs, evaluate, fixed_subset, target_fl

H, POSENC = 12, "grid"
_SFX = os.environ.get("DRIFT_SUFFIX", "")  # e.g. "_smoke" keeps test runs away from real results
OUT = ROOT / f"results_drift{_SFX}"
CKPT = ROOT / f"checkpoints_drift{_SFX}"
MODELS = ROOT / f"models{_SFX}"
FIRST_YEAR, N_WEEKS = 2009, 56  # 14 years x 4 weeks of May
SPEED_CAP_KN = torch.tensor([30.0, 30.0, 45.0, 30.0, 45.0])  # cargo, tanker, passenger, service, military/law
MAX_TARGET_GAP_S = 600
NOISE_CFGS = {"none": (False, 0.0), "clean": (True, 0.0), "trunc": (False, 0.02), "clean+trunc": (True, 0.02)}


def week_ids(years, days):
    return (years.astype(np.int64) - FIRST_YEAR) * 4 + np.minimum((days - 1) // 7, 3)


def positions(mask):
    return torch.from_numpy(np.flatnonzero(mask))


def clean_mask(data, pos, bs=262144):
    """True for samples whose history+target steps respect the per-type speed cap and whose target
    point is not interpolated across a reporting gap > MAX_TARGET_GAP_S."""
    caps = SPEED_CAP_KN.to(data.device)
    off = torch.arange(-H, 2, device=data.device)
    out = []
    for i in range(0, len(pos), bs):
        s = data.samples[pos[i : i + bs].to(data.device)]
        ix = s[:, None] + off[None]
        la, lo = data.lat[ix], data.lon[ix]
        dx, dy = disp_km(la[:, :-1], lo[:, :-1], la[:, 1:], lo[:, 1:])
        kn = torch.hypot(dx, dy).float() / KN_TO_KM_PER_STEP
        ok = (kn.max(1).values <= caps[data.v_group[data.vid[s]]]) & (data.gap[s + 1] <= MAX_TARGET_GAP_S)
        out.append(ok.cpu())
    return torch.cat(out)


def sample_loss(model, data, s, centers, week=None):
    w = data.window(s, H)
    seq, flat, group, cog_cur, lat, lon = batch_inputs(w, H)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(seq, flat, group, lat, lon, week=week)
    logp = F.log_softmax(logits.float(), -1)
    q = soft_labels(target_fl(w, H, cog_cur), centers)
    return -(q * logp).sum(-1)


def eval_sets(data, pos, n10, n60, nbon, seed):
    p = pos.to(data.device)
    return (p[fixed_subset(len(p), n10, seed).to(data.device)],
            p[fixed_subset(len(p), n60, seed + 1).to(data.device)],
            p[fixed_subset(len(p), nbon, seed + 2).to(data.device)])


def ev(model, sets, centers, n_samples=16):
    """sets = (data, idx10, idx60, idx_bon)."""
    return evaluate(model, sets[0], H, *sets[1:], centers, n_samples=n_samples)


def brief(res):
    return {"10min_mean": res["10min"]["mean_km"], "10min_median": res["10min"]["median_km"],
            "60min_mean": res["rollout_mean"]["at60"]["mean_km"], "60min_median": res["rollout_mean"]["at60"]["median_km"],
            "60min_best16": res["rollout_best_of_16"]["at60"]["mean_km"] if "rollout_best_of_16" in res else None,
            "full": res}


def train_model(data, train_pos, val_sets, centers, *, seed, steps, trunc, drift_dim, week_of_sample, last_week,
                batch=4096, lr=2e-3, eval_every=2000):
    torch.manual_seed(seed)
    dev = data.device
    model = Net(H, POSENC, n_weeks=N_WEEKS, drift_dim=drift_dim).to(dev)
    emb = [p for n, p in model.named_parameters() if n.startswith("pos.tables")]
    other = [p for n, p in model.named_parameters() if not n.startswith("pos.tables")]
    opt = torch.optim.AdamW([{"params": other, "weight_decay": 1e-4}, {"params": emb, "weight_decay": 0.0, "lr": lr * 5}],
                            lr=lr, betas=(0.9, 0.99), eps=1e-10)
    base = [g["lr"] for g in opt.param_groups]
    warm = min(500, steps // 10)
    tp = train_pos.to(dev)
    wk_all = week_of_sample.to(dev) if drift_dim else None
    best, best_val, log = None, float("inf"), []
    t0 = time.time()
    for step in range(1, steps + 1):
        sched = step / warm if step <= warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / (steps - warm)))
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * max(sched, 0.02)
        pos = tp[torch.randint(len(tp), (batch,), device=dev)]
        l = sample_loss(model, data, data.samples[pos], centers, week=wk_all[pos] if drift_dim else None)
        if trunc and step > 2 * warm:
            l = l.topk(int(len(l) * (1 - trunc)), largest=False).values
        loss = l.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % eval_every == 0 or step == steps:
            if drift_dim:
                model.test_vec = model.drift.weight[last_week].detach().clone()
            r = ev(model, val_sets, centers, n_samples=8)
            v = r["rollout_mean"]["at60"]["mean_km"]
            log.append({"step": step, "loss": loss.item(), "val_10min": r["10min"]["mean_km"], "val_60min": v,
                        "t": time.time() - t0})
            print(f"    step {step} loss {loss.item():.3f} val10 {r['10min']['mean_km']:.4f} val60 {v:.3f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
            if v < best_val:
                best_val, best = v, {k: x.detach().clone() for k, x in model.state_dict().items()}
    model.load_state_dict(best)
    return model, {"train_s": time.time() - t0, "best_val_60min": best_val, "log": log}


def fit_vector(model, data, pos, centers, init, steps=300, lr=1e-2, batch=4096):
    """New drift vector fitted on labelled recent samples; all model weights frozen."""
    for p in model.parameters():
        p.requires_grad_(False)
    v = init.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([v], lr=lr)
    p_dev = pos.to(data.device)
    for _ in range(steps):
        model.test_vec = v
        s = data.samples[p_dev[torch.randint(len(p_dev), (batch,), device=data.device)]]
        loss = sample_loss(model, data, s, centers).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    for p in model.parameters():
        p.requires_grad_(True)
    model.test_vec = v.detach()
    return v.detach()


def fine_tune(model, data, pos, centers, steps=300, lr=1e-4, batch=4096):
    """Reference for encoding (c): plain fine-tune of all weights on the same labelled week."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    p_dev = pos.to(data.device)
    for _ in range(steps):
        s = data.samples[p_dev[torch.randint(len(p_dev), (batch,), device=data.device)]]
        loss = sample_loss(model, data, s, centers).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


class Ctx:
    """Loads the full train/val split and the test year once."""

    def __init__(self, dev, need_train=True):
        self.dev = dev
        self.centers = bin_centers(dev)
        self.test = Tracks([2023], dev)
        day = self.test.sample_day
        self.test_all = positions(np.ones(len(day), bool))
        self.test_fit = positions(day <= 7)
        self.test_score = positions(day >= 8)
        self.sets_all = eval_sets(self.test, self.test_all, 400_000, 150_000, 20_000, 11)
        self.sets_score = eval_sets(self.test, self.test_score, 400_000, 150_000, 20_000, 21)
        if need_train:
            d = Tracks(list(range(FIRST_YEAR, 2023)), dev)
            y, dd = d.sample_year, d.sample_day
            self.data = d
            self.week = torch.from_numpy(week_ids(y, dd))
            self.train_pos = positions((y <= 2021) | ((y == 2022) & (dd <= 21)))
            self.val_pos = positions((y == 2022) & (dd >= 22))
            self.val_fit = positions((y == 2022) & (dd >= 22) & (dd <= 25))
            self.val_score = positions((y == 2022) & (dd >= 26))
            self.val_sets = (d, *eval_sets(d, self.val_pos, 100_000, 30_000, 1_000, 31))
            self.val_score_sets = (d, *eval_sets(d, self.val_score, 100_000, 30_000, 1_000, 41))
            self.last_week = int(week_ids(np.array([2022]), np.array([21]))[0])  # May 15-21 2022
            self.weeks_2022 = list(range(self.last_week - 2, self.last_week + 1))
            self._clean = None

    def clean_train_pos(self):
        if self._clean is None:
            m = clean_mask(self.data, self.train_pos)
            print(f"  cleaning keeps {m.float().mean().item() * 100:.2f}% of training samples", flush=True)
            self._clean = self.train_pos[m]
        return self._clean


def run_main(ctx, name, seed, *, clean, trunc, drift_dim, steps, phase):
    out = OUT / phase / f"{name}_s{seed}.json"
    if out.exists():
        print(f"skip {phase}/{name} s{seed}", flush=True)
        return
    print(f"== {phase}/{name} seed {seed}", flush=True)
    tp = ctx.clean_train_pos() if clean else ctx.train_pos
    model, info = train_model(ctx.data, tp, ctx.val_sets, ctx.centers, seed=seed, steps=steps, trunc=trunc,
                              drift_dim=drift_dim, week_of_sample=ctx.week, last_week=ctx.last_week)
    res = {"name": name, "seed": seed, "clean": clean, "trunc": trunc, "drift_dim": drift_dim, "steps": steps,
           "train_rows": len(tp), **info}
    c = ctx.centers
    t = ctx.test
    t0 = time.time()
    if drift_dim:
        W = model.drift.weight.detach()
        vecs = {"a_last_week": W[ctx.last_week].clone(), "b_mean_2022": W[ctx.weeks_2022].mean(0)}
        # validation analogue of the three test-time encodings (fit on May 22-25 2022, score May 26-31)
        res["val"] = {}
        for k, v in vecs.items():
            model.test_vec = v
            res["val"][k] = brief(ev(model, ctx.val_score_sets, c, n_samples=8))
        fit_vector(model, ctx.data, ctx.val_fit, c, vecs["a_last_week"])
        res["val"]["c_fit_recent"] = brief(ev(model, ctx.val_score_sets, c, n_samples=8))
        # test
        res["test_score"], res["test_all"] = {}, {}
        for k, v in vecs.items():
            model.test_vec = v
            res["test_score"][k] = brief(ev(model, (t, *ctx.sets_score), c))
            res["test_all"][k] = brief(ev(model, (t, *ctx.sets_all), c))
        vecs["c_fit_recent"] = fit_vector(model, t, ctx.test_fit, c, vecs["a_last_week"])
        res["test_score"]["c_fit_recent"] = brief(ev(model, (t, *ctx.sets_score), c))
        model.test_vec = vecs["a_last_week"]
        state = {"state_dict": model.state_dict(), "test_vecs": {k: v.cpu() for k, v in vecs.items()}}
    else:
        res["val"] = {"plain": brief(ev(model, ctx.val_score_sets, c, n_samples=8))}
        res["test_score"] = {"plain": brief(ev(model, (t, *ctx.sets_score), c))}
        res["test_all"] = {"plain": brief(ev(model, (t, *ctx.sets_all), c))}
        state = {"state_dict": {k: v.clone() for k, v in model.state_dict().items()}, "test_vecs": {}}
        fine_tune(model, t, ctx.test_fit, c)
        res["test_score"]["finetune_week1"] = brief(ev(model, (t, *ctx.sets_score), c))
    res["eval_s"] = time.time() - t0
    state["cfg"] = {"hist": H, "posenc": POSENC, "n_weeks": N_WEEKS, "drift_dim": drift_dim, "clean": clean,
                    "trunc": trunc, "seed": seed, "steps": steps}
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save(state, CKPT / f"{phase}_{name}_s{seed}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    sc = res["test_score"]
    print(f"  done {name} s{seed}: " + ", ".join(f"{k} 60m {v['60min_mean']:.3f}" for k, v in sc.items()), flush=True)


def mean_over_seeds(phase, name, getter):
    vals = [getter(json.loads(f.read_text())) for f in sorted((OUT / phase).glob(f"{name}_s*.json"))]
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


def best_noise(handling_only=False):
    """Config with the lowest val 60-min mean; handling_only skips "none" (for the 2x2 study)."""
    names = [n for n in NOISE_CFGS if not (handling_only and n == "none")]
    scores = {n: mean_over_seeds("noise", n, lambda r: r["val"]["plain"]["60min_mean"]) for n in names}
    print("noise configs, val 60-min mean:", {k: v for k, v in scores.items()}, flush=True)
    return min((v[0], k) for k, v in scores.items() if v[0] is not None)[1]


def best_drift():
    best = None
    for dim in (8, 32):
        for enc in ("a_last_week", "b_mean_2022", "c_fit_recent"):
            m, n = mean_over_seeds("drift", f"drift{dim}", lambda r: r["val"][enc]["60min_mean"])
            if m is not None:
                print(f"  drift{dim} {enc}: val 60-min mean {m:.4f} ({n} seeds)", flush=True)
                if best is None or m < best[0]:
                    best = (m, dim, enc)
    return best[1], best[2]


def run_age(ctx, seeds, steps):
    """One model per May (days 1-21 train, 22-31 val), equal-size subsample, tested on all of May 2023."""
    sizes = {}
    for y in range(FIRST_YEAR, 2023):
        z = np.load(ROOT / "data" / "tracks" / f"{y}.npz")
        tt = z["t"][z["samples"]].astype("datetime64[s]")
        day = (tt.astype("datetime64[D]") - tt.astype("datetime64[M]")).astype(np.int64) + 1
        sizes[y] = int((day <= 21).sum())
    n_eq = min(sizes.values())
    clean, trunc = NOISE_CFGS[best_noise()]
    print(f"age: {n_eq:,} train samples per year; noise handling clean={clean} trunc={trunc}", flush=True)
    for y in range(FIRST_YEAR, 2023):
        todo = [s for s in seeds if not (OUT / "age" / f"may{y}_s{s}.json").exists()]
        if not todo:
            print(f"skip age {y}", flush=True)
            continue
        d = Tracks([y], ctx.dev)
        day = d.sample_day
        tr = positions(day <= 21)
        if clean:
            tr = tr[clean_mask(d, tr)]
        tr = tr[fixed_subset(len(tr), n_eq, 7)]  # same subset for every seed
        vp = positions(day >= 22)
        vs = (d, *eval_sets(d, vp, 50_000, 20_000, 500, 51))
        for s in todo:
            print(f"== age May {y} seed {s}", flush=True)
            model, info = train_model(d, tr, vs, ctx.centers, seed=s, steps=steps, trunc=trunc, drift_dim=0,
                                      week_of_sample=None, last_week=None, eval_every=max(steps // 8, 1))
            res = {"year": y, "seed": s, "train_rows": len(tr), **info,
                   "test_all": brief(ev(model, (ctx.test, *ctx.sets_all), ctx.centers))}
            (OUT / "age").mkdir(parents=True, exist_ok=True)
            (OUT / "age" / f"may{y}_s{s}.json").write_text(json.dumps(res))
            print(f"  done May {y} s{s}: 2023 10m {res['test_all']['10min_mean']:.4f} 60m {res['test_all']['60min_mean']:.3f}", flush=True)
        del d
        torch.cuda.empty_cache()


def factorial_summary():
    noise = best_noise(handling_only=True)
    dim, enc = best_drift()
    cells = {
        "neither": ("noise", "none", "plain"),
        "noise_only": ("noise", noise, "plain"),
        "drift_only": ("drift", f"drift{dim}", enc),
        "noise+drift": ("factorial", f"drift{dim}_{noise}", enc),
    }
    MODELS.mkdir(exist_ok=True)
    summary = {"noise_cfg": noise, "drift_dim": dim, "test_encoding": enc, "cells": {}}
    for cell, (phase, name, key) in cells.items():
        rows = [json.loads(f.read_text()) for f in sorted((OUT / phase).glob(f"{name}_s*.json"))]
        m = lambda k, sub="test_score", kk=key: float(np.mean([r[sub][kk][k] for r in rows]))
        sd = lambda k, sub="test_score", kk=key: float(np.std([r[sub][kk][k] for r in rows]))
        summary["cells"][cell] = {
            "run": f"{phase}/{name}", "encoding": key, "seeds": len(rows),
            **{f"{k}": m(k) for k in ("10min_mean", "10min_median", "60min_mean", "60min_median", "60min_best16")},
            **{f"{k}_std": sd(k) for k in ("10min_mean", "60min_mean")},
        }
        if key == "plain":
            summary["cells"][cell]["finetune_week1_60min_mean"] = m("60min_mean", kk="finetune_week1")
        for r in rows:
            shutil.copy(CKPT / f"{phase}_{name}_s{r['seed']}.pt", MODELS / f"{cell}_s{r['seed']}.pt")
    (OUT / "factorial_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["noise", "age", "drift", "factorial"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=16000)
    ap.add_argument("--age-steps", type=int, default=4000)
    ap.add_argument("--configs", nargs="*")
    args = ap.parse_args()
    if os.name == "nt":  # keep Windows awake while this process runs
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    dev = torch.device("cuda")
    if args.phase == "age":
        run_age(Ctx(dev, need_train=False), args.seeds, args.age_steps)
        return
    ctx = Ctx(dev)
    print(f"train {len(ctx.train_pos):,}  val {len(ctx.val_pos):,}  test {len(ctx.test_all):,} "
          f"(score {len(ctx.test_score):,})", flush=True)
    if args.phase == "noise":
        for name in args.configs or NOISE_CFGS:
            clean, trunc = NOISE_CFGS[name]
            for s in args.seeds:
                run_main(ctx, name, s, clean=clean, trunc=trunc, drift_dim=0, steps=args.steps, phase="noise")
    elif args.phase == "drift":
        for dim in [int(c) for c in (args.configs or ["8", "32"])]:
            for s in args.seeds:
                run_main(ctx, f"drift{dim}", s, clean=False, trunc=0.0, drift_dim=dim, steps=args.steps, phase="drift")
    else:
        noise = best_noise(handling_only=True)
        dim, enc = best_drift()
        clean, trunc = NOISE_CFGS[noise]
        print(f"factorial: noise={noise}, drift_dim={dim}, encoding={enc}", flush=True)
        for s in args.seeds:
            run_main(ctx, f"drift{dim}_{noise}", s, clean=clean, trunc=trunc, drift_dim=dim, steps=args.steps,
                     phase="factorial")
        factorial_summary()


if __name__ == "__main__":
    main()
