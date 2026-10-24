"""Follow-up drift-embedding experiments (all with window cleaning, grid / 2 h history).

  synth  injected label drift: training targets of May Y are scaled by 1 + 0.3 * (2022 - Y) / 13
         (2009 +30% ... 2022 +0%); validation and test keep real labels. Configs: plain, week32, year32_sd.
  real   real labels: year32 (per-year vector) and year32_sd (+ smoothness penalty + vector dropout).

Results go to results_drift/<phase>2/<name>_s<seed>.json; finished runs are skipped.
python src/drift2.py --phase synth --seeds 0 1 2
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from aismodel import Net, soft_labels
from drift import CKPT, FIRST_YEAR, H, OUT, POSENC, Ctx, brief, ev, fit_vector, week_ids
from train import batch_inputs, target_fl

N_YEARS = 2023 - FIRST_YEAR  # 2009..2022
CONFIGS = {
    # name: (period, drift_dim, smooth_lambda, vec_dropout, clean)
    "plain": (None, 0, 0.0, 0.0, True),
    "week32": ("week", 32, 0.0, 0.0, True),
    "year32": ("year", 32, 0.0, 0.0, True),
    "year32_sd": ("year", 32, 1.0, 0.3, True),
    "year32_sd_noclean": ("year", 32, 1.0, 0.3, False),
}
SYNTH_AMP = 0.3


def sample_loss(model, data, s, centers, period=None, scale=None):
    w = data.window(s, H)
    seq, flat, group, cog_cur, lat, lon = batch_inputs(w, H)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(seq, flat, group, lat, lon, week=period)
    logp = F.log_softmax(logits.float(), -1)
    fl = target_fl(w, H, cog_cur)
    if scale is not None:
        fl = fl * scale[:, None]
    return -(soft_labels(fl, centers) * logp).sum(-1)


def train_model(ctx, train_pos, *, seed, steps, period_ids, n_rows, last_id, unknown_id, drift_dim, smooth, vdrop,
                scale_of_sample, batch=4096, lr=2e-3, eval_every=2000):
    torch.manual_seed(seed)
    data, dev, centers = ctx.data, ctx.dev, ctx.centers
    model = Net(H, POSENC, n_weeks=n_rows, drift_dim=drift_dim).to(dev)
    emb = [p for n, p in model.named_parameters() if n.startswith("pos.tables")]
    other = [p for n, p in model.named_parameters() if not n.startswith("pos.tables")]
    opt = torch.optim.AdamW([{"params": other, "weight_decay": 1e-4}, {"params": emb, "weight_decay": 0.0, "lr": lr * 5}],
                            lr=lr, betas=(0.9, 0.99), eps=1e-10)
    base = [g["lr"] for g in opt.param_groups]
    warm = min(500, steps // 10)
    tp = train_pos.to(dev)
    pid = period_ids.to(dev) if drift_dim else None
    scl = scale_of_sample.to(dev) if scale_of_sample is not None else None
    best, best_val, log = None, float("inf"), []
    t0 = time.time()
    for step in range(1, steps + 1):
        sched = step / warm if step <= warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / (steps - warm)))
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * max(sched, 0.02)
        pos = tp[torch.randint(len(tp), (batch,), device=dev)]
        per = None
        if drift_dim:
            per = pid[pos]
            if vdrop:
                per = torch.where(torch.rand(len(per), device=dev) < vdrop, torch.full_like(per, unknown_id), per)
        loss = sample_loss(model, data, data.samples[pos], centers, period=per,
                           scale=scl[pos] if scl is not None else None).mean()
        if smooth:
            W = model.drift.weight[:N_YEARS]
            loss = loss + smooth * ((W[1:] - W[:-1]) ** 2).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % eval_every == 0 or step == steps:
            if drift_dim:
                model.test_vec = model.drift.weight[last_id].detach().clone()
            r = ev(model, ctx.val_sets, centers, n_samples=8)
            v = r["rollout_mean"]["at60"]["mean_km"]
            log.append({"step": step, "loss": loss.item(), "val_10min": r["10min"]["mean_km"], "val_60min": v,
                        "t": time.time() - t0})
            print(f"    step {step} loss {loss.item():.3f} val10 {r['10min']['mean_km']:.4f} val60 {v:.3f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
            if v < best_val:
                best_val, best = v, {k: x.detach().clone() for k, x in model.state_dict().items()}
    model.load_state_dict(best)
    return model, {"train_s": time.time() - t0, "best_val_60min": best_val, "log": log}


def run(ctx, phase, name, seed, steps):
    out = OUT / f"{phase}2" / f"{name}_s{seed}.json"
    if out.exists():
        print(f"skip {phase}/{name} s{seed}", flush=True)
        return
    print(f"== {phase}/{name} seed {seed}", flush=True)
    period, dim, smooth, vdrop, clean = CONFIGS[name]
    d = ctx.data
    years, days = d.sample_year.astype(np.int64), d.sample_day
    if period == "week":
        ids, n_rows, last_id, unknown_id = week_ids(years, days), N_YEARS * 4, ctx.last_week, None
        others = {"b_mean_2022": list(range(ctx.last_week - 2, ctx.last_week + 1))}
    else:
        ids, n_rows, last_id, unknown_id = years - FIRST_YEAR, N_YEARS + 1, N_YEARS - 1, N_YEARS
        others = {"b_unknown": [unknown_id]} if vdrop else {}
    scale = None
    if phase == "synth":
        scale = torch.from_numpy((1 + SYNTH_AMP * (2022 - years) / (2022 - FIRST_YEAR)).astype(np.float32))
    tp = ctx.clean_train_pos() if clean else ctx.train_pos
    model, info = train_model(ctx, tp, seed=seed, steps=steps, period_ids=torch.from_numpy(ids), n_rows=n_rows,
                              last_id=last_id, unknown_id=unknown_id, drift_dim=dim, smooth=smooth, vdrop=vdrop,
                              scale_of_sample=scale)
    res = {"phase": phase, "name": name, "seed": seed, "period": period, "drift_dim": dim, "smooth": smooth, "clean": clean,
           "vec_dropout": vdrop, "synth_amp": SYNTH_AMP if phase == "synth" else 0.0, **info}
    c, t = ctx.centers, ctx.test
    t0 = time.time()
    res["val"], res["test_score"], res["test_all"] = {}, {}, {}
    if dim:
        W = model.drift.weight.detach()
        vecs = {"a_last": W[last_id].clone(), **{k: W[v].mean(0) for k, v in others.items()}}
        for k, v in vecs.items():
            model.test_vec = v
            res["val"][k] = brief(ev(model, ctx.val_score_sets, c, n_samples=8))
            res["test_score"][k] = brief(ev(model, (t, *ctx.sets_score), c))
            res["test_all"][k] = brief(ev(model, (t, *ctx.sets_all), c))
        fit_vector(model, d, ctx.val_fit, c, vecs["a_last"])
        res["val"]["c_fit_recent"] = brief(ev(model, ctx.val_score_sets, c, n_samples=8))
        vecs["c_fit_recent"] = fit_vector(model, t, ctx.test_fit, c, vecs["a_last"])
        res["test_score"]["c_fit_recent"] = brief(ev(model, (t, *ctx.sets_score), c))
        # how far apart the learned vectors are (year-to-year steps for per-year configs)
        if period == "year":
            Wy = W[:N_YEARS]
            res["vec_step_norms"] = (Wy[1:] - Wy[:-1]).norm(dim=-1).tolist()
            res["vec_norms"] = Wy.norm(dim=-1).tolist()
        model.test_vec = vecs["a_last"]
        test_vecs = {k: v.cpu() for k, v in vecs.items()}
    else:
        res["val"]["plain"] = brief(ev(model, ctx.val_score_sets, c, n_samples=8))
        res["test_score"]["plain"] = brief(ev(model, (t, *ctx.sets_score), c))
        res["test_all"]["plain"] = brief(ev(model, (t, *ctx.sets_all), c))
        test_vecs = {}
    res["eval_s"] = time.time() - t0
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "test_vecs": test_vecs,
                "cfg": {"hist": H, "posenc": POSENC, "n_weeks": n_rows, "drift_dim": dim, "period": period,
                        "smooth": smooth, "vec_dropout": vdrop, "clean": clean, "seed": seed, "steps": steps,
                        "synth_amp": res["synth_amp"]}},
               CKPT / f"{phase}2_{name}_s{seed}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    print(f"  done {name} s{seed}: " + ", ".join(f"{k} 60m {v['60min_mean']:.3f}" for k, v in res["test_score"].items()),
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["synth", "real"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=16000)
    ap.add_argument("--configs", nargs="*")
    args = ap.parse_args()
    if os.name == "nt":  # keep Windows awake while this process runs
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    ctx = Ctx(torch.device("cuda"))
    default = ["plain", "week32", "year32_sd"] if args.phase == "synth" else ["year32", "year32_sd"]
    for name in args.configs or default:
        for s in args.seeds:
            run(ctx, args.phase, name, s, args.steps)


if __name__ == "__main__":
    main()
