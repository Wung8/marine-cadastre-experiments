"""Train one (position encoding, history length) model and evaluate it on the test year.

python src/train.py --posenc fourier --hist 6 --steps 20000
python src/train.py --baselines          # dead-reckoning numbers on the same test subset
"""
import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from aismodel import (
    GROUPS, KN_TO_KM_PER_STEP, MAX_HIST, N_ANG, N_FUTURE, N_MAG, ROOT, SPEED_CENTERS_KN, Net, Tracks,
    bin_centers, disp_km, featurize, from_frame, haversine_km, move, soft_labels, to_frame,
)

TRAIN_YEARS = list(range(2009, 2022))
VAL_YEARS = [2022]
TEST_YEARS = [2023]
RESULTS = ROOT / "results"
CKPT = ROOT / "checkpoints"


def fixed_subset(n_total, n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n_total, generator=g)[: min(n, n_total)]


def batch_inputs(w, H):
    """Inputs from the history part (first H+1 columns) of a window."""
    k = H + 1
    return featurize(w["lat"][:, :k], w["lon"][:, :k], w["sog"][:, :k], w["cog"][:, :k], w["group"], w["length"], w["draft"])


def target_fl(w, H, cog_cur):
    dx, dy = disp_km(w["lat"][:, H], w["lon"][:, H], w["lat"][:, H + 1], w["lon"][:, H + 1])
    f, l = to_frame(dx.float(), dy.float(), cog_cur)
    return torch.stack([f, l], -1)


def step_loss(model, w, H, centers):
    seq, flat, group, cog_cur, lat, lon = batch_inputs(w, H)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(seq, flat, group, lat, lon)
    logp = F.log_softmax(logits.float(), -1)
    q = soft_labels(target_fl(w, H, cog_cur), centers)
    return -(q * logp).sum(-1).mean()


def decode_mean(logits, centers):
    p = torch.softmax(logits.float(), -1)
    return p @ centers


def decode_sample(logits, gen):
    p = torch.softmax(logits.float(), -1)
    cls = torch.multinomial(p, 1, generator=gen).squeeze(-1)
    a, m = cls // N_MAG, cls % N_MAG
    # jitter uniformly inside the bin so samples are not stuck on bin centers
    ang = (a.float() + torch.rand(a.shape, device=a.device, generator=gen) - 0.5) * (2 * math.pi / N_ANG)
    sc = SPEED_CENTERS_KN.to(a.device)
    lo = torch.where(m > 0, 0.5 * (sc[m] + sc[(m - 1).clamp(min=0)]), 0.0)
    hi = torch.where(m < N_MAG - 1, 0.5 * (sc[m] + sc[(m + 1).clamp(max=N_MAG - 1)]), sc[m] + 5)
    r = (lo + (hi - lo) * torch.rand(a.shape, device=a.device, generator=gen)) * KN_TO_KM_PER_STEP
    return torch.stack([r * torch.cos(ang), r * torch.sin(ang)], -1)


@torch.no_grad()
def rollout(model, w, H, centers, n_steps, sample=False, gen=None):
    """Autoregressive rollout. Returns predicted (lat, lon) of shape (B, n_steps)."""
    k = H + 1
    lat, lon = w["lat"][:, :k].clone(), w["lon"][:, :k].clone()
    sog, cog = w["sog"][:, :k].clone(), w["cog"][:, :k].clone()
    out_lat, out_lon = [], []
    for _ in range(n_steps):
        seq, flat, group, cog_cur, la, lo = featurize(lat, lon, sog, cog, w["group"], w["length"], w["draft"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(seq, flat, group, la, lo)
        fl = decode_sample(logits, gen) if sample else decode_mean(logits, centers)
        dx, dy = from_frame(fl[:, 0], fl[:, 1], cog_cur)
        nlat, nlon = move(la, lo, dx.double(), dy.double())
        n_sog = torch.hypot(dx, dy) / KN_TO_KM_PER_STEP
        n_cog = torch.rad2deg(torch.atan2(dx, dy)) % 360
        # a near-stationary prediction has no meaningful course; keep the previous one
        n_cog = torch.where(n_sog > 0.3, n_cog, cog_cur)
        lat = torch.cat([lat[:, 1:], nlat[:, None]], 1)
        lon = torch.cat([lon[:, 1:], nlon[:, None]], 1)
        sog = torch.cat([sog[:, 1:], n_sog[:, None]], 1)
        cog = torch.cat([cog[:, 1:], n_cog[:, None]], 1)
        out_lat.append(nlat)
        out_lon.append(nlon)
    return torch.stack(out_lat, 1), torch.stack(out_lon, 1)


def summarize(err, groups):
    """err: (B,) km. Mean/median overall and per vessel group."""
    d = {"mean_km": err.mean().item(), "median_km": err.median().item(), "p90_km": err.quantile(0.9).item()}
    for gi, g in enumerate(GROUPS):
        m = groups == gi
        if m.any():
            d[f"{g}_mean_km"] = err[m].mean().item()
    return d


@torch.no_grad()
def evaluate(model, data, H, idx10, idx60, idx_bon, centers, n_samples=16, bs=8192):
    model.eval()
    res = {}
    # 10-minute: mean-decoded point error + NLL of the true bin (hard label)
    errs, nll, grp = [], [], []
    for i in range(0, len(idx10), bs):
        w = data.window(data.samples[idx10[i : i + bs]], H)
        seq, flat, group, cog_cur, lat, lon = batch_inputs(w, H)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(seq, flat, group, lat, lon)
        fl = decode_mean(logits, centers)
        dx, dy = from_frame(fl[:, 0], fl[:, 1], cog_cur)
        plat, plon = move(lat, lon, dx.double(), dy.double())
        errs.append(haversine_km(plat, plon, w["lat"][:, H + 1], w["lon"][:, H + 1]))
        true_cls = ((target_fl(w, H, cog_cur)[:, None, :] - centers[None]) ** 2).sum(-1).argmin(-1)
        nll.append(-F.log_softmax(logits.float(), -1).gather(1, true_cls[:, None]).squeeze(1))
        grp.append(group)
    err, grp = torch.cat(errs), torch.cat(grp)
    res["10min"] = summarize(err, grp) | {"nll_true_bin": torch.cat(nll).mean().item()}

    # 60-minute: deterministic mean-decoded rollout
    e10, e30, e60, ade, grp = [], [], [], [], []
    for i in range(0, len(idx60), bs):
        w = data.window(data.samples[idx60[i : i + bs]], H)
        plat, plon = rollout(model, w, H, centers, N_FUTURE)
        tl, tn = w["lat"][:, H + 1 :], w["lon"][:, H + 1 :]
        e = haversine_km(plat, plon, tl, tn)
        e10.append(e[:, 0]); e30.append(e[:, 2]); e60.append(e[:, 5]); ade.append(e.mean(1)); grp.append(w["group"])
    grp = torch.cat(grp)
    res["rollout_mean"] = {
        "at10_mean_km": torch.cat(e10).mean().item(),
        "at30_mean_km": torch.cat(e30).mean().item(),
        "ade60_mean_km": torch.cat(ade).mean().item(),
        "at60": summarize(torch.cat(e60), grp),
    }

    # 60-minute: best-of-N sampled rollouts
    gen = torch.Generator(device=data.device).manual_seed(0)
    best, grp = [], []
    for i in range(0, len(idx_bon), bs // n_samples):
        w = data.window(data.samples[idx_bon[i : i + bs // n_samples]], H)
        wr = {k: v.repeat_interleave(n_samples, 0) for k, v in w.items()}
        plat, plon = rollout(model, wr, H, centers, N_FUTURE, sample=True, gen=gen)
        e = haversine_km(plat[:, -1], plon[:, -1], wr["lat"][:, -1], wr["lon"][:, -1]).view(-1, n_samples)
        best.append(e.min(1).values)
        grp.append(w["group"])
    res[f"rollout_best_of_{n_samples}"] = {"at60": summarize(torch.cat(best), torch.cat(grp))}
    model.train()
    return res


@torch.no_grad()
def baselines(data, idx10, idx60):
    """Dead reckoning from AIS SOG/COG, and constant velocity from the last 10-min displacement."""
    res = {}
    H = MAX_HIST
    for name, idx in (("10min", idx10), ("60min", idx60)):
        dr, cv, grp = [], [], []
        for i in range(0, len(idx), 65536):
            w = data.window(data.samples[idx[i : i + 65536]], H)
            lat, lon = w["lat"][:, H], w["lon"][:, H]
            k = 1 if name == "10min" else N_FUTURE
            tl, tn = w["lat"][:, H + k], w["lon"][:, H + k]
            dist = w["sog"][:, H] * KN_TO_KM_PER_STEP * k
            th = torch.deg2rad(w["cog"][:, H])
            plat, plon = move(lat, lon, (dist * torch.sin(th)).double(), (dist * torch.cos(th)).double())
            dr.append(haversine_km(plat, plon, tl, tn))
            dx, dy = disp_km(w["lat"][:, H - 1], w["lon"][:, H - 1], lat, lon)
            plat, plon = move(lat, lon, dx * k, dy * k)
            cv.append(haversine_km(plat, plon, tl, tn))
            grp.append(w["group"])
        grp = torch.cat(grp)
        res[name] = {"dead_reckoning": summarize(torch.cat(dr), grp), "const_velocity": summarize(torch.cat(cv), grp)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posenc", choices=["fourier", "grid"], default="fourier")
    ap.add_argument("--hist", type=int, default=6)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--n10", type=int, default=500_000)
    ap.add_argument("--n60", type=int, default=200_000)
    ap.add_argument("--nbon", type=int, default=20_000)
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--bench", type=int, default=0, help="time N training steps and exit")
    ap.add_argument("--train-years", type=int, nargs="*", default=TRAIN_YEARS)
    ap.add_argument("--val-years", type=int, nargs="*", default=VAL_YEARS)
    ap.add_argument("--test-years", type=int, nargs="*", default=TEST_YEARS)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    assert 0 <= args.hist <= MAX_HIST
    dev = torch.device("cuda")
    torch.manual_seed(0)
    RESULTS.mkdir(exist_ok=True)
    CKPT.mkdir(exist_ok=True)
    centers = bin_centers(dev)

    test = Tracks(args.test_years, dev)
    n_test = len(test.samples)
    idx10 = fixed_subset(n_test, args.n10, 1).to(dev)
    idx60 = fixed_subset(n_test, args.n60, 2).to(dev)
    idx_bon = fixed_subset(n_test, args.nbon, 3).to(dev)

    if args.baselines:
        res = baselines(test, idx10, idx60)
        (RESULTS / f"baselines{args.tag}.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        return

    train = Tracks(args.train_years, dev)
    val = Tracks(args.val_years, dev)
    vidx10 = fixed_subset(len(val.samples), 200_000, 4).to(dev)
    vidx60 = fixed_subset(len(val.samples), 50_000, 5).to(dev)
    vbon = fixed_subset(len(val.samples), 2_000, 6).to(dev)
    print(f"train samples {len(train.samples):,}  val {len(val.samples):,}  test {n_test:,}", flush=True)

    H = args.hist
    model = Net(H, args.posenc).to(dev)
    emb = [p for n, p in model.named_parameters() if n.startswith("pos.tables")]
    other = [p for n, p in model.named_parameters() if not n.startswith("pos.tables")]
    opt = torch.optim.AdamW(
        [{"params": other, "weight_decay": 1e-4}, {"params": emb, "weight_decay": 0.0, "lr": args.lr * 5}],
        lr=args.lr, betas=(0.9, 0.99), eps=1e-10,
    )
    base_lrs = [g["lr"] for g in opt.param_groups]
    n_params = sum(p.numel() for p in model.parameters())
    name = f"{args.posenc}_h{H}{args.tag}"
    print(f"{name}: {n_params:,} params", flush=True)

    total = args.bench or args.steps
    warm = min(500, total // 10)
    log = []
    t0 = time.time()
    n_train = len(train.samples)
    for step in range(1, total + 1):
        sched = step / warm if step <= warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / (total - warm)))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * max(sched, 0.02)
        s = train.samples[torch.randint(n_train, (args.batch,), device=dev)]
        loss = step_loss(model, train.window(s, H), H, centers)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if args.bench and step == 50:
            torch.cuda.synchronize()
            t_b = time.time()
        if step % 200 == 0 and not args.bench:
            el = time.time() - t0
            print(f"step {step}/{total} loss {loss.item():.4f}  {el:.0f}s elapsed, ~{el / step * (total - step) / 60:.1f} min left", flush=True)
        if not args.bench and (step % args.eval_every == 0 or step == total):
            vr = evaluate(model, val, H, vidx10, vidx60, vbon, centers, n_samples=8)
            log.append({"step": step, "time_s": time.time() - t0, "loss": loss.item(),
                        "val_10min_mean_km": vr["10min"]["mean_km"], "val_nll": vr["10min"]["nll_true_bin"],
                        "val_60min_mean_km": vr["rollout_mean"]["at60"]["mean_km"]})
            print("  val", json.dumps(log[-1]), flush=True)
    if args.bench:
        torch.cuda.synchronize()
        dt = (time.time() - t_b) / (args.bench - 50)
        print(f"BENCH {name}: {dt * 1000:.1f} ms/step at batch {args.batch} -> {args.batch / dt:,.0f} samples/s", flush=True)
        return

    train_time = time.time() - t0
    torch.save(model.state_dict(), CKPT / f"{name}.pt")
    del train
    torch.cuda.empty_cache()
    t1 = time.time()
    res = evaluate(model, test, H, idx10, idx60, idx_bon, centers)
    out = {"name": name, "posenc": args.posenc, "hist": H, "steps": args.steps, "batch": args.batch,
           "params": n_params, "train_time_s": train_time, "eval_time_s": time.time() - t1,
           "test": res, "val_log": log}
    (RESULTS / f"{name}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["test"], indent=2), flush=True)


if __name__ == "__main__":
    main()
