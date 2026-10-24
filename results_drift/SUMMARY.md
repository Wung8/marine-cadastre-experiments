# Drift and noise study (grid position encoding, 2 h history)

Errors are great-circle km on **May 8–31 2023** unless noted. Each number is the mean ± std of 3 seeds.

**Splits**

| | |
|---|---|
| Train | May 2009–2021 + May 1–21 2022 (13.0M samples) |
| Validation | May 22–31 2022. Used for early stopping and for every choice below; the 2023 test set was never used for selection. |
| Test | May 2023 |

May 1–7 2023 is held out of the test score because one test-time option (c) fits on it.

**Training:** 16k steps × batch 4096, with the best validation checkpoint kept. Each full run takes about 5 min on the laptop RTX 4070. Total compute was 2 h 45 min (2:47–5:32 am, 63 runs).

## 1. Is there drift? Yes

One model per May, each trained on an equal-size sample (439k samples from days 1–21, cleaning on) and tested on all of May 2023. "Own-year" is the model's error on May 22–31 of its own training year.

| Train on May… | 2023 10-min | 2023 60-min | own-year 60-min | gap |
|---|---|---|---|---|
| 2009 | 0.568 ± 0.008 | 4.65 ± 0.04 | 3.82 | +0.83 |
| 2010 | 0.578 ± 0.007 | 4.67 ± 0.04 | 3.87 | +0.81 |
| 2011 | 0.532 ± 0.009 | 4.32 ± 0.04 | 3.67 | +0.66 |
| 2012 | 0.479 ± 0.003 | 4.04 ± 0.01 | 3.63 | +0.42 |
| 2013 | 0.479 ± 0.001 | 4.09 ± 0.02 | 3.53 | +0.56 |
| 2014 | 0.463 ± 0.003 | 3.93 ± 0.03 | 3.45 | +0.49 |
| 2015 | 0.455 ± 0.002 | 3.92 ± 0.00 | 3.38 | +0.54 |
| 2016 | 0.438 ± 0.001 | 3.83 ± 0.03 | 3.53 | +0.31 |
| 2017 | 0.416 ± 0.001 | 3.70 ± 0.01 | 3.50 | +0.20 |
| 2018 | 0.406 ± 0.002 | 3.68 ± 0.02 | 3.49 | +0.19 |
| 2019 | 0.402 ± 0.003 | 3.67 ± 0.01 | 3.26 | +0.41 |
| 2020 | 0.451 ± 0.000 | 3.93 ± 0.04 | 3.24 | +0.68 |
| 2021 | 0.394 ± 0.004 | 3.68 ± 0.01 | 3.33 | +0.35 |
| 2022 | 0.375 ± 0.004 | 3.63 ± 0.02 | 3.39 | +0.24 |

- **Older data predicts 2023 worse, by a lot and almost every year.** 10-minute error falls by a third, from 0.57 to 0.375 km, between 2009 and 2022.
- **Part of this is that older years are harder in general**, since own-year error falls too. But the gap shrinks from about +0.8 to about +0.2 km, which is the drift-specific part.
- **COVID-era May 2020 breaks the trend:** it scores well on its own year but transfers badly to 2023 (+0.68).
- **Caveat:** older years also have fewer vessels and sparser receiver coverage (2,159 vessels in 2009 vs 4,047 in 2023). So "age" also includes "less diverse data", even at equal sample counts.

## 2. Noise handling

Configs:
- **clean:** drop windows with any 10-min step faster than 30 kn (cargo, tanker, service) or 45 kn (passenger, military), and drop targets interpolated across a reporting gap longer than 10 min. This removes 0.7% of samples.
- **trunc:** after warm-up, ignore the 2% of each batch with the highest loss.

| Config | 10-min mean | 10-min median | 60-min mean | 60-min median | 60-min best-of-16 | val 60-min |
|---|---|---|---|---|---|---|
| none | 0.282 ± 0.001 | 0.117 | 2.756 ± 0.004 | 1.455 | **1.174** | 2.610 |
| **clean** | 0.283 ± 0.001 | 0.116 | **2.738 ± 0.007** | 1.405 | 1.188 | **2.588** |
| trunc | **0.279 ± 0.001** | 0.112 | 2.802 ± 0.008 | 1.407 | 1.267 | 2.643 |
| clean + trunc | 0.280 ± 0.001 | **0.111** | 2.800 ± 0.008 | **1.375** | 1.281 | 2.629 |

- **Cleaning is the only config that improves the 60-minute mean** (−0.6%; −3.4% on the median). Validation picked it.
- **Truncation is a trade-off.** It makes typical 10-minute predictions sharper (median −4%) but hurts the 60-minute mean (+1.7%) and best-of-16 (+8%). By ignoring the hardest samples, the model stops putting probability on rare maneuvers, so its sampled rollouts no longer cover them.

## 3. Drift embedding

A learned vector for each week of each May (weeks 1–7, 8–14, 15–21 and 22–31), trained without noise handling.

Test-time vector options:
- **(a)** the last training week's vector (May 15–21 2022)
- **(b)** the mean of the three 2022 vectors
- **(c)** a new vector fitted on May 1–7 2023, with the model frozen

| Model | Test vector | 60-min mean | 60-min median | best-of-16 | val 60-min |
|---|---|---|---|---|---|
| no embedding | – | **2.756 ± 0.004** | **1.455** | **1.174** | 2.610 |
| 8-d | (a) last week | 2.780 ± 0.018 | 1.476 | 1.208 | 2.629 |
| 8-d | (b) mean of 2022 | 2.779 ± 0.017 | 1.472 | 1.208 | 2.628 |
| 8-d | (c) fit on May 1–7 2023 | 2.788 ± 0.018 | 1.484 | 1.206 | 2.633 |
| 32-d | (a) last week | 2.774 ± 0.007 | 1.477 | 1.214 | 2.619 |
| 32-d | (b) mean of 2022 | 2.774 ± 0.007 | 1.477 | 1.215 | 2.622 |
| 32-d | (c) fit on May 1–7 2023 | 2.786 ± 0.008 | 1.491 | 1.217 | 2.629 |
| no embedding, fine-tuned on May 1–7 2023 | – | 2.781 ± 0.004 | – | – | – |

- **No drift-embedding variant beats the plain model.** All are 0.02–0.03 km worse at 60 minutes and 3–4% worse on best-of-16. The 10-minute error is identical (0.281–0.283) everywhere.
- **The 32-d version is more stable across seeds than 8-d** (std 0.007 vs 0.018). Validation chose 32-d with vector (a).
- **Test-time options (a) and (b) are interchangeable:** the 2022 vectors are nearly the same.
- **Using a week of 2023 labels doesn't help either.** Option (c) is the worst variant, and plain fine-tuning on the same week also hurts (2.781). One week of new data is too little to shift a model trained on 13M samples, and it doesn't represent weeks 2–4 of May anyway.

## 4. 2×2: noise handling × drift embedding

The noise handling used is **clean**. The drift embedding used is **32-d with test vector (a)**. Both were chosen on validation.

| | No drift embedding | Drift embedding |
|---|---|---|
| **No noise handling** | 2.756 ± 0.004 (median 1.455) | 2.774 ± 0.007 (median 1.477) |
| **Cleaning** | **2.738 ± 0.007** (median 1.405) | 2.755 ± 0.010 (median 1.434) |

The two effects are additive, with no interaction:
- **Cleaning:** −0.018 km with or without the drift embedding.
- **Drift embedding:** +0.018 km with or without cleaning.

**Best model: cleaning, no drift embedding.**

## Takeaways

1. **The drift is real and large across years.** A model trained only on old data loses up to a third of its accuracy on 2023.
2. **Training on all years at once already handles most of it.** The full model (13M samples) beats the best single-year model by a wide margin (0.28 vs 0.375 km at 10 minutes). Per-week drift vectors don't add anything on top, and nor does adapting to the first week of the test month.
3. **This matches your LaDe finding.** The drift here is real but isn't a trend a per-week vector can carry forward. The 2022→2023 change is small next to the noise from week to week.
4. **For noise, clean the data rather than the loss.** Removing physically impossible steps and badly interpolated targets helps a little; dropping high-loss samples hurts the 60-minute rollouts.

## 5. Follow-up: does the drift embedding work at all, and do per-year vectors help?

Script: `src/drift2.py`. Every run here uses cleaning.

**Injected drift.** Training targets of May *Y* are scaled by 1 + 0.3 × (2022 − *Y*)/13, so 2009 is +30% and 2022 is +0%. Validation and test keep real labels, so a method that copes with the drift should land near its real-data error.

New design (**per-year + s/d**): one 32-d vector per year, plus:
- **smoothness:** a penalty λ = 1 on ‖v_y − v_{y−1}‖²
- **vector dropout:** 30% of samples see a shared "unknown" vector instead of their year's

| Model | Test vector | Injected drift, 60-min | Real labels, 60-min |
|---|---|---|---|
| plain | – | **7.067 ± 0.084** | 2.738 ± 0.007 |
| per-week 32-d | last week | 2.775 ± 0.016 | 2.755 ± 0.010 |
| per-year + s/d | 2022 vector | 2.757 ± 0.008 (2 seeds) | 2.759 ± 0.013 |
| per-year + s/d | "unknown" vector | 7.664 ± 0.031 (2 seeds) | 2.753 ± 0.015 |
| per-year + s/d | fit on May 1–7 2023 | 2.831 ± 0.023 (2 seeds) | 2.787 ± 0.022 |

- **The drift embedding works when the drift is the kind it can model.** Under injected drift the plain model's 60-minute error goes up 2.6× (2.74 → 7.07). Both embedding designs land back at their real-data score (2.76–2.78).
- **The "unknown" vector shows what kind of drift it is facing.** It learns the drift averaged over all years, so it's wrong for trending drift (7.66) but fine on real data (2.753). On real data it's the best per-year option, a hint that real drift doesn't trend.
- **The learned vectors confirm it.**
  - Under injected drift, the per-year vectors walk an even path (each step norm about 0.013).
  - On real data they shrink to almost nothing (norms 0.01–0.04, below their initial ~0.06). The model finds no per-year conditional change worth encoding, even though single-year models show large drift (section 1). The real drift is mostly in *which* traffic and coverage each year has (inputs), and the plain model already conditions on those.
- **Fitting a vector on a week of 2023 labels is the worst option in every setting.**
- **Validation chose per-week 32-d with the last week's vector** (2.5998 vs 2.6029 km for per-year with the unknown vector; a tie in practice). The 2×2 in section 4 therefore stands as is.

**Final 2×2** (May 8–31 2023, 60-min mean; drift method = per-week 32-d, last week's vector):

| | No drift embedding | Drift embedding |
|---|---|---|
| No noise handling | 2.756 ± 0.004 | 2.774 ± 0.007 |
| Cleaning | **2.738 ± 0.007** | 2.755 ± 0.010 |

## Saved models

All 12 models are written to `models/` as `{neither,noise_only,drift_only,noise+drift}_s{0,1,2}.pt`, each holding `state_dict`, `cfg` and `test_vecs`. Only `noise_only_s*.pt`, the best model, is in the GitHub repo; `bash src/run_drift.sh` rebuilds the rest. To load one:

```python
import torch
from aismodel import Net

ck = torch.load("models/noise_only_s0.pt")
m = Net(ck["cfg"]["hist"], ck["cfg"]["posenc"], n_weeks=ck["cfg"]["n_weeks"], drift_dim=ck["cfg"]["drift_dim"])
m.load_state_dict(ck["state_dict"])
if ck["cfg"]["drift_dim"]:
    m.test_vec = ck["test_vecs"]["a_last_week"]
```

## Files and reproducing

- **Results:** per-run JSONs in `results_drift/{noise,age,drift,factorial}/`, plus `results_drift/factorial_summary.json`.
- **Logs:** `logs/drift_*.log`.
- **Reproduce:** run `bash src/run_drift.sh`. It is resumable and skips finished runs.
