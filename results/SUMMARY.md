# Ship position prediction: position encoding × history length

Test set: **May 2023**, East Coast, 1.20M samples. Errors are great-circle distance in km.
Train on May 2009–2021 (12.3M samples), validate on May 2022 (1.09M samples). May 2024 is not currently hosted by Marine Cadastre.

## Results

| model | pos enc | history | 10m mean | 10m median | 60m mean | 60m median | 60m best-of-16 | train min |
|---|---|---|---|---|---|---|---|---|
| dead reckoning | – | – | 0.651 | 0.196 | 7.325 | 4.030 | – | – |
| const velocity | – | – | 0.800 | 0.282 | 6.770 | 3.817 | – | – |
| fourier_h0 | fourier | 0 | 0.336 | 0.139 | 3.126 | 1.735 | 1.367 | 2.5 |
| fourier_h3 | fourier | 30 min | 0.306 | 0.123 | 2.991 | 1.556 | 1.335 | 3.1 |
| fourier_h6 | fourier | 60 min | 0.299 | 0.121 | 2.908 | 1.522 | 1.279 | 3.3 |
| fourier_h12 | fourier | 120 min | 0.295 | 0.121 | 2.835 | 1.497 | 1.250 | 4.1 |
| grid_h0 | grid | 0 | 0.336 | 0.139 | 3.058 | 1.689 | 1.290 | 7.7 |
| grid_h3 | grid | 30 min | 0.300 | 0.120 | 2.888 | 1.519 | 1.238 | 8.2 |
| grid_h6 | grid | 60 min | 0.294 | 0.119 | 2.813 | 1.479 | 1.201 | 8.4 |
| **grid_h12** | grid | 120 min | **0.288** | **0.118** | **2.744** | **1.444** | **1.158** | 9.2 |

60-minute mean error by vessel group:

| model | cargo | tanker | passenger | service | military/law |
|---|---|---|---|---|---|
| dead reckoning | 3.85 | 3.23 | 15.43 | 6.69 | 10.18 |
| fourier_h12 | 2.53 | 2.35 | 3.47 | 2.81 | 5.42 |
| grid_h12 | 2.48 | 2.27 | 3.32 | 2.71 | 5.64 |

## Findings

- **Every model beats dead reckoning by a wide margin.** The best model (grid, 2 h history) cuts the 60-minute mean error by 63% (7.33 → 2.74 km) and the 10-minute mean error by 56% (0.65 → 0.29 km). The largest gain is for passenger vessels: 15.4 → 3.3 km at 60 minutes.
- **More history always helps; most of the gain comes in the first 30 minutes.** Going from no history to 30 min cuts the 10-minute error by about 10%. Going from 30 min to 2 h cuts it by only another ~4%. At 60 minutes, the longer histories keep helping a little more.
- **The learned grid beats Fourier features at 60 minutes for every history length** (about 2–5% lower mean error, 3–7% lower best-of-16). At 10 minutes the two are nearly tied. The grid's gains come from passenger and service vessels, which operate near ports. On cargo ships and tankers at 10 minutes it is very slightly worse, because those ships mostly cross open water, where geography matters little.
- **The cost of the grid:** about 10× the parameters (11.8M vs 1.3M, almost all in the hash tables) and 2–3× the training time. grid_h0's validation error rose slightly at the end (2.93 → 2.98 km), a mild sign of overfitting.
- **Sampling matters at 60 minutes.** The best of 16 sampled rollouts is about 2.4× closer than the single mean-decoded rollout. When the model is unsure, its mean prediction often falls between two likely paths.

## Caveats

- **One seed per configuration.** Differences of 1–3% between neighboring rows may be within seed-to-seed noise; the trends across rows are more reliable than any single gap.
- **One test year (2023).**
- **Military/law enforcement is a small group,** with sparse labels after 2018 (likely a coding change), so its numbers are noisy.
- **Means are dominated by hard cases** (turning ferries, tugs); medians are reported alongside.
- **During the rollout, a predicted step's speed and course are its 10-minute average,** while observed steps use the instantaneous AIS values, so inputs at later rollout steps differ slightly from training inputs.
- **The RBF encoding was not tested.**

## Setup

- **Region:** lat 24–46°N, lon 82–65°W, excluding the Great Lakes, the St. Lawrence and Florida's Gulf coast.
- **Vessel types:** cargo, tanker, passenger, service (tug/tow, pilot, SAR, dredging, port tender…) and military/law enforcement. Both AIS codes and Marine Cadastre's 1001–1025 codes are mapped.
- **Resampling:** a 10-minute grid, linearly interpolated, with no grid point placed across a reporting gap of more than 20 min. A sample needs 2 h of past and 1 h of future points in one continuous segment, and SOG ≥ 1 kn.
- **Output:** 36 turn-angle bins (relative to current COG) × 32 distance bins (0–30 kn in 1-kn steps, plus one bin at 35 kn), trained with Gaussian soft labels in x/y space (σ = 0.15 km + 5% of the distance travelled). Point predictions use the probability-weighted mean of the bin centers.
- **Model:** a GRU (128) over the history steps, then an MLP (3 × 512) taking the GRU output, current speed/course, position encoding and static info (vessel group, length, draft).
- **Position encodings:**
  - **Fourier:** sin/cos features at wavelengths from 2500 km down to 1.2 km.
  - **Grid:** 4 levels (64, 16, 4 and 1 km cells), 8 dims per level, bilinear interpolation; the 1 km level is hashed into a 1M-entry table.
- **Training:** 30k steps × batch 4096 (about 10 epochs), AdamW with a cosine schedule. All runs use the same budget.
- **60-minute prediction:** 6 autoregressive 10-minute steps.

Reproduce:

```bash
python src/fetch_filter.py --workers 2
python src/build_dataset.py
bash src/run_all.sh
python src/compare.py --md
```
