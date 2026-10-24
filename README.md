# Ship position prediction from AIS (Marine Cadastre, US East Coast)

Predict where a ship will be **10 minutes** ahead from its AIS track, then roll the model out six times to
predict **60 minutes** ahead. The data is May of every year 2009–2023 from
[Marine Cadastre](https://marinecadastre.gov/ais/). The experiments compare position encodings and history
lengths, then test the best model for **concept drift** across years and for **noise handling**.

## Results at a glance

May 2023 test set, East Coast; errors are great-circle km.

| | 10-min mean | 60-min mean | 60-min median |
|---|---|---|---|
| Dead reckoning (AIS speed + course) | 0.651 | 7.33 | 4.03 |
| Fourier position encoding, 2 h history | 0.295 | 2.84 | 1.50 |
| Learned multi-resolution grid, 2 h history | 0.288 | 2.74 | 1.44 |
| **Grid + window cleaning** (May 8–31, 3 seeds) | **0.283** | **2.738** | **1.405** |

- **Encoding and history** ([results/SUMMARY.md](results/SUMMARY.md)):
  - The grid encoding beats Fourier features at 60 minutes for every history length.
  - More history always helps; most of the gain comes in the first 30 minutes.
- **Drift and noise** ([results_drift/SUMMARY.md](results_drift/SUMMARY.md)), 3 seeds per setting:
  - **Drift is real.** A model trained on May of one year predicts May 2023 worse the older that year is: 0.57 km at 10 minutes for 2009 vs 0.375 km for 2022.
  - **Training on all years already absorbs it.** Per-week and per-year drift embeddings fully remove an *injected* label drift (7.07 → 2.76 km at 60 minutes) but don't beat plain training on the real data.
  - **Noise:** cleaning windows of physically impossible steps helps slightly. Loss truncation (dropping the highest-loss samples) hurts the 60-minute rollouts.

## Task and model

**Samples**
- **Vessels:** cargo, tanker, passenger, service (tug/tow, pilot, SAR, dredging…) and military/law enforcement, moving at ≥ 1 kn.
- **Resampling:** tracks are put on a 10-minute grid by linear interpolation.
- **Windows:** each sample needs 2 h of history and 1 h of future in one continuous segment.

**Output:** a class distribution over **36 turn-angle bins** (relative to the current course) × **32 distance bins** (0–30 kn in 1-kn steps, plus one bin at 35 kn).
- **Training labels:** Gaussian soft labels in x/y space.
- **Point prediction:** the probability-weighted mean of the bin centers.
- **60 minutes:** six autoregressive steps, decoded by mean, or sampled 16 times with the best sample scored ("best-of-16").

**Model**
- A GRU over the history steps, then an MLP that also takes the current speed/course, a position encoding, and vessel group/length/draft.
- **Position encodings:**
  - **Fourier:** sin/cos at wavelengths from 2500 km to 1.2 km.
  - **Grid:** 64 / 16 / 4 / 1 km cells, bilinear interpolation, with the finest level hashed (Instant-NGP style).

**Splits:** train on May 2009–2021 (+ May 1–21 2022 in the drift study), validate on May 2022, test on May 2023. May 2024 isn't currently hosted by Marine Cadastre.

## Quick start

```bash
pip install -r requirements.txt
python src/fetch_filter.py --workers 2    # download + filter May 2009-2023 (~83 GB download, ~3 GB kept)
python src/build_dataset.py               # 10-minute tracks -> data/tracks/{year}.npz
bash src/run_all.sh                       # encoding x history sweep -> results/
python src/compare.py --md                # results table
bash src/run_drift.sh                     # noise ablation, data age, drift embedding, 2x2 -> results_drift/
python src/drift2.py --phase synth        # injected-drift check
python src/drift2.py --phase real         # per-year drift vectors on real data
python src/final_2x2.py                   # final 2x2 summary
```

**Notes for running**
- Every script skips runs that are already recorded, so you can stop and restart them.
- `fetch_filter.py` streams one file at a time and deletes each raw download after filtering it. Keep `--workers` at 2–3 on a 32 GB machine.
- On a laptop RTX 4070 (8 GB), one full training run takes about 5 minutes; the whole drift study took about 3 hours.

## Saved model

`models/noise_only_s{0,1,2}.pt` holds the best model (grid, 2 h history, window cleaning, no drift embedding), 3 seeds. To load it:

```python
import torch
from aismodel import Net

ck = torch.load("models/noise_only_s0.pt")
m = Net(ck["cfg"]["hist"], ck["cfg"]["posenc"], n_weeks=ck["cfg"]["n_weeks"], drift_dim=ck["cfg"]["drift_dim"])
m.load_state_dict(ck["state_dict"])
```

The other 2×2 models are not in the repo; `bash src/run_drift.sh` rebuilds them.

## Data notes

These quirks were found and handled while building the dataset:

- **2009–2014:** monthly UTM-zone geodatabases (zones 17–19). In 2013–2014 the layer names carry a prefix (`Zone17_2013_05_Broadcast`). SOG and COG are already in knots and degrees (checked against speed and bearing computed from positions); only draught is stored in tenths.
- **2015–2017:** mostly Marine Cadastre's own 1001–1025 vessel codes, mapped with NOAA's VesselTypeCodes2018 crosswalk.
- **Malformed CSVs:** a few 2015 and 2023 daily files have quoted newlines or rows with an extra field; the fallback reader skips bad rows.
- **Military/law enforcement:** samples drop sharply from 2018 on, likely a coding change.
- **Region:** lat 24–46°N, lon 82–65°W, excluding the Great Lakes, the St. Lawrence and Florida's Gulf coast.

## Files

| File | What it does |
|---|---|
| `src/fetch_filter.py` | download May files, filter by region and vessel type → `data/filtered/` |
| `src/build_dataset.py` | resample to 10-minute tracks, cut sample windows → `data/tracks/` |
| `src/aismodel.py` | track loading, featurization, polar bins, Fourier / grid encodings, model |
| `src/train.py` | encoding × history experiments, evaluation (10-min, 60-min rollout, best-of-16), baselines |
| `src/compare.py` | results table for `results/` |
| `src/drift.py` | noise ablation, data age, drift embedding, noise × drift 2×2 |
| `src/drift2.py` | injected-drift check, per-year drift vectors with smoothness and vector dropout |
| `src/final_2x2.py` | picks the drift method on validation and writes the final 2×2 |
| `results/`, `results_drift/` | per-run JSONs and the two write-ups |
