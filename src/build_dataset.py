"""Turn the filtered per-file parquets into 10-minute resampled tracks per year.

Output: data/tracks/{year}.npz with
  lat, lon (f64), sog (kn, f32), cog (deg, f32, nan = unknown), vid (i32), t (i64),
  gap (i16, seconds between the raw reports a point was interpolated from)
      one row per valid 10-minute grid point, sorted by (vid, t), only points
      that fall inside some sample window are kept
  samples (i64): index of the "current" point of each sample; every sample has
      MAX_HIST valid points before it and N_FUTURE valid points after it, all in
      one contiguous segment
  v_mmsi, v_group, v_length, v_draft: per-vessel static info, indexed by vid

Usage: python src/build_dataset.py 2015 2016 ...   (no args = every year present)
"""
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
FILTERED = ROOT / "data" / "filtered"
TRACKS = ROOT / "data" / "tracks"

STEP = 600  # seconds
MAX_GAP = 1200  # max gap between the raw reports bracketing a grid point
MAX_HIST = 12  # longest history we test (2 h)
N_FUTURE = 6  # 60 minutes
MIN_SOG = 1.0  # only samples where the ship is underway
MAX_SPEED_KN = 60.0
KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON = 111.320


def local_dist_km(lat1, lon1, lat2, lon2):
    mlat = np.radians(0.5 * (lat1 + lat2))
    dx = (lon2 - lon1) * KM_PER_DEG_LON * np.cos(mlat)
    dy = (lat2 - lat1) * KM_PER_DEG_LAT
    return np.hypot(dx, dy)


def drop_jumps(vid, t, lat, lon):
    """Mask of raw points to keep: drop points that imply > MAX_SPEED_KN both
    coming in and going out (single-point GPS glitches)."""
    same = vid[1:] == vid[:-1]
    dt = np.maximum(t[1:] - t[:-1], 1) / 3600.0
    spd = local_dist_km(lat[:-1], lon[:-1], lat[1:], lon[1:]) / 1.852 / dt
    fast = same & (spd > MAX_SPEED_KN)
    bad_in = np.r_[False, fast]
    bad_out = np.r_[fast, False]
    return ~(bad_in & bad_out)


def resample(vid, t, lat, lon, sog, cog):
    n = len(t)
    # per-vessel spans
    starts = np.r_[0, np.flatnonzero(vid[1:] != vid[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    g0 = -(-t[starts] // STEP) * STEP
    g1 = (t[ends - 1] // STEP) * STEP
    cnt = np.maximum((g1 - g0) // STEP + 1, 0)
    gv = np.repeat(vid[starts], cnt)
    gt = np.repeat(g0, cnt) + STEP * (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))

    key_raw = vid.astype(np.int64) * (1 << 34) + t
    key_g = gv.astype(np.int64) * (1 << 34) + gt
    j = np.searchsorted(key_raw, key_g, side="right")  # first raw point strictly after grid time
    p = j - 1  # last raw point at or before grid time
    nx = np.minimum(j, n - 1)
    exact = t[p] == gt
    ok_next = (j < n) & (vid[nx] == gv)
    gap = t[nx] - t[p]
    valid = exact | (ok_next & (gap <= MAX_GAP))

    w = np.where(exact | ~ok_next, 0.0, (gt - t[p]) / np.maximum(gap, 1))
    nx = np.where(exact | ~ok_next, p, nx)
    glat = lat[p] + w * (lat[nx] - lat[p])
    glon = lon[p] + w * (lon[nx] - lon[p])
    s0, s1 = sog[p], sog[nx]
    gsog = np.where(np.isfinite(s0) & np.isfinite(s1), s0 + w * (s1 - s0), np.where(np.isfinite(s0), s0, s1))
    near_first = w <= 0.5
    c_near = np.where(near_first, cog[p], cog[nx])
    c_far = np.where(near_first, cog[nx], cog[p])
    gcog = np.where(np.isfinite(c_near), c_near, c_far)

    # seconds between the raw reports this point was interpolated from (0 = exact report)
    ggap = np.where(exact | ~ok_next, 0, gap)
    m = valid
    return gv[m], gt[m], glat[m], glon[m], gsog[m].astype(np.float32), gcog[m].astype(np.float32), ggap[m]


def build_year(year: int):
    t0 = time.time()
    files = sorted((FILTERED / str(year)).glob("*.parquet"))
    df = pl.concat([pl.read_parquet(f) for f in files])
    n_raw = len(df)
    df = df.unique(["mmsi", "t"]).sort(["mmsi", "t"])

    static = (
        df.group_by("mmsi")
        .agg(
            pl.col("group").mode().first().alias("group"),
            pl.col("length").filter((pl.col("length") > 0) & (pl.col("length") < 500)).median().alias("length"),
            pl.col("draft").filter((pl.col("draft") > 0) & (pl.col("draft") < 30)).median().alias("draft"),
        )
        .sort("mmsi")
        .with_row_index("vid")
    )
    df = df.join(static.select("mmsi", "vid"), on="mmsi").sort(["vid", "t"])

    vid = df["vid"].to_numpy().astype(np.int32)
    t = df["t"].to_numpy()
    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()
    sog = df["sog"].to_numpy().astype(np.float64)
    cog = df["cog"].to_numpy().astype(np.float64)
    del df
    sog[(sog < 0) | (sog > 102.2)] = np.nan
    cog[(cog < 0) | (cog >= 360)] = np.nan

    for _ in range(2):
        keep = drop_jumps(vid, t, lat, lon)
        vid, t, lat, lon, sog, cog = (a[keep] for a in (vid, t, lat, lon, sog, cog))

    gv, gt, glat, glon, gsog, gcog, ggap = resample(vid, t, lat, lon, sog, cog)
    n_grid = len(gt)

    # contiguous segments of valid 10-minute points
    new_seg = np.r_[True, (gv[1:] != gv[:-1]) | (gt[1:] - gt[:-1] != STEP)]
    # a 10-minute step faster than MAX_SPEED_KN also breaks the segment
    step_spd = np.r_[0.0, local_dist_km(glat[:-1], glon[:-1], glat[1:], glon[1:]) / 1.852 * 6]
    new_seg |= step_spd > MAX_SPEED_KN
    seg = np.cumsum(new_seg)

    idx = np.arange(MAX_HIST, n_grid - N_FUTURE)
    ok = (seg[idx - MAX_HIST] == seg[idx]) & (seg[idx + N_FUTURE] == seg[idx])
    ok &= (gsog[idx] >= MIN_SOG) & np.isfinite(gcog[idx])
    samples = idx[ok]

    # keep only points that some sample window touches
    mark = np.zeros(n_grid + 1, dtype=np.int32)
    np.add.at(mark, samples - MAX_HIST, 1)
    np.add.at(mark, samples + N_FUTURE + 1, -1)
    keep = np.cumsum(mark[:-1]) > 0
    remap = np.cumsum(keep) - 1
    samples = remap[samples]

    TRACKS.mkdir(parents=True, exist_ok=True)
    np.savez(
        TRACKS / f"{year}.npz",
        lat=glat[keep],
        lon=glon[keep],
        sog=gsog[keep],
        cog=gcog[keep],
        vid=gv[keep],
        t=gt[keep],
        gap=np.minimum(ggap[keep], 32767).astype(np.int16),
        samples=samples,
        v_mmsi=static["mmsi"].to_numpy(),
        v_group=static["group"].to_numpy().astype(np.int8),
        v_length=static["length"].fill_null(np.nan).to_numpy().astype(np.float32),
        v_draft=static["draft"].fill_null(np.nan).to_numpy().astype(np.float32),
    )
    grp = static["group"].to_numpy()[gv[keep][samples]]
    print(
        f"{year}: {len(files)} files, {n_raw:,} raw rows -> {n_grid:,} grid pts -> "
        f"{len(samples):,} samples ({keep.sum():,} pts kept), vessels {len(static):,}, "
        f"by group {np.bincount(grp, minlength=5).tolist()}  [{time.time() - t0:.0f}s]",
        flush=True,
    )


if __name__ == "__main__":
    years = [int(y) for y in sys.argv[1:]] or sorted(int(p.name) for p in FILTERED.iterdir() if p.is_dir())
    for y in years:
        build_year(y)
