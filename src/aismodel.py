"""Shared pieces: track loading, featurization, polar bins, position encodings, model."""
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
TRACKS = ROOT / "data" / "tracks"

KM_LAT = 110.574
KM_LON = 111.320
KN_TO_KM_PER_STEP = 1.852 / 6.0  # 1 knot for 10 minutes, in km
MAX_HIST = 12
N_FUTURE = 6
GROUPS = ["cargo", "tanker", "passenger", "service", "military_law"]

# ---- polar bins: turn angle relative to current COG x distance travelled in 10 min ----
N_ANG = 36
ANG_CENTERS = torch.arange(N_ANG) * (2 * math.pi / N_ANG)  # 0 = straight ahead, + = to the left
SPEED_CENTERS_KN = torch.tensor([float(v) for v in range(31)] + [35.0])  # 32 magnitude bins
N_MAG = len(SPEED_CENTERS_KN)
N_CLS = N_ANG * N_MAG
# soft-label width in km (x/y space), grows with distance travelled
SIGMA0_KM, SIGMA_SLOPE = 0.15, 0.05


def bin_centers(device):
    """(N_CLS, 2) forward/left km of each class center; class = ang * N_MAG + mag."""
    r = (SPEED_CENTERS_KN * KN_TO_KM_PER_STEP).to(device)
    a = ANG_CENTERS.to(device)
    f = torch.cos(a)[:, None] * r[None, :]
    l = torch.sin(a)[:, None] * r[None, :]
    return torch.stack([f.reshape(-1), l.reshape(-1)], -1)


def soft_labels(fl, centers):
    """fl: (B,2) true forward/left km. Gaussian in x/y space evaluated at bin centers."""
    r = fl.norm(dim=-1, keepdim=True)
    sigma = SIGMA0_KM + SIGMA_SLOPE * r
    d2 = ((fl[:, None, :] - centers[None]) ** 2).sum(-1)
    logits = -d2 / (2 * sigma**2)
    return torch.softmax(logits, -1)


# ---- geometry helpers (degrees, km, COG clockwise from north) ----
def disp_km(lat0, lon0, lat1, lon1):
    mlat = torch.deg2rad(0.5 * (lat0 + lat1))
    return (lon1 - lon0) * KM_LON * torch.cos(mlat), (lat1 - lat0) * KM_LAT


def to_frame(dx, dy, cog_deg):
    th = torch.deg2rad(cog_deg)
    s, c = torch.sin(th), torch.cos(th)
    return dx * s + dy * c, -dx * c + dy * s


def from_frame(f, l, cog_deg):
    th = torch.deg2rad(cog_deg)
    s, c = torch.sin(th), torch.cos(th)
    return f * s - l * c, f * c + l * s


def move(lat, lon, dx, dy):
    nlat = lat + dy / KM_LAT
    mlat = torch.deg2rad(0.5 * (lat + nlat))
    return nlat, lon + dx / (KM_LON * torch.cos(mlat))


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = torch.deg2rad(lat1.double()), torch.deg2rad(lat2.double())
    dp = p2 - p1
    dl = torch.deg2rad((lon2 - lon1).double())
    a = torch.sin(dp / 2) ** 2 + torch.cos(p1) * torch.cos(p2) * torch.sin(dl / 2) ** 2
    return (2 * 6371.0088 * torch.asin(torch.sqrt(a.clamp(0, 1)))).float()


# ---- data ----
class Tracks:
    """All years' resampled points on the GPU. Positions are float64 so a 60-minute
    rollout accumulates no rounding error; everything else is float32."""

    def __init__(self, years, device):
        cols = {k: [] for k in ("lat", "lon", "sog", "cog", "vid", "t", "gap")}
        samples, v_group, v_len, v_draft, sample_year = [], [], [], [], []
        p_off = v_off = 0
        for y in years:
            z = np.load(TRACKS / f"{y}.npz")
            for k in cols:
                a = z[k]
                cols[k].append(a + v_off if k == "vid" else a)
            samples.append(z["samples"] + p_off)
            sample_year.append(np.full(len(z["samples"]), y, dtype=np.int16))
            v_group.append(z["v_group"])
            v_len.append(z["v_length"])
            v_draft.append(z["v_draft"])
            p_off += len(z["lat"])
            v_off += len(z["v_group"])
        cat = np.concatenate
        self.lat = torch.from_numpy(cat(cols["lat"])).to(device)
        self.lon = torch.from_numpy(cat(cols["lon"])).to(device)
        self.sog = torch.from_numpy(cat(cols["sog"]).astype(np.float32)).to(device)
        self.cog = torch.from_numpy(cat(cols["cog"]).astype(np.float32)).to(device)
        self.vid = torch.from_numpy(cat(cols["vid"]).astype(np.int64)).to(device)
        self.t = cat(cols["t"])
        self.gap = torch.from_numpy(cat(cols["gap"]).astype(np.int32)).to(device)
        samples_np = cat(samples)
        self.samples = torch.from_numpy(samples_np).to(device)
        self.sample_day = (self.t[samples_np].astype("datetime64[s]").astype("datetime64[D]")
                           - self.t[samples_np].astype("datetime64[s]").astype("datetime64[M]")).astype(np.int64) + 1
        self.sample_year = cat(sample_year)
        self.v_group = torch.from_numpy(cat(v_group).astype(np.int64)).to(device)
        self.v_len = torch.from_numpy(cat(v_len)).to(device)
        self.v_draft = torch.from_numpy(cat(v_draft)).to(device)
        self.device = device

    def window(self, s, H):
        """Points s-H .. s+N_FUTURE for sample point indices s. Returns dict of (B, H+1+N_FUTURE)."""
        off = torch.arange(-H, N_FUTURE + 1, device=self.device)
        ix = s[:, None] + off[None]
        v = self.vid[s]
        return dict(
            lat=self.lat[ix], lon=self.lon[ix], sog=self.sog[ix], cog=self.cog[ix],
            group=self.v_group[v], length=self.v_len[v], draft=self.v_draft[v],
        )


def featurize(lat, lon, sog, cog, group, length, draft):
    """History tensors are (B, H+1) ending at the current point. Returns
    (seq (B,H,6) or None, flat (B,F), cog_cur, lat_cur, lon_cur)."""
    H = lat.shape[1] - 1
    cog_cur = cog[:, -1]
    flat = [
        (sog[:, -1] / 20.0)[:, None],
        torch.sin(torch.deg2rad(cog_cur))[:, None],
        torch.cos(torch.deg2rad(cog_cur))[:, None],
    ]
    ln_ok = torch.isfinite(length)
    dr_ok = torch.isfinite(draft)
    flat += [
        torch.where(ln_ok, torch.log1p(length.clamp(min=0)) / 6.0, 0.0)[:, None],
        ln_ok.float()[:, None],
        torch.where(dr_ok, draft / 15.0, 0.0)[:, None],
        dr_ok.float()[:, None],
    ]
    seq = None
    if H > 0:
        dx, dy = disp_km(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:])
        f, l = to_frame(dx.float(), dy.float(), cog_cur[:, None])
        c = cog[:, 1:]
        c_ok = torch.isfinite(c)
        rel = torch.deg2rad(torch.where(c_ok, c - cog_cur[:, None], 0.0))
        seq = torch.stack(
            [f / 3.0, l / 3.0, torch.nan_to_num(sog[:, 1:], 0.0) / 20.0,
             torch.where(c_ok, torch.sin(rel), 0.0), torch.where(c_ok, torch.cos(rel), 0.0), c_ok.float()],
            -1,
        )
    return seq, torch.cat(flat, -1), group, cog_cur, lat[:, -1], lon[:, -1]


# ---- position encodings ----
LON0, LAT0 = -73.5, 35.0
COS0 = math.cos(math.radians(LAT0))


def global_xy(lat, lon):
    """Equirectangular km around (35N, 73.5W); only used for position encoding."""
    return ((lon - LON0) * KM_LON * COS0).float(), ((lat - LAT0) * KM_LAT).float()


class FourierPos(nn.Module):
    def __init__(self, n_freq=12, max_wavelength_km=2500.0):
        super().__init__()
        wl = max_wavelength_km * 0.5 ** torch.arange(n_freq)  # 2500 km .. ~1.2 km
        self.register_buffer("w", 2 * math.pi / wl)
        self.out_dim = 2 + 4 * n_freq

    def forward(self, lat, lon):
        x, y = global_xy(lat, lon)
        ax, ay = x[:, None] * self.w, y[:, None] * self.w
        return torch.cat([x[:, None] / 1000, y[:, None] / 1000, ax.sin(), ax.cos(), ay.sin(), ay.cos()], -1)


class HashGridPos(nn.Module):
    """Multi-resolution learned grid with bilinear interpolation (Instant-NGP style);
    levels with more cells than `table` are hashed."""

    X0, Y0, W, Hh = -800.0, -1250.0, 1600.0, 2500.0

    def __init__(self, cell_km=(64.0, 16.0, 4.0, 1.0), dim=8, table=2**20):
        super().__init__()
        self.cell_km = cell_km
        self.tables = nn.ModuleList()
        self.dense = []
        self.ny = []
        for c in cell_km:
            nx, ny = int(self.W / c) + 2, int(self.Hh / c) + 2
            n = nx * ny
            dense = n <= table
            emb = nn.Embedding(n if dense else table, dim)
            nn.init.uniform_(emb.weight, -1e-4, 1e-4)
            self.tables.append(emb)
            self.dense.append(dense)
            self.ny.append(ny)
        self.table = table
        self.out_dim = 2 + dim * len(cell_km)

    def forward(self, lat, lon):
        x, y = global_xy(lat, lon)
        outs = [x[:, None] / 1000, y[:, None] / 1000]
        for emb, c, dense, ny in zip(self.tables, self.cell_km, self.dense, self.ny):
            gx = ((x - self.X0) / c).clamp(0, self.W / c)
            gy = ((y - self.Y0) / c).clamp(0, self.Hh / c)
            ix, iy = gx.floor().long(), gy.floor().long()
            fx, fy = (gx - ix)[:, None], (gy - iy)[:, None]
            acc = 0
            for dx_, dy_, wgt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
                cx, cy = ix + dx_, iy + dy_
                if dense:
                    k = cx * ny + cy
                else:
                    k = ((cx * 73856093) ^ (cy * 19349663)) % self.table
                acc = acc + wgt * emb(k)
            outs.append(acc)
        return torch.cat(outs, -1)


class Net(nn.Module):
    """drift_dim > 0 adds a learned vector per training week. forward() takes week ids during
    training; without them it uses `self.test_vec`, which the caller sets before evaluation."""

    def __init__(self, hist, posenc, d=512, d_gru=128, n_weeks=0, drift_dim=0):
        super().__init__()
        self.hist = hist
        self.drift = nn.Embedding(n_weeks, drift_dim) if drift_dim > 0 else None
        if self.drift is not None:
            nn.init.normal_(self.drift.weight, std=0.01)
            self.register_buffer("test_vec", torch.zeros(drift_dim))
        self.pos = FourierPos() if posenc == "fourier" else HashGridPos()
        self.group_emb = nn.Embedding(len(GROUPS), 8)
        self.gru = nn.GRU(6, d_gru, batch_first=True) if hist > 0 else None
        in_dim = (d_gru if hist > 0 else 0) + 7 + self.pos.out_dim + 8 + drift_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d), nn.GELU(),
            nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(),
            nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(),
            nn.Linear(d, N_CLS),
        )

    def forward(self, seq, flat, group, lat, lon, week=None):
        parts = [flat, self.pos(lat, lon), self.group_emb(group)]
        if self.drift is not None:
            parts.append(self.drift(week) if week is not None else self.test_vec.expand(len(flat), -1))
        if self.gru is not None:
            _, h = self.gru(seq)
            parts.insert(0, h[-1])
        return self.mlp(torch.cat(parts, -1))
