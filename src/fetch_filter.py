"""Download May AIS files from Marine Cadastre, keep only East Coast + selected
vessel types, and write one small parquet per source file. Raw downloads are
deleted as soon as they are filtered, so disk usage stays at a few GB.

Usage: python src/fetch_filter.py [--workers 4] [--years 2015 2016 ...]
"""
import argparse
import io
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "filtered"
BASE = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler"

# Vessel groups: 0 cargo, 1 tanker, 2 passenger, 3 service, 4 military/law enforcement
GROUP_NAMES = ["cargo", "tanker", "passenger", "service", "military_law"]
SERVICE_CODES = {31, 32, 33, 34, 50, 51, 52, 53, 54, 58}
MILITARY_CODES = {35, 55}
# 2015-2017 files mostly use Marine Cadastre's 1001-1025 codes (VesselTypeCodes2018 crosswalk)
MC_CODES = {
    1003: 0, 1004: 0, 1016: 0,  # freight barge / ship, public freight
    1017: 1, 1024: 1,  # public tankship, tank ship
    1012: 2, 1013: 2, 1014: 2, 1015: 2,  # passenger (+ barges)
    1010: 3, 1011: 3, 1023: 3, 1025: 3,  # offshore supply, oil recovery, tank barge (tug-tow), towing
    1018: 4,  # public vessel, unclassified
}

# 2009-2014 geodatabases store vessel type as text
TEXT_GROUPS = {
    "cargo ships": 0,
    "tanker(s)": 1,
    "passenger ships": 2,
    "tugs": 3,
    "towing": 3,
    "towing and length of the tow exceeds 200 m or breadth exceeds 25 m": 3,
    "engaged in dredging or underwater operations": 3,
    "engaged in diving operations": 3,
    "pilot vessel": 3,
    "search and rescue vessels": 3,
    "port tender": 3,
    "vessels with anti-pollution facilities or equipment": 3,
    "medical transports (as defined in the 1949 geneva conventions and additional protocols)": 3,
    "engaged in military operations": 4,
    "law enforcement vessels": 4,
}


def code_to_group(code: np.ndarray) -> np.ndarray:
    g = np.full(code.shape, -1, dtype=np.int8)
    g[(code >= 70) & (code <= 79)] = 0
    g[(code >= 80) & (code <= 89)] = 1
    g[(code >= 60) & (code <= 69)] = 2
    g[np.isin(code, list(SERVICE_CODES))] = 3
    g[np.isin(code, list(MILITARY_CODES))] = 4
    for c, grp in MC_CODES.items():
        g[code == c] = grp
    return g


def in_region(lat: pl.Expr, lon: pl.Expr) -> pl.Expr:
    """US East Coast box, minus the Great Lakes / St. Lawrence and the Gulf side of Florida."""
    box = (lat > 24.0) & (lat < 46.0) & (lon > -82.0) & (lon < -65.0)
    great_lakes = (lat > 41.3) & (lon < -74.8)
    st_lawrence = (lat > 44.3) & (lon < -70.5)
    gulf = (lat < 29.0) & (lon < -80.9)
    return box & ~great_lakes & ~st_lawrence & ~gulf


SCHEMA_COLS = ["mmsi", "t", "lat", "lon", "sog", "cog", "heading", "group", "vtype", "length", "draft"]


def finalize(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.filter(in_region(pl.col("lat"), pl.col("lon")))
        .filter(pl.col("group") >= 0)
        .filter((pl.col("mmsi") >= 200_000_000) & (pl.col("mmsi") < 800_000_000))
        .select(
            pl.col("mmsi").cast(pl.Int64),
            pl.col("t").cast(pl.Int64),
            pl.col("lat").cast(pl.Float64),
            pl.col("lon").cast(pl.Float64),
            pl.col("sog").cast(pl.Float32),
            pl.col("cog").cast(pl.Float32),
            pl.col("heading").cast(pl.Float32),
            pl.col("group").cast(pl.Int8),
            pl.col("vtype").cast(pl.Int16),
            pl.col("length").cast(pl.Float32),
            pl.col("draft").cast(pl.Float32),
        )
    )


USE_COLS = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "Heading", "VesselType", "Length", "Draft"]


def process_daily_csv(zip_path: Path) -> pl.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        data = z.read(name)
    try:
        df = pl.read_csv(io.BytesIO(data), infer_schema_length=0, columns=USE_COLS)
    except pl.exceptions.ComputeError:
        # some files have line breaks inside quoted vessel names
        import pyarrow.csv as pacsv

        header = data[: data.index(b"\n")].decode().strip().split(",")
        tab = pacsv.read_csv(
            io.BytesIO(data),
            parse_options=pacsv.ParseOptions(newlines_in_values=True, invalid_row_handler=lambda row: "skip"),
            convert_options=pacsv.ConvertOptions(
                column_types={c: "string" for c in header}, include_columns=USE_COLS
            ),
        )
        df = pl.from_arrow(tab)
    del data
    f = lambda c: pl.col(c).cast(pl.Float64, strict=False)
    df = df.select(
        pl.col("MMSI").cast(pl.Int64, strict=False).alias("mmsi"),
        pl.col("BaseDateTime").str.to_datetime(strict=False).dt.epoch("s").alias("t"),
        f("LAT").alias("lat"),
        f("LON").alias("lon"),
        f("SOG").alias("sog"),
        f("COG").alias("cog"),
        f("Heading").alias("heading"),
        pl.col("VesselType").cast(pl.Int32, strict=False).fill_null(-1).alias("vtype"),
        f("Length").alias("length"),
        f("Draft").alias("draft"),
    ).drop_nulls(["mmsi", "t", "lat", "lon"])
    groups = code_to_group(df["vtype"].to_numpy())
    return finalize(df.with_columns(pl.Series("group", groups)))


def process_zone_gdb(gdb_path: Path) -> pl.DataFrame:
    import pyogrio

    names = [n for n, _ in pyogrio.list_layers(gdb_path)]
    layer = lambda suffix: next(n for n in names if n == suffix or n.endswith("_" + suffix))
    meta, tab = pyogrio.read_arrow(
        gdb_path, layer=layer("Broadcast"), columns=["SOG", "COG", "Heading", "BaseDateTime", "VoyageID", "MMSI"]
    )
    geom = tab[meta["geometry_name"]].combine_chunks()
    off = np.frombuffer(geom.buffers()[1], dtype=np.int32)[geom.offset : geom.offset + len(geom) + 1]
    buf = np.frombuffer(geom.buffers()[2], dtype=np.uint8)
    if not np.all(np.diff(off) == 21):
        raise ValueError("unexpected WKB layout")
    start = off[:-1]
    idx = start[:, None] + np.arange(5, 21)[None, :]
    xy = buf[idx].copy().view("<f8").reshape(-1, 2)  # little-endian 2D points
    b = pl.from_arrow(tab.drop([meta["geometry_name"]]))
    b = b.with_columns(pl.Series("lon", xy[:, 0]), pl.Series("lat", xy[:, 1]))

    _, vtab = pyogrio.read_arrow(gdb_path, layer=layer("Vessel"), read_geometry=False)
    v = pl.from_arrow(vtab).select(
        pl.col("MMSI").cast(pl.Int64),
        pl.col("VesselType").cast(pl.Utf8).str.to_lowercase().str.strip_chars().alias("vt"),
        pl.col("Length").cast(pl.Float64, strict=False).alias("length"),
    ).unique("MMSI")
    v = v.with_columns(pl.col("vt").replace_strict(TEXT_GROUPS, default=-1, return_dtype=pl.Int8).alias("group"))

    _, voytab = pyogrio.read_arrow(gdb_path, layer=layer("Voyage"), read_geometry=False)
    voy = pl.from_arrow(voytab).select(
        pl.col("VoyageID").cast(pl.Int64),
        pl.col("MMSI").cast(pl.Int64),
        (pl.col("Draught").cast(pl.Float64, strict=False) / 10.0).alias("draft"),
    ).unique(["VoyageID", "MMSI"])

    b = b.select(
        pl.col("MMSI").cast(pl.Int64).alias("mmsi"),
        pl.col("VoyageID").cast(pl.Int64),
        pl.col("BaseDateTime").cast(pl.Datetime("ms")).dt.epoch("s").alias("t"),
        "lat",
        "lon",
        pl.col("SOG").cast(pl.Float64).alias("sog"),
        pl.col("COG").cast(pl.Float64).alias("cog"),
        pl.col("Heading").cast(pl.Float64).alias("heading"),
    )
    b = b.join(v.rename({"MMSI": "mmsi"}), on="mmsi", how="inner")
    b = b.join(voy.rename({"MMSI": "mmsi"}), on=["VoyageID", "mmsi"], how="left")
    b = b.with_columns(pl.lit(-1).alias("vtype"))
    return finalize(b)


def job(url: str) -> str:
    fname = url.rsplit("/", 1)[1]
    stem = fname.replace(".gdb.zip", "").replace(".zip", "")
    year = stem.split("_")[1]
    out = OUT / year / f"{stem}.parquet"
    if out.exists():
        return f"skip {stem}"
    work = RAW / stem
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    zpath = work / fname
    t0 = time.time()
    for attempt in range(5):
        r = subprocess.run(
            ["curl", "-sS", "-L", "--fail", "-A", "Mozilla/5.0", "--retry", "3", "-o", str(zpath), url],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            break
        time.sleep(10 * (attempt + 1))
    else:
        shutil.rmtree(work, ignore_errors=True)
        return f"FAIL download {stem}: {r.stderr.strip()}"
    t1 = time.time()
    mb = zpath.stat().st_size / 1e6
    try:
        if stem.startswith("AIS_"):
            df = process_daily_csv(zpath)
        else:
            with zipfile.ZipFile(zpath) as z:
                z.extractall(work)
            zpath.unlink()
            gdb = next(work.glob("*.gdb"))
            df = process_zone_gdb(gdb)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        df.write_parquet(tmp)
        tmp.replace(out)
    except Exception as e:
        return f"FAIL process {stem}: {e!r}"
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return f"ok {stem}: {mb:.0f} MB in {t1 - t0:.0f}s dl, {time.time() - t1:.0f}s proc, {len(df):,} rows kept"


def all_urls():
    urls = []
    for y in range(2015, 2024):
        urls += [f"{BASE}/{y}/AIS_{y}_05_{d:02d}.zip" for d in range(1, 32)]
    for z in (17, 18, 19):
        urls.append(f"{BASE}/2009/05_May_2009/Zone{z}_2009_05.zip")
        urls.append(f"{BASE}/2010/05_May_2010/Zone{z}_2010_05.zip")
        urls += [f"{BASE}/{y}/05/Zone{z}_{y}_05.gdb.zip" for y in (2011, 2012, 2013)]
        urls.append(f"{BASE}/2014/05/Zone{z}_2014_05.zip")
    return urls


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--years", type=int, nargs="*")
    ap.add_argument("--only", nargs="*", help="substring filter on file names")
    args = ap.parse_args()
    urls = all_urls()
    if args.years:
        urls = [u for u in urls if any(f"_{y}_" in u for y in args.years)]
    if args.only:
        urls = [u for u in urls if any(s in u for s in args.only)]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(job, u) for u in urls]
        for i, f in enumerate(as_completed(futs), 1):
            print(f"[{i}/{len(urls)} {time.time() - t0:.0f}s] {f.result()}", flush=True)
