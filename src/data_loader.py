"""Download and load raw data.

Sources (all public, no API key needed):
  * Binance USD-M futures 1m klines        data.binance.vision  futures/um/monthly|daily/klines
  * Binance USD-M futures 5m metrics       data.binance.vision  futures/um/daily/metrics
      (open interest, long/short account ratios, taker volume ratio)
  * Binance funding rate history           data.binance.vision  futures/um/monthly/fundingRate
  * Binance premium index 5m klines        data.binance.vision  futures/um/monthly/premiumIndexKlines (basis proxy)

Raw zip files are stored untouched under data/raw/<dataset>/ and are never
overwritten once downloaded. Processed, schema-normalised UTC parquet files go
to data/processed/.

Optional external liquidation maps (e.g. a paid Coinglass export) can be dropped
into data/raw/external_liquidation_map/ as CSV with columns
(timestamp, price, amount, side); see load_external_liquidation_map().
API keys, if ever needed, are read from .env via python-dotenv - never hard-coded.
"""
from __future__ import annotations

import io
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    return {k: v for k, v in os.environ.items() if k.endswith("_API_KEY")}


# ----------------------------------------------------------------------------- download

def _months(start: str, end: str):
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def _days(start: str, end: str):
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="D")]


def _fetch(url: str, dest: Path, session: requests.Session) -> str:
    if dest.exists() and dest.stat().st_size > 0:
        return "cached"
    r = session.get(url, timeout=60)
    if r.status_code == 404:
        return "missing"
    r.raise_for_status()
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(r.content)
    tmp.rename(dest)
    return "downloaded"


def build_download_list(cfg: dict) -> list[tuple[str, Path]]:
    d = cfg["data"]
    sym, base = d["symbol"], d["base_url"]
    end = pd.Timestamp(d["end_date"])
    last_full_month = (end + pd.Timedelta(days=1)).to_period("M") - 1
    jobs = []
    # 1m klines: monthly files up to last complete month, daily files after it
    for m in _months(d["klines_start"], str(last_full_month)):
        jobs.append((f"{base}/monthly/klines/{sym}/1m/{sym}-1m-{m}.zip",
                     RAW / "klines_1m" / f"{sym}-1m-{m}.zip"))
    for day in _days(str((last_full_month + 1).start_time.date()), d["end_date"]):
        jobs.append((f"{base}/daily/klines/{sym}/1m/{sym}-1m-{day}.zip",
                     RAW / "klines_1m" / f"{sym}-1m-{day}.zip"))
    # metrics: daily only
    for day in _days(d["metrics_start"], d["end_date"]):
        jobs.append((f"{base}/daily/metrics/{sym}/{sym}-metrics-{day}.zip",
                     RAW / "metrics" / f"{sym}-metrics-{day}.zip"))
    # funding + premium index: monthly, then daily premium for the current month
    for m in _months(d["klines_start"], str(last_full_month)):
        jobs.append((f"{base}/monthly/fundingRate/{sym}/{sym}-fundingRate-{m}.zip",
                     RAW / "funding" / f"{sym}-fundingRate-{m}.zip"))
        jobs.append((f"{base}/monthly/premiumIndexKlines/{sym}/5m/{sym}-5m-{m}.zip",
                     RAW / "premium_5m" / f"{sym}-5m-{m}.zip"))
    for day in _days(str((last_full_month + 1).start_time.date()), d["end_date"]):
        jobs.append((f"{base}/daily/premiumIndexKlines/{sym}/5m/{sym}-5m-{day}.zip",
                     RAW / "premium_5m" / f"{sym}-5m-{day}.zip"))
    return jobs


def download_all(cfg: dict) -> pd.DataFrame:
    jobs = build_download_list(cfg)
    for _, dest in jobs:
        dest.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    with ThreadPoolExecutor(cfg["data"]["download_workers"]) as ex:
        status = list(ex.map(lambda j: _fetch(j[0], j[1], session), jobs))
    log = pd.DataFrame({"url": [j[0] for j in jobs], "file": [str(j[1]) for j in jobs],
                        "status": status})
    log.to_csv(RAW / "download_log.csv", index=False)
    # funding after the last monthly file comes from the REST endpoint (public)
    _download_recent_funding(cfg, session)
    return log


def _download_recent_funding(cfg, session):
    d = cfg["data"]
    end = pd.Timestamp(d["end_date"]) + pd.Timedelta(days=1)
    start = (end.to_period("M")).start_time - pd.Timedelta(days=3)
    dest = RAW / "funding" / f"rest_fundingRate_{start.date()}_{end.date()}.json"
    if dest.exists():
        return
    r = session.get(f"{d['rest_url']}/fapi/v1/fundingRate",
                    params={"symbol": d["symbol"], "startTime": int(start.timestamp() * 1000),
                            "endTime": int(end.timestamp() * 1000), "limit": 1000}, timeout=30)
    r.raise_for_status()
    dest.write_text(r.text)


# ----------------------------------------------------------------------------- parsing

def _read_zip_csv(path: Path, names=None) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0])
    first = raw[:200].decode(errors="ignore").split("\n")[0]
    has_header = any(c.isalpha() for c in first.split(",")[0])
    if names is not None:
        return pd.read_csv(io.BytesIO(raw), header=0 if has_header else None, names=names)
    return pd.read_csv(io.BytesIO(raw))


def _to_utc_ms(s: pd.Series) -> pd.Series:
    s = s.astype("int64")
    # Binance switched some datasets to microseconds in 2025; normalise
    s = np.where(s > 10**14, s // 1000, s)
    return pd.to_datetime(s, unit="ms", utc=True)


def build_price_1m() -> pd.DataFrame:
    files = sorted((RAW / "klines_1m").glob("*.zip"))
    df = pd.concat([_read_zip_csv(f, KLINE_COLS) for f in files], ignore_index=True)
    df["ts"] = _to_utc_ms(df["open_time"])
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume"]:
        df[c] = df[c].astype(float)
    df = df[["ts", "open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume"]]
    return df.sort_values("ts").reset_index(drop=True)


def build_metrics_5m() -> pd.DataFrame:
    files = sorted((RAW / "metrics").glob("*.zip"))
    df = pd.concat([_read_zip_csv(f) for f in files], ignore_index=True)
    df["ts"] = pd.to_datetime(df["create_time"], utc=True)
    df = df.rename(columns={"sum_open_interest": "oi_btc", "sum_open_interest_value": "oi_usd",
                            "count_long_short_ratio": "acct_ls_ratio",
                            "sum_toptrader_long_short_ratio": "top_pos_ls_ratio",
                            "count_toptrader_long_short_ratio": "top_acct_ls_ratio",
                            "sum_taker_long_short_vol_ratio": "taker_ls_ratio"})
    cols = ["ts", "oi_btc", "oi_usd", "acct_ls_ratio", "top_pos_ls_ratio", "top_acct_ls_ratio",
            "taker_ls_ratio"]
    df = df[cols]
    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("ts").reset_index(drop=True)


def build_funding() -> pd.DataFrame:
    parts = []
    for f in sorted((RAW / "funding").glob("*.zip")):
        d = _read_zip_csv(f)
        d.columns = [c.strip().lower() for c in d.columns]
        tcol = "calc_time" if "calc_time" in d.columns else d.columns[0]
        rcol = [c for c in d.columns if "rate" in c][0]
        parts.append(pd.DataFrame({"ts": _to_utc_ms(d[tcol]), "funding": d[rcol].astype(float)}))
    for f in sorted((RAW / "funding").glob("rest_*.json")):
        d = pd.read_json(f)
        if len(d):
            parts.append(pd.DataFrame({"ts": _to_utc_ms(d["fundingTime"]),
                                       "funding": d["fundingRate"].astype(float)}))
    df = pd.concat(parts, ignore_index=True)
    # calc_time can carry a few ms of jitter - snap to the minute
    df["ts"] = df["ts"].dt.round("min")
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def build_premium_5m() -> pd.DataFrame:
    files = sorted((RAW / "premium_5m").glob("*.zip"))
    df = pd.concat([_read_zip_csv(f, KLINE_COLS) for f in files], ignore_index=True)
    df["ts"] = _to_utc_ms(df["open_time"])
    df["premium"] = df["close"].astype(float)
    return df[["ts", "premium"]].sort_values("ts").reset_index(drop=True)


def build_processed():
    PROC.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, fn in [("price_1m", build_price_1m), ("metrics_5m", build_metrics_5m),
                     ("funding", build_funding), ("premium_5m", build_premium_5m)]:
        df = fn()
        df.to_parquet(PROC / f"{name}_raw_parsed.parquet", index=False)
        out[name] = df
    return out


def load_external_liquidation_map(path: Path | None = None) -> pd.DataFrame | None:
    """Optional: bucketed liquidation-map export (e.g. Coinglass heatmap model).

    Expected columns (names auto-mapped, case-insensitive):
      timestamp, price (liquidation price level), amount (USD), side (long/short)
    Returns None if nothing is provided. Not used by the baseline pipeline because
    no such export was supplied for this study.
    """
    path = path or RAW / "external_liquidation_map"
    files = sorted(Path(path).glob("*.csv")) if Path(path).exists() else []
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    mapping = {}
    for c in df.columns:
        lc = c.lower()
        if "time" in lc:
            mapping[c] = "timestamp"
        elif "price" in lc:
            mapping[c] = "price"
        elif any(k in lc for k in ("amount", "value", "intensity", "usd")):
            mapping[c] = "amount"
        elif any(k in lc for k in ("side", "type", "direction")):
            mapping[c] = "side"
    df = df.rename(columns=mapping)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df
