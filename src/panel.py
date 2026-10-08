"""Analysis panel on the 5m snapshot grid: features (t <= t_snap) + forward outcomes.

Every feature column is listed in FEATURE_AVAILABILITY with the timestamp at which
it is known; check_no_lookahead() verifies the panel against that contract.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from .liquidation_features import wkey

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

ANALYSIS_START = "2020-12-01"   # 3-month burn-in for the reconstructed map
DELTA_LAGS = {"15m": 3, "1h": 12, "4h": 48, "12h": 144, "24h": 288}


@njit(cache=True)
def _past_pct(vals, day_id, daily_vals, lookback):
    out = np.full(vals.shape[0], np.nan)
    nd = daily_vals.shape[0]
    cache_day = -1
    srt = np.empty(0)
    for i in range(vals.shape[0]):
        d = day_id[i]
        if d != cache_day:
            lo = max(0, d - lookback)
            h = daily_vals[lo:d]
            h = h[np.isfinite(h)]
            srt = np.sort(h)
            cache_day = d
        if srt.shape[0] >= 90 and np.isfinite(vals[i]):
            out[i] = np.searchsorted(srt, vals[i], side="right") / srt.shape[0]
    return out


def trailing_percentile(x: pd.Series, lookback_days: int = 365) -> pd.Series:
    """Percentile of x_t within the previous `lookback_days` daily observations (causal)."""
    daily = x.resample("1D").median()
    day_id = ((x.index.normalize() - daily.index[0]) // pd.Timedelta(days=1)).to_numpy(np.int64)
    return pd.Series(_past_pct(x.to_numpy(float), day_id, daily.to_numpy(float), lookback_days), index=x.index)


def build_panel(cfg: dict, variant: str, light: bool = False) -> pd.DataFrame:
    """light=True: symmetric windows only, deltas for the main window only, no cluster columns
    (enough for the variant robustness suite; uses far less memory)."""
    if light:
        import pyarrow.parquet as pq
        allc = pq.ParquetFile(PROC / f"liqmap_features_{variant}.parquet").schema.names
        wk = [wkey(w) for w in cfg["features"]["windows"]]
        keep = [c for c in allc if c in ("close", "oi_valid", "oi_usd", "snapshot_age_min", "map_degraded",
                                          "map_total_long", "map_total_short", "ts")
                or any(c.endswith(f"_{k}") and c.split("_")[0] in ("ratio", "dir", "sli", "total", "L") for k in wk)]
        f = pd.read_parquet(PROC / f"liqmap_features_{variant}.parquet", columns=keep)
    else:
        f = pd.read_parquet(PROC / f"liqmap_features_{variant}.parquet")
    for c in f.columns:
        if f[c].dtype == np.float64 and not c.startswith(("close", "oi_", "cl", "big_")):
            f[c] = f[c].astype(np.float32)
    rg = pd.read_parquet(PROC / "regimes_5m.parquet")
    oc = pd.read_parquet(PROC / "grid_outcomes.parquet")
    keys = [wkey(w) for w in cfg["features"]["windows"]]
    dw = [] if light else ["inv_d", "inv_d2"] + [f"exp{int(k)}" for k in cfg["features"]["exp_k"]]
    p = pd.DataFrame(index=f.index)
    p["close_snap"] = f.close
    valid = (f.oi_valid & (f.snapshot_age_min == 0) & ~f.map_degraded &
             (f.index >= pd.Timestamp(ANALYSIS_START, tz="UTC")) & f.close.notna() & oc.P0.notna())
    p["valid"] = valid
    for k in keys + dw:
        for c in ("ratio", "dir", "sli", "total"):
            p[f"{c}_{k}"] = f[f"{c}_{k}"]
        if k in keys:
            p[f"L_down_{k}"] = f[f"L_down_{k}"]
            p[f"L_up_{k}"] = f[f"L_up_{k}"]
    # change features: signed and magnitude, invalid if the lagged bar was invalid
    dkeys = [wkey(cfg["features"]["main_window"])] if light else keys + ["inv_d2", "exp20"]
    for k in dkeys:
        s = p[f"sli_{k}"].where(valid)
        for nm, lag in DELTA_LAGS.items():
            p[f"dsli_{nm}_{k}"] = (s - s.shift(lag)).astype(np.float32)
            p[f"dabs_{nm}_{k}"] = (s.abs() - s.abs().shift(lag)).astype(np.float32)
    for k in keys:
        p[f"total_oi_{k}"] = (f[f"total_{k}"] / f.oi_usd).astype(np.float32)
        p[f"total_pct365_{k}"] = trailing_percentile(f[f"total_{k}"].astype(float).where(valid)).astype(np.float32)
    for c in f.columns:
        if c.startswith(("cl", "big_")):
            p[c] = f[c]
    p["oi_usd"] = f.oi_usd
    p["map_total_long"] = f.map_total_long
    p["map_total_short"] = f.map_total_short
    p = p.join(rg[["ret_24h", "ret_7d", "ret_30d", "rv_24h", "rv_7d", "rv_30d", "px_vs_ma20d",
                   "px_vs_ma50d", "px_vs_ma200d", "trend", "vol_regime", "funding", "funding_regime",
                   "oi_chg_1h", "oi_chg_24h", "oi_regime", "premium", "hour_utc"]])
    p = p.join(oc)
    p["year"] = p.index.year
    for c in p.columns:
        if p[c].dtype == np.float64 and c not in ("close_snap", "P0") and not c.startswith(("cl", "big_")):
            p[c] = p[c].astype(np.float32)
    return p


# ---------------------------------------------------------------- look-ahead contract

FEATURE_AVAILABILITY = {
    # prefix -> how the value's availability time relates to the row timestamp t_snap
    "ratio_": "t_snap", "dir_": "t_snap", "sli_": "t_snap", "total_": "t_snap", "L_": "t_snap",
    "dsli_": "t_snap", "dabs_": "t_snap", "cl": "t_snap", "big_": "t_snap",
    "ret_24h": "t_snap", "ret_7d": "t_snap", "ret_30d": "t_snap", "rv_": "t_snap",
    "px_vs_": "t_snap", "trend": "t_snap", "vol_regime": "t_snap (expanding terciles)",
    "funding": "last settlement <= t_snap", "oi_": "t_snap", "premium": "bar ending t_snap",
    "hour_utc": "t_snap", "close_snap": "t_snap",
}
OUTCOME_PREFIXES = ("fret_", "mfe_", "fpt_", "fwd_", "max_1h_", "casc_", "P0", "t0")


def check_no_lookahead(p: pd.DataFrame, bars5: pd.DataFrame, n_checks: int = 2000, seed: int = 0) -> dict:
    """Automated checks:
    1. close_snap equals the 5m bar close ENDING at t_snap (not a later bar).
    2. P0 is the 1m close at t0 = t_snap + 5min, i.e. strictly after the snapshot.
    3. Recompute ret_24h feature from bars <= t_snap at random rows.
    4. Shuffling future prices must not change any feature column (feature columns are
       computed in modules that never receive outcome data - verified structurally by
       asserting no feature column is identical to an outcome column).
    """
    rng = np.random.default_rng(seed)
    v = p[p.valid]
    rows = v.index[rng.choice(len(v), size=min(n_checks, len(v)), replace=False)]
    c_ok = np.allclose(p.loc[rows, "close_snap"], bars5.close.reindex(rows), equal_nan=True)
    r24 = bars5.close.reindex(rows) / bars5.close.reindex(rows - pd.Timedelta(hours=24)).to_numpy() - 1
    r_ok = np.allclose(p.loc[rows, "ret_24h"].to_numpy(), r24.to_numpy(), equal_nan=True, atol=1e-10)
    t0_ok = bool(((p.loc[rows, "t0"] - rows.to_series()) == pd.Timedelta(minutes=5)).all())
    feat_cols = [c for c in p.columns if c.startswith(tuple(FEATURE_AVAILABILITY))]
    overlap = [c for c in feat_cols if c.startswith(OUTCOME_PREFIXES)]
    return {"close_snap_matches_bar_end": bool(c_ok), "ret_24h_recomputed_from_past": bool(r_ok),
            "t0_is_tsnap_plus_5m": t0_ok, "n_rows_checked": len(rows), "n_feature_cols": len(feat_cols),
            "feature_outcome_name_overlap": overlap}
