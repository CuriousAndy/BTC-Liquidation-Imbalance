"""Build event tables (one row per event) with direction-adjusted outcomes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import event_detection as ed
from .liquidation_features import wkey
from .price_outcomes import fp_outcome

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

H_LABEL = {15: "15m", 30: "30m", 60: "1h", 120: "2h", 240: "4h", 480: "8h", 720: "12h",
           1440: "24h", 2880: "48h", 4320: "72h", 10080: "7d"}
TH_TAGS = {0.005: "0p5", 0.01: "1", 0.02: "2", 0.03: "3", 0.05: "5", 0.075: "7p5", 0.10: "10"}

REGIME_COLS = ["ret_24h", "ret_7d", "ret_30d", "rv_24h", "rv_7d", "rv_30d", "px_vs_ma20d", "px_vs_ma50d",
               "px_vs_ma200d", "trend", "vol_regime", "funding", "funding_regime", "oi_chg_1h", "oi_chg_24h",
               "oi_regime", "premium", "hour_utc", "year", "oi_usd"]
OUTCOME_COLS_PREFIX = ("fret_", "mfe_", "fpt_", "fwd_", "max_1h_", "casc_")


def spec_columns(cols: list[str], k: str) -> list[str]:
    own = [c for c in cols if c.endswith(f"_{k}") and c.split("_")[0] in
           ("ratio", "dir", "sli", "total", "L", "dsli", "dabs")]
    own += [c for c in cols if c in (f"L_down_{k}", f"L_up_{k}", f"total_oi_{k}", f"total_pct365_{k}")]
    return sorted(set(own))


def load_panel(variant: str, specs: list[str]) -> pd.DataFrame:
    import pyarrow.parquet as pq
    allcols = pq.ParquetFile(PROC / f"panel_{variant}.parquet").schema.names
    need = {"valid", "close_snap", "P0", "t0"} | set(REGIME_COLS)
    need |= {c for c in allcols if c.startswith(OUTCOME_COLS_PREFIX)}
    need |= {c for c in allcols if c.startswith(("cl", "big_")) and c != "close_snap"}
    for k in specs:
        need |= set(spec_columns(allcols, k))
    need = [c for c in allcols if c in need]
    p = pd.read_parquet(PROC / f"panel_{variant}.parquet", columns=need)
    return p


def bucket_labels(edges):
    labs = []
    for a, b in zip(edges[:-1], edges[1:]):
        labs.append(f">{a:g}" if not np.isfinite(b) else f"{a:g}-{b:g}")
    return labs


def assign_bucket(r: pd.Series, edges) -> pd.Series:
    return pd.cut(r, bins=edges, right=False, labels=bucket_labels(edges))


# ----------------------------------------------------------------------------- event sets

def event_index(P: pd.DataFrame, k: str, method: str, cond: pd.Series | None = None,
                enter: float = 2.5, cooldown_h: float = 24) -> pd.DatetimeIndex:
    r = P[f"ratio_{k}"].astype(float)
    v = P["valid"] & r.notna()
    if method == "A":
        return ed.crossing_events(r.where(v, 0.0), v, enter, 0.8 * enter)
    if method == "B":
        c = v if cond is None else (v & cond)
        return ed.cooldown_events(c, cooldown_h)
    raise ValueError(method)


def episode_table(P: pd.DataFrame, k: str, enter: float = 2.5) -> pd.DataFrame:
    r = P[f"ratio_{k}"].astype(float)
    v = P["valid"] & r.notna()
    return ed.episodes(r, P[f"dir_{k}"], v, enter, P[f"L_up_{k}"], P[f"L_down_{k}"]) if f"L_up_{k}" in P else \
        ed.episodes(r, P[f"dir_{k}"], v, enter, P[f"total_{k}"], P[f"total_{k}"])


def make_events(P: pd.DataFrame, idx, k: str, cfg: dict, bench: dict | None = None,
                label: str = "") -> pd.DataFrame:
    """Event rows at t_snap in idx with direction-adjusted outcomes."""
    idx = pd.DatetimeIndex(idx)
    cols = [c for c in P.columns if c.endswith(f"_{k}")] + REGIME_COLS + ["P0", "t0"] + \
           [c for c in P.columns if c.startswith(OUTCOME_COLS_PREFIX)] + \
           [c for c in P.columns if c.startswith(("cl", "big_")) and c != "close_snap"] + ["close_snap"]
    cols = list(dict.fromkeys(c for c in cols if c in P.columns))
    E = P.loc[idx, cols].copy()
    E.columns = [c[: -len(k) - 1] if c.endswith(f"_{k}") else c for c in E.columns]
    E = E[E["dir"] != 0]
    d = E["dir"].astype(float)
    dn = d.to_numpy()
    N = {"spec": k, "event_set": label, "large_dir": np.where(dn > 0, "UP", "DOWN"),
         "abs_sli": E["sli"].abs(),
         "bucket": assign_bucket(E["ratio"].astype(float), cfg["ratio_buckets"]).astype(str)}
    for h in cfg["outcomes"]["horizons_min"]:
        N[f"dar_{h}"] = E[f"fret_{h}"].astype(float) * d
        if bench is not None:
            N[f"bdar_{h}"] = d * bench["mu"][h]
    for h in cfg["outcomes"]["mfe_horizons_min"]:
        up, dnn = E[f"mfe_up_{h}"].astype(float), E[f"mfe_dn_{h}"].astype(float)
        N[f"mfe_large_{h}"] = np.where(dn > 0, up, -dnn)
        N[f"mfe_small_{h}"] = np.where(dn > 0, -dnn, up)
        N[f"range_{h}"] = up - dnn
    for W, wl in [(cfg["outcomes"]["fp_window_min"], "7d"), (cfg["outcomes"]["fp_window_short_min"], "24h")]:
        for x, tag in TH_TAGS.items():
            fp = fp_outcome(E[f"fpt_up_{tag}"].to_numpy(), E[f"fpt_dn_{tag}"].to_numpy(), W)
            y = np.where(np.isnan(fp) | (fp == 0), np.nan, (fp == dn).astype(float))
            sfx = "" if wl == "7d" else "_24h"
            N[f"ylarge_{tag}{sfx}"] = y
            N[f"tie_{tag}{sfx}"] = (fp == 0)
            N[f"cens_{tag}{sfx}"] = np.isnan(fp)
            if bench is not None:
                pu = bench["p_up"][(tag, wl)]
                N[f"by_{tag}{sfx}"] = np.where(dn > 0, pu, 1 - pu)
    g = lambda c: E[c].astype(float).to_numpy()
    N["casc_large_1440"] = np.where(dn > 0, g("casc_up_1440"), g("casc_dn_1440"))
    N["casc_small_1440"] = np.where(dn > 0, g("casc_dn_1440"), g("casc_up_1440"))
    N["accel_large_24h"] = np.where(dn > 0, g("max_1h_up_24h"), -g("max_1h_dn_24h"))
    N["accel_small_24h"] = np.where(dn > 0, -g("max_1h_dn_24h"), g("max_1h_up_24h"))
    N["swept_large_24h"] = np.where(dn > 0, g("fwd_swept_short_24h"), g("fwd_swept_long_24h"))
    N["swept_small_24h"] = np.where(dn > 0, g("fwd_swept_long_24h"), g("fwd_swept_short_24h"))
    E = pd.concat([E, pd.DataFrame(N, index=E.index)], axis=1)
    E.index.name = "t_snap"
    return E.sort_index()


def benchmarks(P: pd.DataFrame, cfg: dict, mask: pd.Series | None = None) -> dict:
    """Unconditional benchmarks from valid snapshots on the hour (evaluation only)."""
    m = P["valid"] & (P.index.minute == 0)
    if mask is not None:
        m &= mask
    S = P[m]
    mu = {h: float(S[f"fret_{h}"].astype(float).mean()) for h in cfg["outcomes"]["horizons_min"]}
    p_up = {}
    for W, wl in [(cfg["outcomes"]["fp_window_min"], "7d"), (cfg["outcomes"]["fp_window_short_min"], "24h")]:
        for x, tag in TH_TAGS.items():
            fp = fp_outcome(S[f"fpt_up_{tag}"].to_numpy(), S[f"fpt_dn_{tag}"].to_numpy(), W)
            r = fp[np.isfinite(fp) & (fp != 0)]
            p_up[(tag, wl)] = float((r > 0).mean())
    vol = {"fwd_rv_24h": float(S.fwd_rv_24h.astype(float).median()),
           "range_1440": float((S.mfe_up_1440.astype(float) - S.mfe_dn_1440.astype(float)).median())}
    return {"mu": mu, "p_up": p_up, "vol": vol, "n": int(m.sum())}
