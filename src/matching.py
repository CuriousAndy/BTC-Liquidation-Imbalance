"""Historical nearest-neighbour matching for a current liquidation snapshot.

Similarity (standardised Euclidean, direction as a HARD match):
  signed log imbalance (weight 2), log(total / OI) if OI is supplied, log 7d realised vol,
  24h return, 7d return. The window is chosen by the user (default ±10%).
Neighbours are de-overlapped: a candidate is skipped if it lies within `min_gap_h`
of an already selected one, so the k neighbours are k distinct market episodes.
Output: distributions, never a point forecast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .price_outcomes import fp_outcome

HORIZONS = {60: "1h", 240: "4h", 720: "12h", 1440: "24h", 4320: "72h", 10080: "7d"}


def match(P: pd.DataFrame, k: str, down_usd: float, up_usd: float, oi_usd: float | None = None,
          ret_24h: float | None = None, ret_7d: float | None = None, rv_7d: float | None = None,
          n_list=(20, 50, 100), min_gap_h: float = 24) -> dict:
    sli_now = np.log(up_usd / down_usd)
    direction = 1 if up_usd > down_usd else -1
    V = P[P.valid & (P[f"dir_{k}"] == direction)]
    feats = {"sli": (V[f"sli_{k}"].astype(float), sli_now, 2.0)}
    used = {"sli": sli_now}
    if oi_usd:
        feats["log_total_oi"] = (np.log(V[f"total_oi_{k}"].astype(float)), np.log((up_usd + down_usd) / oi_usd), 1.0)
        used["total/OI"] = (up_usd + down_usd) / oi_usd
    last = P[P.valid].iloc[-1]
    for name, val, col in [("log_rv_7d", rv_7d, "rv_7d"), ("ret_24h", ret_24h, "ret_24h"), ("ret_7d", ret_7d, "ret_7d")]:
        src = "user"
        if val is None:
            val = float(last[col])
            src = f"latest in data ({P[P.valid].index[-1]:%Y-%m-%d %H:%M} UTC)"
        x = V[col].astype(float)
        if name == "log_rv_7d":
            x, val = np.log(x), np.log(val)
        feats[name] = (x, val, 1.0)
        used[name] = (val, src)
    X = pd.DataFrame({n: f[0] for n, f in feats.items()}).dropna()
    mu, sd = X.mean(), X.std()
    q = pd.Series({n: f[1] for n, f in feats.items()})
    w = pd.Series({n: f[2] for n, f in feats.items()})
    dist = np.sqrt((((X - q) / sd) ** 2 * w).sum(axis=1)).sort_values()
    # de-overlap
    chosen = []
    times = []
    gap = pd.Timedelta(hours=min_gap_h)
    for t in dist.index:
        if all(abs(t - s) >= gap for s in times):
            chosen.append(t)
            times.append(t)
        if len(chosen) >= max(n_list):
            break
    out = {"query": used, "direction": "DOWN" if direction < 0 else "UP", "ratio": max(up_usd, down_usd) / min(up_usd, down_usd)}
    res = []
    dist_rows = []
    for n in n_list:
        sel = chosen[:n]
        S = P.loc[sel]
        for h, lab in HORIZONS.items():
            r = S[f"fret_{h}"].astype(float).dropna()
            res.append({"k": n, "horizon": lab, "N": len(r), "mean": r.mean(), "p05": r.quantile(.05),
                        "p10": r.quantile(.1), "p25": r.quantile(.25), "median": r.median(), "p75": r.quantile(.75),
                        "p90": r.quantile(.9), "p95": r.quantile(.95), "P_ret_gt_0": (r > 0).mean()})
        for tag, x in [("1", 1), ("2", 2), ("3", 3), ("5", 5)]:
            fp = fp_outcome(S[f"fpt_up_{tag}"].to_numpy(), S[f"fpt_dn_{tag}"].to_numpy(), 10080)
            dist_rows.append({"k": n, "threshold_pct": x, "P_down_first": np.mean(fp == -1), "P_up_first": np.mean(fp == 1),
                              "P_tie_same_minute": np.mean(fp == 0), "P_neither_7d": np.mean(np.isnan(fp))})
    out["returns"] = pd.DataFrame(res)
    out["first_passage"] = pd.DataFrame(dist_rows)
    nb = P.loc[chosen[:max(n_list)], [f"ratio_{k}", f"sli_{k}", f"total_{k}", "rv_7d", "ret_24h", "ret_7d", "fret_1440", "fret_10080"]].copy()
    nb["distance"] = dist.loc[nb.index]
    out["neighbours"] = nb
    out["max_distance"] = {n: float(dist.loc[chosen[:n]].max()) for n in n_list}
    return out
