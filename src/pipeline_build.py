"""Stage 1-3: cleaning, liquidation-map reconstruction, regimes. Saves to data/processed/."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import cleaning, liquidation_features as lf, regimes as rg

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def run(cfg: dict, only_variants=None) -> dict:
    p_raw = pd.read_parquet(PROC / "price_1m_raw_parsed.parquet")
    m_raw = pd.read_parquet(PROC / "metrics_5m_raw_parsed.parquet")
    fund = pd.read_parquet(PROC / "funding_raw_parsed.parquet")
    prem = pd.read_parquet(PROC / "premium_5m_raw_parsed.parquet")

    price, rep_p = cleaning.clean_price(p_raw)
    metrics, rep_m = cleaning.clean_metrics(m_raw, price)
    price.to_parquet(PROC / "price_1m_clean.parquet")
    metrics.to_parquet(PROC / "metrics_5m_clean.parquet")
    bars5 = cleaning.bars_5m(price)
    bars5.to_parquet(PROC / "bars_5m.parquet")
    gaps_oi = cleaning.gap_table(metrics.oi_valid, "5min")
    gaps_oi.to_csv(PROC / "oi_invalid_gaps.csv", index=False)

    # snapshot age: minutes since last valid OI snapshot at each 5m grid point
    ns = metrics.index.as_unit("ns").asi8.astype(float)
    vt = pd.Series(np.where(metrics.oi_valid, ns, np.nan), index=metrics.index).ffill()
    age = (ns - vt) / 6e10
    lmc = cfg["liquidation_map"]
    # degraded: within degraded_cooldown_h after any invalid run longer than max_gap_for_continuity_min
    big = gaps_oi[gaps_oi.duration_min > lmc["max_gap_for_continuity_min"]]
    degraded = pd.Series(False, index=metrics.index)
    for _, g in big.iterrows():
        degraded.loc[g.start: g.end + pd.Timedelta(hours=lmc["degraded_cooldown_h"])] = True

    regimes = rg.build_regimes(bars5, metrics, fund, prem, cfg)
    regimes.to_parquet(PROC / "regimes_5m.parquet")
    ext = rg.extreme_days(bars5, regimes, cfg)
    ext.to_csv(PROC / "extreme_days.csv")

    reps = {"price": rep_p, "metrics": rep_m, "oi_gaps_over_60min": int(len(big))}
    for v in cfg["liquidation_map"]["variants"]:
        if only_variants and v not in only_variants:
            continue
        f = lf.build_map_features(bars5, metrics, cfg, v)
        f = lf.add_imbalance(f, cfg)
        f["snapshot_age_min"] = age
        f["map_degraded"] = degraded
        f.to_parquet(PROC / f"liqmap_features_{v}.parquet")
        reps[f"map_{v}_rows"] = len(f)
    if not only_variants:
        (PROC / "build_report.json").write_text(json.dumps(reps, indent=2, default=str))
    return reps
