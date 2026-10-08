"""Stage 4: forward outcomes on the full 5m grid + analysis panels per map variant."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import panel as pn, price_outcomes as po

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def run(cfg: dict, skip_outcomes: bool = False) -> dict:
    if not skip_outcomes:
        _outcomes(cfg)
    return _panels(cfg)


def _outcomes(cfg):
    price = pd.read_parquet(PROC / "price_1m_clean.parquet")
    metrics = pd.read_parquet(PROC / "metrics_5m_clean.parquet")
    f1 = pd.read_parquet(PROC / "liqmap_features_M1.parquet", columns=["swept_long_usd", "swept_short_usd"])
    oc = po.build_grid_outcomes(price, metrics.index, cfg, f1, metrics)
    oc.to_parquet(PROC / "grid_outcomes.parquet")


def _panels(cfg, only_variants=None, light=False):
    import gc
    bars5 = pd.read_parquet(PROC / "bars_5m.parquet")
    rf = PROC / "panel_report.json"
    rep = json.loads(rf.read_text()) if (only_variants and rf.exists()) else {}
    for v in cfg["liquidation_map"]["variants"]:
        if only_variants and v not in only_variants:
            continue
        p = pn.build_panel(cfg, v, light=light)
        p.to_parquet(PROC / f"panel_{v}.parquet")
        rep[v] = {"rows": len(p), "valid": int(p.valid.sum()), "lookahead_checks": pn.check_no_lookahead(p, bars5)}
        del p
        gc.collect()
    (PROC / "panel_report.json").write_text(json.dumps(rep, indent=2, default=str))
    return rep
