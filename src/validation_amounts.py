"""EXPLORATORY (added after the pre-registered primary metric turned out not to identify
leverage weights): amount-level agreement between modelled sweeps and real liquidations.

  cond_pearson   corr of log(model USD) vs log(real USD) over bar-sides where BOTH > 0
  side_ratio_r   corr of log((model long+1)/(model short+1)) vs log((real long+1)/(real short+1))
                 over 1h windows where both model and real have any liquidation - the
                 quantity closest to the study's long-vs-short imbalance
Fitted on train days, reported on test days, for every spec in the calibration grid.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import liq_validation as lv

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"


def amount_scores(d: pd.DataFrame) -> dict:
    ml, ms, rl, rs = (d[c].to_numpy(float) for c in ("mod_long", "mod_short", "real_long", "real_short"))
    m = np.r_[ml, ms]
    r = np.r_[rl, rs]
    both = (m > 0) & (r > 0)
    cp = np.corrcoef(np.log(m[both]), np.log(r[both]))[0, 1] if both.sum() > 10 else np.nan
    h = d.resample("1h", label="right", closed="right")[["mod_long", "mod_short", "real_long", "real_short"]].sum()
    h = h[(h.mod_long + h.mod_short > 0) & (h.real_long + h.real_short > 0)]
    a = np.log((h.mod_long + 1) / (h.mod_short + 1))
    b = np.log((h.real_long + 1) / (h.real_short + 1))
    sr = np.corrcoef(a, b)[0, 1] if len(h) > 10 else np.nan
    agree = float((np.sign(a) == np.sign(b)).mean()) if len(h) else np.nan
    return {"cond_pearson": cp, "n_both": int(both.sum()), "side_ratio_r": sr, "side_sign_agree": agree, "n_hours": len(h)}


def run(cfg) -> pd.DataFrame:
    days = lv.validation_days()
    real = lv.load_real(days)
    realbars = lv.real_per_bar(real, days)
    day_of = (realbars.index - pd.Timedelta(minutes=5)).floor("1D")
    is_train = day_of.isin(pd.DatetimeIndex(days[0::2]))
    eng = lv.EngineInputs(cfg)
    lm = cfg["liquidation_map"]
    specs = [("M1", lm["leverage_tiers"], tuple(lm["leverage_weights"]), None, "symmetric"),
             ("M2", lm["leverage_tiers"], tuple(lm["leverage_weights"]), 72, "account")]
    specs += [("grid", lv.GRID_TIERS, w, hl, sp) for w in lv.weight_grid(0.25) for hl in [None, 168, 72, 24]
              for sp in ["symmetric", "account"]]
    rows = []
    for name, tiers, w, hl, sp in specs:
        swl, sws, _ = eng.run(tiers, w, hl, sp)
        d = realbars.copy()
        d["mod_long"] = swl.reindex(d.index).to_numpy()
        d["mod_short"] = sws.reindex(d.index).to_numpy()
        d = d.dropna()
        for part, m in [("train", d.index.isin(realbars.index[is_train])), ("test", d.index.isin(realbars.index[~is_train]))]:
            rows.append({"model": name, "tiers": str(tiers), "weights": str(tuple(round(x, 3) for x in w)), "half_life_h": hl,
                         "side_split": sp, "part": part, **amount_scores(d[m])})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "validation_amount_grid_EXPLORATORY.csv", index=False)
    return out
