"""Layer-2 strategy test (only interpretable if Layer-1 found an effect).

Signal at event onset t0; side = +1 trades toward the larger pool, -1 toward the smaller.
Exits: fixed holding periods, or a symmetric bracket (take-profit +x / stop -x, whichever
the price touches first; same-1m-bar ties count as the stop - conservative; neither within
7d -> exit at the 7d close). Costs: (fee + slippage) per side, both sides.
Positions may overlap for long holds; every trade gets equal notional.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import statistics as st


def run_backtest(E: pd.DataFrame, cfg: dict, side: int, label: str) -> pd.DataFrame:
    bc = cfg["backtest"]
    cost = 2 * (bc["fee_bps"] + bc["slippage_bps"]) / 1e4
    d = E.dir.to_numpy(float) * side
    rows = []
    for h in bc["exits_min"]:
        r = d * E[f"fret_{h}"].to_numpy(float) - cost
        rows.append(_stats(r, E.index, f"hold {h // 60}h", label, side, h / 1440))
    for x, tag in [(0.01, "1"), (0.02, "2"), (0.03, "3")]:
        tu = E[f"fpt_up_{tag}"].to_numpy(float)
        td = E[f"fpt_dn_{tag}"].to_numpy(float)
        tu = np.where(tu < 0, np.inf, tu)
        td = np.where(td < 0, np.inf, td)
        t_win = np.where(d > 0, tu, td)
        t_loss = np.where(d > 0, td, tu)
        r = np.where(t_win < t_loss, x, np.where(t_loss <= t_win, -x, 0.0))
        neither = np.isinf(t_win) & np.isinf(t_loss)
        r = np.where(neither, d * E["fret_10080"].to_numpy(float), r) - cost
        hold = np.where(neither, 7.0, np.minimum(t_win, t_loss) / 1440)
        rows.append(_stats(r, E.index, f"bracket +/-{tag}% (max 7d)", label, side, float(np.nanmean(hold))))
    return pd.DataFrame(rows)


def _stats(r, idx, exit_name, label, side, hold_days):
    r = np.asarray(r, float)
    ok = np.isfinite(r)
    r = r[ok]
    n = len(r)
    if n < 5:
        return {"event_set": label, "side": side, "exit": exit_name, "N_trades": n}
    yrs = (idx[ok][-1] - idx[ok][0]).days / 365.25
    s = st.summarize_dar(r, None, 2000, 10, groups=st.month_groups(idx[ok]))
    return {"event_set": label, "side": "toward LARGE pool" if side > 0 else "toward SMALL pool", "exit": exit_name,
            "N_trades": n, "mean_net_ret": r.mean(), "median_net_ret": np.median(r), "hit_rate": (r > 0).mean(),
            "mean_ci_month_cluster": f"[{s['mean_clci_lo']:.4f}, {s['mean_clci_hi']:.4f}]", "p_cluster": s["p_cluster"],
            "sum_ret": r.sum(), "trades_per_year": n / max(yrs, 1e-9), "avg_hold_days": hold_days,
            "per_trade_sharpe_annualised": r.mean() / r.std(ddof=1) * np.sqrt(n / max(yrs, 1e-9)),
            "worst_trade": r.min(), "best_trade": r.max()}
