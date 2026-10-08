"""Market-state features on the 5m grid. Every feature uses data with ts <= t only.

Tercile/percentile classifications use EXPANDING history (min 90 days), never the
full-sample distribution, so a regime label at t never sees the future.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _expanding_tercile(x: pd.Series, min_periods: int) -> pd.Series:
    """Label 0/1/2 by expanding 33/67% quantiles computed on a daily-sampled history."""
    daily = x.resample("1D").last()
    q1 = daily.expanding(min_periods=min_periods).quantile(1 / 3).shift(1)
    q2 = daily.expanding(min_periods=min_periods).quantile(2 / 3).shift(1)
    q1 = q1.reindex(x.index, method="ffill")
    q2 = q2.reindex(x.index, method="ffill")
    lab = pd.Series(np.where(x <= q1, 0, np.where(x <= q2, 1, 2)), index=x.index, dtype=float)
    lab[q1.isna() | x.isna()] = np.nan
    return lab


def build_regimes(bars5: pd.DataFrame, metrics: pd.DataFrame, funding: pd.DataFrame,
                  premium: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rc = cfg["regimes"]
    c = bars5.close
    r = pd.DataFrame(index=bars5.index)
    lr = np.log(c).diff()
    per_day = 288
    for name, n in [("24h", per_day), ("7d", 7 * per_day), ("30d", 30 * per_day)]:
        r[f"ret_{name}"] = c / c.shift(n) - 1
        r[f"rv_{name}"] = lr.rolling(n, min_periods=int(n * 0.9)).std() * np.sqrt(per_day * 365)
    daily_close = c.resample("1D", label="right", closed="right").last()
    for d in (20, 50, 200):
        ma = daily_close.rolling(d).mean()
        # MA of completed days only, made available at the next 5m bars
        r[f"ma{d}d"] = ma.reindex(r.index, method="ffill")
        r[f"px_vs_ma{d}d"] = c / r[f"ma{d}d"] - 1
    ma = r[f"ma{rc['trend_ma_days']}d"]
    slope = ma / ma.shift(rc["slope_days"] * per_day) - 1
    up = (c > ma) & (slope > 0)
    dn = (c < ma) & (slope < 0)
    r["trend"] = np.where(up, "uptrend", np.where(dn, "downtrend", "sideways"))
    r.loc[ma.isna() | slope.isna(), "trend"] = np.nan
    r["vol_regime_num"] = _expanding_tercile(r["rv_7d"], rc["vol_min_history_days"])
    r["vol_regime"] = r.vol_regime_num.map({0: "low_vol", 1: "mid_vol", 2: "high_vol"})
    # funding: last settled rate at or before t
    fr = funding.set_index("ts").funding.sort_index()
    r["funding"] = fr.reindex(r.index, method="ffill")
    r["funding_age_h"] = (r.index.to_series() -
                          pd.Series(fr.index, index=fr.index).reindex(r.index, method="ffill")).dt.total_seconds() / 3600
    r.loc[r.funding_age_h > 9, "funding"] = np.nan
    r["funding_regime"] = np.where(r.funding < 0, "negative",
                                   np.where(r.funding > 0.0001, "positive", "neutral"))
    r.loc[r.funding.isna(), "funding_regime"] = np.nan
    # OI change (valid snapshots only, no fill across long gaps)
    oi = metrics.oi_btc.where(metrics.oi_valid).reindex(r.index)
    oi_f = oi.ffill(limit=3)
    r["oi_chg_1h"] = oi_f / oi_f.shift(12) - 1
    r["oi_chg_24h"] = oi_f / oi_f.shift(per_day) - 1
    r["oi_regime"] = np.where(r.oi_chg_24h > 0.01, "oi_rising",
                              np.where(r.oi_chg_24h < -0.01, "oi_falling", "oi_flat"))
    r.loc[r.oi_chg_24h.isna(), "oi_regime"] = np.nan
    # basis proxy: premium index, bar labelled by end time
    pm = premium.set_index("ts").premium
    pm.index = pm.index + pd.Timedelta(minutes=5)
    r["premium"] = pm.reindex(r.index)
    r["hour_utc"] = r.index.hour
    r["volume_24h"] = bars5.quote_volume.rolling(per_day).sum()
    return r


def extreme_days(bars5: pd.DataFrame, regimes: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Statistical (not news-based) extreme-day flags, UTC calendar days."""
    st = cfg["stats"]
    dc = bars5.close.resample("1D").last()
    dret = dc.pct_change()
    hi = bars5.high.resample("1D").max()
    lo = bars5.low.resample("1D").min()
    drange = hi / lo - 1
    rv = regimes.rv_24h.resample("1D").max()
    thr = rv.quantile(st["extreme_vol_pct"])
    e = pd.DataFrame({"ret": dret, "range": drange, "rv24h_max": rv})
    e["extreme"] = (dret.abs() > st["extreme_daily_move"]) | (rv > thr)
    e.attrs["rv_threshold"] = float(thr)
    return e
