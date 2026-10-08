"""Cleaning + alignment. Everything is UTC; no silent forward-fill.

Rules (all causal - a value's validity is decided with data up to its own timestamp):
  * metrics: drop exact duplicate rows; OI <= 0 -> missing; implied price
    (oi_usd / oi_btc) more than 3% away from the traded close -> invalid;
    |log(OI_t / OI_last_valid)| > 0.25 -> invalid (no real 5m OI move ever exceeded
    ~14%; the 2021-05-22 zero-then-ramp artefact is the motivating case).
  * price: 1m klines checked for duplicates, gaps, OHLC consistency. Zero-volume
    flat bars (exchange maintenance fills) are kept but flagged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def clean_price(p: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rep = {"rows_raw": len(p), "dup_ts": int(p.ts.duplicated().sum())}
    p = p.drop_duplicates("ts").sort_values("ts").set_index("ts")
    assert p.index.tz is not None and str(p.index.tz) == "UTC"
    full = pd.date_range(p.index.min(), p.index.max(), freq="1min")
    rep["missing_bars"] = int(len(full.difference(p.index)))
    bad = (p.high < p[["open", "close"]].max(axis=1)) | (p.low > p[["open", "close"]].min(axis=1))
    rep["ohlc_inconsistent"] = int(bad.sum())
    p["flat_zero_vol"] = (p.volume == 0) & (p.high == p.low)
    rep["flat_zero_vol_bars"] = int(p.flat_zero_vol.sum())
    p = p.reindex(full)  # gaps (none observed) stay NaN, never filled
    p.index.name = "ts"
    return p, rep


def clean_metrics(m: pd.DataFrame, price_1m: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rep = {"rows_raw": len(m), "exact_dups": int(m.duplicated().sum())}
    m = m.drop_duplicates()
    rep["conflicting_ts_dups"] = int(m.ts.duplicated().sum())
    m = m.drop_duplicates("ts", keep="last").sort_values("ts").set_index("ts")
    rep["off_grid_ts"] = int(((m.index.minute % 5 != 0) | (m.index.second != 0)).sum())
    m = m[(m.index.minute % 5 == 0) & (m.index.second == 0)]
    # traded price at the snapshot instant = close of the 1m bar that opens 1 minute earlier
    px = price_1m["close"].reindex(m.index - pd.Timedelta(minutes=1)).to_numpy()
    oi = m.oi_btc.to_numpy(float)
    implied = np.where(oi > 0, m.oi_usd.to_numpy(float) / np.where(oi > 0, oi, 1), np.nan)
    valid = (oi > 0) & np.isfinite(implied) & (np.abs(np.log(implied / px)) < 0.03)
    rep["oi_nonpositive"] = int((oi <= 0).sum())
    rep["oi_implied_price_mismatch"] = int(((oi > 0) & ~valid).sum())
    # causal jump filter against last valid value
    last = np.nan
    jump_bad = 0
    for i in range(len(oi)):
        if not valid[i]:
            continue
        if np.isfinite(last) and abs(np.log(oi[i] / last)) > 0.25:
            valid[i] = False
            jump_bad += 1
            continue
        last = oi[i]
    rep["oi_jump_rejected"] = jump_bad
    m["oi_valid"] = valid
    full = pd.date_range(m.index.min(), m.index.max(), freq="5min")
    rep["missing_5m_snapshots"] = int(len(full.difference(m.index)))
    m = m.reindex(full)
    m.index.name = "ts"
    m["oi_valid"] = m["oi_valid"].fillna(False).astype(bool)
    rep["valid_snapshots"] = int(m.oi_valid.sum())
    rep["expected_snapshots"] = len(full)
    return m, rep


def gap_table(valid: pd.Series, step: str) -> pd.DataFrame:
    """Contiguous runs where valid is False."""
    s = (~valid).astype(int)
    grp = (s.diff().fillna(s.iloc[0]) != 0).cumsum()
    runs = pd.DataFrame({"bad": s, "grp": grp}).loc[s == 1]
    if runs.empty:
        return pd.DataFrame(columns=["start", "end", "n_missing", "duration_min"])
    t = runs.groupby("grp").apply(lambda x: pd.Series({"start": x.index.min(), "end": x.index.max(),
                                                       "n_missing": len(x)}))
    t["duration_min"] = t.n_missing * pd.Timedelta(step).total_seconds() / 60
    return t.sort_values("n_missing", ascending=False).reset_index(drop=True)


def bars_5m(price_1m: pd.DataFrame) -> pd.DataFrame:
    """5m bars labelled by their END time t (bar covers [t-5m, t))."""
    p = price_1m
    g = p.resample("5min", label="right", closed="left")
    b = pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min(),
                      "close": g.close.last(), "volume": g.volume.sum(),
                      "quote_volume": g.quote_volume.sum(),
                      "taker_buy_volume": g.taker_buy_volume.sum()})
    b["typical"] = (b.high + b.low + b.close) / 3
    return b
