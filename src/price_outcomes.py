"""Forward price outcomes, computed once for every 5m grid point from 1m data.

Timing convention (no look-ahead):
  t_snap  = liquidation-map snapshot time (state built from data <= t_snap)
  t0      = t_snap + publication lag (5 min) = time the signal is usable
  P0      = traded price at t0 (close of the 1m bar ending at t0)
  outcomes use 1m bars that START at or after t0.

First passage: first 1m bar whose high >= P0(1+x) (up) or low <= P0(1-x) (down).
If both happen inside the same 1m bar the order is unknown -> 'tie' (excluded
from hit-first probabilities, counted separately). Not hit inside the window ->
censored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

PUB_LAG_MIN = 5


@njit(cache=True)
def _block_max(a, b):
    n = (a.shape[0] + b - 1) // b
    out = np.empty(n)
    for i in range(n):
        out[i] = a[i * b:min((i + 1) * b, a.shape[0])].max()
    return out


@njit(cache=True)
def _block_min(a, b):
    n = (a.shape[0] + b - 1) // b
    out = np.empty(n)
    for i in range(n):
        out[i] = a[i * b:min((i + 1) * b, a.shape[0])].min()
    return out


@njit(cache=True)
def _first_hit(arr, b1, b2, starts, levels, maxwin, up):
    """Minutes from start until arr crosses level (>= if up else <=); -1 if not within maxwin.
    b1/b2 are block extrema at 60 and 1440 minutes for fast skipping."""
    n = arr.shape[0]
    res = np.full(starts.shape[0], -1, np.int64)
    for q in range(starts.shape[0]):
        s = starts[q]
        lv = levels[q]
        if s < 0 or not np.isfinite(lv):
            continue
        end = min(s + maxwin, n)
        j = s
        while j < end:
            if j % 1440 == 0 and j + 1440 <= end:
                v = b2[j // 1440]
                if (up and v < lv) or ((not up) and v > lv):
                    j += 1440
                    continue
            if j % 60 == 0 and j + 60 <= end:
                v = b1[j // 60]
                if (up and v < lv) or ((not up) and v > lv):
                    j += 60
                    continue
            v = arr[j]
            if (up and v >= lv) or ((not up) and v <= lv):
                res[q] = j - s
                break
            j += 1
    return res


class PricePath:
    """1m price arrays + fast first-passage queries."""

    def __init__(self, price_1m: pd.DataFrame):
        self.t_first = price_1m.index[0]
        self.high = price_1m.high.to_numpy(float)
        self.low = price_1m.low.to_numpy(float)
        self.close = price_1m.close.to_numpy(float)
        self.hb1, self.hb2 = _block_max(self.high, 60), _block_max(self.high, 1440)
        self.lb1, self.lb2 = _block_min(self.low, 60), _block_min(self.low, 1440)

    def idx(self, ts) -> np.ndarray:
        ts = pd.DatetimeIndex(ts)
        return ((ts - self.t_first) // pd.Timedelta(minutes=1)).to_numpy(np.int64)

    def p0(self, t0) -> np.ndarray:
        i = self.idx(t0) - 1
        return self.close[i]

    def first_hit(self, t0, levels, maxwin, up: bool) -> np.ndarray:
        s = self.idx(t0)
        s = np.where((s >= 0) & (s < len(self.high)), s, -1)
        arr, b1, b2 = (self.high, self.hb1, self.hb2) if up else (self.low, self.lb1, self.lb2)
        return _first_hit(arr, b1, b2, s, np.asarray(levels, float), int(maxwin), up)


def fp_outcome(t_up: np.ndarray, t_dn: np.ndarray, window: int) -> np.ndarray:
    """+1 up first, -1 down first, 0 tie, nan censored (neither within window)."""
    u = np.where((t_up >= 0) & (t_up < window), t_up, np.inf)
    d = np.where((t_dn >= 0) & (t_dn < window), t_dn, np.inf)
    out = np.where(u < d, 1.0, np.where(d < u, -1.0, 0.0))
    out[np.isinf(u) & np.isinf(d)] = np.nan
    return out


def build_grid_outcomes(price_1m: pd.DataFrame, grid: pd.DatetimeIndex, cfg: dict,
                        map_feats: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    oc = cfg["outcomes"]
    pp = PricePath(price_1m)
    t0 = grid + pd.Timedelta(minutes=PUB_LAG_MIN)
    i0 = pp.idx(t0)
    n1 = len(pp.close)
    ok = (i0 >= 1) & (i0 < n1)
    i0c = np.clip(i0, 1, n1 - 1)
    P0 = np.where(ok, pp.close[i0c - 1], np.nan)
    out = pd.DataFrame(index=grid)
    out["t0"] = t0
    out["P0"] = P0
    for h in oc["horizons_min"]:
        j = i0c - 1 + h
        v = np.where(ok & (j < n1), pp.close[np.clip(j, 0, n1 - 1)], np.nan)
        out[f"fret_{h}"] = (v / P0 - 1).astype(np.float32)
    # forward extremes over [i0, i0+h)
    hi = pd.Series(pp.high)
    lo = pd.Series(pp.low)
    for h in oc["mfe_horizons_min"]:
        fmax = hi[::-1].rolling(h, min_periods=h).max()[::-1].to_numpy()
        fmin = lo[::-1].rolling(h, min_periods=h).min()[::-1].to_numpy()
        out[f"mfe_up_{h}"] = (np.where(ok, fmax[i0c], np.nan) / P0 - 1).astype(np.float32)
        out[f"mfe_dn_{h}"] = (np.where(ok, fmin[i0c], np.nan) / P0 - 1).astype(np.float32)
    W = oc["fp_window_min"]
    for x in oc["thresholds"]:
        tu = pp.first_hit(t0, P0 * (1 + x), W, True)
        td = pp.first_hit(t0, P0 * (1 - x), W, False)
        tag = f"{x * 100:g}".replace(".", "p")
        out[f"fpt_up_{tag}"] = tu.astype(np.int32)
        out[f"fpt_dn_{tag}"] = td.astype(np.int32)
    # realised volatility over the next 24h (1m log returns, annualised)
    lr = np.diff(np.log(pp.close), prepend=np.nan)
    s2 = pd.Series(lr ** 2)
    fwd_var = s2[::-1].rolling(1440, min_periods=1400).sum()[::-1].to_numpy()
    out["fwd_rv_24h"] = (np.sqrt(np.where(ok, fwd_var[i0c], np.nan) * 365)).astype(np.float32)
    # largest 1h move up / down inside the next 24h (acceleration proxy), on 5m grid
    c5 = pd.Series(P0, index=grid)
    r1h = c5.shift(-12) / c5 - 1          # 1h return starting at each grid point
    out["max_1h_up_24h"] = r1h[::-1].rolling(288, min_periods=280).max()[::-1].astype(np.float32)
    out["max_1h_dn_24h"] = r1h[::-1].rolling(288, min_periods=280).min()[::-1].astype(np.float32)
    # cascade proxy: 1h window with OI drop >= 3% and |return| >= 2% (observed OI, not modelled)
    oi = metrics.oi_btc.where(metrics.oi_valid).reindex(grid)
    oi1h = oi / oi.shift(12) - 1
    pr1h = c5 / c5.shift(12) - 1
    casc_dn = ((oi1h <= -0.03) & (pr1h <= -0.02)).astype(float)
    casc_up = ((oi1h <= -0.03) & (pr1h >= 0.02)).astype(float)
    for h, n in [(1440, 288), (4320, 864)]:
        # any 1h cascade window lying fully after t0 (windows ending in [t0+1h, t0+h+1h))
        out[f"casc_dn_{h}"] = casc_dn[::-1].rolling(n, min_periods=1).max()[::-1].shift(-13).astype(np.float32)
        out[f"casc_up_{h}"] = casc_up[::-1].rolling(n, min_periods=1).max()[::-1].shift(-13).astype(np.float32)
    # modelled forced liquidation swept in the next 24h, per side (USD)
    sl = map_feats.swept_long_usd.reindex(grid)
    ss = map_feats.swept_short_usd.reindex(grid)
    out["fwd_swept_long_24h"] = sl[::-1].rolling(288, min_periods=280).sum()[::-1].shift(-2).astype(np.float32)
    out["fwd_swept_short_24h"] = ss[::-1].rolling(288, min_periods=280).sum()[::-1].shift(-2).astype(np.float32)
    return out
