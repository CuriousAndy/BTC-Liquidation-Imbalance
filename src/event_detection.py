"""Event construction - de-overlapping persistent imbalance states.

Method A  threshold crossing with hysteresis: onset when ratio goes from < enter
          to >= enter; armed again only after ratio < exit (exit = 0.8 * enter).
Method B  cooldown: any valid bar satisfying the condition, if no event of the
          same condition in the previous `cooldown` hours.
Method C  episode: maximal run of consecutive valid bars with ratio >= enter and
          an unchanged large-pool direction. Start = onset (tradable). Peak = bar
          of maximum ratio (EX-POST definition - uses the future inside the
          episode, reported only as a descriptive check, never as a signal).

Invalid bars (stale/degraded map) break the state machine: after an invalid
stretch the crossing is not "seen", so no onset is generated until the ratio is
observed below the arming level again (Method A) and episodes touching an
invalid stretch at their start are flagged left_censored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True)
def _crossing(r, valid, enter, exit_):
    n = r.shape[0]
    ev = np.zeros(n, np.bool_)
    state = 2  # 0 = armed (below), 1 = in event, 2 = unknown
    for i in range(n):
        if not valid[i]:
            state = 2
            continue
        x = r[i]
        if state == 0:
            if x >= enter:
                ev[i] = True
                state = 1
        elif state == 1:
            if x < exit_:
                state = 0
        else:
            # unknown after a gap: arm once the ratio is actually observed below `enter`
            if x < enter:
                state = 0
    return ev


def crossing_events(ratio: pd.Series, valid: pd.Series, enter: float, exit_: float | None = None) -> pd.DatetimeIndex:
    exit_ = 0.8 * enter if exit_ is None else exit_
    ev = _crossing(ratio.to_numpy(float), valid.to_numpy(bool), float(enter), float(exit_))
    return ratio.index[ev]


@njit(cache=True)
def _cooldown(cond, tsec, cool_sec):
    n = cond.shape[0]
    ev = np.zeros(n, np.bool_)
    last = -1e18
    for i in range(n):
        if cond[i] and tsec[i] - last >= cool_sec:
            ev[i] = True
            last = tsec[i]
    return ev


def cooldown_events(cond: pd.Series, cooldown_h: float) -> pd.DatetimeIndex:
    tsec = (pd.DatetimeIndex(cond.index).as_unit("ns").asi8 // 10**9).astype(np.float64)
    ev = _cooldown(cond.to_numpy(bool), tsec, float(cooldown_h * 3600))
    return cond.index[ev]


def episodes(ratio: pd.Series, direction: pd.Series, valid: pd.Series, enter: float,
             L_up: pd.Series, L_down: pd.Series) -> pd.DataFrame:
    on = (ratio >= enter) & valid
    key = on.astype(int).astype(str) + direction.astype(str)
    brk = (key != key.shift()) | ~on
    grp = brk.cumsum()
    df = pd.DataFrame({"ratio": ratio, "dir": direction, "grp": grp, "L_up": L_up, "L_down": L_down})[on]
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("grp")
    ep = pd.DataFrame({
        "start": g.apply(lambda x: x.index[0]),
        "end": g.apply(lambda x: x.index[-1]),
        "n_bars": g.size(),
        "initial_ratio": g.ratio.first(),
        "max_ratio": g.ratio.max(),
        "mean_ratio": g.ratio.mean(),
        "peak_time": g.ratio.idxmax(),
        "direction": g.dir.first(),
        "max_L_up": g.L_up.max(),
        "max_L_down": g.L_down.max(),
    }).reset_index(drop=True)
    ep["duration_h"] = (ep.end - ep.start).dt.total_seconds() / 3600 + 5 / 60
    prev_idx = ratio.index.get_indexer(ep.start) - 1
    prev_valid = valid.to_numpy()[np.clip(prev_idx, 0, None)]
    ep["left_censored"] = (prev_idx < 0) | ~prev_valid
    return ep
