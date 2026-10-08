"""Reconstruct a historical BTC liquidation map and derive imbalance features.

No free source publishes historical liquidation-map snapshots (Coinglass/Hyblock
maps are proprietary *models*, not observed data). We rebuild the same kind of
model transparently from Binance BTCUSDT-perp 5m open interest:

  for each 5m step t (bar [t-5m, t)):
    1. optional exponential decay of all open modelled positions (variant M2)
    2. sweep: long liquidation levels >= bar low and short levels <= bar high are
       removed (they would have been liquidated inside the bar); swept notional is
       recorded as modelled forced liquidation
    3. dOI > 0: new notional dOI * typical_price is opened on both sides and spread
       over leverage tiers; liq price long = P(1 - 1/L + mmr), short = P(1 + 1/L - mmr)
       dOI < 0: positions are closed pro rata on both sides (net of what the sweep
       already removed)
  The state after step t is the map "as of" t. It only uses data with ts <= t.

This is an estimate of where leveraged positions *would* be liquidated, not an
order book and not observed liquidations. All results inherit this model risk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

FEATURE_WINDOWS = [0.03, 0.05, 0.075, 0.10, 0.125, 0.15]


def wkey(w: float) -> str:
    return f"w{w * 100:g}".replace(".", "p")


@njit(cache=True)
def _run_map(high, low, typ, close, doi, oi_prev, valid, long_share, decay,
             lnmin, step, nbins, lev, levw, mmr, windows, cl_range, dfloor, ks, cum_levels,
             out_win, out_dw, out_cl, out_big, out_misc):
    L = np.zeros(nbins)
    S = np.zeros(nbins)
    nw = windows.shape[0]
    nk = ks.shape[0]
    nc = cum_levels.shape[0]
    for t in range(high.shape[0]):
        if not np.isfinite(close[t]):
            out_misc[t, :] = np.nan
            continue
        if decay < 1.0:
            L *= decay
            S *= decay
        # --- sweep
        ilo = int((np.log(low[t]) - lnmin) / step)
        ihi = int((np.log(high[t]) - lnmin) / step)
        swl = 0.0
        for i in range(max(ilo, 0), nbins):
            swl += L[i]
            L[i] = 0.0
        sws = 0.0
        for i in range(0, min(ihi + 1, nbins)):
            sws += S[i]
            S[i] = 0.0
        # --- OI flow
        if valid[t]:
            p = typ[t]
            if doi[t] > 0:
                notional = doi[t] * p
                la = 2.0 * notional * long_share[t]
                sa = 2.0 * notional * (1.0 - long_share[t])
                for j in range(lev.shape[0]):
                    ql = p * (1.0 - 1.0 / lev[j] + mmr)
                    qs = p * (1.0 + 1.0 / lev[j] - mmr)
                    bl = int((np.log(ql) - lnmin) / step)
                    bs = int((np.log(qs) - lnmin) / step)
                    if 0 <= bl < nbins:
                        L[bl] += la * levw[j]
                    if 0 <= bs < nbins:
                        S[bs] += sa * levw[j]
            elif doi[t] < 0 and oi_prev[t] > 0:
                closed = -doi[t] * p
                net = closed - 0.5 * (swl + sws)
                if net > 0:
                    frac = min(net / (oi_prev[t] * p), 1.0)
                    L *= (1.0 - frac)
                    S *= (1.0 - frac)
        # --- features relative to the bar close
        P0 = close[t]
        b0 = int((np.log(P0) - lnmin) / step)
        for k in range(nw):
            w = windows[k]
            bd = int((np.log(P0 * (1.0 - w)) - lnmin) / step)
            bu = int((np.log(P0 * (1.0 + w)) - lnmin) / step)
            sd = 0.0
            for i in range(max(bd, 0), min(b0 + 1, nbins)):
                sd += L[i]
            su = 0.0
            for i in range(max(b0, 0), min(bu + 1, nbins)):
                su += S[i]
            out_win[t, k, 0] = sd
            out_win[t, k, 1] = su
        # distance weighted within +-cl_range: weights 1/d, 1/d^2, exp(-k d)...
        bd = int((np.log(P0 * (1.0 - cl_range)) - lnmin) / step)
        bu = int((np.log(P0 * (1.0 + cl_range)) - lnmin) / step)
        for q in range(2 + nk):
            out_dw[t, q, 0] = 0.0
            out_dw[t, q, 1] = 0.0
        for i in range(max(bd, 0), min(bu + 1, nbins)):
            pc = np.exp(lnmin + (i + 0.5) * step)
            d = abs(pc / P0 - 1.0)
            if d < dfloor:
                d = dfloor
            for side in range(2):
                v = L[i] if side == 0 else S[i]
                if v == 0.0:
                    continue
                out_dw[t, 0, side] += v / d
                out_dw[t, 1, side] += v / (d * d)
                for kk in range(nk):
                    out_dw[t, 2 + kk, side] += v * np.exp(-ks[kk] * d)
        # cumulative cluster levels within +-cl_range (distance from P0, log)
        totd = 0.0
        for i in range(max(bd, 0), min(b0 + 1, nbins)):
            totd += L[i]
        totu = 0.0
        for i in range(max(b0, 0), min(bu + 1, nbins)):
            totu += S[i]
        for c in range(nc):
            out_cl[t, c, 0] = np.nan
            out_cl[t, c, 1] = np.nan
        if totd > 0:
            acc = 0.0
            c = 0
            for i in range(min(b0, nbins - 1), max(bd, 0) - 1, -1):
                acc += L[i]
                while c < nc and acc >= cum_levels[c] * totd:
                    out_cl[t, c, 0] = np.exp(lnmin + (i + 0.5) * step)
                    c += 1
        if totu > 0:
            acc = 0.0
            c = 0
            for i in range(max(b0, 0), min(bu + 1, nbins)):
                acc += S[i]
                while c < nc and acc >= cum_levels[c] * totu:
                    out_cl[t, c, 1] = np.exp(lnmin + (i + 0.5) * step)
                    c += 1
        # largest cluster: max of 5-bin (~0.5%) rolling sum
        best = 0.0
        bi = -1
        for i in range(max(bd, 0), min(b0 + 1, nbins) - 4):
            s5 = L[i] + L[i + 1] + L[i + 2] + L[i + 3] + L[i + 4]
            if s5 > best:
                best = s5
                bi = i + 2
        out_big[t, 0, 0] = np.exp(lnmin + (bi + 0.5) * step) if bi >= 0 else np.nan
        out_big[t, 0, 1] = best
        best = 0.0
        bi = -1
        for i in range(max(b0, 0), min(bu + 1, nbins) - 4):
            s5 = S[i] + S[i + 1] + S[i + 2] + S[i + 3] + S[i + 4]
            if s5 > best:
                best = s5
                bi = i + 2
        out_big[t, 1, 0] = np.exp(lnmin + (bi + 0.5) * step) if bi >= 0 else np.nan
        out_big[t, 1, 1] = best
        out_misc[t, 0] = L.sum()
        out_misc[t, 1] = S.sum()
        out_misc[t, 2] = swl
        out_misc[t, 3] = sws


def build_map_features(bars5: pd.DataFrame, metrics: pd.DataFrame, cfg: dict, variant: str) -> pd.DataFrame:
    lm = cfg["liquidation_map"]
    fc = cfg["features"]
    vcfg = lm["variants"][variant]
    idx = metrics.index
    b = bars5.reindex(idx)
    oi = metrics.oi_btc.where(metrics.oi_valid)
    # dOI against the last valid snapshot (causal). Large gaps are flagged downstream.
    last_valid = oi.ffill().shift(1)
    doi = (oi - last_valid).where(metrics.oi_valid, 0.0).fillna(0.0)
    if vcfg["side_split"] == "symmetric":
        ls = np.full(len(idx), 0.5)
    else:
        r = metrics.acct_ls_ratio.where(metrics.oi_valid).ffill(limit=3).fillna(1.0)
        ls = (r / (1 + r)).clip(0.2, 0.8).to_numpy()
    hl = vcfg["decay_half_life_h"]
    decay = 0.5 ** (5 / 60 / hl) if hl else 1.0

    lnmin = np.log(lm["grid_min_price"])
    step = lm["grid_step_log"]
    nbins = int((np.log(lm["grid_max_price"]) - lnmin) / step) + 1
    windows = np.array(fc["windows"], float)
    ks = np.array(fc["exp_k"], float)
    cum = np.array(fc["cluster_cum_levels"], float)
    n = len(idx)
    out_win = np.full((n, len(windows), 2), np.nan)
    out_dw = np.full((n, 2 + len(ks), 2), np.nan)
    out_cl = np.full((n, len(cum), 2), np.nan)
    out_big = np.full((n, 1, 2, 2), np.nan).reshape(n, 2, 2)
    out_misc = np.full((n, 4), np.nan)
    _run_map(b.high.to_numpy(float), b.low.to_numpy(float), b.typical.to_numpy(float),
             b.close.to_numpy(float), doi.to_numpy(float), last_valid.fillna(0).to_numpy(float),
             metrics.oi_valid.to_numpy(bool), ls, decay, lnmin, step, nbins,
             np.array(vcfg.get("leverage_tiers", lm["leverage_tiers"]), float),
             np.array(vcfg.get("leverage_weights", lm["leverage_weights"]), float)
             / np.sum(vcfg.get("leverage_weights", lm["leverage_weights"])),
             lm["maintenance_margin"], windows, fc["cluster_range"], fc["distance_floor"], ks, cum,
             out_win, out_dw, out_cl, out_big, out_misc)

    f = pd.DataFrame(index=idx)
    f["close"] = b.close
    for k, w in enumerate(windows):
        f[f"L_down_{wkey(w)}"] = out_win[:, k, 0]
        f[f"L_up_{wkey(w)}"] = out_win[:, k, 1]
    names = ["inv_d", "inv_d2"] + [f"exp{int(k)}" for k in ks]
    for q, nm in enumerate(names):
        f[f"DW_down_{nm}"] = out_dw[:, q, 0]
        f[f"DW_up_{nm}"] = out_dw[:, q, 1]
    for c, lv in enumerate(cum):
        f[f"cl{int(lv * 100)}_down_price"] = out_cl[:, c, 0]
        f[f"cl{int(lv * 100)}_up_price"] = out_cl[:, c, 1]
    f["big_down_price"] = out_big[:, 0, 0]
    f["big_down_amt"] = out_big[:, 0, 1]
    f["big_up_price"] = out_big[:, 1, 0]
    f["big_up_amt"] = out_big[:, 1, 1]
    f["map_total_long"] = out_misc[:, 0]
    f["map_total_short"] = out_misc[:, 1]
    f["swept_long_usd"] = out_misc[:, 2]
    f["swept_short_usd"] = out_misc[:, 3]
    f["oi_btc"] = oi
    f["oi_usd"] = metrics.oi_usd.where(metrics.oi_valid)
    f["oi_valid"] = metrics.oi_valid
    return f


def add_imbalance(f: pd.DataFrame, cfg: dict, clip: float = np.log(50)) -> pd.DataFrame:
    """Ratio, direction, signed log imbalance for every window and every distance weighting."""
    eps = 1.0  # $1 floor so an empty side gives a huge-but-finite ratio
    specs = [wkey(w) for w in cfg["features"]["windows"]]
    for k in specs:
        d, u = f[f"L_down_{k}"], f[f"L_up_{k}"]
        _imb(f, k, d, u, eps, clip)
    for nm in ["inv_d", "inv_d2"] + [f"exp{int(x)}" for x in cfg["features"]["exp_k"]]:
        _imb(f, nm, f[f"DW_down_{nm}"], f[f"DW_up_{nm}"], eps, clip)
    return f


def _imb(f, k, d, u, eps, clip):
    dd, uu = d.clip(lower=eps), u.clip(lower=eps)
    f[f"ratio_raw_{k}"] = dd / uu
    f[f"ratio_{k}"] = np.maximum(dd, uu) / np.minimum(dd, uu)
    f[f"dir_{k}"] = np.where(uu > dd, 1, np.where(dd > uu, -1, 0))  # +1 = large pool UP
    f[f"sli_{k}"] = np.log(uu / dd).clip(-clip, clip)
    f[f"total_{k}"] = d + u
