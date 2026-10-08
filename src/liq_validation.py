"""Validate and calibrate the reconstructed liquidation map against REAL liquidations.

Real data: Tardis.dev free CSV (first day of each month, no API key), Binance USDT-M
BTCUSDT liquidation orders since 2021-09. side=sell -> a LONG was liquidated,
side=buy -> a SHORT was liquidated; amount is in BTC.
Caveat: since 2021 Binance's public liquidation stream pushes at most the latest
liquidation per symbol per second, so real volumes are a LOWER BOUND, worst in bursts.

PRE-REGISTERED criteria (fixed before looking at validation results):
  primary   Pearson corr of log1p(modelled swept USD) vs log1p(real liquidated USD),
            per 5m bar and side, pooled over both sides, on TEST days only.  pass >= 0.5
  secondary share of real liquidated USD that falls in bars where the model swept > 0.
            pass > 50 %
Additional (stricter, added on the same day before results): the model must add
information over a price-only baseline = OI x adverse excursion of the bar
(how far price moved toward that side's liquidations), measured as partial R^2.

Calibration: leverage-weight profiles x decay half-life x side split are fitted on
TRAIN days (odd months) and evaluated on TEST days (even months).
"""
from __future__ import annotations

import gzip
import io
import itertools
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "tardis_liquidations"
PROC = ROOT / "data" / "processed"
URL = "https://datasets.tardis.dev/v1/binance-futures/liquidations/{y}/{m:02d}/{d:02d}/BTCUSDT.csv.gz"

GRID_TIERS = [5, 10, 20, 50, 100]


# ----------------------------------------------------------------------------- real data

def validation_days(start="2021-09-01", end="2026-10-01") -> list[pd.Timestamp]:
    return list(pd.date_range(start, end, freq="MS", tz="UTC"))


def download(days) -> pd.DataFrame:
    RAW.mkdir(parents=True, exist_ok=True)

    def one(day):
        f = RAW / f"BTCUSDT_{day:%Y-%m-%d}.csv.gz"
        if f.exists() and f.stat().st_size > 0:
            return "cached"
        r = requests.get(URL.format(y=day.year, m=day.month, d=day.day), timeout=60)
        if r.status_code != 200:
            return f"http {r.status_code}"
        f.write_bytes(r.content)
        return "downloaded"

    with ThreadPoolExecutor(6) as ex:
        st = list(ex.map(one, days))
    log = pd.DataFrame({"day": days, "status": st})
    log.to_csv(RAW / "download_log.csv", index=False)
    return log


def load_real(days) -> pd.DataFrame:
    parts = []
    for day in days:
        f = RAW / f"BTCUSDT_{day:%Y-%m-%d}.csv.gz"
        if not f.exists():
            continue
        d = pd.read_csv(io.BytesIO(gzip.decompress(f.read_bytes())))
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    bad = ~(pd.to_numeric(df["price"], errors="coerce") > 0) | ~(pd.to_numeric(df["amount"], errors="coerce") > 0)
    load_real.n_dropped = int(bad.sum())  # recorded for the audit
    df = df[~bad]
    df["ts"] = pd.to_datetime(df["timestamp"], unit="us", utc=True)
    df["usd"] = df["price"].astype(float) * df["amount"].astype(float)
    df["liq_side"] = df["side"].map({"sell": "long", "buy": "short"})
    return df[["ts", "liq_side", "price", "amount", "usd"]].sort_values("ts").reset_index(drop=True)


def real_per_bar(real: pd.DataFrame, days) -> pd.DataFrame:
    """Real liquidated USD per 5m bar labelled by bar END (bar = [t-5m, t)), every bar of every day."""
    idx = pd.DatetimeIndex([])
    for d in days:
        idx = idx.append(pd.date_range(d + pd.Timedelta(minutes=5), d + pd.Timedelta(days=1), freq="5min"))
    lab = real.ts.dt.floor("5min") + pd.Timedelta(minutes=5)
    g = real.assign(lab=lab).pivot_table(index="lab", columns="liq_side", values="usd", aggfunc="sum").fillna(0.0)
    out = pd.DataFrame(index=idx)
    out["real_long"] = g.get("long", pd.Series(dtype=float)).reindex(idx).fillna(0.0)
    out["real_short"] = g.get("short", pd.Series(dtype=float)).reindex(idx).fillna(0.0)
    return out


# ----------------------------------------------------------------------------- light engine

@njit(cache=True)
def _sweeps(high, low, typ, doi, oi_prev, valid, long_share, decay, lnmin, step, nbins,
            lev, levw, mmr, flag, out_bin_t, out_bin_i, out_bin_side, out_bin_v):
    """Same mechanics as liquidation_features._run_map but only returns per-step swept USD.
    Lazy global scale g implements pro-rata closes and decay in O(1) per step.
    For flagged steps it also records the swept amount per price bin (sparse)."""
    n = high.shape[0]
    L = np.zeros(nbins)
    S = np.zeros(nbins)
    g = 1.0
    hiL = -1
    loS = nbins
    swl = np.zeros(n)
    sws = np.zeros(n)
    k = 0
    cap = out_bin_t.shape[0]
    for t in range(n):
        if not np.isfinite(high[t]):
            continue
        g *= decay
        ilo = int((np.log(low[t]) - lnmin) / step)
        ihi = int((np.log(high[t]) - lnmin) / step)
        a = 0.0
        if ilo < 0:
            ilo = 0
        for i in range(ilo, hiL + 1):
            if L[i] != 0.0:
                v = L[i] * g
                a += v
                if flag[t] and k < cap:
                    out_bin_t[k] = t; out_bin_i[k] = i; out_bin_side[k] = 0; out_bin_v[k] = v; k += 1
                L[i] = 0.0
        if ilo <= hiL:
            hiL = ilo - 1
        b = 0.0
        top = min(ihi, nbins - 1)
        for i in range(loS, top + 1):
            if S[i] != 0.0:
                v = S[i] * g
                b += v
                if flag[t] and k < cap:
                    out_bin_t[k] = t; out_bin_i[k] = i; out_bin_side[k] = 1; out_bin_v[k] = v; k += 1
                S[i] = 0.0
        if top >= loS:
            loS = top + 1
        swl[t] = a
        sws[t] = b
        if valid[t]:
            p = typ[t]
            if doi[t] > 0:
                notional = doi[t] * p
                la = 2.0 * notional * long_share[t] / g
                sa = 2.0 * notional * (1.0 - long_share[t]) / g
                for j in range(lev.shape[0]):
                    if levw[j] == 0.0:
                        continue
                    bl = int((np.log(p * (1.0 - 1.0 / lev[j] + mmr)) - lnmin) / step)
                    bs = int((np.log(p * (1.0 + 1.0 / lev[j] - mmr)) - lnmin) / step)
                    if 0 <= bl < nbins:
                        L[bl] += la * levw[j]
                        if bl > hiL:
                            hiL = bl
                    if 0 <= bs < nbins:
                        S[bs] += sa * levw[j]
                        if bs < loS:
                            loS = bs
            elif doi[t] < 0 and oi_prev[t] > 0:
                net = -doi[t] * p - 0.5 * (a + b)
                if net > 0:
                    g *= 1.0 - min(net / (oi_prev[t] * p), 1.0)
        if g < 1e-100:  # renormalise
            for i in range(nbins):
                L[i] *= g
                S[i] *= g
            g = 1.0
    return swl, sws, k


class EngineInputs:
    def __init__(self, cfg):
        bars5 = pd.read_parquet(PROC / "bars_5m.parquet")
        m = pd.read_parquet(PROC / "metrics_5m_clean.parquet")
        b = bars5.reindex(m.index)
        oi = m.oi_btc.where(m.oi_valid)
        last = oi.ffill().shift(1)
        self.index = m.index
        self.high, self.low = b.high.to_numpy(float), b.low.to_numpy(float)
        self.typ, self.close = b.typical.to_numpy(float), b.close.to_numpy(float)
        self.doi = (oi - last).where(m.oi_valid, 0.0).fillna(0.0).to_numpy(float)
        self.oi_prev = last.fillna(0).to_numpy(float)
        self.valid = m.oi_valid.to_numpy(bool)
        r = m.acct_ls_ratio.where(m.oi_valid).ffill(limit=3).fillna(1.0)
        self.ls_acct = (r / (1 + r)).clip(0.2, 0.8).to_numpy()
        self.ls_sym = np.full(len(m), 0.5)
        self.oi_usd = m.oi_usd.where(m.oi_valid).ffill(limit=3).to_numpy(float)
        lm = cfg["liquidation_map"]
        self.lnmin = np.log(lm["grid_min_price"])
        self.step = lm["grid_step_log"]
        self.nbins = int((np.log(lm["grid_max_price"]) - self.lnmin) / self.step) + 1
        self.mmr = lm["maintenance_margin"]

    def run(self, tiers, weights, half_life_h, split, flag=None, cap=1):
        decay = 0.5 ** (5 / 60 / half_life_h) if half_life_h else 1.0
        flag = np.zeros(len(self.index), bool) if flag is None else flag
        bt = np.zeros(cap, np.int64); bi = np.zeros(cap, np.int64); bs = np.zeros(cap, np.int8); bv = np.zeros(cap)
        ls = self.ls_sym if split == "symmetric" else self.ls_acct
        swl, sws, k = _sweeps(self.high, self.low, self.typ, self.doi, self.oi_prev, self.valid, ls, decay,
                              self.lnmin, self.step, self.nbins, np.array(tiers, float),
                              np.array(weights, float) / np.sum(weights), self.mmr, flag, bt, bi, bs, bv)
        sparse = pd.DataFrame({"t": bt[:k], "bin": bi[:k], "side": bs[:k], "usd": bv[:k]}) if cap > 1 else None
        return pd.Series(swl, index=self.index), pd.Series(sws, index=self.index), sparse


# ----------------------------------------------------------------------------- metrics

def baseline(eng: EngineInputs) -> pd.DataFrame:
    """Price-only baseline: OI x adverse excursion of the bar relative to the previous close."""
    prev = pd.Series(eng.close, index=eng.index).shift(1)
    dn = (prev - pd.Series(eng.low, index=eng.index)).clip(lower=0) / prev
    up = (pd.Series(eng.high, index=eng.index) - prev).clip(lower=0) / prev
    oi = pd.Series(eng.oi_usd, index=eng.index)
    return pd.DataFrame({"base_long": dn * oi, "base_short": up * oi})


def score(df: pd.DataFrame, mcol_long="mod_long", mcol_short="mod_short") -> dict:
    """df has real_long/real_short, model and baseline columns for a set of bars."""
    y = np.log1p(np.concatenate([df.real_long, df.real_short]))
    x = np.log1p(np.concatenate([df[mcol_long], df[mcol_short]]))
    z = np.log1p(np.concatenate([df.base_long, df.base_short]))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    y, x, z = y[ok], x[ok], z[ok]
    r = np.corrcoef(x, y)[0, 1]
    rs = pd.Series(x).corr(pd.Series(y), method="spearman")
    rb = np.corrcoef(z, y)[0, 1]
    # partial R^2 of model over baseline
    Xb = np.column_stack([np.ones_like(z), z])
    Xf = np.column_stack([np.ones_like(z), z, x])
    rss_b = np.sum((y - Xb @ np.linalg.lstsq(Xb, y, rcond=None)[0]) ** 2)
    rss_f = np.sum((y - Xf @ np.linalg.lstsq(Xf, y, rcond=None)[0]) ** 2)
    real_l, real_s = df.real_long.to_numpy(), df.real_short.to_numpy()
    hit = (real_l * (df[mcol_long].to_numpy() > 0)).sum() + (real_s * (df[mcol_short].to_numpy() > 0)).sum()
    tot = real_l.sum() + real_s.sum()
    bars_with_real = ((real_l > 0) | (real_s > 0)).mean()
    return {"pearson_log": r, "spearman": rs, "baseline_pearson_log": rb,
            "partial_R2_over_baseline": 1 - rss_f / rss_b, "R2_baseline": 1 - rss_b / np.sum((y - y.mean()) ** 2),
            "share_real_usd_in_model_sweep_bars": hit / tot if tot > 0 else np.nan,
            "share_bars_model_sweep": float(((df[mcol_long] > 0) | (df[mcol_short] > 0)).mean()),
            "share_bars_with_real_liq": float(bars_with_real), "n_bar_sides": int(ok.sum()),
            "real_usd_total": float(tot)}


def weight_grid(step=0.25):
    units = int(round(1 / step))
    out = []
    for c in itertools.product(range(units + 1), repeat=len(GRID_TIERS)):
        if sum(c) == units:
            out.append(tuple(x / units for x in c))
    return out


def level_test(sparse: pd.DataFrame, real: pd.DataFrame, eng: EngineInputs, bin_width_log=0.0025) -> pd.DataFrame:
    """Within bars where the model swept something AND real liquidations happened on the same side:
    do real liquidation prices concentrate in the price buckets where the model put its swept
    notional? Lift = share of real USD in model-occupied buckets / share of the traversed range
    those buckets cover. Coarse 0.25% buckets because Binance reports the liquidation ORDER price,
    which sits beyond the trigger (bankruptcy price)."""
    sp = sparse.copy()
    sp["ts"] = eng.index[sp.t.to_numpy()]
    sp["price"] = np.exp(eng.lnmin + (sp["bin"] + 0.5) * eng.step)
    real = real.copy()
    real["ts_bar"] = real.ts.dt.floor("5min") + pd.Timedelta(minutes=5)
    real["side"] = (real.liq_side == "short").astype(int)
    rows = []
    hi = pd.Series(eng.high, index=eng.index)
    lo = pd.Series(eng.low, index=eng.index)
    real_groups = {k: v for k, v in real.groupby(["ts_bar", "side"])}
    for (ts, side), g in sp.groupby(["ts", "side"]):
        r = real_groups.get((ts, side))
        if r is None:
            continue
        h, l = hi.at[ts], lo.at[ts]
        # extend the range by 1% to allow for order-price vs trigger offset
        lo_e, hi_e = np.log(l * 0.99), np.log(h * 1.01)
        nb = max(int(np.ceil((hi_e - lo_e) / bin_width_log)), 1)
        mb = np.clip(((np.log(g.price) - lo_e) / bin_width_log).astype(int), 0, nb - 1)
        rb = np.clip(((np.log(r.price.astype(float)) - lo_e) / bin_width_log).astype(int), 0, nb - 1)
        occupied = np.zeros(nb, bool)
        occupied[np.unique(mb)] = True
        # allow one bucket of slack toward the bankruptcy side
        occ2 = occupied.copy()
        if side == 0:
            occ2[:-1] |= occupied[1:]
        else:
            occ2[1:] |= occupied[:-1]
        share_real = r.usd[occ2[rb]].sum() / r.usd.sum()
        rows.append({"ts": ts, "side": "long" if side == 0 else "short", "share_real_in_model_buckets": share_real,
                     "share_range_model_buckets": occ2.mean(), "real_usd": r.usd.sum(), "model_usd": g.usd.sum()})
    return pd.DataFrame(rows)


def baseline_breakout(eng: EngineInputs, lookback_bars: int = 288) -> pd.DataFrame:
    """Stronger price-only baseline (no map): how far the bar broke the prior 24h low/high,
    times OI - the classic 'stop run' proxy."""
    hi = pd.Series(eng.high, index=eng.index)
    lo = pd.Series(eng.low, index=eng.index)
    ph = hi.shift(1).rolling(lookback_bars, min_periods=lookback_bars // 2).max()
    pl = lo.shift(1).rolling(lookback_bars, min_periods=lookback_bars // 2).min()
    oi = pd.Series(eng.oi_usd, index=eng.index)
    return pd.DataFrame({"brk_long": (pl - lo).clip(lower=0) / pl * oi,
                         "brk_short": (hi - ph).clip(lower=0) / ph * oi})


def incremental_over(df: pd.DataFrame, base_cols: list[tuple[str, str]], mcols=("mod_long", "mod_short")) -> dict:
    """Partial R^2 of log1p(model) over a set of log1p baselines (pooled sides) + side dummy."""
    y = np.log1p(np.concatenate([df.real_long, df.real_short]))
    x = np.log1p(np.concatenate([df[mcols[0]], df[mcols[1]]]))
    Z = [np.log1p(np.concatenate([df[a], df[b]])) for a, b in base_cols]
    side = np.r_[np.zeros(len(df)), np.ones(len(df))]
    Xb = np.column_stack([np.ones_like(y), side] + Z)
    ok = np.all(np.isfinite(Xb), axis=1) & np.isfinite(x) & np.isfinite(y)
    y, x, Xb = y[ok], x[ok], Xb[ok]
    Xf = np.column_stack([Xb, x])
    tss = np.sum((y - y.mean()) ** 2)
    rss_b = np.sum((y - Xb @ np.linalg.lstsq(Xb, y, rcond=None)[0]) ** 2)
    rss_f = np.sum((y - Xf @ np.linalg.lstsq(Xf, y, rcond=None)[0]) ** 2)
    return {"R2_baselines": 1 - rss_b / tss, "R2_baselines_plus_model": 1 - rss_f / tss,
            "partial_R2_model": 1 - rss_f / rss_b}
