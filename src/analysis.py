"""Stage 5-10: descriptive stats, first passage, event tables, results matrix,
robustness (methods, regimes, size, change, extremes, time), cluster analysis.
All tables are written to results/tables/. Event frames to data/processed/events/."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as ss

from . import events_table as et
from . import statistics as st
from .liquidation_features import wkey
from .price_outcomes import PricePath, fp_outcome

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
EVD = ROOT / "data" / "processed" / "events"
H_MAIN = [60, 240, 720, 1440, 4320, 10080]
TH_MAIN = ["0p5", "1", "2", "3", "5", "7p5", "10"]
TH_PCT = {"0p5": 0.5, "1": 1, "2": 2, "3": 3, "5": 5, "7p5": 7.5, "10": 10}


def all_specs(cfg):
    return [wkey(w) for w in cfg["features"]["windows"]] + ["inv_d", "inv_d2"] + \
        [f"exp{int(k)}" for k in cfg["features"]["exp_k"]]


def blk(n):
    return int(max(2, min(10, n // 5)))


# ----------------------------------------------------------------------------- summary rows

def summary_row(E: pd.DataFrame, cfg: dict, n_boot: int) -> dict:
    n = len(E)
    r = {"N_events": n, "large_pool_up_N": int((E.dir > 0).sum()), "large_pool_down_N": int((E.dir < 0).sum()),
         "sample_flag": st.size_flag(n)}
    if n < 5:
        return r
    b = blk(n)
    gm = st.month_groups(E.index)
    r["n_months"] = int(len(np.unique(gm)))
    for tag in ["1", "2", "3", "5"]:
        s = st.summarize_fp(E[f"ylarge_{tag}"].to_numpy(), E[f"by_{tag}"].to_numpy(), n_boot, b, groups=gm)
        r[f"P_large_first_{tag}pct"] = s.get("p_large_first")
        r[f"N_resolved_{tag}pct"] = s.get("N_resolved")
        r[f"p_binom_{tag}pct"] = s.get("p_binom_vs_50")
        if tag == "2":
            r["P2_cluster_ci"] = f"[{s.get('cluster_ci_lo', np.nan):.3f}, {s.get('cluster_ci_hi', np.nan):.3f}]"
            r["P2_p_cluster_vs_50"] = s.get("p_cluster_vs_50")
            r["P2_benchmark"] = s.get("p_benchmark")
            r["P2_p_cluster_vs_benchmark"] = s.get("p_cluster_vs_benchmark")
            r["ties_2pct"] = int(E["tie_2"].sum())
            r["censored_2pct"] = int(E["cens_2"].sum())
    for h, lab in [(240, "4h"), (720, "12h"), (1440, "24h"), (4320, "72h"), (10080, "7d")]:
        r[f"median_DAR_{lab}"] = float(np.nanmedian(E[f"dar_{h}"]))
    s = st.summarize_dar(E["dar_1440"].to_numpy(), E["bdar_1440"].to_numpy(), n_boot, b, groups=gm)
    r["mean_DAR_24h"] = s.get("mean")
    r["mean_DAR_24h_cluster_ci"] = f"[{s.get('mean_clci_lo', np.nan):.4f}, {s.get('mean_clci_hi', np.nan):.4f}]"
    r["p_value_DAR24_cluster"] = s.get("p_cluster")
    r["p_value_DAR24_wilcoxon"] = s.get("p_wilcoxon")
    r["excess_DAR24_vs_uncond"] = s.get("mean_excess_vs_uncond")
    r["median_range_24h"] = float(np.nanmedian(E["range_1440"]))
    r["median_fwd_rv_24h"] = float(np.nanmedian(E["fwd_rv_24h"]))
    r["median_past_rv_24h"] = float(np.nanmedian(E["rv_24h"]))
    r["cascade_large_dir_24h"] = float(np.nanmean(E["casc_large_1440"]))
    r["cascade_small_dir_24h"] = float(np.nanmean(E["casc_small_1440"]))
    r["median_mfe_large_24h"] = float(np.nanmedian(E["mfe_large_1440"]))
    r["median_mfe_small_24h"] = float(np.nanmedian(E["mfe_small_1440"]))
    r["median_accel_large_24h"] = float(np.nanmedian(E["accel_large_24h"]))
    r["median_accel_small_24h"] = float(np.nanmedian(E["accel_small_24h"]))
    return r


def table_by(E: pd.DataFrame, by: list[str], cfg, n_boot, order=None) -> pd.DataFrame:
    rows = []
    for key, g in E.groupby(by, observed=True, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), **summary_row(g.sort_index(), cfg, n_boot)})
    t = pd.DataFrame(rows)
    if order is not None and len(t):
        t["_o"] = t[by[0]].map({v: i for i, v in enumerate(order)})
        t = t.sort_values(["_o"] + by[1:]).drop(columns="_o")
    return t


# ----------------------------------------------------------------------------- event sets

def bucket_events(P, k, cfg, B, cooldown_h=24):
    labs = et.bucket_labels(cfg["ratio_buckets"])
    b = et.assign_bucket(P[f"ratio_{k}"].astype(float), cfg["ratio_buckets"]).astype(str)
    out = []
    for lab in labs:
        idx = et.event_index(P, k, "B", cond=(b == lab), cooldown_h=cooldown_h)
        out.append(et.make_events(P, idx, k, cfg, B, label=f"B{cooldown_h}_bucket"))
    return pd.concat(out).sort_index()


def core_events(P, k, cfg, B, method="B", cooldown_h=24, lo=2.5, hi=5.0):
    r = P[f"ratio_{k}"].astype(float)
    if method == "B":
        idx = et.event_index(P, k, "B", cond=(r >= lo) & (r <= hi), cooldown_h=cooldown_h)
        E = et.make_events(P, idx, k, cfg, B, label=f"B{cooldown_h}_core")
    elif method == "A":
        idx = et.event_index(P, k, "A", enter=lo)
        E = et.make_events(P, idx, k, cfg, B, label="A_cross")
        E = E[(E.ratio >= lo) & (E.ratio <= hi)]
    elif method in ("C_start", "C_peak"):
        ep = et.episode_table(P, k, lo)
        ep = ep[~ep.left_censored]
        if method == "C_start":
            ep = ep[(ep.initial_ratio >= lo) & (ep.initial_ratio <= hi)]
            idx = ep.start
        else:
            ep = ep[(ep.max_ratio >= lo) & (ep.max_ratio <= hi)]
            idx = ep.peak_time
        E = et.make_events(P, pd.DatetimeIndex(idx), k, cfg, B, label=method)
    return E


# ----------------------------------------------------------------------------- sections

def section_core(P, k, cfg, B, variant, n_boot):
    E = bucket_events(P, k, cfg, B)
    EVD.mkdir(parents=True, exist_ok=True)
    E.to_parquet(EVD / f"events_B24_bucket_{variant}_{k}.parquet")
    order = et.bucket_labels(cfg["ratio_buckets"])
    core = table_by(E, ["bucket"], cfg, n_boot, order)
    C = core_events(P, k, cfg, B)
    C.to_parquet(EVD / f"events_B24_core_{variant}_{k}.parquet")
    row = {"bucket": "CORE 2.5-5 (pooled, B24)", **summary_row(C, cfg, n_boot)}
    core = pd.concat([core, pd.DataFrame([row])], ignore_index=True)
    core["core_bucket"] = core.bucket.isin(cfg["core_buckets"] + ["CORE 2.5-5 (pooled, B24)"])
    asym = table_by(E, ["large_dir", "bucket"], cfg, n_boot)
    asym["_o"] = asym.bucket.map({v: i for i, v in enumerate(order)})
    asym = asym.sort_values(["large_dir", "_o"]).drop(columns="_o")
    ac = table_by(C, ["large_dir"], cfg, n_boot)
    ac.insert(1, "bucket", "CORE 2.5-5 (pooled, B24)")
    asym = pd.concat([asym, ac], ignore_index=True)
    return E, C, core, asym


def section_matrix(P, specs, cfg, B, variant, n_boot):
    """Full grid: spec x bucket x horizon (DAR) and spec x bucket x threshold (first passage)."""
    rows_d, rows_f = [], []
    for k in specs:
        E = bucket_events(P, k, cfg, B)
        for lab, g in E.groupby("bucket", observed=True):
            g = g.sort_index()
            gm = st.month_groups(g.index)
            for h in H_MAIN:
                s = st.summarize_dar(g[f"dar_{h}"].to_numpy(), g[f"bdar_{h}"].to_numpy(), n_boot, blk(len(g)), groups=gm)
                rows_d.append({"variant": variant, "spec": k, "bucket": lab, "horizon": et.H_LABEL[h], **s})
            for tag in TH_MAIN:
                s = st.summarize_fp(g[f"ylarge_{tag}"].to_numpy(), g[f"by_{tag}"].to_numpy(), n_boot, blk(len(g)), groups=gm)
                rows_f.append({"variant": variant, "spec": k, "bucket": lab, "threshold_pct": TH_PCT[tag],
                               "N_events": len(g), **s})
    return pd.DataFrame(rows_d), pd.DataFrame(rows_f)


def finalize_matrix(D, F, specs, cfg):
    order_b = et.bucket_labels(cfg["ratio_buckets"])
    win_specs = [s for s in specs if s.startswith("w")]
    D["q_fdr"] = st.bh_fdr(D["p_cluster"])
    F["eff"] = F["p_large_first"] - 0.5
    F["q_fdr"] = st.bh_fdr(F["p_cluster_vs_50"])
    # Neighbours = adjacent window and adjacent ratio bucket at the SAME horizon/threshold.
    # Adjacent horizons/thresholds reuse the same events, so they are not independent support.
    for M, val, pcol, ax3 in [(D, "mean", "p_cluster", "horizon"), (F, "eff", "p_cluster_vs_50", "threshold_pct")]:
        M["stability"] = ""
        wmask = M.spec.isin(win_specs)
        M.loc[wmask, "stability"] = st.stability_flags(M[wmask].reset_index(drop=True), val, pcol,
                                                       {"spec": win_specs, "bucket": order_b}, fixed=[ax3]).to_numpy()
        dmask = ~wmask
        if dmask.any():
            M.loc[dmask, "stability"] = st.stability_flags(M[dmask].reset_index(drop=True), val, pcol,
                                                           {"bucket": order_b}, fixed=[ax3, "spec"]).to_numpy()
    return D, F


def section_methods(P, k, cfg, B, n_boot):
    rows = []
    sets = [("A crossing (enter 2.5 / exit 2.0)", dict(method="A"))]
    sets += [(f"B cooldown {h}h", dict(method="B", cooldown_h=h)) for h in cfg["events"]["cooldowns_h"]]
    sets += [("C episode start", dict(method="C_start")), ("C episode peak (EX-POST, not tradable)", dict(method="C_peak"))]
    for name, kw in sets:
        E = core_events(P, k, cfg, B, **kw)
        rows.append({"event_definition": name, "subset": "CORE 2.5-5", **summary_row(E, cfg, n_boot)})
        for lab in cfg["core_buckets"]:
            g = E[E.bucket == lab]
            rows.append({"event_definition": name, "subset": lab, **summary_row(g, cfg, n_boot)})
        for dname in ["UP", "DOWN"]:
            g = E[E.large_dir == dname]
            rows.append({"event_definition": name, "subset": f"CORE large={dname}", **summary_row(g, cfg, n_boot)})
    ep = et.episode_table(P, k, 2.5)
    return pd.DataFrame(rows), ep


def section_ladder(P, k, cfg, B, n_boot):
    rows = []
    for th in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.5]:
        idx = et.event_index(P, k, "A", enter=th)
        E = et.make_events(P, idx, k, cfg, B, label=f"A_{th}")
        rows.append({"crossing_threshold": th, "exit": 0.8 * th, **summary_row(E, cfg, n_boot)})
    return pd.DataFrame(rows)


def section_monotonic(Eb: pd.DataFrame, cfg) -> pd.DataFrame:
    """Does direction predictability change monotonically with ratio? (all-bucket B24 events)"""
    import statsmodels.api as sm
    rows = []
    order = et.bucket_labels(cfg["ratio_buckets"])
    E = Eb.sort_index()
    E = E.assign(bidx=E.bucket.map({v: i for i, v in enumerate(order)}), lr=np.log(E.ratio.astype(float)))
    for tag in ["1", "2", "3", "5"]:
        g = E[np.isfinite(E[f"ylarge_{tag}"])]
        X = sm.add_constant(g[["lr"]].astype(float))
        m = sm.GLM(g[f"ylarge_{tag}"].astype(float), X, family=sm.families.Binomial()).fit(
            cov_type="HAC", cov_kwds={"maxlags": 7})
        bp = g.groupby("bidx")[f"ylarge_{tag}"].mean()
        rho = ss.spearmanr(bp.index, bp.values)
        rows.append({"outcome": f"P(large first) {tag}%", "logit_coef_log_ratio": m.params["lr"],
                     "z_HAC": m.tvalues["lr"], "p_HAC": m.pvalues["lr"], "N": len(g),
                     "spearman_bucket_rho": rho.statistic, "spearman_p": rho.pvalue})
    for h in [240, 1440, 4320]:
        g = E[np.isfinite(E[f"dar_{h}"])]
        X = sm.add_constant(g[["lr"]].astype(float))
        m = sm.OLS(g[f"dar_{h}"].astype(float), X).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
        rows.append({"outcome": f"DAR {et.H_LABEL[h]}", "ols_coef_log_ratio": m.params["lr"],
                     "z_HAC": m.tvalues["lr"], "p_HAC": m.pvalues["lr"], "N": len(g)})
    g = E[np.isfinite(E["range_1440"])]
    X = sm.add_constant(g[["lr"]].astype(float))
    m = sm.OLS(np.log(g["range_1440"].astype(float)), X).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    rows.append({"outcome": "log range 24h (unconditional on past vol)", "ols_coef_log_ratio": m.params["lr"],
                 "z_HAC": m.tvalues["lr"], "p_HAC": m.pvalues["lr"], "N": len(g)})
    return pd.DataFrame(rows)


def section_regimes(C, cfg, n_boot):
    tabs = []
    for col in ["trend", "vol_regime", "funding_regime", "oi_regime"]:
        t = table_by(C.dropna(subset=[col]), [col, "large_dir"], cfg, n_boot)
        t.insert(0, "regime_type", col)
        t = t.rename(columns={col: "regime"})
        tabs.append(t)
        t2 = table_by(C.dropna(subset=[col]), [col], cfg, n_boot)
        t2.insert(0, "regime_type", col)
        t2 = t2.rename(columns={col: "regime"})
        t2["large_dir"] = "ALL"
        tabs.append(t2)
    crowd = C[(C.large_dir == "DOWN") & (C.funding_regime == "positive") & (C.oi_regime == "oi_rising")]
    rest = C[(C.large_dir == "DOWN") & ~((C.funding_regime == "positive") & (C.oi_regime == "oi_rising"))]
    sq = C[(C.large_dir == "UP") & (C.funding_regime == "negative") & (C.oi_regime == "oi_rising")]
    tabs.append(pd.DataFrame([
        {"regime_type": "combo", "regime": "long-crowded: large DOWN + funding>0.01% + OI rising", "large_dir": "DOWN", **summary_row(crowd, cfg, n_boot)},
        {"regime_type": "combo", "regime": "large DOWN, not long-crowded", "large_dir": "DOWN", **summary_row(rest, cfg, n_boot)},
        {"regime_type": "combo", "regime": "short-crowded: large UP + funding<0 + OI rising", "large_dir": "UP", **summary_row(sq, cfg, n_boot)},
    ]))
    return pd.concat(tabs, ignore_index=True)


SIZE_BINS = [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0001]
SIZE_LABS = ["0-25%", "25-50%", "50-75%", "75-90%", "90-95%", "95-99%", "99%+"]
RGROUP_EDGES = [1.0, 1.5, 2.5, 3.5, 5.0, np.inf]
RGROUP_LABS = ["1-1.5", "1.5-2.5", "2.5-3.5", "3.5-5", ">5"]


def section_size(P, k, cfg, B, n_boot):
    """ratio group x absolute-size percentile (trailing 365d, causal), B24 per cell."""
    r = P[f"ratio_{k}"].astype(float)
    rg = pd.cut(r, RGROUP_EDGES, right=False, labels=RGROUP_LABS).astype(str)
    sz = pd.cut(P[f"total_pct365_{k}"].astype(float), SIZE_BINS, right=False, labels=SIZE_LABS).astype(str)
    rows = []
    for a in RGROUP_LABS:
        for b in SIZE_LABS:
            idx = et.event_index(P, k, "B", cond=(rg == a) & (sz == b), cooldown_h=24)
            E = et.make_events(P, idx, k, cfg, B)
            rows.append({"ratio_group": a, "size_pct365": b, **summary_row(E, cfg, n_boot)})
    # coarse: high ratio x small vs large size
    for a, lo_hi in [("2.5-5", (2.5, 5.0)), (">5", (5.0, np.inf))]:
        for b, cond_s in [("size<50%", P[f"total_pct365_{k}"] < 0.5), ("size>=90%", P[f"total_pct365_{k}"] >= 0.9)]:
            idx = et.event_index(P, k, "B", cond=(r >= lo_hi[0]) & (r < lo_hi[1]) & cond_s, cooldown_h=24)
            E = et.make_events(P, idx, k, cfg, B)
            rows.append({"ratio_group": f"COARSE {a}", "size_pct365": b, **summary_row(E, cfg, n_boot)})
    return pd.DataFrame(rows)


def section_change(C, cfg, n_boot):
    """Level vs speed of change. C = core events (2.5-5)."""
    rows = []
    defs = {
        "fast build: |log imbalance| rose >= 0.5 in past 1h": C["dabs_1h"] >= 0.5,
        "fast build: rose >= 0.7 in past 4h": C["dabs_4h"] >= 0.7,
        "stable: |change| < 0.2 over past 24h": C["dabs_24h"].abs() < 0.2,
        "building over 24h: rose >= 0.5": C["dabs_24h"] >= 0.5,
        "fading over 24h: fell <= -0.3": C["dabs_24h"] <= -0.3,
    }
    for name, m in defs.items():
        rows.append({"state": name, **summary_row(C[m.fillna(False)], cfg, n_boot)})
    for lag in ["15m", "1h", "4h", "12h", "24h"]:
        q = pd.qcut(C[f"dabs_{lag}"].astype(float), 3, labels=["low", "mid", "high"])
        for lab in ["low", "mid", "high"]:
            rows.append({"state": f"tercile of d|sli| {lag}: {lab}", **summary_row(C[q == lab], cfg, n_boot)})
    return pd.DataFrame(rows)


def section_extreme(C, ext: pd.DataFrame, cfg, n_boot):
    xd = set(ext.index[ext.extreme].normalize())
    day = C.index.normalize()
    prev = day - pd.Timedelta(days=1)
    ex_ante = np.array([(d in xd) or (p in xd) for d, p in zip(day, prev)])
    t0 = pd.DatetimeIndex(C.t0)
    ex_post = np.array([any(((t + pd.Timedelta(days=j)).normalize() in xd) for j in range(0, 4)) for t in t0])
    rows = [{"sample": "all events", **summary_row(C, cfg, n_boot)},
            {"sample": "excl. events on/after an extreme day (ex-ante filter)", **summary_row(C[~ex_ante], cfg, n_boot)},
            {"sample": "excl. events whose next 72h touch an extreme day (ex-post filter)", **summary_row(C[~ex_post], cfg, n_boot)},
            {"sample": "ONLY events whose next 72h touch an extreme day", **summary_row(C[ex_post], cfg, n_boot)}]
    for dname in ["UP", "DOWN"]:
        m = C.large_dir == dname
        rows.append({"sample": f"large={dname} all", **summary_row(C[m], cfg, n_boot)})
        rows.append({"sample": f"large={dname} excl. ex-post extreme", **summary_row(C[m & ~ex_post], cfg, n_boot)})
    t = pd.DataFrame(rows)
    t.attrs["n_extreme_days"] = len(xd)
    return t


def section_time(C, Eb, cfg, n_boot):
    yr = table_by(C, ["year"], cfg, n_boot)
    yr2 = table_by(C, ["year", "large_dir"], cfg, n_boot)
    roll = []
    months = pd.date_range(C.index.min().normalize(), C.index.max(), freq="MS", tz="UTC")
    for win in [6, 12]:
        for m in months:
            lo = m - pd.DateOffset(months=win)
            g = C[(C.index >= lo) & (C.index < m)]
            if len(g) < 10:
                continue
            y = g["ylarge_2"].dropna()
            roll.append({"window_months": win, "end": m, "N": len(g), "P_large_first_2pct": y.mean(),
                         "P_ci_halfwidth": 1.96 * np.sqrt(y.mean() * (1 - y.mean()) / max(len(y), 1)),
                         "mean_DAR_24h": g.dar_1440.mean(), "median_DAR_24h": g.dar_1440.median()})
    return yr, yr2, pd.DataFrame(roll)


def section_clusters(C: pd.DataFrame, pp: PricePath, cfg, n_boot):
    """First touch of liquidation clusters at their actual prices, with distance control.
    Benchmark: driftless random walk -> P(hit level a before b) = d_b / (d_a + d_b) in log distance."""
    rows = []
    W = cfg["outcomes"]["fp_window_min"]
    P0 = C.P0.to_numpy(float)
    d = C.dir.to_numpy(float)
    per_event = []
    for lev in ["cl25", "cl50", "cl75", "big"]:
        up = C[f"{lev}_up_price"].to_numpy(float)
        dn = C[f"{lev}_down_price"].to_numpy(float)
        ok = np.isfinite(up) & np.isfinite(dn) & (up > P0) & (dn < P0)
        tu = pp.first_hit(C.t0[ok], up[ok], W, True).astype(float)
        td = pp.first_hit(C.t0[ok], dn[ok], W, False).astype(float)
        fp = fp_outcome(tu, td, W)
        dd = d[ok]
        y = np.where(np.isnan(fp) | (fp == 0), np.nan, (fp == dd).astype(float))
        du = np.log(up[ok] / P0[ok])
        ddn = np.log(P0[ok] / dn[ok])
        d_large = np.where(dd > 0, du, ddn)
        d_small = np.where(dd > 0, ddn, du)
        bm = d_small / (d_large + d_small)
        t_large = np.where(dd > 0, tu, td)
        t_small = np.where(dd > 0, td, tu)
        for sub, m in [("ALL", np.ones(ok.sum(), bool)), ("large=UP", dd > 0), ("large=DOWN", dd < 0)]:
            s = st.summarize_fp(y[m], bm[m], n_boot, blk(int(m.sum())), groups=st.month_groups(C.index[ok][m]))
            hit_l = [(np.isfinite(t_large[m]) & (t_large[m] >= 0) & (t_large[m] < h)).mean() for h in (1440, 4320, 10080)]
            hit_s = [(np.isfinite(t_small[m]) & (t_small[m] >= 0) & (t_small[m] < h)).mean() for h in (1440, 4320, 10080)]
            tl = t_large[m][t_large[m] >= 0]
            ts = t_small[m][t_small[m] >= 0]
            rows.append({"level": lev, "subset": sub, "N": int(m.sum()), **s,
                         "median_dist_large_pct": 100 * np.median(np.exp(d_large[m]) - 1),
                         "median_dist_small_pct": 100 * np.median(np.exp(d_small[m]) - 1),
                         "P_reach_large_24h": hit_l[0], "P_reach_large_72h": hit_l[1], "P_reach_large_7d": hit_l[2],
                         "P_reach_small_24h": hit_s[0], "P_reach_small_72h": hit_s[1], "P_reach_small_7d": hit_s[2],
                         "median_time_to_large_h": np.median(tl) / 60 if len(tl) else np.nan,
                         "median_time_to_small_h": np.median(ts) / 60 if len(ts) else np.nan})
        per_event.append(pd.DataFrame({"t_snap": C.index[ok], "level": lev, "dir": dd, "y_large_first": y,
                                       "bm_prob": bm, "log_dist_large": d_large, "log_dist_small": d_small,
                                       "t_large_min": t_large, "t_small_min": t_small}))
    T = pd.DataFrame(rows)
    PE = pd.concat(per_event, ignore_index=True)
    # logistic with distance control
    import statsmodels.api as sm
    lr = []
    for lev, g in PE.groupby("level"):
        g = g[np.isfinite(g.y_large_first)].sort_values("t_snap")
        X = pd.DataFrame({"log_dist_ratio": np.log(g.log_dist_large / g.log_dist_small),
                          "large_up": (g.dir > 0).astype(float)})
        X = sm.add_constant(X)
        m = sm.GLM(g.y_large_first, X, family=sm.families.Binomial()).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
        # intercept = log-odds of hitting the large side first when distances are EQUAL and large=DOWN
        lr.append({"level": lev, "N": len(g), **{f"coef_{c}": m.params[c] for c in X.columns},
                   **{f"p_{c}": m.pvalues[c] for c in X.columns},
                   "P_large_first_at_equal_distance_largeDOWN": 1 / (1 + np.exp(-m.params["const"])),
                   "P_large_first_at_equal_distance_largeUP": 1 / (1 + np.exp(-(m.params["const"] + m.params["large_up"])))})
    return T, pd.DataFrame(lr), PE


def section_oos(P, k, cfg, B_all, n_boot):
    """Time split: first 70% of the analysis period = in-sample, last 30% = out-of-sample."""
    v = P.index[P.valid]
    t_split = v[0] + (v[-1] - v[0]) * cfg["stats"]["oos_split"]
    rows = []
    for name, m in [("in-sample", P.index < t_split), ("out-of-sample", P.index >= t_split)]:
        Bm = et.benchmarks(P, cfg, mask=pd.Series(m, index=P.index))
        Ps = P[m]
        E = bucket_events(Ps, k, cfg, Bm)
        C = core_events(Ps, k, cfg, Bm)
        rows.append({"period": name, "subset": "CORE 2.5-5", **summary_row(C, cfg, n_boot)})
        for dname in ["UP", "DOWN"]:
            rows.append({"period": name, "subset": f"CORE large={dname}", **summary_row(C[C.large_dir == dname], cfg, n_boot)})
        for lab in cfg["core_buckets"] + ["5-7.5", ">7.5", "1-1.25"]:
            rows.append({"period": name, "subset": lab, **summary_row(E[E.bucket == lab], cfg, n_boot)})
    t = pd.DataFrame(rows)
    t.attrs["split"] = str(t_split)
    return t, t_split


def section_percentile(P, specs, ratio=3.06, direction=-1):
    rows = []
    for k in specs:
        r = P[f"ratio_{k}"][P.valid].astype(float)
        dirs = P[f"dir_{k}"][P.valid]
        rows.append({"spec": k, "pct_of_ratio_all_snapshots": (r < ratio).mean(),
                     "pct_within_same_direction": (r[dirs == direction] < ratio).mean(),
                     "share_time_ratio_ge_2.5": (r >= 2.5).mean(), "share_time_ratio_ge_3.06": (r >= ratio).mean(),
                     "share_time_large_down": (dirs == -1).mean(),
                     "median_ratio": r.median(), "p90_ratio": r.quantile(0.9), "p99_ratio": r.quantile(0.99)})
    return pd.DataFrame(rows)


def save(df: pd.DataFrame, name: str):
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=False, float_format="%.6g")


def section_matched_control(P, C, k, cfg, n_boot, tol=0.0025):
    """Is post-event drift explained by the prior 24h move alone?
    For each event, controls = valid hourly snapshots with ratio < 2 (same window) whose 24h
    return is within +-tol of the event's, and which are > 72h away from the event. The
    control's direction-adjusted return uses the EVENT's large-pool direction."""
    m = P.valid & (P.index.minute == 0) & (P[f"ratio_{k}"].astype(float) < 2.0)
    S = P.loc[m, ["ret_24h"] + [f"fret_{h}" for h in H_MAIN]].dropna(subset=["ret_24h"]).astype(float)
    S = S.sort_values("ret_24h")
    r = S.ret_24h.to_numpy()
    t = S.index
    rows = []
    for ts, e in C.iterrows():
        lo, hi = np.searchsorted(r, e.ret_24h - tol), np.searchsorted(r, e.ret_24h + tol)
        sub = S.iloc[lo:hi]
        sub = sub[np.abs((t[lo:hi] - ts) / pd.Timedelta(hours=1)) > 72]
        row = {"t_snap": ts, "dir": e.dir, "n_controls": len(sub)}
        for h in H_MAIN:
            row[f"dar_{h}"] = e[f"dar_{h}"]
            row[f"ctrl_dar_{h}"] = e.dir * sub[f"fret_{h}"].mean() if len(sub) else np.nan
        rows.append(row)
    M = pd.DataFrame(rows).set_index("t_snap")
    out = []
    gm = st.month_groups(M.index)
    for h in H_MAIN:
        d = (M[f"dar_{h}"] - M[f"ctrl_dar_{h}"]).to_numpy(float)
        s = st.summarize_dar(d, None, n_boot, 10, groups=gm)
        out.append({"horizon": et.H_LABEL[h], "N": s["N"], "mean_event_DAR": M[f"dar_{h}"].mean(),
                    "mean_matched_control_DAR": M[f"ctrl_dar_{h}"].mean(), "mean_difference": s.get("mean"),
                    "diff_cluster_ci": f"[{s.get('mean_clci_lo', np.nan):.4f}, {s.get('mean_clci_hi', np.nan):.4f}]",
                    "p_cluster": s.get("p_cluster"), "median_n_controls": M.n_controls.median()})
    return pd.DataFrame(out)
