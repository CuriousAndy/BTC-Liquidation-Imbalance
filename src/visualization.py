"""Figures (static PNG, light surface). Palette = dataviz reference instance:
categorical blue / orange / aqua in fixed order, diverging blue<->red with gray midpoint,
thin 2px lines, recessive grid, text in ink colours (never series colours)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"

C_UP, C_DOWN, C_ALL = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])  # red(-) gray blue(+)

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    "savefig.dpi": 140, "savefig.bbox": "tight",
})


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def ratio_distribution(P: pd.DataFrame, specs: list[str], core_events: pd.DataFrame, k_main: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    r = P.loc[P.valid, f"ratio_{k_main}"].astype(float)
    bins = np.logspace(0, np.log10(20), 60)
    axes[0].hist(r.clip(upper=20), bins=bins, color=C_UP, alpha=0.85, edgecolor=SURF, linewidth=0.5)
    axes[0].set_xscale("log")
    for x in (2.5, 5):
        axes[0].axvline(x, color=INK2, ls="--", lw=1)
    axes[0].set_title(f"Imbalance ratio, all valid 5m snapshots ({k_main})")
    axes[0].set_xlabel("max/min ratio (log scale, clipped at 20)")
    axes[0].set_ylabel("snapshots")
    for k, c in zip(specs, plt.cm.Greys(np.linspace(0.35, 0.95, len(specs)))):
        rr = np.sort(P.loc[P.valid, f"ratio_{k}"].astype(float).clip(upper=20).to_numpy())
        axes[1].plot(rr, 1 - np.arange(len(rr)) / len(rr), color=c, lw=1.6, label=k)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].axvline(2.5, color=INK2, ls="--", lw=1)
    axes[1].set_title("Survival P(ratio >= x) by window / weighting")
    axes[1].set_xlabel("ratio")
    axes[1].legend(ncol=2, fontsize=8)
    _save(fig, "01_ratio_distribution")


def sli_scatter(S: pd.DataFrame, k: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, h, lab in [(axes[0], "fret_1440", "24h"), (axes[1], "fret_4320", "72h")]:
        d = S[["sli", h]].dropna()
        ax.scatter(d.sli, 100 * d[h], s=4, color=INK2, alpha=0.15, linewidths=0)
        q = pd.qcut(d.sli, 20, duplicates="drop")
        g = d.groupby(q, observed=True).agg(x=("sli", "mean"), m=(h, "mean"), s=(h, "sem"))
        ax.errorbar(g.x, 100 * g.m, yerr=196 * g.s, color=C_UP, marker="o", ms=5, lw=2, capsize=2,
                    label="vigintile mean ± 95% CI")
        ax.axhline(0, color=INK2, lw=1)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_ylim(np.nanpercentile(100 * d[h], [1, 99]))
        ax.set_xlabel("signed log imbalance  log(L_up / L_down)   (>0 = short pool above larger)")
        ax.set_ylabel(f"future {lab} return, %")
        ax.set_title(f"Signed imbalance vs {lab} return (4h-spaced snapshots, {k})")
        ax.legend(loc="upper left")
    _save(fig, "02_sli_vs_future_return")


def bucket_prob(core: pd.DataFrame, core_buckets, k: str):
    t = core[~core.bucket.str.startswith("CORE")]
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    cols = {"1": "#86b6ef", "2": C_UP, "3": "#1c5cab", "5": "#0d366b"}
    for tag, c in cols.items():
        p = t[f"P_large_first_{tag}pct"].to_numpy(float)
        n = t[f"N_resolved_{tag}pct"].to_numpy(float)
        se = np.sqrt(p * (1 - p) / np.maximum(n, 1))
        ax.errorbar(x + (int(tag) - 2.5) * 0.06, p, yerr=1.96 * se, color=c, marker="o", ms=5, lw=1.6, capsize=2,
                    label=f"±{tag}%")
    ax.axhline(0.5, color=INK2, ls="--", lw=1)
    for i, b in enumerate(t.bucket):
        if b in core_buckets:
            ax.axvspan(i - 0.45, i + 0.45, color="#f0efec", zorder=0)
    ax.set_xticks(x, [f"{b}\nN={n}" for b, n in zip(t.bucket, t.N_events)], fontsize=8)
    ax.set_ylabel("P(large pool side hit first)")
    ax.set_title(f"First passage toward the larger pool, by ratio bucket ({k}, B24 events; shaded = core 2.5-5)")
    ax.legend(title="symmetric threshold", ncol=4)
    _save(fig, "03_bucket_vs_P_large_first")


def bucket_dar(E: pd.DataFrame, cfg, core_buckets, k: str):
    order = [b for b in __import__("src.events_table", fromlist=["x"]).bucket_labels(cfg["ratio_buckets"])]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for j, (h, c) in enumerate([(240, "#86b6ef"), (1440, C_UP), (4320, "#0d366b")]):
        meds, lo, hi = [], [], []
        rng = np.random.default_rng(0)
        for b in order:
            v = E.loc[E.bucket == b, f"dar_{h}"].dropna().to_numpy()
            if len(v) < 5:
                meds.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
            bs = np.median(v[rng.integers(0, len(v), (1000, len(v)))], axis=1)
            meds.append(np.median(v)); lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
        meds, lo, hi = 100 * np.array(meds), 100 * np.array(lo), 100 * np.array(hi)
        x = np.arange(len(order)) + (j - 1) * 0.12
        ax.errorbar(x, meds, yerr=[meds - lo, hi - meds], color=c, marker="o", ms=5, lw=1.6, capsize=2,
                    label={240: "4h", 1440: "24h", 4320: "72h"}[h])
    ax.axhline(0, color=INK2, ls="--", lw=1)
    for i, b in enumerate(order):
        if b in core_buckets:
            ax.axvspan(i - 0.45, i + 0.45, color="#f0efec", zorder=0)
    ax.set_xticks(np.arange(len(order)), order, fontsize=8)
    ax.set_ylabel("median direction-adjusted return, %\n(>0 = toward larger pool)")
    ax.set_title(f"Median DAR by ratio bucket ({k}, B24 events, iid bootstrap 95% CI)")
    ax.legend(title="horizon", ncol=3)
    _save(fig, "04_bucket_vs_median_DAR")


def event_study(C: pd.DataFrame, price_1m: pd.DataFrame, k: str):
    close = price_1m.close.to_numpy(float)
    offs = np.arange(-24 * 60, 72 * 60 + 1, 15)
    t0 = pd.DatetimeIndex(C.t0)
    i0 = ((t0 - price_1m.index[0]) // pd.Timedelta(minutes=1)).to_numpy(np.int64) - 1  # bar ending at t0
    J = i0[:, None] + offs[None, :]
    ok = (J >= 0) & (J < len(close))
    paths = np.where(ok, close[np.clip(J, 0, len(close) - 1)], np.nan) / close[i0][:, None] - 1
    d = C.dir.to_numpy(float)
    xh = offs / 60
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for lab, m, c in [("large pool UP", d > 0, C_UP), ("large pool DOWN", d < 0, C_DOWN)]:
        mu = np.nanmean(paths[m], 0) * 100
        se = np.nanstd(paths[m], 0) / np.sqrt(m.sum()) * 100
        axes[0].plot(xh, mu, color=c, label=f"{lab} (N={m.sum()})")
        axes[0].fill_between(xh, mu - 1.96 * se, mu + 1.96 * se, color=c, alpha=0.15, lw=0)
    axes[0].set_title(f"Mean BTC path around event onset ({k}, core 2.5-5, B24)")
    axes[0].set_ylabel("% vs P0")
    norm = paths * d[:, None]
    mu = np.nanmean(norm, 0) * 100
    se = np.nanstd(norm, 0) / np.sqrt(len(C)) * 100
    med = np.nanmedian(norm, 0) * 100
    axes[1].plot(xh, mu, color=C_ALL, label="mean ± 95% CI (iid)")
    axes[1].fill_between(xh, mu - 1.96 * se, mu + 1.96 * se, color=C_ALL, alpha=0.18, lw=0)
    axes[1].plot(xh, med, color=INK2, lw=1.4, ls="--", label="median")
    axes[1].set_title("Direction-normalised path (+ = toward larger pool)")
    for ax in axes:
        ax.axvline(0, color=INK2, lw=1)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xlabel("hours from t0")
        ax.legend()
    _save(fig, "05_event_study")
    return pd.DataFrame({"hours": xh, "mean_norm_pct": mu, "median_norm_pct": med, "se_pct": se,
                         "mean_up_pct": np.nanmean(paths[d > 0], 0) * 100,
                         "mean_down_pct": np.nanmean(paths[d < 0], 0) * 100})


def heatmaps(D: pd.DataFrame, F: pd.DataFrame, k: str, order_b, horizons, thresholds):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for ax, val, title, data, cols in [
        (axes[0], "mean", "mean DAR, %", D, horizons), (axes[1], "median", "median DAR, %", D, horizons),
        (axes[2], "p_large_first", "P(large first) - 0.5", F, thresholds)]:
        sub = data[(data.spec == k)]
        col = "horizon" if data is D else "threshold_pct"
        M = sub.pivot(index=col, columns="bucket", values=val).reindex(index=cols, columns=order_b)
        N = sub.pivot(index=col, columns="bucket", values="N" if data is D else "N_resolved").reindex(index=cols, columns=order_b)
        A = M.to_numpy(float) * (100 if data is D else 1)
        if data is F:
            A = A - 0.5
        lim = np.nanpercentile(np.abs(A), 95) or 1
        im = ax.imshow(A, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if np.isfinite(A[i, j]):
                    n = N.iloc[i, j]
                    ax.text(j, i, f"{A[i, j]:.2f}" + ("*" if n < 30 else ""), ha="center", va="center", fontsize=7, color=INK)
        ax.set_xticks(range(len(order_b)), order_b, rotation=45, fontsize=8)
        ax.set_yticks(range(len(cols)), [str(c) + ("%" if data is F else "") for c in cols], fontsize=8)
        ax.set_title(f"{title} ({k})")
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Ratio bucket × horizon/threshold (B24 events). * = N<30. Blue = toward larger pool, red = toward smaller",
                 fontsize=10, color=INK2)
    _save(fig, f"06_heatmap_{k}")


def window_stability(F: pd.DataFrame, D: pd.DataFrame, specs, order_b):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    shades = plt.cm.Blues(np.linspace(0.35, 1.0, len(specs)))
    for k, c in zip(specs, shades):
        f = F[(F.spec == k) & (F.threshold_pct == 2)].set_index("bucket").reindex(order_b)
        axes[0].plot(range(len(order_b)), f.p_large_first, color=c, marker="o", ms=4, lw=1.6, label=k)
        d = D[(D.spec == k) & (D.horizon == "24h")].set_index("bucket").reindex(order_b)
        axes[1].plot(range(len(order_b)), 100 * d["mean"], color=c, marker="o", ms=4, lw=1.6, label=k)
    axes[0].axhline(0.5, color=INK2, ls="--", lw=1)
    axes[1].axhline(0, color=INK2, ls="--", lw=1)
    axes[0].set_title("P(large first, ±2%) by bucket, each window")
    axes[1].set_title("mean DAR 24h (%) by bucket, each window")
    for ax in axes:
        ax.set_xticks(range(len(order_b)), order_b, rotation=45, fontsize=8)
        ax.legend(ncol=3, fontsize=8)
    _save(fig, "07_window_stability")


def rolling_plot(R: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 4))
    for w, c in [(6, "#86b6ef"), (12, C_UP)]:
        g = R[R.window_months == w]
        ax.plot(g.end, g.P_large_first_2pct, color=c, label=f"rolling {w}m")
        ax.fill_between(g.end, g.P_large_first_2pct - g.P_ci_halfwidth, g.P_large_first_2pct + g.P_ci_halfwidth,
                        color=c, alpha=0.15, lw=0)
    ax.axhline(0.5, color=INK2, ls="--", lw=1)
    ax.set_title("Rolling P(large pool first, ±2%), core 2.5-5 events")
    ax.legend()
    _save(fig, "08_rolling_stability")


def volatility_by_bucket(core: pd.DataFrame, bench_range: float):
    t = core[~core.bucket.str.startswith("CORE")]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(t)), 100 * t.median_range_24h, color=C_UP, width=0.7)
    ax.plot(range(len(t)), 100 * t.median_past_rv_24h / np.sqrt(365) * 4, color=INK2, lw=1.4, ls="--",
            marker="o", ms=4, label="past 24h realised vol scaled (×4/√365) - for reference")
    ax.axhline(100 * bench_range, color=C_DOWN, lw=1.6, label="unconditional median 24h range")
    ax.set_xticks(range(len(t)), t.bucket, fontsize=8)
    ax.set_ylabel("median 24h high-low range, %")
    ax.set_title("Future 24h range by ratio bucket (raw, not controlled for past vol)")
    ax.legend()
    _save(fig, "09_range_by_bucket")
