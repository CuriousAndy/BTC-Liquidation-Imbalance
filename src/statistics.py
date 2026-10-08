"""Event-level statistics: DAR summaries, first-passage probabilities, bootstrap,
binomial / Wilcoxon tests, sample-size flags, parameter-stability flags, FDR."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as ss

RNG_SEED = 20261007


def size_flag(n: int) -> str:
    if n < 30:
        return "very small sample"
    if n <= 100:
        return "small sample"
    return "reasonable"


def _block_indices(n: int, n_boot: int, block: int, rng) -> np.ndarray:
    """Moving-block bootstrap indices over time-ordered events."""
    block = max(1, min(block, n))
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    return idx


def month_groups(idx) -> np.ndarray:
    """Calendar-month cluster id for each (UTC) timestamp."""
    idx = pd.DatetimeIndex(idx)
    return (idx.year * 12 + idx.month).to_numpy()


def _cluster_means(x: np.ndarray, groups: np.ndarray, n_boot: int, rng) -> np.ndarray | None:
    """Bootstrap means resampling whole clusters (months) with replacement."""
    u, inv = np.unique(groups, return_inverse=True)
    m = len(u)
    if m < 8:
        return None
    sums = np.bincount(inv, weights=x, minlength=m)
    cnts = np.bincount(inv, minlength=m).astype(float)
    draw = rng.integers(0, m, size=(n_boot, m))
    return sums[draw].sum(1) / cnts[draw].sum(1)


def boot_ci(x: np.ndarray, stat=np.mean, n_boot=2000, block=10, alpha=0.05, seed=RNG_SEED, groups=None):
    """Returns (iid CI, dependence-robust CI). The dependence-robust CI is a month-cluster
    bootstrap when `groups` is given (mean only), otherwise a moving-block bootstrap."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x = x[ok]
    n = len(x)
    if n < 5:
        return (np.nan, np.nan), (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    iid = x[rng.integers(0, n, size=(n_boot, n))]
    a = iid.mean(1) if stat is np.mean else np.median(iid, 1)
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    if groups is not None and stat is np.mean:
        b = _cluster_means(x, np.asarray(groups)[ok], n_boot, rng)
        if b is None:
            return tuple(np.percentile(a, q)), (np.nan, np.nan)
    else:
        blk = x[_block_indices(n, n_boot, block, rng)]
        b = blk.mean(1) if stat is np.mean else np.median(blk, 1)
    return tuple(np.percentile(a, q)), tuple(np.percentile(b, q))


def block_boot_pvalue_mean(x: np.ndarray, n_boot=2000, block=10, seed=RNG_SEED, groups=None) -> float:
    """Two-sided p-value for mean = 0 from the centred bootstrap distribution
    (month-cluster bootstrap if groups given and >= 8 months, else moving-block)."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x = x[ok]
    n = len(x)
    if n < 5:
        return np.nan
    rng = np.random.default_rng(seed)
    xc = x - x.mean()
    if groups is not None:
        m = _cluster_means(xc, np.asarray(groups)[ok], n_boot, rng)
        if m is None:
            return np.nan
    else:
        m = xc[_block_indices(n, n_boot, block, rng)].mean(1)
    return float((np.abs(m) >= abs(x.mean())).mean())


def summarize_dar(dar: np.ndarray, base: np.ndarray | None = None, n_boot=2000, block=10, groups=None) -> dict:
    """dar: direction-adjusted returns of time-ordered events.
    base: per-event expected DAR under the unconditional benchmark (dir * mean return)."""
    x = np.asarray(dar, float)
    ok = np.isfinite(x)
    x = x[ok]
    g = None if groups is None else np.asarray(groups)[ok]
    n = len(x)
    out = {"N": n, "sample_flag": size_flag(n), "n_months": int(len(np.unique(g))) if g is not None else np.nan}
    if n < 5:
        return out
    (lo, hi), (blo, bhi) = boot_ci(x, np.mean, n_boot, block, groups=g)
    (mlo, mhi), _ = boot_ci(x, np.median, n_boot, block)
    t = ss.ttest_1samp(x, 0.0)
    try:
        w = ss.wilcoxon(x).pvalue
    except ValueError:
        w = np.nan
    out.update({"mean": x.mean(), "median": np.median(x), "p_pos": (x > 0).mean(), "std": x.std(ddof=1),
                "mean_ci_lo": lo, "mean_ci_hi": hi, "mean_clci_lo": blo, "mean_clci_hi": bhi,
                "median_ci_lo": mlo, "median_ci_hi": mhi, "t_stat": t.statistic, "p_t": t.pvalue,
                "p_wilcoxon": w, "p_cluster": block_boot_pvalue_mean(x, n_boot, block, groups=g),
                "p_sign": ss.binomtest(int((x > 0).sum()), int((x != 0).sum()), 0.5).pvalue if (x != 0).sum() else np.nan})
    if base is not None:
        b = np.asarray(base, float)[ok]
        ex = x - b
        out["mean_excess_vs_uncond"] = ex.mean()
        out["p_cluster_excess"] = block_boot_pvalue_mean(ex, n_boot, block, groups=g)
    return out


def summarize_fp(y_large_first: np.ndarray, base_p: np.ndarray | None = None, n_boot=2000, block=10, groups=None) -> dict:
    """y: 1 = large-pool side hit first, 0 = small side first, nan = censored/tie (dropped).
    base_p: per-event benchmark probability of the large side being hit first."""
    y = np.asarray(y_large_first, float)
    ok = np.isfinite(y)
    yy = y[ok]
    g = None if groups is None else np.asarray(groups)[ok]
    n = len(yy)
    out = {"N_resolved": n, "sample_flag": size_flag(n)}
    if n < 5:
        return out
    k = int(yy.sum())
    bt = ss.binomtest(k, n, 0.5)
    (lo, hi), (blo, bhi) = boot_ci(yy, np.mean, n_boot, block, groups=g)
    out.update({"p_large_first": k / n, "ci_lo": lo, "ci_hi": hi, "cluster_ci_lo": blo, "cluster_ci_hi": bhi,
                "p_binom_vs_50": bt.pvalue,
                "p_cluster_vs_50": block_boot_pvalue_mean(yy - 0.5, n_boot, block, groups=g)})
    if base_p is not None:
        b = np.asarray(base_p, float)[ok]
        out["p_benchmark"] = np.nanmean(b)
        out["excess_vs_benchmark"] = np.nanmean(yy - b)
        out["p_cluster_vs_benchmark"] = block_boot_pvalue_mean(yy - b, n_boot, block, groups=g)
    return out


def bh_fdr(p: pd.Series) -> pd.Series:
    """Benjamini-Hochberg q-values (nan-safe)."""
    q = pd.Series(np.nan, index=p.index)
    v = p.dropna()
    if v.empty:
        return q
    order = v.sort_values()
    m = len(order)
    ranks = np.arange(1, m + 1)
    qq = (order.to_numpy() * m / ranks)
    qq = np.minimum.accumulate(qq[::-1])[::-1].clip(max=1)
    q[order.index] = qq
    return q


def stability_flags(mat: pd.DataFrame, value: str, pcol: str, axes: dict[str, list], fixed=()) -> pd.Series:
    """For each significant cell (p<0.05) look at neighbours along each ordered axis in `axes`
    (other `axes` and all `fixed` columns held equal). 'likely unstable / data-mined' if no
    neighbour has the same sign with p<0.10."""
    flags = pd.Series("", index=mat.index, dtype=object)
    key_cols = list(axes) + list(fixed)
    lookup = mat.set_index(key_cols)
    for i, row in mat.iterrows():
        p = row[pcol]
        if not (np.isfinite(p) and p < 0.05):
            continue
        sign = np.sign(row[value])
        support = False
        for ax, levels in axes.items():
            j = levels.index(row[ax])
            for nb in (j - 1, j + 1):
                if 0 <= nb < len(levels):
                    key = tuple(levels[nb] if c == ax else row[c] for c in key_cols)
                    if key in lookup.index:
                        r2 = lookup.loc[key]
                        if isinstance(r2, pd.DataFrame):
                            r2 = r2.iloc[0]
                        if np.isfinite(r2[pcol]) and r2[pcol] < 0.10 and np.sign(r2[value]) == sign:
                            support = True
        flags[i] = "stable (neighbour supports)" if support else "likely unstable / data-mined"
    return flags
