"""Stage: validate + calibrate the liquidation map on real Binance liquidations (Tardis free days)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import liq_validation as lv

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
MOD = ROOT / "results" / "models"
FIG = ROOT / "results" / "figures"


def _frame(eng, swl, sws, realbars, base):
    d = realbars.copy()
    d["mod_long"] = swl.reindex(d.index).to_numpy()
    d["mod_short"] = sws.reindex(d.index).to_numpy()
    d["base_long"] = base.base_long.reindex(d.index).to_numpy()
    d["base_short"] = base.base_short.reindex(d.index).to_numpy()
    return d.dropna()


def run(cfg, reuse_grid: bool = True) -> dict:
    days = lv.validation_days()
    lv.download(days)
    real = lv.load_real(days)
    realbars = lv.real_per_bar(real, days)
    # bars ending exactly at midnight belong to the previous day; train = every other validation day
    day_of = (realbars.index - pd.Timedelta(minutes=5)).floor("1D")
    is_train = day_of.isin(pd.DatetimeIndex(days[0::2]))
    train_idx, test_idx = realbars.index[is_train], realbars.index[~is_train]
    eng = lv.EngineInputs(cfg)
    base = lv.baseline(eng)

    per_day = real.groupby([real.ts.dt.floor("1D"), "liq_side"]).usd.agg(["size", "sum"]).unstack().fillna(0)
    per_day.columns = [f"{a}_{b}" for a, b in per_day.columns]
    per_day.to_csv(TAB / "validation_real_liquidations_per_day.csv")

    def evaluate(name, tiers, w, hl, split):
        swl, sws, _ = eng.run(tiers, w, hl, split)
        d = _frame(eng, swl, sws, realbars, base)
        rows = []
        for part, m in [("train", d.index.isin(train_idx)), ("test", d.index.isin(test_idx)),
                        ("all", np.ones(len(d), bool))]:
            rows.append({"model": name, "tiers": str(tiers), "weights": str(tuple(round(x, 3) for x in w)),
                         "half_life_h": hl, "side_split": split, "part": part, **lv.score(d[m])})
        return rows, d

    ref_rows = []
    lm = cfg["liquidation_map"]
    for name, v in lm["variants"].items():
        r, d = evaluate(name, lm["leverage_tiers"], lm["leverage_weights"], v["decay_half_life_h"],
                        "symmetric" if v["side_split"] == "symmetric" else "account")
        ref_rows += r
        if name == "M1":
            d_m1 = d
    ref = pd.DataFrame(ref_rows)
    ref.to_csv(TAB / "validation_reference_models.csv", index=False)

    gfile = TAB / "validation_grid_all.csv"
    if reuse_grid and gfile.exists():
        G = pd.read_csv(gfile)
    else:
        grid_rows = []
        for w in lv.weight_grid(0.25):
            for hl in [None, 168, 72, 24]:
                for split in ["symmetric", "account"]:
                    r, _ = evaluate("grid", lv.GRID_TIERS, w, hl, split)
                    grid_rows += r
        G = pd.DataFrame(grid_rows)
        G.to_csv(gfile, index=False)
    tr = G[G.part == "train"].reset_index(drop=True)
    te = G[G.part == "test"].reset_index(drop=True)
    best_i = int(tr.pearson_log.idxmax())
    best = tr.iloc[best_i]
    best_test = te.iloc[best_i]
    # rank stability: correlation of train vs test score across the grid
    rank_corr = pd.Series(tr.pearson_log).corr(pd.Series(te.pearson_log), method="spearman")
    w_best = tuple(float(x) for x in best.weights.strip("()").split(","))
    hl_best = None if pd.isna(best.half_life_h) else float(best.half_life_h)
    calib = {"tiers": lv.GRID_TIERS, "weights": w_best, "half_life_h": hl_best, "side_split": best.side_split,
             "train_pearson_log": float(best.pearson_log), "test_pearson_log": float(best_test.pearson_log),
             "train_test_rank_corr_over_grid": float(rank_corr), "grid_size": int(len(tr))}
    (MOD / "calibrated_map.json").write_text(json.dumps(calib, indent=1))

    # level test: M1 and calibrated, all validation bars flagged
    flag = np.asarray(eng.index.isin(realbars.index))
    lt = []
    for name, tiers, w, hl, split in [("M1", lm["leverage_tiers"], lm["leverage_weights"], None, "symmetric"),
                                      ("M3_calibrated", lv.GRID_TIERS, w_best, hl_best, best.side_split)]:
        _, _, sparse = eng.run(tiers, w, hl, split, flag=flag, cap=3_000_000)
        L = lv.level_test(sparse, real, eng)
        L["model"] = name
        lt.append(L)
    LT = pd.concat(lt, ignore_index=True)
    LT.to_csv(TAB / "validation_level_test_bars.csv", index=False)
    lsum = LT.groupby(["model", "side"]).apply(lambda g: pd.Series({
        "n_bars": len(g), "usd_weighted_share_real_in_model_buckets": np.average(g.share_real_in_model_buckets, weights=g.real_usd),
        "mean_share_range_covered": g.share_range_model_buckets.mean(),
        "lift": np.average(g.share_real_in_model_buckets, weights=g.real_usd) / g.share_range_model_buckets.mean()})).reset_index()
    lsum.to_csv(TAB / "validation_level_test_summary.csv", index=False)

    # stronger price-only baseline: breakout of prior 24h low/high (added before seeing these results)
    brk = lv.baseline_breakout(eng)
    inc = []
    for name, tiers, w, hl, split in [("M1", lm["leverage_tiers"], lm["leverage_weights"], None, "symmetric"),
                                      ("M3_calibrated", lv.GRID_TIERS, w_best, hl_best, best.side_split)]:
        swl, sws, _ = eng.run(tiers, w, hl, split)
        d = _frame(eng, swl, sws, realbars, base)
        d["brk_long"] = brk.brk_long.reindex(d.index).to_numpy()
        d["brk_short"] = brk.brk_short.reindex(d.index).to_numpy()
        d = d.dropna()
        for part, idx in [("train", train_idx), ("test", test_idx)]:
            dd = d[d.index.isin(idx)]
            for bname, cols in [("excursion", [("base_long", "base_short")]),
                                ("breakout_24h", [("brk_long", "brk_short")]),
                                ("excursion+breakout", [("base_long", "base_short"), ("brk_long", "brk_short")])]:
                inc.append({"model": name, "part": part, "baseline": bname, **lv.incremental_over(dd, cols),
                            **({"pearson_log": lv.score(dd)["pearson_log"],
                                "share_real_usd_in_model_sweep_bars": lv.score(dd)["share_real_usd_in_model_sweep_bars"]}
                               if bname == "excursion" else {})})
    pd.DataFrame(inc).to_csv(TAB / "validation_incremental_over_baselines.csv", index=False)

    _figures(d_m1, eng, realbars, base, calib, G)
    return {"reference": ref, "calib": calib, "level": lsum}


def _figures(d_m1, eng, realbars, base, calib, G):
    from . import visualization as vz  # palette + rcParams
    import matplotlib.pyplot as plt
    swl, sws, _ = eng.run(lv.GRID_TIERS, calib["weights"], calib["half_life_h"], calib["side_split"])
    d3 = _frame(eng, swl, sws, realbars, base)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, d, title in [(axes[0], d_m1, "M1 (assumed leverage)"), (axes[1], d3, "M3 (calibrated on train days)")]:
        x = np.log10(1 + np.concatenate([d.mod_long, d.mod_short]))
        y = np.log10(1 + np.concatenate([d.real_long, d.real_short]))
        ax.scatter(x, y, s=4, alpha=0.2, color=vz.INK2, linewidths=0)
        ax.set_xlabel("log10(1 + modelled swept USD) per 5m bar & side")
        ax.set_ylabel("log10(1 + real liquidated USD)")
        ax.set_title(f"{title}: r = {np.corrcoef(x, y)[0, 1]:.2f}")
    tr = G[G.part == "train"].pearson_log.to_numpy()
    te = G[G.part == "test"].pearson_log.to_numpy()
    axes[2].scatter(tr, te, s=10, color=vz.C_UP, alpha=0.6, linewidths=0)
    axes[2].set_xlabel("train-day correlation")
    axes[2].set_ylabel("test-day correlation")
    axes[2].set_title(f"Calibration grid ({len(tr)} specs): train vs test")
    lim = [min(tr.min(), te.min()) - 0.01, max(tr.max(), te.max()) + 0.01]
    axes[2].plot(lim, lim, color=vz.INK2, lw=1, ls="--")
    vz._save(fig, "10_map_validation")
