"""BTC liquidation imbalance study - full pipeline.

    python main.py                      # all stages (download is cached)
    python main.py --stages analyze     # re-run selected stages
    python main.py --match --down 5.5e9 --up 1.8e9 --price 85549 [--window 0.10] [--oi 6e10]

Stages: download, build, outcomes, audit, analyze (includes panelreg), panelreg, validate, figures
    python main.py --stages none --variant-suite M3   # calibrated-map re-run
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from src import analysis as an  # noqa: E402
from src import backtest as bt  # noqa: E402
from src import data_audit, data_loader, events_table as et, matching, pipeline_build, pipeline_outcomes  # noqa: E402
from src import regressions as rg  # noqa: E402
from src import visualization as vz  # noqa: E402
from src.price_outcomes import PricePath  # noqa: E402

TAB = ROOT / "results" / "tables"
MOD = ROOT / "results" / "models"


def log(msg):
    print(f"[{pd.Timestamp.now():%H:%M:%S}] {msg}", flush=True)


def stage_analyze(cfg):
    seed = cfg["seed"]
    np.random.seed(seed)
    random.seed(seed)
    nb = cfg["stats"]["n_boot"]
    specs = an.all_specs(cfg)
    k = an.wkey(cfg["features"]["main_window"])
    summary = {}

    # ---- full results matrix + percentile, one spec at a time (memory)
    for variant in cfg["liquidation_map"]["variants"]:
        Ds, Fs, pct = [], [], []
        vspecs = specs if variant == "M1" else [s for s in specs if s.startswith("w")]
        for s in vspecs:
            log(f"matrix {variant} {s}")
            P = et.load_panel(variant, [s])
            B = et.benchmarks(P, cfg)
            D, F = an.section_matrix(P, [s], cfg, B, variant, nb)
            Ds.append(D)
            Fs.append(F)
            pct.append(an.section_percentile(P, [s]))
            del P
            gc.collect()
        D, F = an.finalize_matrix(pd.concat(Ds, ignore_index=True), pd.concat(Fs, ignore_index=True), vspecs, cfg)
        an.save(D, f"results_matrix_DAR_{variant}")
        an.save(F, f"results_matrix_first_passage_{variant}")
        an.save(pd.concat(pct, ignore_index=True), f"Q10_ratio_percentiles_{variant}")

    # ---- main spec, M1 and M2 core tables
    for variant in cfg["liquidation_map"]["variants"]:
        log(f"core tables {variant}")
        P = et.load_panel(variant, [k, "inv_d2"])
        B = et.benchmarks(P, cfg)
        (MOD / f"benchmarks_{variant}.json").write_text(json.dumps(
            {"mu": B["mu"], "p_up": {f"{a}|{b}": v for (a, b), v in B["p_up"].items()}, "vol": B["vol"], "n": B["n"]}, indent=1))
        E, C, core, asym = an.section_core(P, k, cfg, B, variant, nb)
        an.save(core, f"T17_core_summary_{variant}_{k}")
        an.save(asym, f"T18_asymmetric_{variant}_{k}")
        if variant != "M1":
            del P
            gc.collect()
            continue
        log("methods / ladder / monotonic")
        meth, ep = an.section_methods(P, k, cfg, B, nb)
        an.save(meth, f"event_definition_robustness_{k}")
        an.save(ep, f"episodes_{k}")
        an.save(an.section_ladder(P, k, cfg, B, nb), f"threshold_ladder_{k}")
        an.save(an.section_monotonic(E, cfg), f"monotonicity_{k}")
        log("regimes / size / change / extremes / time")
        an.save(an.section_regimes(C, cfg, nb), f"regimes_{k}")
        an.save(an.section_size(P, k, cfg, B, nb), f"size_interaction_{k}")
        an.save(an.section_change(C, cfg, nb), f"change_speed_{k}")
        ext = pd.read_csv(ROOT / "data" / "processed" / "extreme_days.csv", index_col=0, parse_dates=True)
        ext.index = pd.to_datetime(ext.index, utc=True)
        xt = an.section_extreme(C, ext, cfg, nb)
        an.save(xt, f"extreme_days_robustness_{k}")
        an.save(ext[ext.extreme].reset_index(), "extreme_days_list")
        yr, yr2, roll = an.section_time(C, E, cfg, nb)
        an.save(yr, f"by_year_{k}")
        an.save(yr2, f"by_year_direction_{k}")
        an.save(roll, f"rolling_{k}")
        log("clusters")
        price = pd.read_parquet(ROOT / "data" / "processed" / "price_1m_clean.parquet", columns=["high", "low", "close"])
        pp = PricePath(price)
        Eall = E[E.ratio >= 2.5]
        # placebo: one snapshot per day at 00:00 UTC regardless of ratio, and balanced days only
        daily = P.index[P.valid & (P.index.hour == 0) & (P.index.minute == 0)]
        PL = et.make_events(P, daily, k, cfg, B, label="placebo_daily")
        for name, ev in [("core", C), ("all_ratio_ge_2.5", Eall), ("placebo_daily_all", PL),
                         ("placebo_daily_ratio_lt_1.5", PL[PL.ratio < 1.5])]:
            T, LR, PE = an.section_clusters(ev, pp, cfg, nb)
            an.save(T, f"clusters_first_touch_{name}_{k}")
            an.save(LR, f"clusters_logit_distance_{name}_{k}")
        an.save(an.section_matched_control(P, C, k, cfg, nb), f"matched_prior_move_control_{k}")
        log("event regressions")
        er = pd.concat([rg.event_models(C, P, "core 2.5-5 B24"), rg.event_models(E, P, "all buckets B24")])
        an.save(er, f"event_regressions_{k}")
        log("out-of-sample")
        oos, t_split = an.section_oos(P, k, cfg, B, nb)
        an.save(oos, f"oos_split_{k}")
        summary["oos_split"] = str(t_split)
        log("backtest")
        bts = []
        for side in (+1, -1):
            bts.append(bt.run_backtest(C, cfg, side, "core 2.5-5 B24 (full)"))
            bts.append(bt.run_backtest(C[C.index >= t_split], cfg, side, "core 2.5-5 B24 (OOS only)"))
            for dname in ("UP", "DOWN"):
                bts.append(bt.run_backtest(C[C.large_dir == dname], cfg, side, f"core large={dname}"))
        an.save(pd.concat(bts, ignore_index=True), f"backtest_{k}")
        log("figures (main)")
        Pk = P
        vz.bucket_prob(core, cfg["core_buckets"], k)
        vz.bucket_dar(E, cfg, cfg["core_buckets"], k)
        es = vz.event_study(C, price, k)
        an.save(es, f"event_study_path_{k}")
        vz.rolling_plot(roll)
        vz.volatility_by_bucket(core, B["vol"]["range_1440"])
        S = rg.panel_sample(Pk, k)
        vz.sli_scatter(S, k)
        del price, pp, P, Pk
        gc.collect()

    (MOD / "run_summary.json").write_text(json.dumps(summary, indent=1))
    stage_panelreg(cfg)


def oos_split_time(cfg) -> pd.Timestamp:
    P = et.load_panel("M1", [an.wkey(cfg["features"]["main_window"])])
    v = P.index[P.valid]
    return v[0] + (v[-1] - v[0]) * cfg["stats"]["oos_split"]


def stage_panelreg(cfg):
    """Panel regressions for every spec (continuous variable) + walk-forward."""
    specs = an.all_specs(cfg)
    k = an.wkey(cfg["features"]["main_window"])
    t_split = oos_split_time(cfg)
    rows, wf = [], []
    for s in specs:
        log(f"panel regressions {s}")
        P = et.load_panel("M1", [s, "inv_d2"])
        S = rg.panel_sample(P, s)
        rows.append(rg.panel_models(S, s))
        if s == k:
            S.to_parquet(ROOT / "data" / "processed" / f"panel_4h_sample_{s}.parquet")
        if s in (k, "w5", "w15", "inv_d2"):
            wf.append(rg.walk_forward(S, t_split).assign(spec=s))
        del P
        gc.collect()
    an.save(pd.concat(rows, ignore_index=True), "panel_regressions_all_specs")
    an.save(pd.concat(wf, ignore_index=True), "walk_forward")


def stage_variant_suite(cfg, variant):
    """Core study re-run on one alternative map variant (used for the calibrated map M3)."""
    nb = cfg["stats"]["n_boot"]
    k = an.wkey(cfg["features"]["main_window"])
    wspecs = [s for s in an.all_specs(cfg) if s.startswith("w")]
    Ds, Fs, pct = [], [], []
    for s in wspecs:
        log(f"{variant} matrix {s}")
        P = et.load_panel(variant, [s])
        B = et.benchmarks(P, cfg)
        D, F = an.section_matrix(P, [s], cfg, B, variant, nb)
        Ds.append(D)
        Fs.append(F)
        pct.append(an.section_percentile(P, [s]))
        del P
        gc.collect()
    D, F = an.finalize_matrix(pd.concat(Ds, ignore_index=True), pd.concat(Fs, ignore_index=True), wspecs, cfg)
    an.save(D, f"results_matrix_DAR_{variant}")
    an.save(F, f"results_matrix_first_passage_{variant}")
    an.save(pd.concat(pct, ignore_index=True), f"Q10_ratio_percentiles_{variant}")
    log(f"{variant} core suite")
    P = et.load_panel(variant, [k])
    B = et.benchmarks(P, cfg)
    E, C, core, asym = an.section_core(P, k, cfg, B, variant, nb)
    an.save(core, f"T17_core_summary_{variant}_{k}")
    an.save(asym, f"T18_asymmetric_{variant}_{k}")
    meth, _ = an.section_methods(P, k, cfg, B, nb)
    an.save(meth, f"event_definition_robustness_{k}_{variant}")
    an.save(an.section_ladder(P, k, cfg, B, nb), f"threshold_ladder_{k}_{variant}")
    an.save(an.section_monotonic(E, cfg), f"monotonicity_{k}_{variant}")
    an.save(an.section_matched_control(P, C, k, cfg, nb), f"matched_prior_move_control_{k}_{variant}")
    oos, _ = an.section_oos(P, k, cfg, B, nb)
    an.save(oos, f"oos_split_{k}_{variant}")


def stage_figures(cfg):
    specs = an.all_specs(cfg)
    k = an.wkey(cfg["features"]["main_window"])
    D = pd.read_csv(TAB / "results_matrix_DAR_M1.csv")
    F = pd.read_csv(TAB / "results_matrix_first_passage_M1.csv")
    order_b = et.bucket_labels(cfg["ratio_buckets"])
    for s in [k, "w5", "w3", "inv_d2"]:
        vz.heatmaps(D, F, s, order_b, [et.H_LABEL[h] for h in an.H_MAIN], [0.5, 1, 2, 3, 5, 7.5, 10])
    vz.window_stability(F, D, [s for s in specs if s.startswith("w")], order_b)
    P = et.load_panel("M1", [s for s in specs if s.startswith("w")])
    C = pd.read_parquet(ROOT / "data" / "processed" / "events" / f"events_B24_core_M1_{k}.parquet")
    vz.ratio_distribution(P, [s for s in specs if s.startswith("w")], C, k)


def run_match(cfg, a):
    k = an.wkey(a.window)
    P = et.load_panel("M1", [k])
    res = matching.match(P, k, a.down, a.up, a.oi, a.ret24h, a.ret7d, a.rv7d)
    pct = an.section_percentile(P, [k], ratio=max(a.up, a.down) / min(a.up, a.down), direction=1 if a.up > a.down else -1)
    tag = f"match_{k}_{int(a.down / 1e6)}M_{int(a.up / 1e6)}M"
    an.save(res["returns"], f"{tag}_returns")
    an.save(res["first_passage"], f"{tag}_first_passage")
    an.save(res["neighbours"].reset_index(), f"{tag}_neighbours")
    an.save(pct, f"{tag}_percentile")
    meta = {"query": {kk: (v if not isinstance(v, tuple) else list(v)) for kk, v in res["query"].items()},
            "direction": res["direction"], "ratio": res["ratio"], "price": a.price, "window": a.window,
            "max_distance": res["max_distance"]}
    (TAB / f"{tag}_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps(meta, indent=1, default=str))
    print(pct.to_string())
    print(res["returns"].round(4).to_string())
    print(res["first_passage"].round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="download,build,outcomes,audit,analyze,figures")
    ap.add_argument("--match", action="store_true")
    ap.add_argument("--variant-suite", dest="variant_suite", default=None,
                    help="rebuild one map variant and re-run the core study on it, e.g. M3")
    ap.add_argument("--price", type=float, default=85549)
    ap.add_argument("--down", type=float, default=5.5e9)
    ap.add_argument("--up", type=float, default=1.8e9)
    ap.add_argument("--window", type=float, default=0.10)
    ap.add_argument("--oi", type=float, default=None, help="aggregate OI in USD matching the map source (optional)")
    ap.add_argument("--ret24h", type=float, default=None)
    ap.add_argument("--ret7d", type=float, default=None)
    ap.add_argument("--rv7d", type=float, default=None, help="annualised 7d realised vol, e.g. 0.45")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "config" / "config.yaml"))
    for d in ["results/tables", "results/figures", "results/models", "data/processed/events"]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    data_loader.load_env()
    if a.match:
        run_match(cfg, a)
        return
    st = a.stages.split(",")
    if "download" in st:
        log("download")
        data_loader.download_all(cfg)
        data_loader.build_processed()
    if "build" in st:
        log("build (clean + liquidation map + regimes)")
        pipeline_build.run(cfg)
    if "outcomes" in st:
        log("outcomes + panels")
        pipeline_outcomes.run(cfg)
    if "audit" in st:
        log("audit")
        data_audit.write_audit(cfg)
    if "analyze" in st:
        stage_analyze(cfg)
    elif "panelreg" in st:
        log("panel regressions")
        stage_panelreg(cfg)
    if a.variant_suite:
        log(f"variant suite {a.variant_suite}")
        pipeline_build.run(cfg, only_variants=[a.variant_suite])
        pipeline_outcomes._panels(cfg, only_variants=[a.variant_suite], light=True)
        stage_variant_suite(cfg, a.variant_suite)
    if "validate" in st:
        log("map validation on real liquidations")
        from src import validation_run, validation_amounts
        validation_run.run(cfg)
        validation_amounts.run(cfg)
    if "figures" in st:
        log("figures")
        stage_figures(cfg)
    log("done")


if __name__ == "__main__":
    main()
