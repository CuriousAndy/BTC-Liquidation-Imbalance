"""Regressions with HAC (Newey-West) standard errors.

Panel models use non-overlapping-ish snapshots every 4h (valid rows at hh:00, hh % 4 == 0)
and the CONTINUOUS signed_log_imbalance (sli), not buckets. HAC lags cover the outcome
overlap (horizon / 4h + 1). Event-level models use the time-ordered event list.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, log_loss

from .price_outcomes import fp_outcome

CONTROLS = ["ret_24h", "ret_7d", "ret_30d", "px_vs_ma50d", "log_rv_24h", "log_rv_7d", "funding_bp",
            "oi_chg_24h", "premium_bp", "hour_sin", "hour_cos"]


def panel_sample(P: pd.DataFrame, k: str, extra_specs=("inv_d2",)) -> pd.DataFrame:
    m = P.valid & (P.index.minute == 0) & (P.index.hour % 4 == 0)
    S = pd.DataFrame(index=P.index[m])
    g = lambda c: P.loc[m, c].astype(float)
    S["sli"] = g(f"sli_{k}")
    S["abs_sli"] = S.sli.abs()
    if f"total_oi_{k}" in P:
        S["log_total_oi"] = np.log(g(f"total_oi_{k}").clip(lower=1e-6))
        S["size_pct365"] = g(f"total_pct365_{k}")
    else:  # distance-weighted specs: weighted total relative to OI
        S["log_total_oi"] = np.log((g(f"total_{k}") / g("oi_usd")).clip(lower=1e-9))
        S["size_pct365"] = np.nan
    for e in extra_specs:
        if f"sli_{e}" in P and e != k:
            S[f"sli_{e}"] = g(f"sli_{e}")
    for lag in ["1h", "24h"]:
        S[f"dsli_{lag}"] = g(f"dsli_{lag}_{k}") if f"dsli_{lag}_{k}" in P else np.nan
    S["sli_x_dsli_1h"] = S.sli * S.dsli_1h
    S["ret_24h"], S["ret_7d"], S["ret_30d"] = g("ret_24h"), g("ret_7d"), g("ret_30d")
    S["px_vs_ma50d"] = g("px_vs_ma50d")
    S["log_rv_24h"], S["log_rv_7d"] = np.log(g("rv_24h")), np.log(g("rv_7d"))
    S["funding_bp"] = g("funding") * 1e4
    S["oi_chg_24h"] = g("oi_chg_24h")
    S["premium_bp"] = g("premium") * 1e4
    S["hour_sin"], S["hour_cos"] = np.sin(2 * np.pi * S.index.hour / 24), np.cos(2 * np.pi * S.index.hour / 24)
    for h in (240, 1440, 4320):
        S[f"fret_{h}"] = g(f"fret_{h}")
    S["log_fwd_rv_24h"] = np.log(g("fwd_rv_24h"))
    S["log_range_24h"] = np.log((g("mfe_up_1440") - g("mfe_dn_1440")).clip(lower=1e-5))
    S["accel_asym_24h"] = g("max_1h_up_24h") + g("max_1h_dn_24h")
    S["casc_any_24h"] = ((g("casc_up_1440") > 0) | (g("casc_dn_1440") > 0)).astype(float)
    fp = fp_outcome(P.loc[m, "fpt_up_2"].to_numpy(), P.loc[m, "fpt_dn_2"].to_numpy(), 10080)
    S["up_first_2"] = np.where(np.isnan(fp) | (fp == 0), np.nan, (fp > 0).astype(float))
    return S


def _ols(y, X, lags):
    d = pd.concat([y, X], axis=1).dropna()
    if len(d) < 50:
        return None
    return sm.OLS(d.iloc[:, 0], sm.add_constant(d.iloc[:, 1:])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def _logit(y, X, lags):
    d = pd.concat([y, X], axis=1).dropna()
    if len(d) < 50:
        return None
    return sm.GLM(d.iloc[:, 0], sm.add_constant(d.iloc[:, 1:]), family=sm.families.Binomial()).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags})


def _pseudo_r2(m):
    return 1 - m.deviance / m.null_deviance


def _row(model_name, k, m, mc, key_vars, kind):
    if m is None:
        return {"model": model_name, "spec": k}
    r = {"model": model_name, "spec": k, "N": int(m.nobs)}
    for v in key_vars:
        if v in m.params:
            r[f"coef_{v}"] = m.params[v]
            r[f"z_{v}"] = m.tvalues[v]
            r[f"p_{v}"] = m.pvalues[v]
    if kind == "ols":
        r["R2_full"] = m.rsquared
        r["R2_controls_only"] = mc.rsquared if mc is not None else np.nan
    else:
        r["R2_full"] = _pseudo_r2(m)
        r["R2_controls_only"] = _pseudo_r2(mc) if mc is not None else np.nan
    r["incremental_R2_of_imbalance"] = r["R2_full"] - r["R2_controls_only"]
    return r


def panel_models(S: pd.DataFrame, k: str) -> pd.DataFrame:
    rows = []
    C = S[CONTROLS]
    # Direction
    for h in (240, 1440, 4320):
        lags = h // 240 + 1
        y = S[f"fret_{h}"]
        m0 = _ols(y, S[["sli"]], lags)
        rows.append(_row(f"DIR fret_{h} ~ sli", k, m0, None, ["sli"], "ols"))
        m1 = _ols(y, pd.concat([S[["sli", "abs_sli", "log_total_oi"]], C], axis=1), lags)
        mc = _ols(y, pd.concat([S[["abs_sli", "log_total_oi"]], C], axis=1), lags)
        rows.append(_row(f"DIR fret_{h} ~ sli + size + controls", k, m1, mc, ["sli", "abs_sli", "log_total_oi"], "ols"))
        extra = [c for c in S.columns if c.startswith("sli_") and c not in ("sli_x_dsli_1h",)]
        m2 = _ols(y, pd.concat([S[["sli", "dsli_1h", "dsli_24h", "sli_x_dsli_1h", "abs_sli", "log_total_oi"] + extra], C], axis=1), lags)
        mc2 = _ols(y, pd.concat([S[["abs_sli", "log_total_oi"]], C], axis=1), lags)
        rows.append(_row(f"DIR fret_{h} ~ level + change + level*change + distance-wtd + controls", k, m2, mc2,
                         ["sli", "dsli_1h", "dsli_24h", "sli_x_dsli_1h"] + extra, "ols"))
    y = S["up_first_2"]
    m0 = _logit(y, S[["sli"]], 12)
    rows.append(_row("DIR logit up_first_2pct ~ sli", k, m0, None, ["sli"], "logit"))
    m1 = _logit(y, pd.concat([S[["sli", "abs_sli", "log_total_oi"]], C], axis=1), 12)
    mc = _logit(y, pd.concat([S[["abs_sli", "log_total_oi"]], C], axis=1), 12)
    rows.append(_row("DIR logit up_first_2pct ~ sli + size + controls", k, m1, mc, ["sli", "abs_sli", "log_total_oi"], "logit"))
    # Volatility / magnitude / acceleration / cascade
    Cv = S[CONTROLS]
    for name, yv, kind in [("VOL log fwd_rv_24h", "log_fwd_rv_24h", "ols"), ("MAG log range_24h", "log_range_24h", "ols"),
                           ("CASCADE any 24h (logit)", "casc_any_24h", "logit")]:
        f = _ols if kind == "ols" else _logit
        m1 = f(S[yv], pd.concat([S[["abs_sli", "log_total_oi"]], Cv], axis=1), 6)
        mc = f(S[yv], pd.concat([S[["log_total_oi"]], Cv], axis=1), 6)
        rows.append(_row(f"{name} ~ |sli| + size + controls(incl. past vol)", k, m1, mc, ["abs_sli", "log_total_oi"], kind))
        m0 = f(S[yv], S[["abs_sli"]], 6)
        rows.append(_row(f"{name} ~ |sli| only", k, m0, None, ["abs_sli"], kind))
    m1 = _ols(S["accel_asym_24h"], pd.concat([S[["sli", "abs_sli", "log_total_oi"]], Cv], axis=1), 6)
    mc = _ols(S["accel_asym_24h"], pd.concat([S[["abs_sli", "log_total_oi"]], Cv], axis=1), 6)
    rows.append(_row("ACCEL (max 1h up + max 1h down) 24h ~ sli + controls", k, m1, mc, ["sli"], "ols"))
    return pd.DataFrame(rows)


def event_models(E: pd.DataFrame, P: pd.DataFrame, label: str) -> pd.DataFrame:
    """Event-level: Y = large pool first (2%), DAR 24h; regressors direction-aligned."""
    E = E.sort_index()
    d = E.dir.astype(float)
    X = pd.DataFrame(index=E.index)
    X["abs_sli"] = E.abs_sli.astype(float)
    X["log_total_oi"] = np.log(E.total_oi.astype(float).clip(lower=1e-6))
    X["large_up"] = (d > 0).astype(float)
    X["dir_x_ret_24h"] = d * E.ret_24h.astype(float)
    X["dir_x_ret_7d"] = d * E.ret_7d.astype(float)
    X["log_rv_7d"] = np.log(E.rv_7d.astype(float))
    X["dir_x_funding_bp"] = d * E.funding.astype(float) * 1e4
    X["oi_chg_24h"] = E.oi_chg_24h.astype(float)
    X["dabs_1h"] = E.dabs_1h.astype(float)
    X["dabs_24h"] = E.dabs_24h.astype(float)
    if "sli_inv_d2" in P:
        X["dir_x_sli_inv_d2"] = d * P.loc[E.index, "sli_inv_d2"].astype(float)
    X["hour_sin"] = np.sin(2 * np.pi * E.index.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * E.index.hour / 24)
    rows = []
    for yname, y, kind in [("ylarge_2", E.ylarge_2.astype(float), "logit"), ("ylarge_1", E.ylarge_1.astype(float), "logit"),
                           ("dar_1440", E.dar_1440.astype(float), "ols"), ("dar_4320", E.dar_4320.astype(float), "ols")]:
        f = _logit if kind == "logit" else _ols
        m = f(y, X, 7)
        if m is None:
            continue
        for v in m.params.index:
            rows.append({"event_set": label, "y": yname, "model": kind, "var": v, "coef": m.params[v],
                         "z_HAC": m.tvalues[v], "p_HAC": m.pvalues[v], "N": int(m.nobs)})
    return pd.DataFrame(rows)


def walk_forward(S: pd.DataFrame, t_split) -> pd.DataFrame:
    """Fit on in-sample (time < split), evaluate strictly out-of-sample."""
    rows = []
    tr, te = S[S.index < t_split], S[S.index >= t_split]
    C = CONTROLS
    specs = {"constant": [], "controls_only": C, "sli_only": ["sli"], "sli+controls": ["sli", "abs_sli", "log_total_oi"] + C}
    y = "up_first_2"
    for name, cols in specs.items():
        a = tr[[y] + cols].dropna()
        b = te[[y] + cols].dropna()
        if name == "constant":
            p_hat = np.full(len(b), a[y].mean())
            coef = np.nan
        else:
            m = sm.GLM(a[y], sm.add_constant(a[cols]), family=sm.families.Binomial()).fit()
            p_hat = m.predict(sm.add_constant(b[cols], has_constant="add"))
            coef = m.params.get("sli", np.nan)
        rows.append({"target": "up_first_2pct", "model": name, "N_train": len(a), "N_test": len(b),
                     "coef_sli_train": coef,
                     "AUC_test": roc_auc_score(b[y], p_hat) if name != "constant" else 0.5,
                     "logloss_test": log_loss(b[y], np.clip(p_hat, 1e-6, 1 - 1e-6))})
    for h in (1440, 4320):
        yv = f"fret_{h}"
        for name, cols in [("sli_only", ["sli"]), ("sli+controls", ["sli", "abs_sli", "log_total_oi"] + C), ("controls_only", C)]:
            a = tr[[yv] + cols].dropna()
            b = te[[yv] + cols].dropna()
            m = sm.OLS(a[yv], sm.add_constant(a[cols])).fit()
            pred = m.predict(sm.add_constant(b[cols], has_constant="add"))
            oos_r2 = 1 - ((b[yv] - pred) ** 2).sum() / ((b[yv] - a[yv].mean()) ** 2).sum()
            m_te = sm.OLS(b[yv], sm.add_constant(b[cols])).fit(cov_type="HAC", cov_kwds={"maxlags": h // 240 + 1})
            rows.append({"target": yv, "model": name, "N_train": len(a), "N_test": len(b),
                         "coef_sli_train": m.params.get("sli", np.nan), "coef_sli_test": m_te.params.get("sli", np.nan),
                         "p_sli_test_HAC": m_te.pvalues.get("sli", np.nan), "OOS_R2": oos_r2,
                         "sign_hit_rate_test": float((np.sign(pred) == np.sign(b[yv])).mean())})
    for yv in ["log_fwd_rv_24h", "log_range_24h"]:
        for name, cols in [("controls_only", ["log_total_oi"] + C), ("abs_sli+controls", ["abs_sli", "log_total_oi"] + C)]:
            a = tr[[yv] + cols].dropna()
            b = te[[yv] + cols].dropna()
            m = sm.OLS(a[yv], sm.add_constant(a[cols])).fit()
            pred = m.predict(sm.add_constant(b[cols], has_constant="add"))
            oos_r2 = 1 - ((b[yv] - pred) ** 2).sum() / ((b[yv] - a[yv].mean()) ** 2).sum()
            rows.append({"target": yv, "model": name, "N_train": len(a), "N_test": len(b),
                         "coef_abs_sli_train": m.params.get("abs_sli", np.nan), "OOS_R2": oos_r2})
    return pd.DataFrame(rows)
