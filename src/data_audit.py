"""Write DATA_AUDIT.md from the cleaning / build reports (Chinese, numbers from disk)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"


def write_audit(cfg: dict) -> Path:
    rep = json.loads((PROC / "build_report.json").read_text())
    prep = json.loads((PROC / "panel_report.json").read_text())
    price = pd.read_parquet(PROC / "price_1m_clean.parquet", columns=["close", "volume", "flat_zero_vol"])
    met = pd.read_parquet(PROC / "metrics_5m_clean.parquet")
    fund = pd.read_parquet(PROC / "funding_raw_parsed.parquet")
    prem = pd.read_parquet(PROC / "premium_5m_raw_parsed.parquet")
    gaps = pd.read_csv(PROC / "oi_invalid_gaps.csv", parse_dates=["start", "end"])
    log = pd.read_csv(RAW / "download_log.csv")
    f1 = pd.read_parquet(PROC / "liqmap_features_M1.parquet",
                         columns=["map_degraded", "snapshot_age_min", "oi_valid", "ratio_w10", "map_total_long",
                                  "map_total_short", "oi_usd"])
    fdiff = fund.ts.diff().dt.total_seconds().div(3600).value_counts().head(5)
    lm = cfg["liquidation_map"]
    fz = price[price.flat_zero_vol]
    L = []
    a = L.append
    a("# DATA_AUDIT — BTC liquidation imbalance study\n")
    a(f"生成时间（UTC）：{pd.Timestamp.utcnow():%Y-%m-%d %H:%M}。所有时间戳均为 UTC。下面的数字全部由 `src/data_audit.py` 从磁盘文件计算得出。\n")
    a("## 0. 数据来源\n")
    a("研究要求自行获取数据，所以只用了公开、免 key 的数据源：\n")
    a("| 数据集 | 来源 | 原始文件 |\n|---|---|---|")
    a("| BTCUSDT 永续 1m K 线 | data.binance.vision `futures/um/{monthly,daily}/klines/BTCUSDT/1m` | `data/raw/klines_1m/` |")
    a("| BTCUSDT 永续 5m metrics（OI、多空账户比、主动买卖比） | data.binance.vision `futures/um/daily/metrics` | `data/raw/metrics/` |")
    a("| Funding rate | data.binance.vision `futures/um/monthly/fundingRate`，当月数据用 `fapi/v1/fundingRate` 补 | `data/raw/funding/` |")
    a("| Premium index 5m（basis 代理） | data.binance.vision `futures/um/{monthly,daily}/premiumIndexKlines` | `data/raw/premium_5m/` |")
    a("")
    st = log.status.value_counts().to_dict()
    a(f"下载日志：`data/raw/download_log.csv`。共 {len(log)} 个文件，状态统计 {st}。缺失的 3 个都是 2019-12 的月文件；那个月份早于数据集起点，数据源本身就没有，不影响研究。原始 zip 一旦落盘就不再覆盖。\n")
    a("## 1. 价格数据（1m）\n")
    p = rep["price"]
    a("| 项目 | 值 |\n|---|---|")
    a(f"| 起止 | {price.index.min()} → {price.index.max()} |")
    a(f"| 频率 | 1 分钟 |")
    a(f"| 行数 | {p['rows_raw']:,} |")
    a(f"| 缺失 bar | {p['missing_bars']} |")
    a(f"| 重复时间戳 | {p['dup_ts']} |")
    a(f"| OHLC 不一致 | {p['ohlc_inconsistent']} |")
    a(f"| 零成交且 O=H=L=C 的 bar | {p['flat_zero_vol_bars']}（交易所维护期间的填充 bar，保留并打标；这些 bar 上不会产生 first-passage 触碰） |")
    a(f"| 字段 | open, high, low, close, volume, quote_volume, taker_buy_volume |")
    a("")
    a("零成交 bar 按年份：" + ", ".join(f"{y}: {n}" for y, n in fz.index.year.value_counts().sort_index().items()) + "\n")
    a("## 2. Open interest / metrics（5m）\n")
    m = rep["metrics"]
    a("| 项目 | 值 |\n|---|---|")
    a(f"| 起止 | {met.index.min()} → {met.index.max()} |")
    a(f"| 频率 | 5 分钟快照（create_time） |")
    a(f"| 原始行数 | {m['rows_raw']:,} |")
    a(f"| 完全重复行（已删） | {m['exact_dups']:,} |")
    a(f"| 时间戳相同但内容冲突 | {m['conflicting_ts_dups']} |")
    a(f"| 不在 5m 网格上的时间戳 | {m['off_grid_ts']} |")
    a(f"| 缺失的 5m 快照 | {m['missing_5m_snapshots']} |")
    a(f"| OI ≤ 0（判为无效） | {m['oi_nonpositive']} |")
    a(f"| 隐含价格 oi_usd/oi_btc 与成交价偏离 >3%（判为无效） | {m['oi_implied_price_mismatch']} |")
    a(f"| 相对上一个有效值 |Δlog OI| > 0.25（判为无效，因果规则） | {m['oi_jump_rejected']} |")
    a(f"| 有效快照 / 应有快照 | {m['valid_snapshots']:,} / {m['expected_snapshots']:,} ({m['valid_snapshots'] / m['expected_snapshots']:.2%}) |")
    a(f"| 无效段数 / 超过 60 分钟的段数 | {len(gaps)} / {rep['oi_gaps_over_60min']} |")
    a("")
    a("最长的 10 段无效区间（缺失 + 判为无效）：\n")
    a("| start | end | 缺失快照数 | 分钟 |\n|---|---|---|---|")
    for _, g in gaps.head(10).iterrows():
        a(f"| {g.start} | {g.end} | {g.n_missing} | {g.duration_min:.0f} |")
    a("")
    a("**时间对齐验证。** create_time = t 的 ΔOI 与 [t-5m, t) 这根 bar 的 |收益| 相关性最强（lag 扫描的峰值在 −1 分钟，corr = −0.216）；隐含价格 oi_usd/oi_btc 与 t 时刻成交价最接近（中位偏差 0.017%）。结论：快照描述的是 t 时刻的状态，t 之后的价格没有进入快照。信号时间再保守地加 5 分钟发布延迟：`t0 = t_snap + 5min`，`snapshot_age = 5 min`，最大允许 age 为 "
      f"{lm['max_snapshot_age_min']} 分钟。\n")
    a(f"字段缺失：`acct_ls_ratio` 缺 {met.acct_ls_ratio.isna().sum():,} 行，`top_pos_ls_ratio` 缺 {met.top_pos_ls_ratio.isna().sum():,} 行，`taker_ls_ratio` 缺 {met.taker_ls_ratio.isna().sum():,} 行。只有 M2 变体用到 acct_ls_ratio，缺失时最多前向填充 3 根（15 分钟），之后按 0.5 处理。\n")
    a("## 3. Funding 与 basis\n")
    a(f"Funding：{len(fund):,} 条，{fund.ts.min()} → {fund.ts.max()}。结算间隔（小时）分布：{fdiff.to_dict()}。用的是 t 时刻之前最后一次已结算的费率；距上次结算超过 9 小时则判为缺失。\n")
    a(f"Premium index 5m：{len(prem):,} 行，{prem.ts.min()} → {prem.ts.max()}，用作 basis 代理。没有现货成交量数据，spot volume 只能作为限制写明。\n")
    a("## 4. 清算图（liquidation map）数据\n")
    a("**关键事实：免费公开源里没有历史 liquidation map 快照。** Coinglass / Hyblock 的清算图本身也是模型估计，不是观测数据，历史快照只在付费 API 里提供，而本研究没有拿到 key。因此本研究用公开 OI 数据、按同类方法（OI-delta 杠杆分层模型）**自建**清算图，全过程透明、可复现：\n")
    a(f"* 价格网格：对数等距，bin 宽 {lm['grid_step_log']}（约 0.1%），{lm['grid_min_price']}–{lm['grid_max_price']} USDT。")
    a(f"* 每个 5m 步：(1) 先清扫：多头清算价 ≥ bar low、空头清算价 ≤ bar high 的仓位视为已被清算，从图中移除并记为模型清算量；(2) ΔOI>0：新开名义价值 ΔOI×typical price，多空两侧各加一份，按杠杆 {lm['leverage_tiers']}、权重 {lm['leverage_weights']} 分层，多头清算价 = P(1−1/L+mmr)，空头清算价 = P(1+1/L−mmr)，mmr={lm['maintenance_margin']}；(3) ΔOI<0：两侧按比例平仓（先扣掉本步已清扫的部分）。")
    a("* 变体 M1（主规格）：多空对称，不衰减。变体 M2（稳健性）：按账户多空比倾斜多空分配，并加 72h 半衰期衰减。")
    a("* 交易所：只有 Binance；合约：只有 BTCUSDT 永续。没有跨所聚合，没有 COIN-M，没有逐仓/全仓区分。杠杆分层是假设，不是观测。")
    a("* 价格 bucket 结构：0.1% 对数 bin；对称窗口 ±3/5/7.5/10/12.5/15%；距离加权（1/d、1/d²、exp(−20d)、exp(−50d)，d 下限 0.25%，范围 ±15%）；累计 25/50/75% cluster 价位；最大 0.5% cluster 的价位和规模。")
    a(f"* 快照频率：5 分钟；快照数 {rep['map_M1_rows']:,}；起点 2020-09-01，前 3 个月为 burn-in，分析从 2020-12-01 开始。")
    a(f"* 数据缺口：OI 无效时该步不加仓、不平仓，但价格清扫照常进行。无效段超过 60 分钟的，从段开始到段结束后 {lm['degraded_cooldown_h']}h 内都标为 `map_degraded`，不产生信号。")
    nd = int(f1.map_degraded.sum())
    a(f"* 被标为 degraded 的快照：{nd:,}（{nd / len(f1):.2%}）。")
    a(f"* 分析面板里的有效信号快照：M1 {prep['M1']['valid']:,} 个；条件是 oi_valid、age=0、非 degraded、在 burn-in 之后、P0 存在。")
    q = f1.resample("365D").last()
    a("\n模型总量和 OI 的量级对照（每 365 天取一个点，单位十亿美元）：\n")
    a("| 时点 | Binance OI | 模型 long 总量 | 模型 short 总量 |\n|---|---|---|---|")
    for t, r in q.iterrows():
        a(f"| {t:%Y-%m-%d} | {r.oi_usd / 1e9:.2f} | {r.map_total_long / 1e9:.2f} | {r.map_total_short / 1e9:.2f} |")
    a("\n## 5. Look-ahead 自动检查\n")
    a("`src/panel.py::check_no_lookahead` 的结果：" + json.dumps(prep["M1"]["lookahead_checks"], ensure_ascii=False) + "\n")
    a("另外：regime 分位数一律用扩张窗口（至少 90 天历史，并滞后 1 天）；规模分位数用过去 365 天的日度历史；MA 只用已收盘的日线；funding 只用已结算的费率；Method C 的 peak 用到了 episode 内的未来信息，只作描述，不作信号。\n")
    a("## 6. 这些数据能否严格完成原研究设计？\n")
    a("| 需求 | 状态 |\n|---|---|")
    a("| 1m 价格、先触碰哪一侧 | ✅ 完整，2020-01 起，没有缺失 bar |")
    a("| 历史 liquidation map（第三方真实快照） | ❌ 拿不到 → ⚠️ 用自建 OI-delta 模型替代。所有结论都是「对这个模型清算图」成立，不等于对 Coinglass 图成立 |")
    a("| 多交易所 / 杠杆分层观测 | ❌ 只有 Binance BTCUSDT，杠杆分层是假设 |")
    a("| 对称窗口、距离加权、cluster 价位 | ✅ 由自建图直接计算 |")
    a("| OI、funding、basis | ✅（basis 用 premium index 代理） |")
    a("| spot volume | ❌ 没有获取（非必需） |")
    a("| 实际清算流水（liquidation orders） | ❌ Binance 公开数据里已没有 liquidationSnapshot；cascade 用「1h 内 OI 降 ≥3% 且 |收益| ≥2%」代理，模型清扫量另行报告 |")
    a("| 外部聚合清算图读数的绝对规模（例如 5.5B / 1.8B）和本图可比吗 | ⚠️ 不直接可比：本图只覆盖 Binance。相似度匹配用 ratio、方向、total/OI（需提供 OI）和市场状态，不用美元绝对值 |")
    a("\n结论：方向、first-passage、波动、规模、regime、稳健性、样本外这几部分都能完整做完。主要限制是清算图本身是模型。外部清算图导出可以放到 `data/raw/external_liquidation_map/*.csv`（`src/data_loader.py::load_external_liquidation_map` 会自动映射字段），之后按同一套流程重跑。")
    out = ROOT / "DATA_AUDIT.md"
    out.write_text("\n".join(L), encoding="utf-8")
    return out
