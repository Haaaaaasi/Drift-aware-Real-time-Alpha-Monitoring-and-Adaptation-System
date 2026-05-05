"""Demo：用 2026 Q1 模擬資料展示「4 層 monitoring + 3 種 adaptation」的實際輸出。

設定
----
* Reference 窗口：2025-10-01 → 2025-12-31（93 個交易日，作為訓練分布基準）
* Production 窗口：2026-01-01 → 2026-04-17（66 個交易日，剛剛模擬的時段）
* Universe：10 檔權值股（與策略 C 一致）

對照三件事
----------
1. 4 層 monitor 各自會回報什麼 metric / 觸發什麼 alert
2. 3 種 adaptation policy 在這段期間會不會觸發、為什麼
3. 這些數字在 Grafana 4 個面板上各自出現在哪
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pipelines.daily_batch_pipeline import compute_python_alphas, load_csv_data
from src.adaptation.performance_trigger import PerformanceTriggeredAdapter
from src.adaptation.recurring_concept import RecurringConceptPool
from src.adaptation.scheduler import ScheduledRetrainer
from src.common.logging import get_logger, setup_logging
from src.labeling.label_generator import LabelGenerator
from src.meta_signal.ml_meta_model import MLMetaModel
from src.monitoring.alpha_monitor import AlphaMonitor
from src.monitoring.data_monitor import DataMonitor
from src.monitoring.model_monitor import ModelMonitor
from src.monitoring.strategy_monitor import StrategyMonitor

setup_logging()
logger = get_logger("demo_monitoring_2026")

CSV = "data/tw_stocks_tej.parquet"
BLUECHIPS = ["1301", "2002", "2303", "2308", "2317", "2330", "2357", "2382", "2412", "2454"]
EFFECTIVE_ALPHAS = json.loads(Path("reports/alpha_ic_analysis/effective_alphas.json").read_text())["effective_alphas"]

REF_START, REF_END = pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31")
PROD_START, PROD_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-17")

OUT = Path("reports/demo_monitoring_2026")
OUT.mkdir(parents=True, exist_ok=True)


def section(title: str) -> None:
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def main() -> None:
    # ---------- 載入並切兩個窗口 ----------
    bars = load_csv_data(CSV)
    bars["security_id"] = bars["security_id"].astype(str)
    bars = bars[bars["security_id"].isin(BLUECHIPS)].reset_index(drop=True)

    alphas = compute_python_alphas(bars)
    alphas = alphas[alphas["alpha_id"].isin(EFFECTIVE_ALPHAS)]

    label_gen = LabelGenerator(horizons=[5], bar_type="daily")
    labels = label_gen.generate_labels(bars[["security_id", "tradetime", "close"]])
    fwd5 = labels[labels["horizon"] == 5].set_index(
        ["security_id", "signal_time"]
    )["forward_return"].dropna()
    fwd5.index = fwd5.index.set_names(["security_id", "tradetime"])

    bars_ref = bars[(bars["tradetime"] >= REF_START) & (bars["tradetime"] <= REF_END)]
    bars_prod = bars[(bars["tradetime"] >= PROD_START) & (bars["tradetime"] <= PROD_END)]
    alphas_ref = alphas[(alphas["tradetime"] >= REF_START) & (alphas["tradetime"] <= REF_END)]
    alphas_prod = alphas[(alphas["tradetime"] >= PROD_START) & (alphas["tradetime"] <= PROD_END)]
    fwd5_ref = fwd5[(fwd5.index.get_level_values("tradetime") >= REF_START)
                    & (fwd5.index.get_level_values("tradetime") <= REF_END)]
    fwd5_prod = fwd5[(fwd5.index.get_level_values("tradetime") >= PROD_START)
                     & (fwd5.index.get_level_values("tradetime") <= PROD_END)]

    print(f"Reference: {REF_START.date()} → {REF_END.date()}  bars={len(bars_ref)}  alphas={len(alphas_ref)}")
    print(f"Production: {PROD_START.date()} → {PROD_END.date()}  bars={len(bars_prod)}  alphas={len(alphas_prod)}")

    # ---------- 訓練模型於 reference 期間 ----------
    section("0. 訓練 reference model（用 2025-10 ~ 2025-12 資料）")
    model = MLMetaModel(feature_columns=EFFECTIVE_ALPHAS)
    train_panel = alphas[alphas["tradetime"] <= REF_END]
    train_labels = fwd5[fwd5.index.get_level_values("tradetime") <= REF_END]
    info = model.train(train_panel, train_labels)
    print(f"  訓練樣本數: {info['n_train']:,}")
    print(f"  Holdout IC : {info['holdout_metrics']['ic']:.4f}")
    print(f"  Holdout rank-IC: {info['holdout_metrics']['rank_ic']:.4f}")

    # 對 production 期間 predict（拿來算 model monitor）
    pred_prod = model.predict(alphas_prod).rename(columns={"tradetime": "signal_time"})
    pred_ref = model.predict(alphas_ref).rename(columns={"tradetime": "signal_time"})

    all_metrics: list[dict] = []

    # ---------- Layer 1: Data Monitor ----------
    section("1️⃣  Data Monitor — 原始資料品質")
    data_mon = DataMonitor()
    # 用 reference 期間的 close 當基準分布
    ref_close = bars_ref["close"].dropna().values
    metrics_data = data_mon.run(bars_prod, reference_features=ref_close)
    all_metrics.extend(metrics_data)
    _print_metrics(metrics_data, "data_monitor")

    # ---------- Layer 2: Alpha Monitor ----------
    section("2️⃣  Alpha Monitor — 因子分布漂移與相關性")
    alpha_mon = AlphaMonitor()
    # 為每個 alpha 建立 reference 分布
    ref_alpha_vals = {
        aid: grp["alpha_value"].dropna().values
        for aid, grp in alphas_ref.groupby("alpha_id")
    }
    # baseline 相關矩陣
    pivot_ref = alphas_ref.pivot_table(
        index=["security_id", "tradetime"], columns="alpha_id", values="alpha_value"
    )
    base_corr = pivot_ref.corr()
    metrics_alpha = alpha_mon.run(
        alphas_prod, fwd5_prod,
        baseline_corr_matrix=base_corr,
        reference_alpha_values=ref_alpha_vals,
    )
    all_metrics.extend(metrics_alpha)
    _print_metrics(metrics_alpha, "alpha_monitor", limit=15)

    # ---------- Layer 3: Model Monitor ----------
    section("3️⃣  Model Monitor — 模型預測分布、校準、方向準確率")
    model_mon = ModelMonitor()
    # 對齊 prediction 與 forward return
    pred_idx = pred_prod.set_index(["security_id", "signal_time"])["signal_score"]
    pred_idx.index = pred_idx.index.set_names(["security_id", "tradetime"])
    common = pred_idx.index.intersection(fwd5_prod.index)
    metrics_model = model_mon.run(
        predictions=pred_idx.loc[common],
        actuals=fwd5_prod.loc[common],
        reference_predictions=pred_ref["signal_score"].dropna().values,
    )
    all_metrics.extend(metrics_model)
    _print_metrics(metrics_model, "model_monitor")

    # ---------- Layer 4: Strategy Monitor ----------
    section("4️⃣  Strategy Monitor — 策略層級績效退化")
    strat_mon = StrategyMonitor()
    # 用策略 C 的 daily_pnl
    pnl = pd.read_csv("reports/simulations/sim_20260101_20260417_top10_rt5_bluechip/daily_pnl.csv")
    daily_ret = pd.Series(
        pnl["net_return"].values,
        index=pd.to_datetime(pnl["date"]),
    )
    metrics_strat = strat_mon.run(daily_ret)
    all_metrics.extend(metrics_strat)
    _print_metrics(metrics_strat, "strategy_monitor")

    # ---------- 統計 alert 嚴重度 ----------
    section("📊 Alert 統計（4 層 monitor 加總）")
    severities: dict[str, int] = {"CRITICAL": 0, "WARNING": 0, "INFO": 0}
    for m in all_metrics:
        s = m.get("severity")
        if s:
            severities[s] = severities.get(s, 0) + 1
    print(f"  CRITICAL: {severities['CRITICAL']}")
    print(f"  WARNING : {severities['WARNING']}")
    print(f"  INFO    : {severities['INFO']}")
    print(f"  總 metric 數: {len(all_metrics)}（無 severity = 純資訊不需告警）")

    # ---------- Adaptation: 3 條 policy ----------
    section("🔄  Adaptation — 三條 policy 在這段期間會做什麼")

    # Policy 1: Scheduled
    print("\n[Policy 1] Scheduled Retrain（每 7 日重訓）")
    sched = ScheduledRetrainer(retrain_interval_days=7)
    n_days = (PROD_END - PROD_START).days
    n_retrain = n_days // 7
    print(f"  → 從 {PROD_START.date()} 到 {PROD_END.date()} 共 {n_days} 日")
    print(f"  → 預期會重訓 {n_retrain} 次（不論模型表現好壞，固定排程）")

    # Policy 2: Performance-triggered
    print("\n[Policy 2] Performance-triggered（IC 連續退化或 Critical alert 累積）")
    perf = PerformanceTriggeredAdapter(
        ic_threshold=0.0,
        ic_consecutive_days=5,
        sharpe_threshold=0.0,
        sharpe_consecutive_days=10,
        critical_alert_limit=3,
    )
    # 用 production 的 daily IC 當訊號
    daily_ics = []
    for d, grp in pred_prod.groupby("signal_time"):
        actual_d = fwd5_prod[fwd5_prod.index.get_level_values("tradetime") == d]
        if len(actual_d) >= 3 and len(grp) >= 3:
            merged = grp.set_index("security_id")["signal_score"].reindex(
                actual_d.index.get_level_values("security_id")
            ).dropna()
            if len(merged) >= 3:
                ic_d = np.corrcoef(merged.values, actual_d.values[:len(merged)])[0, 1]
                if not np.isnan(ic_d):
                    daily_ics.append(ic_d)
    rolling_ic_recent = pd.Series(daily_ics).rolling(10).mean().dropna()
    rolling_sharpe = daily_ret.rolling(20).apply(
        lambda x: (x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0
    ).dropna()
    n_critical = severities["CRITICAL"]
    should_trigger, reason = perf.check_trigger(
        rolling_ic_series=pd.Series(daily_ics),
        rolling_sharpe_series=rolling_sharpe,
        critical_alert_count=n_critical,
    )
    print(f"  → Daily IC 樣本數: {len(daily_ics)}, 平均: {np.mean(daily_ics):.4f}")
    print(f"  → 累積 CRITICAL alerts: {n_critical}")
    print(f"  → 是否觸發: {'✅ YES' if should_trigger else '❌ NO'}")
    print(f"  → 原因: {reason}")

    # Policy 3: Recurring concept — 離線示範 fingerprint + cosine similarity
    print("\n[Policy 3] Recurring Concept Pool（找歷史相似 regime）")
    pool = RecurringConceptPool(similarity_threshold=0.7)
    ref_fp = pool.compute_regime_fingerprint(bars_ref)
    prod_fp = pool.compute_regime_fingerprint(bars_prod)
    # 額外抓 2024 Q3 / 2024 Q4 / 2025 H1 三段歷史 regime 做比對
    historical_periods = {
        "2024_Q3": (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-09-30")),
        "2024_Q4": (pd.Timestamp("2024-10-01"), pd.Timestamp("2024-12-31")),
        "2025_H1": (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-30")),
        "2025_Q4": (REF_START, REF_END),
    }
    print(f"  → Production (2026-Q1) fingerprint:")
    for k, v in prod_fp.items():
        print(f"      {k:<22} = {v:>10.5f}")
    print(f"\n  → 與歷史 regime 的 cosine similarity:")
    prod_vec = np.array(list(prod_fp.values()))
    best_sim, best_id = 0.0, None
    for hid, (s, e) in historical_periods.items():
        hist_bars = bars[(bars["tradetime"] >= s) & (bars["tradetime"] <= e)]
        if hist_bars.empty:
            continue
        hist_fp = pool.compute_regime_fingerprint(hist_bars)
        hist_vec = np.array([hist_fp.get(k, 0.0) for k in prod_fp.keys()])
        dot = np.dot(prod_vec, hist_vec)
        norm = np.linalg.norm(prod_vec) * np.linalg.norm(hist_vec)
        sim = float(dot / max(norm, 1e-10))
        flag = "←最相似" if sim > best_sim else ""
        print(f"      {hid}: similarity = {sim:>+.4f}  {flag}")
        if sim > best_sim:
            best_sim, best_id = sim, hid
    print(f"\n  → 最終決策: similarity={best_sim:.3f}, threshold=0.7")
    print(f"  → {'✅ 重用 ' + best_id + ' 的模型' if best_sim >= 0.7 else '❌ 沒有夠相似的歷史 regime → 觸發新訓練'}")

    # ---------- Grafana mapping ----------
    section("📈 Grafana Dashboard Mapping — 哪個數字出現在哪個面板")
    grafana_map = """
  ╔══════════════════════════════════════════════════════════════════╗
  ║ 面板 1: data_monitor.json   →  訪問 http://localhost:3000        ║
  ║   • Missing ratio time-series   ← DataMonitor.missing_ratio      ║
  ║   • Abnormal price ratio        ← DataMonitor.abnormal_*_ratio   ║
  ║   • KS p-value (close)          ← feature_dist_shift_pvalue      ║
  ║   • PSI (close)                 ← feature_dist_shift_psi  ⭐     ║
  ║   • Alert table (last 7 days)   ← alerts table (severity 過濾)   ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║ 面板 2: alpha_monitor.json                                        ║
  ║   • Per-alpha rolling IC        ← AlphaMonitor.alpha_ic          ║
  ║   • Per-alpha rolling rank-IC   ← AlphaMonitor.alpha_rank_ic     ║
  ║   • Alpha turnover              ← AlphaMonitor.alpha_turnover    ║
  ║   • Alpha PSI per factor  ⭐    ← AlphaMonitor.alpha_value_psi   ║
  ║   • Correlation drift           ← AlphaMonitor.alpha_corr_drift  ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║ 面板 3: model_monitor.json                                        ║
  ║   • Directional accuracy        ← ModelMonitor.directional_acc.  ║
  ║   • Calibration ECE  ⭐         ← ModelMonitor.calibration_ece   ║
  ║   • Prediction KS               ← prediction_dist_drift_pvalue   ║
  ║   • 4 個 Stat 摘要              ← 上述 metrics 的 latest          ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║ 面板 4: strategy_monitor.json                                     ║
  ║   • Rolling Sharpe              ← StrategyMonitor.rolling_sharpe ║
  ║   • Max drawdown                ← StrategyMonitor.max_drawdown   ║
  ║   • Realized vs Expected        ← realized_minus_expected        ║
  ║   • Model Registry table        ← model_registry table           ║
  ╚══════════════════════════════════════════════════════════════════╝

  打開 Grafana 的兩個前置條件：
    1. docker-compose up -d        （啟動 PostgreSQL + Grafana）
    2. python -m pipelines.daily_batch_pipeline --data-source tej
       （把 metrics 寫入 PostgreSQL，alert_mgr.persist_metrics() 自動執行）
"""
    print(grafana_map)

    # ---------- 寫檔 ----------
    pd.DataFrame(all_metrics).to_csv(OUT / "all_monitoring_metrics.csv", index=False)
    print(f"\n✅ 完整 metrics 已寫入: {OUT / 'all_monitoring_metrics.csv'}")


def _print_metrics(metrics: list[dict], monitor_name: str, limit: int = 10) -> None:
    if not metrics:
        print(f"  （{monitor_name} 無 metric 產出）")
        return
    rows = []
    for m in metrics[:limit]:
        sev = m.get("severity") or ""
        sev_mark = {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "🔵"}.get(sev, "  ")
        val = m.get("metric_value")
        val_str = f"{val:>10.4f}" if isinstance(val, (int, float)) else f"{'NA':>10}"
        dim = m.get("dimension") or "-"
        rows.append(
            f"  {sev_mark} {m['metric_name']:<32} = {val_str}"
            f"  dim={str(dim):<10} [{sev or 'OK'}]"
        )
    print("\n".join(rows))
    if len(metrics) > limit:
        print(f"  ... 還有 {len(metrics) - limit} 筆未顯示")
    n_alert = sum(1 for m in metrics if m.get("severity"))
    print(f"  Σ {len(metrics)} metrics, {n_alert} 個 alert")


if __name__ == "__main__":
    main()
