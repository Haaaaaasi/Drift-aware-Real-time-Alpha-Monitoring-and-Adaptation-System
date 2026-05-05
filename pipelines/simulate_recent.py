"""Walk-forward 模擬：從 start 走到 end，記錄每日持倉與績效；支援三種 adaptation 策略。

設計原則
--------
* **Alpha 只算一次**：對全歷史一次性計算所有 alpha 值，模擬迴圈中只做切片。
* **Adaptation 策略**：以 ``strategy`` 參數切換三種對照組
  - ``none``     ：僅在起始日訓練一次，之後凍結模型（MVP v3 的 no-adapt baseline）
  - ``scheduled``：每 ``retrain_every`` 個交易日重訓一次（Policy 1）
  - ``triggered``：依 rolling IC / Sharpe 退化觸發重訓（Policy 2），有 ``min_retrain_gap`` 冷卻期
* **無 leakage 保證**：訓練 y 以 label_available_at <= T 為成熟門檻（依實際 trading
  bar 推算，非曆日加法）；訓練 X 以 tradetime <= T - purge_days 為 feature-side purge；
  rolling IC 監控使用 [t-trigger_window_days, t-trigger_eval_gap_days] 雙邊界窗口。
* **執行模型**：T 日收盤建倉 → T+1 日收盤平倉/重平衡（與 PaperTradingEngine 一致）。
* **Slippage**：以 turnover × slippage_bps 模擬，預設 5 bps（與 paper_engine 一致）。
* **Universe-by-day**：cross-section 由 alpha_panel[tradetime==t] 自然定義；下市股
  在最後一筆 OHLCV 之後不再進入 alpha_panel → 自動退出 universe。下市日當天的
  next_return 為 NaN（沒有 t+1 close）→ 在 gross_return 計算中以 0 處理（保守）。
  此規則無需顯式邏輯，TEJ data source 啟用後即生效。

輸出
----
``reports/simulations/<run_id>/``
  * ``holdings.csv``   — 每天持倉明細（date, security_id, weight, signal_score, last_close）
  * ``daily_pnl.csv``  — 每天組合報酬（date, gross_return, commission_cost, tax_cost,
    slippage_cost, net_return, cumulative_value, n_holdings, turnover,
    rolling_ic, rolling_sharpe）
  * ``retrain_log.csv`` — 每次重訓紀錄（date, reason, n_train, train_ic, train_rank_ic）
  * ``summary.txt``    — 累積報酬 / 年化 / Sharpe / Max DD / Win rate / 平均持倉數 / 重訓次數

使用範例
--------
    # 預設：TEJ survivorship-correct 資料 + scheduled 策略，每 5 日重訓一次
    python -m pipelines.simulate_recent

    # Triggered 策略（rolling IC 連 3 日 < 0 或 Sharpe 連 10 日 < 0 就重訓）
    python -m pipelines.simulate_recent \\
        --strategy triggered --start 2022-01-01 --end 2024-12-31

    # No-adapt：訓練一次後凍結
    python -m pipelines.simulate_recent \\
        --strategy none --start 2022-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from pipelines.daily_batch_pipeline import compute_python_alphas, load_csv_data
from src.alpha_engine.alpha_cache import cache_path_for_data_path
from src.adaptation.performance_trigger import PerformanceTriggeredAdapter
from src.adaptation.recurring_concept import compute_alpha_ic_stats
from src.common.logging import get_logger, setup_logging
from src.common.metrics import information_coefficient
from src.config.alpha_selection import EFFECTIVE_ALPHAS_PATH, load_effective_alpha_ids
from src.config.constants import DATA_SOURCE_DEFAULT_PATHS, DEFAULT_DATA_SOURCE
from src.labeling.label_generator import LabelGenerator
from src.meta_signal.ml_meta_model import MLMetaModel
from src.portfolio.constructor import PortfolioConstructor
from src.risk.risk_manager import RiskManager

Strategy = Literal["none", "scheduled", "triggered", "model_pool"]

setup_logging()
logger = get_logger("simulate_recent")


DEFAULT_OUT_DIR = Path("reports/simulations")

DATA_SOURCE_DEFAULTS = DATA_SOURCE_DEFAULT_PATHS


def _load_alphas_from_dolphindb(
    start: date,
    end: date,
    buffer_days: int = 365,
    alpha_ids: list[str] | None = None,
) -> pd.DataFrame:
    """從 DolphinDB ``dfs://darams_alpha`` 的 ``alpha_features`` 表讀取 WQ101 alpha。

    回傳格式與 ``compute_python_alphas`` 相同：``security_id, tradetime, alpha_id, alpha_value``。
    security_id 轉為 str 以對齊 bars。
    """
    from src.common.db import get_dolphindb

    client = get_dolphindb()
    buffer_start = (pd.Timestamp(start) - pd.Timedelta(days=buffer_days)).strftime("%Y.%m.%d")
    end_str = pd.Timestamp(end).strftime("%Y.%m.%d")

    where_alpha = ""
    if alpha_ids:
        ids_str = ",".join(f'"{a}"' for a in alpha_ids)
        where_alpha = f" and alpha_id in [{ids_str}]"

    script = (
        "select security_id, tradetime, alpha_id, alpha_value "
        'from loadTable("dfs://darams_alpha", "alpha_features") '
        f'where tradetime between {buffer_start} : {end_str} '
        'and bar_type = "daily"'
        f'{where_alpha}'
    )
    logger.info("dolphindb_alpha_query", start=buffer_start, end=end_str,
                filter_alphas=len(alpha_ids) if alpha_ids else 0)
    df = client.run(script)
    if df is None or len(df) == 0:
        raise RuntimeError(
            f"DolphinDB alpha_features 在 {buffer_start}~{end_str} 沒有資料；"
            "請先執行 backfill：python -m scripts.backfill_alpha"
        )

    df["security_id"] = df["security_id"].astype(str)
    df["tradetime"] = pd.to_datetime(df["tradetime"])
    df = df.dropna(subset=["alpha_value"])
    logger.info("dolphindb_alpha_loaded", rows=len(df),
                n_alphas=df["alpha_id"].nunique(),
                n_securities=df["security_id"].nunique())
    return df[["security_id", "tradetime", "alpha_id", "alpha_value"]].reset_index(drop=True)


def _load_effective_alphas() -> list[str] | None:
    return load_effective_alpha_ids(EFFECTIVE_ALPHAS_PATH, required=True)


def _trading_days(bars: pd.DataFrame, start: date, end: date) -> list[pd.Timestamp]:
    days = pd.to_datetime(bars["tradetime"]).dt.normalize().drop_duplicates().sort_values()
    days = days[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]
    return list(days)


def _next_day_returns(bars: pd.DataFrame) -> pd.DataFrame:
    """為每個 (security_id, tradetime) 計算 close[t+1]/close[t] - 1。

    Returns
    -------
    DataFrame 索引為 (security_id, tradetime)，欄位 ``next_return``。
    """
    bars_sorted = bars.sort_values(["security_id", "tradetime"]).copy()
    bars_sorted["next_close"] = bars_sorted.groupby("security_id")["close"].shift(-1)
    bars_sorted["next_return"] = bars_sorted["next_close"] / bars_sorted["close"] - 1
    return bars_sorted.set_index(["security_id", "tradetime"])[["next_return"]]


def _filter_universe(
    bars: pd.DataFrame,
    symbols: list[str] | None,
    min_turnover_ntd: float,
    sim_start: date,
    lookback_days: int = 60,
) -> tuple[pd.DataFrame, str]:
    """根據白名單與流動性門檻過濾 universe。

    Liquidity 計算：以 sim_start 前 lookback_days 個交易日的平均成交金額
    （vol × close）為基準。低於門檻者排除。

    Returns
    -------
    (過濾後 bars, 描述字串)
    """
    desc_parts = []
    if symbols:
        sym_set = {str(s) for s in symbols}
        bars["security_id_str"] = bars["security_id"].astype(str)
        bars = bars[bars["security_id_str"].isin(sym_set)].drop(columns="security_id_str")
        desc_parts.append(f"symbols={len(sym_set)}")

    if min_turnover_ntd > 0:
        sim_start_ts = pd.Timestamp(sim_start)
        lookback = bars[
            (bars["tradetime"] < sim_start_ts)
            & (bars["tradetime"] >= sim_start_ts - pd.Timedelta(days=lookback_days * 2))
        ].copy()
        lookback["turnover_value"] = lookback["vol"] * lookback["close"]
        avg_turnover = lookback.groupby("security_id")["turnover_value"].mean()
        keep = set(avg_turnover[avg_turnover >= min_turnover_ntd].index)
        before = bars["security_id"].nunique()
        bars = bars[bars["security_id"].isin(keep)]
        after = bars["security_id"].nunique()
        desc_parts.append(f"min_turnover={min_turnover_ntd:.0f}_NTD ({before}→{after} 檔)")

    return bars.reset_index(drop=True), ", ".join(desc_parts) if desc_parts else "no_filter"


def simulate(
    csv_path: str | Path,
    start: date,
    end: date,
    strategy: Strategy = "scheduled",
    retrain_every: int = 5,
    purge_days: int = 5,
    horizon_days: int = 5,
    top_k: int = 10,
    portfolio_method: str = "equal_weight_topk",
    rebalance_every: int = 1,
    entry_rank: int = 20,
    exit_rank: int = 40,
    max_turnover: float = 1.0,
    min_holding_days: int = 0,
    objective: str = "forward_return",
    capital: float = 10_000_000.0,
    slippage_bps: float = 5.0,
    commission_rate: float = 0.000926,
    tax_rate: float = 0.003,
    round_trip_cost_pct: float | None = None,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    symbols: list[str] | None = None,
    min_turnover_ntd: float = 0.0,
    run_tag: str | None = None,
    trigger_ic_threshold: float = 0.0,
    trigger_ic_days: int = 3,
    trigger_sharpe_threshold: float = 0.0,
    trigger_sharpe_days: int = 10,
    min_retrain_gap: int = 20,
    rolling_window: int = 20,
    trigger_window_days: int = 60,
    trigger_eval_gap_days: int = 20,
    shadow_warmup_days: int = 5,
    alpha_source: Literal["python", "dolphindb"] = "python",
    alpha_ids: list[str] | None = None,
    skip_effective_filter: bool = False,
    similarity_threshold: float = 0.5,  # Phase B-1：對應 exp(-d/2) >= 0.5（d <= 1.4 std）
    pool_regime_window: int = 60,
    shadow_window: int = 20,
    pool_top_k: int = 3,  # Phase B-3：shadow 階段最多納入幾個 reused 候選
    train_window_days: int | None = 500,
    allow_yfinance: bool = False,
) -> dict[str, Any]:
    """跑 walk-forward 模擬並寫出結果檔案。

    Parameters
    ----------
    strategy
        * ``none``      ：僅在起始日訓練一次，之後凍結模型（MVP v3 baseline）
        * ``scheduled`` ：每 ``retrain_every`` 個交易日重訓一次（Policy 1）
        * ``triggered`` ：依 rolling IC / Sharpe 退化觸發重訓（Policy 2）
    retrain_every
        僅在 ``strategy='scheduled'`` 時生效。
    trigger_ic_threshold, trigger_ic_days
        僅在 ``strategy='triggered'`` 時生效：rolling IC 連 ``trigger_ic_days`` 天 <=
        ``trigger_ic_threshold`` 就觸發重訓。
    trigger_sharpe_threshold, trigger_sharpe_days
        同上，但針對 rolling Sharpe。
    min_retrain_gap
        Triggered 策略的冷卻期：兩次重訓至少要間隔 ``min_retrain_gap`` 個交易日，避免過度重訓。
    rolling_window
        （legacy）僅作為樣本數下限的位置切片參考；實際 IC/Sharpe 計算改採 ``trigger_window_days``
        / ``trigger_eval_gap_days`` 雙邊界。
    trigger_window_days, trigger_eval_gap_days
        Trigger 用 rolling IC / Sharpe 的計算範圍：``signal_time ∈ [t-trigger_window_days,
        t-trigger_eval_gap_days]``（calendar days）。預設 [t-60, t-20]，與 model_pool 的 shadow
        eval 窗口 [t-30, t-10] 完全不重疊，避免「trigger 與 shadow 吃同一段樣本」造成 selection bias。
    shadow_warmup_days
        Model_pool 策略的 shadow 候選訓練 cutoff 額外往前推 ``shadow_warmup_days`` 日，
        避免新候選用 shadow window 之內的資料訓練（IS leakage）。
    commission_rate, tax_rate, round_trip_cost_pct
        交易成本：commission per-side、tax sell-side only、slippage 仍由 slippage_bps 控制。
        若 ``round_trip_cost_pct`` 不為 None，將忽略三細項，改用單一 round-trip rate
        （供 cost-sensitivity sweep 使用：例如 0 / 0.2 / 0.4 / 0.6）。
    horizon_days
        Label 前向 horizon，用於判斷哪些過往預測的標籤已成熟可計算 rolling IC。
    train_window_days
        訓練窗口（calendar days）。``None``（預設）= expanding，從第一筆歷史資料擴展至
        purge_cutoff。設為正整數（例如 500）= rolling window，訓練集限制在
        ``[purge_cutoff - train_window_days, purge_cutoff]``。Rolling 模式下模型會忘掉舊
        regime，model_pool 相對優勢預計上升——可作為對照組實驗使用。

    Returns
    -------
    dict
        {run_dir, summary_metrics, holdings_path, daily_pnl_path, retrain_log_path}
    """
    if strategy not in ("none", "scheduled", "triggered", "model_pool"):
        raise ValueError(f"strategy 必須是 none/scheduled/triggered/model_pool，得到 {strategy!r}")

    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    tag = f"_{run_tag}" if run_tag else ""
    strat_suffix = {
        "none": "none",
        "scheduled": f"sched{retrain_every}",
        "triggered": "trig",
        "model_pool": "pool",
    }[strategy]
    run_id = f"sim_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}_top{top_k}_{strat_suffix}{tag}"
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("simulation_start", run_id=run_id, strategy=strategy, period=f"{start} → {end}")

    # --- 1. 一次性載入與計算 ---
    bars = load_csv_data(csv_path, allow_yfinance=allow_yfinance)
    # 統一 security_id 為 str，避免 alpha cache（str）與 CSV（int64）類型不一致
    # 造成下游 alpha_panel 與 labels 的 (security_id, tradetime) 索引 join 失敗
    bars["security_id"] = bars["security_id"].astype(str)
    bars, filter_desc = _filter_universe(bars, symbols, min_turnover_ntd, start)
    logger.info(
        "bars_loaded",
        rows=len(bars),
        symbols=int(bars["security_id"].nunique()),
        filter=filter_desc,
    )
    if bars["security_id"].nunique() == 0:
        raise RuntimeError(f"Universe filter 後沒有任何標的：{filter_desc}")

    if alpha_source == "dolphindb":
        # 若未指定 alpha_ids，預先載入 effective_alphas 過濾條件推入 SQL 查詢
        # 避免拉取全量 53.8M rows 再在 Python 端過濾
        query_alpha_ids = alpha_ids
        if query_alpha_ids is None and not skip_effective_filter:
            query_alpha_ids = _load_effective_alphas()
        alpha_panel = _load_alphas_from_dolphindb(start, end, alpha_ids=query_alpha_ids)
        # 以 bars 的 (security_id, tradetime) 內連接，排除 DolphinDB 端多餘標的
        bars_key = bars[["security_id", "tradetime"]].copy()
        bars_key["security_id"] = bars_key["security_id"].astype(str)
        before = len(alpha_panel)
        alpha_panel = alpha_panel.merge(bars_key, on=["security_id", "tradetime"], how="inner")
        logger.info("alpha_panel_aligned_to_bars", rows_before=before, rows_after=len(alpha_panel))
    else:
        alpha_panel = compute_python_alphas(
            bars,
            cache_path=cache_path_for_data_path(csv_path),
        )

    # 確保 alpha_panel 也用 str 型 security_id（cache 直接回時可能是 str，新算則跟 bars 同 → 已 str）
    alpha_panel["security_id"] = alpha_panel["security_id"].astype(str)

    eff_alphas = None
    if not skip_effective_filter:
        eff_alphas = _load_effective_alphas()
        if eff_alphas:
            alpha_panel = alpha_panel[alpha_panel["alpha_id"].isin(eff_alphas)]
            logger.info("effective_alphas_applied", count=len(eff_alphas))

    label_gen = LabelGenerator(horizons=[horizon_days], bar_type="daily")
    labels_df = label_gen.generate_labels(bars[["security_id", "tradetime", "close"]])
    labels_h = (
        labels_df[labels_df["horizon"] == horizon_days]
        .dropna(subset=["forward_return"])
        .set_index(["security_id", "signal_time"])
        .rename_axis(index=["security_id", "tradetime"])
    )
    # fwd_5: Series used by _compute_rolling_ic (interface unchanged)
    fwd_5 = labels_h["forward_return"]
    # label_avail: authoritative cutoff — label_available_at <= t means the label is mature
    label_avail = labels_h["label_available_at"]

    next_ret = _next_day_returns(bars)
    last_close_lookup = bars.set_index(["security_id", "tradetime"])["close"]

    sim_days = _trading_days(bars, start, end)
    if not sim_days:
        raise RuntimeError(f"模擬期間 {start} → {end} 在 CSV 中沒有交易日")
    logger.info("simulation_days", n=len(sim_days), first=str(sim_days[0].date()), last=str(sim_days[-1].date()))

    # --- 2. 模擬迴圈 ---
    portfolio_constructor = PortfolioConstructor(
        method=portfolio_method,
        top_k=top_k,
        long_only=True,
        entry_rank=entry_rank,
        exit_rank=exit_rank,
        min_holding_days=min_holding_days,
    )
    risk_mgr = RiskManager(max_position_weight=0.10, max_gross_exposure=1.0, max_turnover=max_turnover)

    model: MLMetaModel | None = None
    current_model_id: str | None = None
    last_train_idx = -10**6
    prev_weights: dict[str, float] = {}
    holding_days: dict[str, int] = {}
    last_portfolio_rebalance_idx = -10**6
    holdings_records: list[dict] = []
    pnl_records: list[dict] = []
    retrain_records: list[dict] = []
    past_signal_history: list[pd.DataFrame] = []  # 累積過往預測供 rolling IC 計算
    portfolio_value = capital

    # model_pool 策略的 controller（其他策略為 None）
    pool_ctrl = None
    if strategy == "model_pool":
        from src.adaptation.model_pool_strategy import ModelPoolController
        pool_ctrl = ModelPoolController(
            similarity_threshold=similarity_threshold,
            pool_regime_window=pool_regime_window,
            shadow_window=shadow_window,
            shadow_warmup_days=shadow_warmup_days,
            min_improvement_ic=0.005,
            purge_days=purge_days,
            horizon_days=horizon_days,
            top_k_candidates=pool_top_k,
        )
        pool_ctrl.initialize_run()

    trigger_adapter = PerformanceTriggeredAdapter(
        ic_threshold=trigger_ic_threshold,
        ic_consecutive_days=trigger_ic_days,
        sharpe_threshold=trigger_sharpe_threshold,
        sharpe_consecutive_days=trigger_sharpe_days,
        critical_alert_limit=10**6,  # WP9 不使用 alert-based 觸發
    )

    for i, t in enumerate(sim_days):
        # 2a. 重訓決策
        need_retrain, reason = _decide_retrain(
            strategy=strategy,
            model=model,
            day_idx=i,
            last_train_idx=last_train_idx,
            retrain_every=retrain_every,
            min_retrain_gap=min_retrain_gap,
            pnl_records=pnl_records,
            adapter=trigger_adapter,
            rolling_window=rolling_window,
        )

        if need_retrain:
            purge_cutoff = t - pd.Timedelta(days=purge_days)
            if train_window_days is not None:
                window_start = purge_cutoff - pd.Timedelta(days=train_window_days)
                train_panel = alpha_panel[
                    (alpha_panel["tradetime"] >= window_start) &
                    (alpha_panel["tradetime"] <= purge_cutoff)
                ]
                # label_avail / fwd_5 皆以 (security_id, tradetime) 為 MultiIndex，
                # tradetime 對應 signal_time，與 alpha_panel 的 tradetime 同義。
                train_labels = fwd_5[
                    (label_avail <= t) &
                    (label_avail.index.get_level_values("tradetime") >= window_start)
                ]
            else:
                # expanding window：從最早歷史累積至 purge_cutoff（train_window_days=None 時才走）
                train_panel = alpha_panel[alpha_panel["tradetime"] <= purge_cutoff]
                # Use label_available_at as authoritative maturity gate:
                # a label is safe to train on only when its availability date <= t.
                train_labels = fwd_5[label_avail <= t]
            if len(train_labels) < 100:
                logger.warning("insufficient_train_data", t=str(t.date()), n=len(train_labels))
                if model is None:
                    continue
            elif strategy == "model_pool" and pool_ctrl is not None and model is not None:
                # 非初始訓練：shadow 3-way compare
                # Phase B-2：先算 alpha-side fingerprint stats（最近 60 日成熟標籤）
                alpha_ic_stats = compute_alpha_ic_stats(
                    alpha_panel=alpha_panel,
                    fwd_returns=fwd_5,
                    label_available_at=label_avail,
                    t=t,
                    window_days=60,
                    purge_days=purge_days,
                    horizon_days=horizon_days,
                )
                try:
                    decision = pool_ctrl.decide_on_trigger(
                        t=t,
                        current_model=model,
                        current_model_id=current_model_id,
                        bars=bars,
                        alpha_panel=alpha_panel,
                        fwd_returns=fwd_5,
                        train_panel=train_panel,
                        train_labels=train_labels,
                        eff_alphas=eff_alphas if eff_alphas else None,
                        alpha_ic_stats=alpha_ic_stats,
                    )
                    model = decision.best_model
                    current_model_id = decision.best_model_id
                    last_train_idx = i
                    retrain_records.append({
                        "date": t.strftime("%Y-%m-%d"),
                        "day_idx": i,
                        "reason": decision.reason,
                        "n_train": decision.train_info["n_train"],
                        "train_ic": round(decision.train_info["holdout_metrics"].get("ic", 0.0), 4),
                        "train_rank_ic": round(decision.train_info["holdout_metrics"].get("rank_ic", 0.0), 4),
                        "similarity": round(decision.similarity, 4) if decision.similarity else None,
                    })
                    logger.info(
                        "model_pool_retrain",
                        t=str(t.date()),
                        reason=decision.reason,
                        best_id=decision.best_model_id,
                        n_candidates=len(decision.candidates_evaluated),
                    )
                except Exception as exc:
                    logger.warning("model_pool_decide_failed", t=str(t.date()), error=str(exc))
                    # 降級：訓練新模型並沿用
                    model = MLMetaModel(feature_columns=eff_alphas, objective=objective, proxy_top_k=top_k)
                    train_info = model.train(train_panel, train_labels)
                    current_model_id = train_info["model_id"]
                    last_train_idx = i
                    retrain_records.append({
                        "date": t.strftime("%Y-%m-%d"),
                        "day_idx": i,
                        "reason": "pool_fallback_retrain",
                        "n_train": train_info["n_train"],
                        "train_ic": round(train_info["holdout_metrics"].get("ic", 0.0), 4),
                        "train_rank_ic": round(train_info["holdout_metrics"].get("rank_ic", 0.0), 4),
                        "similarity": None,
                    })
            else:
                # none / scheduled / triggered，以及 model_pool 的初始訓練
                model = MLMetaModel(feature_columns=eff_alphas, objective=objective, proxy_top_k=top_k)
                train_info = model.train(train_panel, train_labels)
                current_model_id = train_info["model_id"]
                last_train_idx = i
                retrain_record = {
                    "date": t.strftime("%Y-%m-%d"),
                    "day_idx": i,
                    "reason": reason,
                    "n_train": train_info["n_train"],
                    "train_ic": round(train_info["holdout_metrics"].get("ic", 0.0), 4),
                    "train_rank_ic": round(train_info["holdout_metrics"].get("rank_ic", 0.0), 4),
                    "similarity": None,
                }
                retrain_records.append(retrain_record)
                # model_pool 的初始訓練：把第一個模型加入 pool
                if strategy == "model_pool" and pool_ctrl is not None:
                    bars_window = bars[bars["tradetime"] <= t]
                    # Phase B-2：alpha-side fingerprint（無歷史時可能 n_alphas=0，這時 3 維歸 0）
                    init_alpha_stats = compute_alpha_ic_stats(
                        alpha_panel=alpha_panel,
                        fwd_returns=fwd_5,
                        label_available_at=label_avail,
                        t=t,
                        window_days=60,
                        purge_days=purge_days,
                        horizon_days=horizon_days,
                    )
                    pool_ctrl.register_initial(model, bars_window, train_info, alpha_ic_stats=init_alpha_stats)
                logger.info(
                    "model_retrained",
                    t=str(t.date()),
                    reason=reason,
                    ic=round(train_info["holdout_metrics"].get("ic", 0.0), 4),
                    rank_ic=round(train_info["holdout_metrics"].get("rank_ic", 0.0), 4),
                    n_train=train_info["n_train"],
                )

        if model is None:
            continue

        # 2b. 對 T 日截面預測
        todays_panel = alpha_panel[alpha_panel["tradetime"] == t]
        if todays_panel.empty:
            logger.warning("no_alpha_for_date", t=str(t.date()))
            continue

        signals = model.predict(todays_panel).rename(columns={"tradetime": "signal_time"})
        past_signal_history.append(signals[["security_id", "signal_time", "signal_score"]].copy())

        # 2c. Portfolio + Risk
        rebalance_due = (
            not prev_weights
            or rebalance_every <= 1
            or (i - last_portfolio_rebalance_idx) >= rebalance_every
        )
        held_from_prev_count = 0
        forced_sells_count = 0
        turnover_cap_applied = False

        if not rebalance_due:
            tradable_secs = set(signals["security_id"].astype(str))
            current_weights = {
                sec: weight for sec, weight in prev_weights.items() if sec in tradable_secs
            }
            held_from_prev_count = len(current_weights)
            forced_sells_count = len(prev_weights) - len(current_weights)
        else:
            targets = portfolio_constructor.construct(
                signals,
                previous_weights=prev_weights,
                holding_days=holding_days,
            )
            held_from_prev_count = int(targets.attrs.get("held_from_prev_count", 0))
            forced_sells_count = int(targets.attrs.get("forced_sells_count", 0))
            if targets.empty:
                current_weights = {}
            else:
                adj = risk_mgr.apply_constraints(targets, previous_weights=prev_weights)
                turnover_cap_applied = bool(adj.attrs.get("turnover_cap_applied", False))
                adj = adj.merge(
                    signals[["security_id", "signal_score"]], on="security_id", how="left"
                )
                current_weights = dict(zip(adj["security_id"].astype(str), adj["target_weight"]))
                last_portfolio_rebalance_idx = i

        signal_lookup = signals.set_index("security_id")["signal_score"]
        for sec, weight in current_weights.items():
            close_t = float(last_close_lookup.get((sec, t), np.nan))
            signal_score = float(signal_lookup.get(sec, np.nan))
            holdings_records.append({
                "date": t.strftime("%Y-%m-%d"),
                "security_id": sec,
                "target_weight": float(weight),
                "signal_score": signal_score,
                "last_close": close_t,
                "target_shares": int(round(weight * portfolio_value / close_t))
                                 if close_t and not np.isnan(close_t) else 0,
            })

        # 2d. 計算 turnover 與三細項成本（commission / tax / slippage）
        # 拆 buys（增倉量）/ sells（減倉量），turnover = max(buys, sells)。
        # 首日（空倉→滿倉）：buys=1.0, sells=0.0 → turnover=1.0（正確）；
        # 舊公式 sum(|Δw|)/2 首日只算到 0.5，且 tax 錯誤徵收 0.5×tax_rate。
        all_secs = set(prev_weights) | set(current_weights)
        buys = sum(
            max(0.0, current_weights.get(s, 0.0) - prev_weights.get(s, 0.0)) for s in all_secs
        )
        sells = sum(
            max(0.0, prev_weights.get(s, 0.0) - current_weights.get(s, 0.0)) for s in all_secs
        )
        turnover = max(buys, sells)

        commission_cost, tax_cost, slippage_cost = _compute_costs(
            buys=buys,
            sells=sells,
            commission_rate=commission_rate,
            tax_rate=tax_rate,
            slippage_bps=slippage_bps,
            round_trip_cost_pct=round_trip_cost_pct,
        )

        # 2e. 計算次日報酬：sum(weight_i × close[T+1]/close[T] - 1)
        gross_return = 0.0
        for sec, w in current_weights.items():
            r = next_ret["next_return"].get((sec, t), np.nan)
            if not np.isnan(r):
                gross_return += w * float(r)

        net_return = gross_return - commission_cost - tax_cost - slippage_cost
        portfolio_value *= (1 + net_return)

        # 2f. 計算 rolling IC / Sharpe — 改用 [t-trigger_window_days, t-trigger_eval_gap_days]
        # 雙邊界，避免與 shadow eval 窗口 [t-30, t-10] 重疊
        rolling_ic_val = _compute_rolling_ic(
            past_signal_history, fwd_5, t,
            purge_days=purge_days,
            horizon_days=horizon_days,
            window_days=trigger_window_days,
            eval_gap_days=trigger_eval_gap_days,
        )
        rolling_sharpe_val = _compute_rolling_sharpe(
            pnl_records, t,
            window_days=trigger_window_days,
            eval_gap_days=trigger_eval_gap_days,
        )

        pnl_records.append({
            "date": t.strftime("%Y-%m-%d"),
            "n_holdings": len(current_weights),
            "gross_exposure": sum(current_weights.values()),
            "turnover": turnover,
            "buys_turnover": buys,
            "sells_turnover": sells,
            "rebalance_flag": bool(rebalance_due),
            "held_from_prev_count": held_from_prev_count,
            "forced_sells_count": forced_sells_count,
            "turnover_cap_applied": bool(turnover_cap_applied),
            "gross_return": gross_return,
            "commission_cost": commission_cost,
            "tax_cost": tax_cost,
            "slippage_cost": slippage_cost,
            "net_return": net_return,
            "cumulative_value": portfolio_value,
            "rolling_ic": rolling_ic_val,
            "rolling_sharpe": rolling_sharpe_val,
        })

        prev_weights = current_weights
        holding_days = {
            sec: (holding_days.get(sec, 0) + 1 if sec in prev_weights else 1)
            for sec in current_weights
            if abs(current_weights.get(sec, 0.0)) > 1e-12
        }

    # --- 3. 寫入結果 ---
    holdings_df = pd.DataFrame(holdings_records)
    pnl_df = pd.DataFrame(pnl_records)
    retrain_df = pd.DataFrame(retrain_records)

    holdings_path = run_dir / "holdings.csv"
    pnl_path = run_dir / "daily_pnl.csv"
    retrain_path = run_dir / "retrain_log.csv"
    holdings_df.to_csv(holdings_path, index=False)
    pnl_df.to_csv(pnl_path, index=False)
    retrain_df.to_csv(retrain_path, index=False)

    summary = _summarize(pnl_df, capital)
    summary["n_retrains"] = len(retrain_records)
    summary["strategy"] = strategy
    if pool_ctrl is not None:
        summary["n_pool_reuses"] = pool_ctrl.n_pool_reuses
        summary["n_pool_misses"] = pool_ctrl.n_pool_misses
        summary["pool_backend"] = pool_ctrl._backend
    else:
        summary["n_pool_reuses"] = 0
        summary["n_pool_misses"] = 0
        summary["pool_backend"] = "n/a"
    summary_path = run_dir / "summary.txt"
    _write_summary(summary_path, run_id, start, end, strategy, retrain_every, top_k, summary)

    logger.info("simulation_complete", **summary)

    return {
        "run_dir": str(run_dir),
        "holdings_path": str(holdings_path),
        "daily_pnl_path": str(pnl_path),
        "retrain_log_path": str(retrain_path),
        "summary_path": str(summary_path),
        "summary_metrics": summary,
        "strategy": strategy,
    }


def _decide_retrain(
    *,
    strategy: Strategy,
    model: MLMetaModel | None,
    day_idx: int,
    last_train_idx: int,
    retrain_every: int,
    min_retrain_gap: int,
    pnl_records: list[dict],
    adapter: PerformanceTriggeredAdapter,
    rolling_window: int,
) -> tuple[bool, str]:
    """決定當日是否重訓模型，回傳 (need_retrain, reason)。"""
    # 起始日一律要訓一次
    if model is None:
        return True, "initial_train"

    # None 策略：訓一次就不再動
    if strategy == "none":
        return False, ""

    # Scheduled 策略：固定週期
    if strategy == "scheduled":
        if (day_idx - last_train_idx) >= retrain_every:
            return True, f"scheduled_every_{retrain_every}d"
        return False, ""

    # Triggered 策略 / model_pool 策略：依 rolling IC / Sharpe 判斷，且有冷卻期
    if strategy in ("triggered", "model_pool"):
        if (day_idx - last_train_idx) < min_retrain_gap:
            return False, ""
        if len(pnl_records) < rolling_window:
            return False, ""
        ic_series = pd.Series(
            [r["rolling_ic"] for r in pnl_records if not np.isnan(r.get("rolling_ic", np.nan))]
        )
        sharpe_series = pd.Series(
            [r["rolling_sharpe"] for r in pnl_records if not np.isnan(r.get("rolling_sharpe", np.nan))]
        )
        triggered, reason = adapter.check_trigger(ic_series, sharpe_series, 0)
        return triggered, reason or ""

    return False, ""


def _compute_costs(
    *,
    buys: float,
    sells: float,
    commission_rate: float,
    tax_rate: float,
    slippage_bps: float,
    round_trip_cost_pct: float | None,
) -> tuple[float, float, float]:
    """計算單日 (commission, tax, slippage) 三細項成本。

    若 ``round_trip_cost_pct`` 不為 None，則覆寫三細項，將總成本歸入 slippage_cost
    （commission/tax 設 0）。此模式供 cost-sensitivity sweep 使用。

    Notes
    -----
    * buys = 正向 Δw 總和（增倉量）；sells = 負向 |Δw| 總和（減倉量）。
    * turnover（headline）= max(buys, sells)；首日全買時正確輸出 1.0 而非 0.5。
    * commission per-side：買賣雙邊收取 → ``(buys + sells) × commission_rate``
    * tax sell-side only：僅對賣出收取 → ``sells × tax_rate``（首日全買 → 0）
    * slippage per-side：買賣雙邊收取 → ``(buys + sells) × slippage_bps / 10000``
    """
    turnover = max(buys, sells)
    if round_trip_cost_pct is not None:
        return 0.0, 0.0, turnover * (round_trip_cost_pct / 100.0)
    commission_cost = (buys + sells) * commission_rate
    tax_cost = sells * tax_rate
    slippage_cost = (buys + sells) * (slippage_bps / 10000.0)
    return commission_cost, tax_cost, slippage_cost


def _compute_rolling_ic(
    past_signals: list[pd.DataFrame],
    fwd: pd.Series,
    current_time: pd.Timestamp,
    *,
    purge_days: int,
    horizon_days: int,
    window_days: int,
    eval_gap_days: int,
) -> float:
    """計算 IC，採 ``signal_time ∈ [current_time - window_days, current_time - eval_gap_days]``
    的 calendar-day 邊界。

    雙邊界用意：trigger 用較舊樣本（[t-60, t-20]）判斷退化，避免與 shadow eval 的近期
    窗口（[t-30, t-10]）重疊造成 selection bias。

    成熟條件：signal_time <= current_time - eval_gap_days - horizon_days - purge_days
    （label 必須已實現再加 purge 緩衝；eval_gap 又把樣本上界推得更早）。
    """
    if not past_signals:
        return np.nan

    upper = current_time - pd.Timedelta(days=eval_gap_days)
    lower = current_time - pd.Timedelta(days=window_days)
    mature_cutoff = upper - pd.Timedelta(days=purge_days + horizon_days)
    effective_upper = min(upper, mature_cutoff)

    in_window = [
        df for df in past_signals
        if not df.empty
        and lower < df["signal_time"].iloc[0] <= effective_upper
    ]
    if len(in_window) < 5:
        return np.nan

    combined = pd.concat(in_window, ignore_index=True)
    sig = combined.set_index(["security_id", "signal_time"])["signal_score"]
    sig.index = sig.index.set_names(["security_id", "tradetime"])
    common = sig.index.intersection(fwd.index)
    if len(common) < 10:
        return np.nan

    ic = information_coefficient(sig.loc[common], fwd.loc[common])
    return float(ic) if not np.isnan(ic) else np.nan


def _compute_rolling_sharpe(
    pnl_records: list[dict],
    current_time: pd.Timestamp,
    *,
    window_days: int,
    eval_gap_days: int,
) -> float:
    """以 ``date ∈ [current_time - window_days, current_time - eval_gap_days]`` 邊界計
    annualized Sharpe（calendar days）。

    與 ``_compute_rolling_ic`` 共用同一視窗，保證 trigger 信號（IC 與 Sharpe）來自相同樣本。
    """
    if not pnl_records:
        return np.nan

    upper = current_time - pd.Timedelta(days=eval_gap_days)
    lower = current_time - pd.Timedelta(days=window_days)
    rets = [
        float(r["net_return"]) for r in pnl_records
        if lower < pd.Timestamp(r["date"]) <= upper
    ]
    if len(rets) < 10:
        return np.nan
    arr = np.array(rets, dtype=float)
    if arr.std() <= 0:
        return 0.0
    return float((arr.mean() / arr.std()) * np.sqrt(252))


def _summarize(pnl_df: pd.DataFrame, initial_capital: float) -> dict[str, float]:
    if pnl_df.empty:
        return {"n_days": 0}
    final_val = float(pnl_df["cumulative_value"].iloc[-1])
    cum_return = final_val / initial_capital - 1
    daily_ret = pnl_df["net_return"].astype(float)
    n_days = len(pnl_df)

    if daily_ret.std() > 0:
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    cum_vals = pnl_df["cumulative_value"].astype(float)
    peak = cum_vals.cummax()
    dd = (cum_vals - peak) / peak
    max_dd = float(dd.min())

    win_rate = float((daily_ret > 0).sum() / n_days)
    avg_holdings = float(pnl_df["n_holdings"].mean())
    avg_turnover = float(pnl_df["turnover"].mean())

    # 成本三細項（bps/day, 每日 turnover-weighted）
    avg_commission_bps = float(pnl_df.get("commission_cost", pd.Series([0.0])).mean()) * 1e4
    avg_tax_bps = float(pnl_df.get("tax_cost", pd.Series([0.0])).mean()) * 1e4
    avg_slippage_bps = float(pnl_df.get("slippage_cost", pd.Series([0.0])).mean()) * 1e4
    total_cost_bps = avg_commission_bps + avg_tax_bps + avg_slippage_bps
    avg_gross_return_bps = float(pnl_df.get("gross_return", pd.Series([0.0])).mean()) * 1e4
    avg_net_return_bps = float(pnl_df.get("net_return", pd.Series([0.0])).mean()) * 1e4

    annualized = (1 + cum_return) ** (252 / n_days) - 1 if n_days > 0 else 0.0

    return {
        "n_days": n_days,
        "cumulative_return_pct": round(cum_return * 100, 3),
        "annualized_return_pct": round(annualized * 100, 3),
        "sharpe": round(float(sharpe), 3),
        "max_drawdown_pct": round(max_dd * 100, 3),
        "win_rate_pct": round(win_rate * 100, 2),
        "avg_holdings": round(avg_holdings, 2),
        "avg_turnover": round(avg_turnover, 4),
        "avg_commission_bps": round(avg_commission_bps, 3),
        "avg_tax_bps": round(avg_tax_bps, 3),
        "avg_slippage_bps": round(avg_slippage_bps, 3),
        "avg_total_cost_bps": round(total_cost_bps, 3),
        "avg_gross_return_bps": round(avg_gross_return_bps, 3),
        "avg_net_return_bps": round(avg_net_return_bps, 3),
        "final_value": round(final_val, 2),
    }


def _write_summary(
    path: Path,
    run_id: str,
    start: date,
    end: date,
    strategy: Strategy,
    retrain_every: int,
    top_k: int,
    summary: dict[str, float],
) -> None:
    strat_desc = {
        "none": "凍結訓練（no-adapt baseline）",
        "scheduled": f"每 {retrain_every} 個交易日重訓",
        "triggered": "依 rolling IC / Sharpe 退化觸發重訓",
        "model_pool": "觸發重訓 + recurring concept pool shadow 3-way compare",
    }[strategy]
    lines = [
        f"=== 模擬摘要：{run_id} ===",
        f"  期間            : {start} → {end}",
        f"  交易日數        : {summary.get('n_days', 0)}",
        f"  Adaptation 策略 : {strategy} ({strat_desc})",
        f"  實際重訓次數    : {summary.get('n_retrains', 0)}",
        f"  Top-K           : {top_k}",
        "",
        "--- 績效 ---",
        f"  累積報酬        : {summary.get('cumulative_return_pct', 0):>8.3f} %",
        f"  年化報酬        : {summary.get('annualized_return_pct', 0):>8.3f} %",
        f"  Sharpe Ratio    : {summary.get('sharpe', 0):>8.3f}",
        f"  Gross avg bps   : {summary.get('avg_gross_return_bps', 0):>8.3f}",
        f"  Net avg bps     : {summary.get('avg_net_return_bps', 0):>8.3f}",
        f"  最大回撤        : {summary.get('max_drawdown_pct', 0):>8.3f} %",
        f"  日勝率          : {summary.get('win_rate_pct', 0):>8.2f} %",
        f"  期末組合價值    : {summary.get('final_value', 0):>14,.2f}",
        "",
        "--- 組合特徵 ---",
        f"  平均持倉數      : {summary.get('avg_holdings', 0):>8.2f}",
        f"  平均週轉率      : {summary.get('avg_turnover', 0):>8.4f}",
        "",
        "--- 成本拆分（bps/日，turnover-weighted） ---",
        f"  Commission      : {summary.get('avg_commission_bps', 0):>8.3f}",
        f"  Tax (sell-side) : {summary.get('avg_tax_bps', 0):>8.3f}",
        f"  Slippage        : {summary.get('avg_slippage_bps', 0):>8.3f}",
        f"  Total           : {summary.get('avg_total_cost_bps', 0):>8.3f}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--data-source",
        choices=["csv", "tej"],
        default=DEFAULT_DATA_SOURCE,
        help=(
            "tej = TEJ ingest 產出的 data/tw_stocks_tej.parquet（預設，含期間下市股，survivorship-correct）; "
            "csv = yfinance 下載的 data/tw_stocks_ohlcv.csv（僅 demo，無下市股且已知 8476 資料污染）"
        ),
    )
    p.add_argument(
        "--csv", default=None,
        help="OHLCV 路徑；省略時依 --data-source 取對應預設（csv→tw_stocks_ohlcv.csv / tej→tw_stocks_tej.parquet）",
    )
    p.add_argument(
        "--allow-yfinance",
        action="store_true",
        help="明確允許使用已知污染的 yfinance CSV（僅 demo/反例使用）",
    )
    p.add_argument("--start", default="2026-01-01", help="模擬起日 YYYY-MM-DD")
    p.add_argument("--end", default=None, help="模擬迄日 YYYY-MM-DD（預設：CSV 最後一日）")
    p.add_argument(
        "--strategy", choices=["none", "scheduled", "triggered", "model_pool"], default="scheduled",
        help="Adaptation 策略：none=凍結 / scheduled=固定週期 / triggered=效能觸發 / model_pool=Recurring Concept Pool",
    )
    p.add_argument("--retrain-every", type=int, default=5, help="scheduled 模式：每 N 個交易日重訓")
    p.add_argument("--purge-days", type=int, default=5)
    p.add_argument("--horizon-days", type=int, default=5)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument(
        "--portfolio-method",
        choices=["equal_weight_topk", "score_proportional", "volatility_scaled", "turnover_aware_topk"],
        default="equal_weight_topk",
        help="Portfolio construction method; turnover_aware_topk keeps existing names with entry/exit buffers.",
    )
    p.add_argument("--rebalance-every", type=int, default=1, help="Portfolio rebalance interval in trading days.")
    p.add_argument("--entry-rank", type=int, default=20, help="turnover_aware_topk entry pool rank cutoff.")
    p.add_argument("--exit-rank", type=int, default=40, help="turnover_aware_topk exit buffer rank cutoff.")
    p.add_argument("--max-turnover", type=float, default=1.0, help="Maximum one-way turnover per rebalance.")
    p.add_argument("--min-holding-days", type=int, default=0, help="Minimum holding age before a rank-based sell.")
    p.add_argument(
        "--objective",
        choices=["forward_return", "net_return_proxy"],
        default="forward_return",
        help="Model evaluation objective metadata; net_return_proxy adds cost-aware holdout diagnostics.",
    )
    p.add_argument("--capital", type=float, default=10_000_000.0)
    p.add_argument("--slippage-bps", type=float, default=5.0,
                   help="per-side 滑點（bps）；只在 round_trip_cost_pct 未提供時生效")
    p.add_argument("--commission-rate", type=float, default=0.000926,
                   help="per-side 手續費率，預設 0.1425%% × 0.65 折扣後")
    p.add_argument("--tax-rate", type=float, default=0.003,
                   help="sell-side only 證交稅，預設 0.3%%")
    p.add_argument("--round-trip-cost-pct", type=float, default=None,
                   help="若指定，覆寫三細項，用單一 round-trip rate（cost-sensitivity sweep 用）")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--trigger-ic-threshold", type=float, default=0.0)
    p.add_argument("--trigger-ic-days", type=int, default=3)
    p.add_argument("--trigger-sharpe-threshold", type=float, default=0.0)
    p.add_argument("--trigger-sharpe-days", type=int, default=10)
    p.add_argument("--min-retrain-gap", type=int, default=20,
                   help="triggered 模式：兩次重訓最少間隔 N 個交易日")
    p.add_argument("--rolling-window", type=int, default=20, help="（legacy）保留作為向後相容")
    p.add_argument("--trigger-window-days", type=int, default=60,
                   help="trigger 用 IC/Sharpe 計算的回看上限（calendar days），預設 [t-60, ...]")
    p.add_argument("--trigger-eval-gap", type=int, default=20,
                   help="trigger 用 IC/Sharpe 計算的近期排除（calendar days），預設 [..., t-20]")
    p.add_argument("--shadow-warmup-days", type=int, default=5,
                   help="model_pool 候選的 shadow 訓練 cutoff 額外往前推 N 日，避免 IS leakage")
    p.add_argument("--train-window-days", type=int, default=500,
                   help="訓練窗口（calendar days）。500=rolling（預設）；None=expanding（模型記得全歷史）")
    p.add_argument(
        "--symbols", nargs="+", default=None,
        help="股票代號白名單（空格分隔），例如 --symbols 2330 2317 2454",
    )
    p.add_argument(
        "--min-turnover-ntd", type=float, default=0.0,
        help="最近 60 日平均成交金額（vol×close）下限，例如 100000000 = 1 億",
    )
    p.add_argument("--run-tag", default=None, help="附加在 run_id 後的標籤，避免覆蓋")
    p.add_argument(
        "--alpha-source", choices=["python", "dolphindb"], default="python",
        help="alpha 來源：python=用 compute_python_alphas 近似版；dolphindb=從 alpha_features 讀真實 WQ101",
    )
    p.add_argument(
        "--alpha-ids", nargs="+", default=None,
        help="alpha 白名單（僅對 dolphindb source 生效），例如 --alpha-ids wq001 wq014 wq041",
    )
    p.add_argument(
        "--skip-effective-filter", action="store_true",
        help="不套用 reports/alpha_ic_analysis/effective_alphas.json（跑全 101 時建議開）",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    csv_path = args.csv or DATA_SOURCE_DEFAULTS[args.data_source]
    logger.info("data_source_resolved", source=args.data_source, path=csv_path)

    if args.end:
        end = datetime.strptime(args.end, "%Y-%m-%d").date()
    else:
        # parquet 與 CSV 的 datetime 欄共用 — 依副檔名 dispatch
        path_obj = Path(csv_path)
        if path_obj.suffix.lower() == ".parquet":
            bars_dates = pd.read_parquet(path_obj, columns=["datetime"])
            bars_dates["datetime"] = pd.to_datetime(bars_dates["datetime"])
        else:
            bars_dates = pd.read_csv(path_obj, usecols=["datetime"], parse_dates=["datetime"])
        end = bars_dates["datetime"].max().date()

    result = simulate(
        csv_path=csv_path,
        start=start,
        end=end,
        strategy=args.strategy,
        retrain_every=args.retrain_every,
        purge_days=args.purge_days,
        horizon_days=args.horizon_days,
        top_k=args.top_k,
        portfolio_method=args.portfolio_method,
        rebalance_every=args.rebalance_every,
        entry_rank=args.entry_rank,
        exit_rank=args.exit_rank,
        max_turnover=args.max_turnover,
        min_holding_days=args.min_holding_days,
        objective=args.objective,
        capital=args.capital,
        slippage_bps=args.slippage_bps,
        commission_rate=args.commission_rate,
        tax_rate=args.tax_rate,
        round_trip_cost_pct=args.round_trip_cost_pct,
        out_dir=args.out_dir,
        symbols=args.symbols,
        min_turnover_ntd=args.min_turnover_ntd,
        run_tag=args.run_tag,
        trigger_ic_threshold=args.trigger_ic_threshold,
        trigger_ic_days=args.trigger_ic_days,
        trigger_sharpe_threshold=args.trigger_sharpe_threshold,
        trigger_sharpe_days=args.trigger_sharpe_days,
        min_retrain_gap=args.min_retrain_gap,
        rolling_window=args.rolling_window,
        trigger_window_days=args.trigger_window_days,
        trigger_eval_gap_days=args.trigger_eval_gap,
        shadow_warmup_days=args.shadow_warmup_days,
        alpha_source=args.alpha_source,
        alpha_ids=args.alpha_ids,
        skip_effective_filter=args.skip_effective_filter,
        train_window_days=args.train_window_days,
        allow_yfinance=args.allow_yfinance,
    )

    print(f"\n結果已寫入: {result['run_dir']}")
    print(Path(result["summary_path"]).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
