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
* **執行模型**：預設保留舊 close-to-close research proxy；可用
  ``--execution-price next_open`` / ``next_vwap`` 改成 T 日收盤訊號、T+1 才成交。
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
import hashlib
import json
import subprocess
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from pipelines.daily_batch_pipeline import compute_python_alphas, load_csv_data
from src.alpha_engine.feature_store import ALPHA_ENGINE_VERSION, FeatureStore
from src.adaptation.performance_trigger import PerformanceTriggeredAdapter
from src.adaptation.recurring_concept import compute_alpha_ic_stats
from src.alpha_selection import (
    RollingTopKSelector,
    SelectorContext,
    StaticISSelector,
    hash_alpha_ids,
    hash_universe,
    stable_hash,
)
from src.alpha_selection.snapshot import write_selection_artifacts
from src.common.logging import get_logger, setup_logging
from src.common.metrics import information_coefficient
from src.config.alpha_selection import (
    EFFECTIVE_ALPHAS_PATH,
    exclude_indclass_cap_alpha_ids,
    load_effective_alpha_ids,
)
from src.config.constants import DATA_SOURCE_DEFAULT_PATHS, DEFAULT_DATA_SOURCE
from src.config.frozen_alpha_selector import load_frozen_alpha_selector
from src.labeling.label_generator import LabelGenerator
from src.meta_signal.ml_meta_model import MLMetaModel
from src.portfolio.constructor import PortfolioConstructor
from src.risk.risk_manager import RiskManager

Strategy = Literal["none", "scheduled", "triggered", "model_pool"]
SelectorName = Literal["static_is", "rolling_topk", "legacy"]
ModelPoolTriggerMode = Literal["triggered", "scheduled"]

setup_logging()
logger = get_logger("simulate_recent")


DEFAULT_OUT_DIR = Path("reports/simulations")

DATA_SOURCE_DEFAULTS = DATA_SOURCE_DEFAULT_PATHS


def _file_sha256(path: str | Path | None) -> str | None:
    """回傳檔案 SHA256；檔案不存在時回傳 None。"""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_metadata() -> dict[str, Any]:
    """取得目前 git sha 與 dirty 狀態；失敗時保留 None。"""
    repo = Path(__file__).resolve().parents[1]
    safe_repo = str(repo).replace("\\", "/")

    def _run_git(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-c", f"safe.directory={safe_repo}", *args],
                cwd=repo,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    sha = _run_git(["rev-parse", "HEAD"])
    status = _run_git(["status", "--porcelain"])
    return {
        "git_sha": sha,
        "dirty_worktree": bool(status) if status is not None else None,
    }


def _bars_snapshot_hash(csv_path: str | Path, bars: pd.DataFrame) -> str:
    """優先用原始檔案 hash；沒有檔案時退回 bars metadata hash。"""
    file_hash = _file_sha256(csv_path)
    if file_hash:
        return file_hash
    return stable_hash(
        {
            "rows": len(bars),
            "min_time": str(pd.to_datetime(bars["tradetime"]).min()),
            "max_time": str(pd.to_datetime(bars["tradetime"]).max()),
            "securities": sorted(bars["security_id"].astype(str).unique().tolist()),
        }
    )


def _make_static_selector(
    *,
    alpha_ids: list[str] | None,
    skip_effective_filter: bool,
    exclude_indclass_cap_alphas: bool,
) -> StaticISSelector:
    return StaticISSelector(
        alpha_ids=tuple(str(a) for a in alpha_ids) if alpha_ids else None,
        skip_effective_filter=skip_effective_filter,
        exclude_indclass_cap=exclude_indclass_cap_alphas,
    )


def _selector_retrain_fields(
    *,
    selector: SelectorName,
    selector_snapshot_hash: str | None,
    feature_store_version: str,
    feature_alpha_ids: list[str] | None,
) -> dict[str, Any]:
    return {
        "selector": selector,
        "selector_snapshot_hash": selector_snapshot_hash,
        "feature_store_version": feature_store_version,
        "feature_columns_hash": hash_alpha_ids(feature_alpha_ids or []),
        "n_feature_alphas": len(feature_alpha_ids) if feature_alpha_ids is not None else None,
    }


def _infer_data_source(csv_path: str | Path, explicit: str | None = None) -> str | None:
    """由 CLI 指定值或預設路徑推斷資料源名稱。"""
    if explicit:
        return explicit
    p = Path(csv_path)
    try:
        resolved = p.resolve()
    except Exception:
        resolved = p
    for source, default_path in DATA_SOURCE_DEFAULTS.items():
        try:
            if resolved == Path(default_path).resolve():
                return source
        except Exception:
            if str(p).replace("\\", "/") == str(default_path).replace("\\", "/"):
                return source
    name = p.name.lower()
    if "tej" in name:
        return "tej"
    if "ohlcv" in name or name.endswith(".csv"):
        return "csv"
    return None

MODEL_POOL_DECISION_COLUMNS = [
    "date",
    "day_idx",
    "current_model_id",
    "shadow_new_model_id",
    "live_model_id",
    "selected_candidate_model_id",
    "applied_model_id",
    "candidate_model_id",
    "candidate_role",
    "selected",
    "selected_role",
    "decision_reason",
    "pool_hit",
    "candidate_similarity",
    "selected_similarity",
    "best_seen_similarity",
    "n_reused_candidates",
    "selection_metric",
    "selection_score",
    "shadow_rank_by_selection_metric",
    "shadow_rank_by_topk_net_return",
    "raw_best_candidate_model_id",
    "raw_best_role",
    "raw_best_score",
    "best_non_reused_model_id",
    "best_non_reused_score",
    "reuse_score_margin_vs_best_non_reused",
    "reuse_guard_min_score",
    "reuse_guard_margin",
    "reuse_guard_passed",
    "reuse_guard_reason",
    "shadow_ic",
    "shadow_hit_rate",
    "shadow_sharpe",
    "shadow_n_samples",
    "shadow_topk_gross_return",
    "shadow_topk_net_return",
    "shadow_topk_turnover",
    "shadow_topk_n_days",
    "proxy_n_days",
    "proxy_gross_return",
    "proxy_net_return",
    "proxy_turnover",
    "proxy_cost",
    "proxy_rank_by_net",
]


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


def _resolve_alpha_ids_for_run(
    *,
    alpha_ids: list[str] | None,
    skip_effective_filter: bool,
    exclude_indclass_cap: bool,
) -> list[str] | None:
    """解析本次模擬實際使用的 alpha universe。"""
    if skip_effective_filter:
        resolved = list(alpha_ids) if alpha_ids else None
    else:
        effective = _load_effective_alphas() or []
        if alpha_ids:
            requested = {str(a) for a in alpha_ids}
            resolved = [a for a in effective if a in requested]
        else:
            resolved = effective

    if exclude_indclass_cap:
        resolved = exclude_indclass_cap_alpha_ids(resolved)
    return resolved


def _trading_days(bars: pd.DataFrame, start: date, end: date) -> list[pd.Timestamp]:
    days = pd.to_datetime(bars["tradetime"]).dt.normalize().drop_duplicates().sort_values()
    days = days[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]
    return list(days)


def _next_day_returns(
    bars: pd.DataFrame,
    execution_price: Literal["close", "next_open", "next_vwap"] = "close",
) -> pd.DataFrame:
    """為每個 (security_id, tradetime) 計算 one-day execution return。

    ``close`` 保留舊版 close-to-close research proxy：close[t+1] / close[t] - 1。
    ``next_open`` / ``next_vwap`` 代表 T 日收盤後產生訊號，隔一個交易日才用
    open 或 vwap 成交，並用同一價格欄位標記到下一個交易日，避免 T close
    signal 與 T close fill 同時發生的樂觀假設。

    Returns
    -------
    DataFrame 索引為 (security_id, tradetime)，欄位 ``next_return``。
    """
    bars_sorted = bars.sort_values(["security_id", "tradetime"]).copy()
    if execution_price == "close":
        price_col = "close"
        bars_sorted["entry_price"] = bars_sorted[price_col]
        bars_sorted["exit_price"] = bars_sorted.groupby("security_id")[price_col].shift(-1)
    elif execution_price in ("next_open", "next_vwap"):
        price_col = "open" if execution_price == "next_open" else "vwap"
        bars_sorted["entry_price"] = bars_sorted.groupby("security_id")[price_col].shift(-1)
        bars_sorted["exit_price"] = bars_sorted.groupby("security_id")[price_col].shift(-2)
    else:
        raise ValueError(
            "execution_price 必須是 close / next_open / next_vwap，"
            f"得到 {execution_price!r}"
        )
    bars_sorted["next_return"] = bars_sorted["exit_price"] / bars_sorted["entry_price"] - 1
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
    selector: SelectorName = "static_is",
    selector_alpha_top_k: int = 30,
    selector_window_days: int = 252,
    selector_min_coverage: float = 0.20,
    selector_min_observations: int = 1000,
    selector_stability_penalty: float = 0.0,
    selector_admission_gate: bool = False,
    admission_max_promoted: int = 4,
    admission_min_score: float = 0.03,
    admission_min_coverage: float | None = None,
    admission_min_observations: int | None = None,
    admission_subwindows: int = 3,
    admission_min_subwindow_passes: int = 2,
    admission_subwindow_min_abs_ic: float = 0.01,
    admission_max_abs_corr_to_live: float | None = 0.98,
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
    exclude_indclass_cap_alphas: bool = False,
    execution_price: Literal["close", "next_open", "next_vwap"] = "close",
    hard_exit_score_threshold: float | None = None,
    hard_exit_min_holding_days: int | None = None,
    tail_cleanup_weight: float = 0.0,
    renormalize_after_exit_cleanup: bool = False,
    placebo_mode: Literal["none", "shuffle_signal"] = "none",
    placebo_seed: int = 0,
    similarity_threshold: float = 0.5,  # Phase B-1：對應 exp(-d/2) >= 0.5（d <= 1.4 std）
    pool_regime_window: int = 60,
    shadow_window: int = 20,
    pool_top_k: int = 3,  # Phase B-3：shadow 階段最多納入幾個 reused 候選
    model_pool_diagnostics: bool = False,
    model_pool_selection_metric: Literal[
        "ic", "hit_rate", "sharpe", "topk_gross_return", "topk_net_return"
    ] = "ic",
    model_pool_reuse_min_score: float | None = None,
    model_pool_reuse_margin: float = 0.0,
    model_pool_trigger_mode: ModelPoolTriggerMode = "triggered",
    train_window_days: int | None = 500,
    data_source: str | None = None,
    allow_yfinance: bool = False,
    frozen_config: str | Path | None = None,
    frozen_execution: str = "primary",
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
    exclude_indclass_cap_alphas
        排除所有需要 placeholder ``indclass`` 或 ``cap`` 的 WQ101 alpha，用於 TEJ
        真實產業分類 / 市值資料接上前的保守 ablation。
    execution_price
        ``close`` 為舊 close-to-close proxy；``next_open`` / ``next_vwap`` 為 T 日收盤
        訊號、T+1 open/vwap 才成交的保守 rerun 模式。

    Returns
    -------
    dict
        {run_dir, summary_metrics, holdings_path, daily_pnl_path, retrain_log_path}
    """
    if strategy not in ("none", "scheduled", "triggered", "model_pool"):
        raise ValueError(f"strategy 必須是 none/scheduled/triggered/model_pool，得到 {strategy!r}")
    if model_pool_trigger_mode not in ("triggered", "scheduled"):
        raise ValueError(
            f"model_pool_trigger_mode 必須是 triggered 或 scheduled，得到 {model_pool_trigger_mode!r}"
        )

    frozen_meta: dict[str, Any] = {}
    if frozen_config is not None:
        frozen_spec = load_frozen_alpha_selector(frozen_config)
        frozen_overrides = frozen_spec.simulation_overrides(frozen_execution)
        csv_path = frozen_overrides["csv_path"]
        data_source = frozen_overrides["data_source"]
        allow_yfinance = frozen_overrides["allow_yfinance"]
        selector = frozen_overrides["selector"]
        selector_alpha_top_k = frozen_overrides["selector_alpha_top_k"]
        selector_window_days = frozen_overrides["selector_window_days"]
        selector_min_coverage = frozen_overrides["selector_min_coverage"]
        selector_min_observations = frozen_overrides["selector_min_observations"]
        selector_stability_penalty = frozen_overrides["selector_stability_penalty"]
        selector_admission_gate = frozen_overrides["selector_admission_gate"]
        alpha_ids = frozen_overrides["alpha_ids"]
        skip_effective_filter = frozen_overrides["skip_effective_filter"]
        exclude_indclass_cap_alphas = frozen_overrides["exclude_indclass_cap_alphas"]
        top_k = frozen_overrides["top_k"]
        portfolio_method = frozen_overrides["portfolio_method"]
        rebalance_every = frozen_overrides["rebalance_every"]
        entry_rank = frozen_overrides["entry_rank"]
        exit_rank = frozen_overrides["exit_rank"]
        max_turnover = frozen_overrides["max_turnover"]
        min_holding_days = frozen_overrides["min_holding_days"]
        tail_cleanup_weight = frozen_overrides["tail_cleanup_weight"]
        objective = frozen_overrides["objective"]
        execution_price = frozen_overrides["execution_price"]
        commission_rate = frozen_overrides["commission_rate"]
        tax_rate = frozen_overrides["tax_rate"]
        slippage_bps = frozen_overrides["slippage_bps"]
        round_trip_cost_pct = frozen_overrides["round_trip_cost_pct"]
        horizon_days = frozen_overrides["horizon_days"]
        purge_days = frozen_overrides["purge_days"]
        if "retrain_every" in frozen_overrides:
            retrain_every = frozen_overrides["retrain_every"]
        if "train_window_days" in frozen_overrides:
            train_window_days = frozen_overrides["train_window_days"]
        frozen_meta = frozen_spec.metadata(frozen_execution)

    if selector not in ("static_is", "rolling_topk", "legacy"):
        raise ValueError(f"selector 必須是 static_is、rolling_topk 或 legacy，收到 {selector!r}")
    if selector_alpha_top_k <= 0:
        raise ValueError("selector_alpha_top_k 必須大於 0")
    if selector_window_days <= 0:
        raise ValueError("selector_window_days 必須大於 0")
    if not 0.0 <= selector_stability_penalty <= 1.0:
        raise ValueError("selector_stability_penalty 必須介於 0 與 1 之間")
    if selector_admission_gate:
        if admission_max_promoted < 0:
            raise ValueError("admission_max_promoted 不可為負")
        if admission_subwindows <= 0:
            raise ValueError("admission_subwindows 必須大於 0")
        if admission_min_subwindow_passes < 0:
            raise ValueError("admission_min_subwindow_passes 不可為負")
        if not 0.0 <= admission_subwindow_min_abs_ic <= 1.0:
            raise ValueError("admission_subwindow_min_abs_ic 必須介於 0 與 1 之間")
        if admission_max_abs_corr_to_live is not None and not 0.0 <= admission_max_abs_corr_to_live <= 1.0:
            raise ValueError("admission_max_abs_corr_to_live 必須介於 0 與 1 之間")

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
    if hard_exit_min_holding_days is None:
        hard_exit_min_holding_days = min_holding_days
    if hard_exit_min_holding_days < 0:
        raise ValueError("hard_exit_min_holding_days must be non-negative")
    if tail_cleanup_weight < 0:
        raise ValueError("tail_cleanup_weight must be non-negative")
    if placebo_mode not in ("none", "shuffle_signal"):
        raise ValueError(f"placebo_mode must be none/shuffle_signal, got {placebo_mode!r}")
    placebo_rng = np.random.default_rng(placebo_seed)
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

    git_meta = _git_metadata()
    bars_snapshot_hash = _bars_snapshot_hash(csv_path, bars)
    universe_hash = hash_universe(bars["security_id"].unique())
    feature_store = FeatureStore.for_data_path(csv_path)
    feature_store_version = (
        feature_store.version
        if alpha_source != "dolphindb"
        else stable_hash(
            {
                "kind": "dolphindb_alpha_features",
                "data_source": _infer_data_source(csv_path, data_source),
                "start": str(start),
                "end": str(end),
            }
        )
    )
    static_selector = (
        _make_static_selector(
            alpha_ids=alpha_ids,
            skip_effective_filter=skip_effective_filter,
            exclude_indclass_cap_alphas=exclude_indclass_cap_alphas,
        )
        if selector == "static_is"
        else None
    )
    rolling_selector = None
    admission_base_alpha_ids = None
    if static_selector is not None:
        feature_alpha_ids = static_selector.selected_alpha_ids()
    else:
        feature_alpha_ids = _resolve_alpha_ids_for_run(
            alpha_ids=alpha_ids,
            skip_effective_filter=skip_effective_filter,
            exclude_indclass_cap=exclude_indclass_cap_alphas,
        )
        if selector == "rolling_topk":
            if selector_admission_gate:
                admission_base_alpha_ids = load_effective_alpha_ids(
                    EFFECTIVE_ALPHAS_PATH,
                    required=True,
                    exclude_indclass_cap=exclude_indclass_cap_alphas,
                )
                if feature_alpha_ids is not None:
                    feature_set = {str(a) for a in feature_alpha_ids}
                    admission_base_alpha_ids = [
                        str(a) for a in admission_base_alpha_ids if str(a) in feature_set
                    ]
                if not admission_base_alpha_ids:
                    raise RuntimeError("admission gate 找不到可用的 incumbent base alpha 清單")
            rolling_selector = RollingTopKSelector(
                candidate_alpha_ids=tuple(feature_alpha_ids or []),
                top_k=selector_alpha_top_k,
                window_days=selector_window_days,
                min_coverage=selector_min_coverage,
                min_observations=selector_min_observations,
                stability_penalty=selector_stability_penalty,
                base_alpha_ids=(
                    tuple(admission_base_alpha_ids)
                    if admission_base_alpha_ids is not None
                    else None
                ),
                admission_enabled=selector_admission_gate,
                admission_max_promoted=admission_max_promoted,
                admission_min_score=admission_min_score,
                admission_min_coverage=admission_min_coverage,
                admission_min_observations=admission_min_observations,
                admission_subwindows=admission_subwindows,
                admission_min_subwindow_passes=admission_min_subwindow_passes,
                admission_subwindow_min_abs_ic=admission_subwindow_min_abs_ic,
                admission_max_abs_corr_to_live=admission_max_abs_corr_to_live,
            )
    if feature_alpha_ids is not None and len(feature_alpha_ids) == 0:
        raise RuntimeError(
            "Alpha universe 為空：請檢查 --alpha-ids、effective_alphas.json "
            "與 --exclude-indclass-cap-alphas 的交集"
        )
    eff_alphas = feature_alpha_ids
    alpha_bars = bars
    if train_window_days is not None:
        alpha_load_start = pd.Timestamp(start) - pd.Timedelta(
            days=train_window_days + max(purge_days, horizon_days, selector_window_days) + 10
        )
        alpha_bars = bars[bars["tradetime"] >= alpha_load_start].reset_index(drop=True)
        logger.info(
            "alpha_load_window_applied",
            start=str(alpha_load_start.date()),
            rows_before=len(bars),
            rows_after=len(alpha_bars),
            train_window_days=train_window_days,
        )

    if alpha_source == "dolphindb":
        # 預先把 alpha universe 推入 SQL 查詢，避免拉全量資料再在 Python 端過濾。
        # 避免拉取全量 53.8M rows 再在 Python 端過濾
        alpha_panel = _load_alphas_from_dolphindb(start, end, alpha_ids=feature_alpha_ids)
        # 以 bars 的 (security_id, tradetime) 內連接，排除 DolphinDB 端多餘標的
        bars_key = bars[["security_id", "tradetime"]].copy()
        bars_key["security_id"] = bars_key["security_id"].astype(str)
        before = len(alpha_panel)
        alpha_panel = alpha_panel.merge(bars_key, on=["security_id", "tradetime"], how="inner")
        logger.info("alpha_panel_aligned_to_bars", rows_before=before, rows_after=len(alpha_panel))
    else:
        if selector in ("static_is", "rolling_topk"):
            alpha_panel = feature_store.load_alpha_panel(alpha_bars, alpha_ids=feature_alpha_ids)
        else:
            alpha_panel = compute_python_alphas(
                alpha_bars,
                alpha_ids=feature_alpha_ids,
                cache_path=feature_store.cache_path,
            )

    # 確保 alpha_panel 也用 str 型 security_id（cache 直接回時可能是 str，新算則跟 bars 同 → 已 str）
    alpha_panel["security_id"] = alpha_panel["security_id"].astype(str)

    if feature_alpha_ids is not None:
        alpha_panel = alpha_panel[alpha_panel["alpha_id"].isin(feature_alpha_ids)]
        logger.info(
            "alpha_universe_applied",
            count=len(feature_alpha_ids),
            exclude_indclass_cap=exclude_indclass_cap_alphas,
        )

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

    next_ret = _next_day_returns(bars, execution_price=execution_price)
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
    model_pool_decision_records: list[dict] = []
    past_signal_history: list[pd.DataFrame] = []  # 累積過往預測供 rolling IC 計算
    selection_snapshots = []
    previous_selector_alpha_ids: list[str] | None = None
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
            selection_metric=model_pool_selection_metric,
            shadow_proxy_top_k=top_k,
            reuse_min_score=model_pool_reuse_min_score,
            reuse_margin=model_pool_reuse_margin,
            commission_rate=commission_rate,
            tax_rate=tax_rate,
            slippage_bps=slippage_bps,
            round_trip_cost_pct=round_trip_cost_pct,
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
            model_pool_trigger_mode=model_pool_trigger_mode,
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
            window_start = None
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
            selection_snapshot = None
            active_alpha_ids = eff_alphas
            if static_selector is not None or rolling_selector is not None:
                selector_config_hash = (
                    static_selector.config_hash
                    if static_selector is not None
                    else rolling_selector.config_hash
                )
                selector_context = SelectorContext(
                    as_of_date=t,
                    label_cutoff=t,
                    train_window_start=window_start,
                    train_window_end=purge_cutoff,
                    label_horizon_days=horizon_days,
                    purge_days=purge_days,
                    label_available_rule="label_available_at <= as_of_date",
                    selector_config_hash=selector_config_hash,
                    feature_store_version=feature_store_version,
                    bars_snapshot_hash=bars_snapshot_hash,
                    universe_hash=universe_hash,
                    alpha_engine_version=ALPHA_ENGINE_VERSION,
                    git_commit=git_meta.get("git_sha"),
                )
                if static_selector is not None:
                    selection_snapshot = static_selector.select(selector_context)
                else:
                    selection_snapshot = rolling_selector.select(
                        selector_context,
                        alpha_panel=train_panel,
                        labels=fwd_5,
                        label_available_at=label_avail,
                        previous_selected_alpha_ids=previous_selector_alpha_ids,
                    )
                selection_snapshots.append(selection_snapshot)
                active_alpha_ids = selection_snapshot.selected_alphas
                if not active_alpha_ids:
                    logger.warning("selector_selected_no_alphas", t=str(t.date()), selector=selector)
                    if model is None:
                        continue
                    active_alpha_ids = getattr(model, "_feature_columns", eff_alphas)
                previous_selector_alpha_ids = list(active_alpha_ids)
                train_panel = train_panel[train_panel["alpha_id"].isin(active_alpha_ids)]
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
                        label_available_at=label_avail,
                        train_panel=train_panel,
                        train_labels=train_labels,
                        eff_alphas=active_alpha_ids if active_alpha_ids else None,
                        alpha_ic_stats=alpha_ic_stats,
                    )
                    if model_pool_diagnostics:
                        model_pool_decision_records.extend(
                            _attach_model_pool_candidate_proxies(
                                records=decision.diagnostic_records,
                                candidate_models=decision.candidate_models,
                                t=t,
                                day_idx=i,
                                sim_days=sim_days,
                                alpha_panel=alpha_panel,
                                next_ret=next_ret,
                                prev_weights=prev_weights,
                                top_k=top_k,
                                rebalance_every=rebalance_every,
                                commission_rate=commission_rate,
                                tax_rate=tax_rate,
                                slippage_bps=slippage_bps,
                                round_trip_cost_pct=round_trip_cost_pct,
                            )
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
                        **_selector_retrain_fields(
                            selector=selector,
                            selector_snapshot_hash=selection_snapshot.snapshot_hash if selection_snapshot is not None else None,
                            feature_store_version=feature_store_version,
                            feature_alpha_ids=active_alpha_ids,
                        ),
                        "feature_columns_hash": decision.train_info.get("feature_columns_hash"),
                        "n_feature_alphas": decision.train_info.get("n_features"),
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
                    model = MLMetaModel(feature_columns=active_alpha_ids, objective=objective, proxy_top_k=top_k)
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
                        **_selector_retrain_fields(
                            selector=selector,
                            selector_snapshot_hash=selection_snapshot.snapshot_hash if selection_snapshot is not None else None,
                            feature_store_version=feature_store_version,
                            feature_alpha_ids=active_alpha_ids,
                        ),
                        "feature_columns_hash": train_info.get("feature_columns_hash"),
                        "n_feature_alphas": train_info.get("n_features"),
                    })
            else:
                # none / scheduled / triggered，以及 model_pool 的初始訓練
                model = MLMetaModel(feature_columns=active_alpha_ids, objective=objective, proxy_top_k=top_k)
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
                    **_selector_retrain_fields(
                        selector=selector,
                        selector_snapshot_hash=selection_snapshot.snapshot_hash if selection_snapshot is not None else None,
                        feature_store_version=feature_store_version,
                        feature_alpha_ids=active_alpha_ids,
                    ),
                    "feature_columns_hash": train_info.get("feature_columns_hash"),
                    "n_feature_alphas": train_info.get("n_features"),
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
        signals = _apply_placebo_to_signals(signals, placebo_mode=placebo_mode, rng=placebo_rng)
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
        current_weights, exit_metrics = _apply_exit_discipline(
            current_weights=current_weights,
            signal_scores=signal_lookup,
            holding_days=holding_days,
            hard_exit_score_threshold=hard_exit_score_threshold,
            hard_exit_min_holding_days=hard_exit_min_holding_days,
            tail_cleanup_weight=tail_cleanup_weight,
            renormalize=renormalize_after_exit_cleanup,
        )
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
            "execution_price": execution_price,
            "turnover": turnover,
            "buys_turnover": buys,
            "sells_turnover": sells,
            "rebalance_flag": bool(rebalance_due),
            "held_from_prev_count": held_from_prev_count,
            "forced_sells_count": forced_sells_count,
            "turnover_cap_applied": bool(turnover_cap_applied),
            **exit_metrics,
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
    decisions_path = run_dir / "model_pool_decisions.csv"
    holdings_df.to_csv(holdings_path, index=False)
    pnl_df.to_csv(pnl_path, index=False)
    retrain_df.to_csv(retrain_path, index=False)
    selection_artifacts = write_selection_artifacts(selection_snapshots, run_dir)
    if strategy == "model_pool" and model_pool_diagnostics:
        decisions_df = pd.DataFrame(model_pool_decision_records)
        for col in MODEL_POOL_DECISION_COLUMNS:
            if col not in decisions_df.columns:
                decisions_df[col] = np.nan
        decisions_df = decisions_df[MODEL_POOL_DECISION_COLUMNS]
        decisions_df.to_csv(decisions_path, index=False)

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
    config_path = run_dir / "config.json"
    effective_alphas_path = None if skip_effective_filter else str(EFFECTIVE_ALPHAS_PATH)
    config_path.write_text(
        json.dumps({
            "run_id": run_id,
            "data_source": _infer_data_source(csv_path, data_source),
            "csv_path": str(csv_path),
            "start": str(start),
            "end": str(end),
            "strategy": strategy,
            "selector": selector,
            "selector_config_hash": (
                static_selector.config_hash
                if static_selector is not None
                else rolling_selector.config_hash
                if rolling_selector is not None
                else stable_hash(
                    {
                        "selector": selector,
                        "alpha_ids": alpha_ids,
                        "skip_effective_filter": skip_effective_filter,
                        "exclude_indclass_cap_alphas": exclude_indclass_cap_alphas,
                    }
                )
            ),
            "selector_alpha_top_k": selector_alpha_top_k,
            "selector_window_days": selector_window_days,
            "selector_min_coverage": selector_min_coverage,
            "selector_min_observations": selector_min_observations,
            "selector_stability_penalty": selector_stability_penalty,
            "selector_admission_gate": selector_admission_gate,
            "admission_base_alpha_ids": admission_base_alpha_ids,
            "admission_max_promoted": admission_max_promoted,
            "admission_min_score": admission_min_score,
            "admission_min_coverage": admission_min_coverage,
            "admission_min_observations": admission_min_observations,
            "admission_subwindows": admission_subwindows,
            "admission_min_subwindow_passes": admission_min_subwindow_passes,
            "admission_subwindow_min_abs_ic": admission_subwindow_min_abs_ic,
            "admission_max_abs_corr_to_live": admission_max_abs_corr_to_live,
            "retrain_every": retrain_every,
            "purge_days": purge_days,
            "horizon_days": horizon_days,
            "top_k": top_k,
            "portfolio_method": portfolio_method,
            "rebalance_every": rebalance_every,
            "entry_rank": entry_rank,
            "exit_rank": exit_rank,
            "max_turnover": max_turnover,
            "min_holding_days": min_holding_days,
            "objective": objective,
            "capital": capital,
            "slippage_bps": slippage_bps,
            "commission_rate": commission_rate,
            "tax_rate": tax_rate,
            "round_trip_cost_pct": round_trip_cost_pct,
            "hard_exit_score_threshold": hard_exit_score_threshold,
            "hard_exit_min_holding_days": hard_exit_min_holding_days,
            "tail_cleanup_weight": tail_cleanup_weight,
            "renormalize_after_exit_cleanup": renormalize_after_exit_cleanup,
            "placebo_mode": placebo_mode,
            "placebo_seed": placebo_seed,
            "trigger_ic_threshold": trigger_ic_threshold,
            "trigger_ic_days": trigger_ic_days,
            "trigger_sharpe_threshold": trigger_sharpe_threshold,
            "trigger_sharpe_days": trigger_sharpe_days,
            "min_retrain_gap": min_retrain_gap,
            "trigger_window_days": trigger_window_days,
            "trigger_eval_gap_days": trigger_eval_gap_days,
            "shadow_warmup_days": shadow_warmup_days,
            "similarity_threshold": similarity_threshold,
            "pool_regime_window": pool_regime_window,
            "shadow_window": shadow_window,
            "pool_top_k": pool_top_k,
            "model_pool_diagnostics": model_pool_diagnostics,
            "model_pool_selection_metric": model_pool_selection_metric,
            "model_pool_reuse_min_score": model_pool_reuse_min_score,
            "model_pool_reuse_margin": model_pool_reuse_margin,
            "model_pool_trigger_mode": model_pool_trigger_mode,
            "train_window_days": train_window_days,
            "symbols": symbols,
            "min_turnover_ntd": min_turnover_ntd,
            "alpha_source": alpha_source,
            "alpha_ids": alpha_ids,
            "skip_effective_filter": skip_effective_filter,
            "exclude_indclass_cap_alphas": exclude_indclass_cap_alphas,
            "n_feature_alphas": len(feature_alpha_ids) if feature_alpha_ids is not None else None,
            "n_candidate_alphas": len(feature_alpha_ids) if feature_alpha_ids is not None else None,
            "n_admission_base_alphas": (
                len(admission_base_alpha_ids)
                if admission_base_alpha_ids is not None
                else None
            ),
            "n_quarantine_alphas": (
                len(set(feature_alpha_ids or []) - set(admission_base_alpha_ids or []))
                if selector_admission_gate
                else None
            ),
            "feature_store_version": feature_store_version,
            "bars_snapshot_hash": bars_snapshot_hash,
            "universe_hash": universe_hash,
            "alpha_engine_version": ALPHA_ENGINE_VERSION,
            "feature_columns_hash": (
                hash_alpha_ids(feature_alpha_ids or [])
                if selector != "rolling_topk"
                else None
            ),
            "candidate_feature_columns_hash": hash_alpha_ids(feature_alpha_ids or []),
            "alpha_selection_snapshots_path": (
                str(selection_artifacts["snapshots_path"])
                if selection_artifacts["snapshots_path"] is not None
                else None
            ),
            "alpha_scores_path": (
                str(selection_artifacts["scores_path"])
                if selection_artifacts["scores_path"] is not None
                else None
            ),
            "alpha_weights_by_date_path": (
                str(selection_artifacts["weights_path"])
                if selection_artifacts["weights_path"] is not None
                else None
            ),
            "execution_price": execution_price,
            "effective_alphas_path": effective_alphas_path,
            "effective_alphas_hash": _file_sha256(effective_alphas_path),
            "pool_backend": summary.get("pool_backend"),
            **frozen_meta,
            **git_meta,
            "allow_yfinance": allow_yfinance,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("simulation_complete", **summary)

    return {
        "run_dir": str(run_dir),
        "holdings_path": str(holdings_path),
        "daily_pnl_path": str(pnl_path),
        "retrain_log_path": str(retrain_path),
        "model_pool_decisions_path": str(decisions_path) if decisions_path.exists() else None,
        "alpha_selection_snapshots_path": (
            str(selection_artifacts["snapshots_path"])
            if selection_artifacts["snapshots_path"] is not None
            else None
        ),
        "alpha_scores_path": (
            str(selection_artifacts["scores_path"])
            if selection_artifacts["scores_path"] is not None
            else None
        ),
        "alpha_weights_by_date_path": (
            str(selection_artifacts["weights_path"])
            if selection_artifacts["weights_path"] is not None
            else None
        ),
        "config_path": str(config_path),
        "summary_path": str(summary_path),
        "summary_metrics": summary,
        "strategy": strategy,
    }


def _decide_retrain(
    *,
    strategy: Strategy,
    model_pool_trigger_mode: ModelPoolTriggerMode = "triggered",
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

    if strategy == "model_pool" and model_pool_trigger_mode == "scheduled":
        if (day_idx - last_train_idx) >= retrain_every:
            return True, f"model_pool_scheduled_every_{retrain_every}d"
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


def _apply_placebo_to_signals(
    signals: pd.DataFrame,
    *,
    placebo_mode: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """套用 placebo signal 轉換；預設不改動。"""
    if placebo_mode == "none":
        return signals
    if placebo_mode != "shuffle_signal":
        raise ValueError(f"unsupported placebo_mode: {placebo_mode!r}")
    if signals.empty or "signal_score" not in signals.columns:
        return signals

    shuffled = signals.copy()
    values = shuffled["signal_score"].to_numpy(copy=True)
    rng.shuffle(values)
    shuffled["signal_score"] = values
    shuffled["signal_direction"] = np.where(values >= 0, 1, -1).astype(int)
    shuffled["confidence"] = np.abs(values)
    return shuffled


def _apply_exit_discipline(
    *,
    current_weights: dict[str, float],
    signal_scores: pd.Series,
    holding_days: dict[str, int],
    hard_exit_score_threshold: float | None,
    hard_exit_min_holding_days: int,
    tail_cleanup_weight: float,
    renormalize: bool,
) -> tuple[dict[str, float], dict[str, float | int | bool]]:
    """依訊號分數與殘餘小權重執行持股淘汰，回傳新權重與診斷欄位。"""
    scores = {str(sec): float(score) for sec, score in signal_scores.items()}
    weights = {str(sec): float(weight) for sec, weight in current_weights.items()}

    gross_before = float(sum(max(0.0, weight) for weight in weights.values()))
    negative_before = float(
        sum(
            max(0.0, weight)
            for sec, weight in weights.items()
            if pd.notna(scores.get(sec, np.nan)) and scores[sec] <= 0.0
        )
    )

    hard_exit_secs: set[str] = set()
    tail_exit_secs: set[str] = set()
    for sec, weight in weights.items():
        age = int(holding_days.get(sec, 0))
        if age < hard_exit_min_holding_days:
            continue
        score = scores.get(sec, np.nan)
        if (
            hard_exit_score_threshold is not None
            and pd.notna(score)
            and float(score) <= hard_exit_score_threshold
        ):
            hard_exit_secs.add(sec)
            continue
        if tail_cleanup_weight > 0 and abs(weight) < tail_cleanup_weight:
            tail_exit_secs.add(sec)

    exit_secs = hard_exit_secs | tail_exit_secs
    next_weights = {
        sec: weight
        for sec, weight in weights.items()
        if sec not in exit_secs and abs(weight) > 1e-12
    }

    gross_after = float(sum(max(0.0, weight) for weight in next_weights.values()))
    did_renormalize = False
    if renormalize and gross_after > 0 and gross_before > gross_after:
        target_gross = min(1.0, gross_before)
        scale = target_gross / gross_after
        next_weights = {sec: weight * scale for sec, weight in next_weights.items()}
        gross_after = float(sum(max(0.0, weight) for weight in next_weights.values()))
        did_renormalize = True

    negative_after = float(
        sum(
            max(0.0, weight)
            for sec, weight in next_weights.items()
            if pd.notna(scores.get(sec, np.nan)) and scores[sec] <= 0.0
        )
    )
    hard_exit_weight = float(sum(max(0.0, weights.get(sec, 0.0)) for sec in hard_exit_secs))
    tail_exit_weight = float(sum(max(0.0, weights.get(sec, 0.0)) for sec in tail_exit_secs))

    metrics: dict[str, float | int | bool] = {
        "hard_exit_count": int(len(hard_exit_secs)),
        "hard_exit_weight": hard_exit_weight,
        "tail_exit_count": int(len(tail_exit_secs)),
        "tail_exit_weight": tail_exit_weight,
        "exit_cleanup_count": int(len(exit_secs)),
        "exit_cleanup_weight": float(hard_exit_weight + tail_exit_weight),
        "exit_cleanup_gross_before": gross_before,
        "exit_cleanup_gross_after": gross_after,
        "exit_cleanup_renormalized": did_renormalize,
        "negative_score_weight_before": negative_before,
        "negative_score_weight_after": negative_after,
    }
    return next_weights, metrics


def _attach_model_pool_candidate_proxies(
    *,
    records: list[dict],
    candidate_models: dict[str, MLMetaModel],
    t: pd.Timestamp,
    day_idx: int,
    sim_days: list[pd.Timestamp],
    alpha_panel: pd.DataFrame,
    next_ret: pd.DataFrame,
    prev_weights: dict[str, float],
    top_k: int,
    rebalance_every: int,
    commission_rate: float,
    tax_rate: float,
    slippage_bps: float,
    round_trip_cost_pct: float | None,
) -> list[dict]:
    """為 model_pool decision records 補下一持股週期的候選 proxy。

    這不是完整 counterfactual 回測；每個候選只在 trigger 日用同一截面產生
    equal-weight top-k，並假設持有至下一次 rebalance。
    """
    if not records:
        return []

    enriched: list[dict] = []
    for rec in records:
        row = dict(rec)
        row["day_idx"] = day_idx
        model = candidate_models.get(str(row.get("candidate_model_id")))
        proxy = _compute_candidate_holding_proxy(
            model=model,
            t=t,
            day_idx=day_idx,
            sim_days=sim_days,
            alpha_panel=alpha_panel,
            next_ret=next_ret,
            prev_weights=prev_weights,
            top_k=top_k,
            rebalance_every=rebalance_every,
            commission_rate=commission_rate,
            tax_rate=tax_rate,
            slippage_bps=slippage_bps,
            round_trip_cost_pct=round_trip_cost_pct,
        )
        row.update(proxy)
        enriched.append(row)

    valid = [
        (idx, float(row["proxy_net_return"]))
        for idx, row in enumerate(enriched)
        if pd.notna(row.get("proxy_net_return"))
    ]
    for rank, (idx, _val) in enumerate(
        sorted(valid, key=lambda x: x[1], reverse=True), start=1
    ):
        enriched[idx]["proxy_rank_by_net"] = rank
    for row in enriched:
        row.setdefault("proxy_rank_by_net", np.nan)
    return enriched


def _compute_candidate_holding_proxy(
    *,
    model: MLMetaModel | None,
    t: pd.Timestamp,
    day_idx: int,
    sim_days: list[pd.Timestamp],
    alpha_panel: pd.DataFrame,
    next_ret: pd.DataFrame,
    prev_weights: dict[str, float],
    top_k: int,
    rebalance_every: int,
    commission_rate: float,
    tax_rate: float,
    slippage_bps: float,
    round_trip_cost_pct: float | None,
) -> dict[str, float]:
    """候選模型在下一個持股週期的簡化 proxy。"""
    base = {
        "proxy_n_days": 0,
        "proxy_gross_return": np.nan,
        "proxy_net_return": np.nan,
        "proxy_turnover": np.nan,
        "proxy_cost": np.nan,
    }
    if model is None:
        return base

    todays_panel = alpha_panel[alpha_panel["tradetime"] == t]
    if todays_panel.empty:
        return base

    try:
        signals = model.predict(todays_panel)
    except Exception:
        return base

    if signals.empty or "signal_score" not in signals.columns:
        return base

    ranked = (
        signals.dropna(subset=["signal_score"])
        .sort_values("signal_score", ascending=False)
        .head(top_k)
    )
    if ranked.empty:
        empty_sells = sum(max(0.0, w) for w in prev_weights.values())
        c_cost, t_cost, s_cost = _compute_costs(
            buys=0.0,
            sells=empty_sells,
            commission_rate=commission_rate,
            tax_rate=tax_rate,
            slippage_bps=slippage_bps,
            round_trip_cost_pct=round_trip_cost_pct,
        )
        return {
            **base,
            "proxy_n_days": 0,
            "proxy_gross_return": 0.0,
            "proxy_net_return": -(c_cost + t_cost + s_cost),
            "proxy_turnover": float(empty_sells),
            "proxy_cost": float(c_cost + t_cost + s_cost),
        }

    weight = 1.0 / len(ranked)
    candidate_weights = {str(sec): weight for sec in ranked["security_id"].astype(str)}

    all_secs = set(prev_weights) | set(candidate_weights)
    buys = sum(
        max(0.0, candidate_weights.get(s, 0.0) - prev_weights.get(s, 0.0))
        for s in all_secs
    )
    sells = sum(
        max(0.0, prev_weights.get(s, 0.0) - candidate_weights.get(s, 0.0))
        for s in all_secs
    )
    commission_cost, tax_cost, slippage_cost = _compute_costs(
        buys=buys,
        sells=sells,
        commission_rate=commission_rate,
        tax_rate=tax_rate,
        slippage_bps=slippage_bps,
        round_trip_cost_pct=round_trip_cost_pct,
    )
    total_cost = commission_cost + tax_cost + slippage_cost

    horizon = max(int(rebalance_every), 1)
    days = sim_days[day_idx:min(day_idx + horizon, len(sim_days))]
    gross_daily: list[float] = []
    for d in days:
        gross = 0.0
        for sec, w in candidate_weights.items():
            r = next_ret["next_return"].get((sec, d), np.nan)
            if not np.isnan(r):
                gross += w * float(r)
        gross_daily.append(gross)

    if not gross_daily:
        return {**base, "proxy_n_days": 0}

    gross_arr = np.array(gross_daily, dtype=float)
    net_arr = gross_arr.copy()
    net_arr[0] -= total_cost
    proxy_gross = float(np.prod(1.0 + gross_arr) - 1.0)
    proxy_net = float(np.prod(1.0 + net_arr) - 1.0)
    return {
        "proxy_n_days": int(len(gross_daily)),
        "proxy_gross_return": proxy_gross,
        "proxy_net_return": proxy_net,
        "proxy_turnover": float(max(buys, sells)),
        "proxy_cost": float(total_cost),
    }


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
    avg_hard_exit_count = float(pnl_df.get("hard_exit_count", pd.Series([0.0])).mean())
    avg_hard_exit_weight = float(pnl_df.get("hard_exit_weight", pd.Series([0.0])).mean())
    avg_tail_exit_count = float(pnl_df.get("tail_exit_count", pd.Series([0.0])).mean())
    avg_tail_exit_weight = float(pnl_df.get("tail_exit_weight", pd.Series([0.0])).mean())
    avg_exit_cleanup_weight = float(pnl_df.get("exit_cleanup_weight", pd.Series([0.0])).mean())
    avg_negative_score_weight_after = float(
        pnl_df.get("negative_score_weight_after", pd.Series([0.0])).mean()
    )

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
        "avg_hard_exit_count": round(avg_hard_exit_count, 3),
        "avg_hard_exit_weight": round(avg_hard_exit_weight, 5),
        "avg_tail_exit_count": round(avg_tail_exit_count, 3),
        "avg_tail_exit_weight": round(avg_tail_exit_weight, 5),
        "avg_exit_cleanup_weight": round(avg_exit_cleanup_weight, 5),
        "avg_negative_score_weight_after": round(avg_negative_score_weight_after, 5),
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
    p.add_argument(
        "--selector",
        choices=["static_is", "rolling_topk", "legacy"],
        default="static_is",
        help="Alpha selection 路徑：static_is=新 snapshot 介面；rolling_topk=成熟 label 動態選 alpha；legacy=舊 effective filter 路徑",
    )
    p.add_argument("--selector-alpha-top-k", type=int, default=30, help="rolling_topk 每次選出的 alpha 數。")
    p.add_argument("--selector-window-days", type=int, default=252, help="rolling_topk rank IC 回看 calendar days。")
    p.add_argument("--selector-min-coverage", type=float, default=0.20, help="rolling_topk alpha 最低有效樣本覆蓋率。")
    p.add_argument("--selector-min-observations", type=int, default=1000, help="rolling_topk alpha 最低有效樣本數。")
    p.add_argument(
        "--selector-stability-penalty",
        type=float,
        default=0.0,
        help="rolling_topk 對新進 alpha 的 score 折扣，0.10 表示新 alpha 分數乘以 0.90。",
    )
    p.add_argument(
        "--selector-admission-gate",
        action="store_true",
        help="啟用 quarantine/admission gate：新增 alpha 通過 point-in-time 品質條件後才可進 live selector。",
    )
    p.add_argument("--admission-max-promoted", type=int, default=4, help="每次 selector event 最多允許幾個 quarantine alpha 升級為可選候選。")
    p.add_argument("--admission-min-score", type=float, default=0.03, help="quarantine alpha 的最低 admission score。")
    p.add_argument("--admission-min-coverage", type=float, default=None, help="quarantine alpha 的最低 coverage；預設沿用 --selector-min-coverage。")
    p.add_argument("--admission-min-observations", type=int, default=None, help="quarantine alpha 的最低樣本數；預設沿用 --selector-min-observations。")
    p.add_argument("--admission-subwindows", type=int, default=3, help="admission stability 檢查要切成幾個時間子窗。")
    p.add_argument("--admission-min-subwindow-passes", type=int, default=2, help="quarantine alpha 至少要通過幾個 stability 子窗。")
    p.add_argument("--admission-subwindow-min-abs-ic", type=float, default=0.01, help="子窗 abs(rank IC) 通過門檻。")
    p.add_argument("--admission-max-abs-corr-to-live", type=float, default=0.98, help="與 incumbent live alpha 的最大允許絕對相關；設為 1.0 幾乎等於關閉 diversity gate。")
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
        "--hard-exit-score-threshold",
        type=float,
        default=None,
        help="持股滿 hard-exit-min-holding-days 後，signal_score <= threshold 即強制賣出。",
    )
    p.add_argument(
        "--hard-exit-min-holding-days",
        type=int,
        default=None,
        help="hard exit / tail cleanup 的最小持倉天數；預設沿用 --min-holding-days。",
    )
    p.add_argument(
        "--tail-cleanup-weight",
        type=float,
        default=0.0,
        help="持股滿最小天數後，權重低於此門檻的殘餘小倉位會被清掉，例如 0.005 = 50 bps。",
    )
    p.add_argument(
        "--renormalize-after-exit-cleanup",
        action="store_true",
        help="清掉持股後將剩餘多頭權重放大回原 gross exposure；預設保留現金以便診斷。",
    )
    p.add_argument(
        "--placebo-mode",
        choices=["none", "shuffle_signal"],
        default="none",
        help="Placebo 模式；shuffle_signal 會每日打亂 signal_score 與股票的對應關係。",
    )
    p.add_argument("--placebo-seed", type=int, default=0, help="Placebo shuffle random seed。")
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
    p.add_argument(
        "--execution-price",
        choices=["close", "next_open", "next_vwap"],
        default="close",
        help=(
            "回測成交價格語意：close=舊 close-to-close proxy；"
            "next_open/next_vwap=T 日收盤訊號、T+1 open/vwap 才成交"
        ),
    )
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
    p.add_argument("--model-pool-diagnostics", action="store_true",
                   help="輸出 model_pool_decisions.csv，包含候選 shadow metrics 與下一持股週期 proxy")
    p.add_argument("--similarity-threshold", type=float, default=0.5,
                   help="model_pool regime similarity threshold")
    p.add_argument("--pool-top-k", type=int, default=3,
                   help="model_pool shadow 階段最多納入幾個 reused 候選")
    p.add_argument("--pool-regime-window", type=int, default=60,
                   help="model_pool regime fingerprint 回看天數")
    p.add_argument("--shadow-window", type=int, default=20,
                   help="model_pool shadow evaluation window 交易日數")
    p.add_argument(
        "--model-pool-selection-metric",
        choices=["ic", "hit_rate", "sharpe", "topk_gross_return", "topk_net_return"],
        default="ic",
        help="model_pool shadow selector 使用的排序指標；topk_net_return 會使用成本感知 top-k proxy",
    )
    p.add_argument(
        "--model-pool-reuse-min-score",
        type=float,
        default=None,
        help="reused candidate 的最低 shadow selector 分數；未達門檻則退回 current/new",
    )
    p.add_argument(
        "--model-pool-reuse-margin",
        type=float,
        default=0.0,
        help="reused candidate 必須比最佳 current/new 高出的 selector margin",
    )
    p.add_argument(
        "--model-pool-trigger-mode",
        choices=["triggered", "scheduled"],
        default="triggered",
        help="model_pool 何時進入 shadow compare；scheduled 會沿用 --retrain-every cadence。",
    )
    p.add_argument(
        "--frozen-config",
        default=None,
        help="套用 frozen alpha selector 規格；會覆寫 selector/portfolio/execution/cost/label 參數。",
    )
    p.add_argument(
        "--frozen-execution",
        choices=["primary", "secondary", "next_vwap", "next_open"],
        default="primary",
        help="frozen config 的執行價格選擇；primary=next_vwap、secondary=next_open。",
    )
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
        help="alpha 來源：python=pandas WQ101 主路徑；dolphindb=從 alpha_features 讀 real mode 備援",
    )
    p.add_argument(
        "--alpha-ids", nargs="+", default=None,
        help="alpha 白名單，例如 --alpha-ids wq001 wq014 wq041",
    )
    p.add_argument(
        "--skip-effective-filter", action="store_true",
        help="不套用 reports/alpha_ic_analysis/effective_alphas.json（跑全 101 時建議開）",
    )
    p.add_argument(
        "--exclude-indclass-cap-alphas",
        action="store_true",
        help="排除需要 placeholder indclass/cap 的 WQ101 alpha，用於保守 ablation",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    if args.frozen_config and args.csv is None:
        frozen_spec = load_frozen_alpha_selector(args.frozen_config)
        csv_path = frozen_spec.simulation_overrides(args.frozen_execution)["csv_path"]
    else:
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
        selector=args.selector,
        selector_alpha_top_k=args.selector_alpha_top_k,
        selector_window_days=args.selector_window_days,
        selector_min_coverage=args.selector_min_coverage,
        selector_min_observations=args.selector_min_observations,
        selector_stability_penalty=args.selector_stability_penalty,
        selector_admission_gate=args.selector_admission_gate,
        admission_max_promoted=args.admission_max_promoted,
        admission_min_score=args.admission_min_score,
        admission_min_coverage=args.admission_min_coverage,
        admission_min_observations=args.admission_min_observations,
        admission_subwindows=args.admission_subwindows,
        admission_min_subwindow_passes=args.admission_min_subwindow_passes,
        admission_subwindow_min_abs_ic=args.admission_subwindow_min_abs_ic,
        admission_max_abs_corr_to_live=args.admission_max_abs_corr_to_live,
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
        execution_price=args.execution_price,
        hard_exit_score_threshold=args.hard_exit_score_threshold,
        hard_exit_min_holding_days=args.hard_exit_min_holding_days,
        tail_cleanup_weight=args.tail_cleanup_weight,
        renormalize_after_exit_cleanup=args.renormalize_after_exit_cleanup,
        placebo_mode=args.placebo_mode,
        placebo_seed=args.placebo_seed,
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
        model_pool_diagnostics=args.model_pool_diagnostics,
        similarity_threshold=args.similarity_threshold,
        pool_top_k=args.pool_top_k,
        pool_regime_window=args.pool_regime_window,
        shadow_window=args.shadow_window,
        model_pool_selection_metric=args.model_pool_selection_metric,
        model_pool_reuse_min_score=args.model_pool_reuse_min_score,
        model_pool_reuse_margin=args.model_pool_reuse_margin,
        model_pool_trigger_mode=args.model_pool_trigger_mode,
        alpha_source=args.alpha_source,
        alpha_ids=args.alpha_ids,
        skip_effective_filter=args.skip_effective_filter,
        exclude_indclass_cap_alphas=args.exclude_indclass_cap_alphas,
        train_window_days=args.train_window_days,
        data_source=args.data_source,
        allow_yfinance=args.allow_yfinance,
        frozen_config=args.frozen_config,
        frozen_execution=args.frozen_execution,
    )

    print(f"\n結果已寫入: {result['run_dir']}")
    print(Path(result["summary_path"]).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
