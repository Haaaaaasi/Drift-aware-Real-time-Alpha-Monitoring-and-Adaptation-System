"""MVP v3 — Adaptation A/B 實驗：三種策略的 walk-forward 對照。

研究問題
--------
論文 RQ3：Adaptation（scheduled / performance-triggered）能否在不同市場 regime 下
量化改善績效？本實驗以同一份資料、同一份 universe、同一組 XGBoost 超參數，跑四組
對照：

  * ``none``         — 訓練一次後凍結（baseline，呈現「完全不 adapt」的效能曲線）
  * ``scheduled_20`` — 每 20 個交易日重訓（中頻 scheduled）
  * ``scheduled_60`` — 每 60 個交易日重訓（低頻 scheduled，接近季度）
  * ``triggered``    — 由 rolling IC / Sharpe 退化觸發，含 20 日冷卻期

各組會呼叫 ``pipelines.simulate_recent.simulate()`` 獨立產出
``reports/simulations/sim_..._<strategy>/``（daily_pnl.csv / retrain_log.csv /
summary.txt）。本腳本再彙整為 ``reports/adaptation_ab/<run_id>/``：

  * ``comparison.csv`` — 四組績效指標對照
  * ``comparison.png`` — 4 面板視覺化（累積報酬 / rolling Sharpe / rolling IC + 重訓時機 / 重訓次數）
  * ``experiment_summary.md`` — 實驗設計與結論摘要（供研究報告引用）

使用範例
--------
    python -m pipelines.ab_experiment \\
        --start 2022-01-01 --end 2024-12-31 --top-k 10

    # 自訂策略子集
    python -m pipelines.ab_experiment \\
        --strategies none triggered
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 強制無頭後端（避開 Windows tk.tcl 殘缺問題）
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from pipelines.simulate_recent import (
    _compute_costs,
    _filter_universe,
    _next_day_returns,
    _summarize,
    _trading_days,
    simulate,
)
from pipelines.daily_batch_pipeline import load_csv_data
from src.common.logging import get_logger, setup_logging
from src.config.constants import DATA_SOURCE_DEFAULT_PATHS, DEFAULT_DATA_SOURCE

setup_logging()
logger = get_logger("ab_experiment")

DEFAULT_OUT_DIR = Path("reports/adaptation_ab")
DATA_SOURCE_DEFAULTS = DATA_SOURCE_DEFAULT_PATHS

# 四組預設策略配置
DEFAULT_STRATEGIES: dict[str, dict] = {
    "none": {
        "strategy": "none",
    },
    "scheduled_20": {
        "strategy": "scheduled",
        "retrain_every": 20,
    },
    "scheduled_60": {
        "strategy": "scheduled",
        "retrain_every": 60,
    },
    "triggered": {
        "strategy": "triggered",
        "trigger_ic_threshold": 0.0,
        "trigger_ic_days": 3,
        "trigger_sharpe_threshold": 0.0,
        "trigger_sharpe_days": 10,
        "min_retrain_gap": 20,
    },
    "model_pool": {
        "strategy": "model_pool",
        "trigger_ic_threshold": 0.0,
        "trigger_ic_days": 3,
        "trigger_sharpe_threshold": 0.0,
        "trigger_sharpe_days": 10,
        "min_retrain_gap": 20,
        # Phase B-1：score = exp(-d_zscored/2) * staleness * perf_gate；
        # 0.5 對應「跨維度平均 ~1.4 std 內」，比舊 cosine 0.8 寬鬆但語意一致
        "similarity_threshold": 0.5,
        "pool_regime_window": 60,
        "shadow_window": 20,
        # Phase B-3：top-k 個 reused 候選都丟進 shadow eval，由 evaluator 自選
        "pool_top_k": 3,
    },
}

# 成本敏感度 sweep 預設四檔：0% / 0.2% / 0.4% / 0.6% round-trip
DEFAULT_COST_SWEEP: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6)


def run_ab_experiment(
    csv_path: str | Path,
    start: date,
    end: date,
    strategies: list[str] | None = None,
    top_k: int = 10,
    portfolio_method: str = "equal_weight_topk",
    rebalance_every: int = 1,
    entry_rank: int = 20,
    exit_rank: int = 40,
    max_turnover: float = 1.0,
    min_holding_days: int = 0,
    objective: str = "forward_return",
    benchmark: str = "none",
    purge_days: int = 5,
    horizon_days: int = 5,
    capital: float = 10_000_000.0,
    slippage_bps: float = 5.0,
    commission_rate: float = 0.000926,
    tax_rate: float = 0.003,
    round_trip_cost_pct: float | None = None,
    trigger_window_days: int = 60,
    trigger_eval_gap_days: int = 20,
    shadow_warmup_days: int = 5,
    symbols: list[str] | None = None,
    min_turnover_ntd: float = 0.0,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    run_tag: str | None = None,
    alpha_source: str = "python",
    alpha_ids: list[str] | None = None,
    skip_effective_filter: bool = False,
    cost_sweep: list[float] | None = None,
    train_window_days: int | None = None,
    allow_yfinance: bool = False,
) -> dict:
    """跑 A/B 實驗，回傳彙整結果。"""
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    tag = f"_{run_tag}" if run_tag else ""
    run_id = f"ab_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}_top{top_k}{tag}"
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if strategies is None:
        strategies = list(DEFAULT_STRATEGIES.keys())
    # 預先驗證策略名稱合法（cost_sweep / 單跑都要）
    for strat_key in strategies:
        if strat_key not in DEFAULT_STRATEGIES:
            raise ValueError(f"未知策略 {strat_key!r}，可選：{list(DEFAULT_STRATEGIES)}")
    logger.info("ab_experiment_start", run_id=run_id, strategies=strategies, cost_sweep=cost_sweep)

    # Sub-run output 寫到 ab run_dir 之內，避免多測試共用全域 reports/simulations/
    # 造成 daily_pnl.csv 互相覆寫
    sim_out_dir = run_dir / "simulations"
    sim_out_dir.mkdir(parents=True, exist_ok=True)

    # 共用參數打包：方便 cost-sweep 子 run 與單跑 baseline 共用
    common_sim_kwargs = dict(
        csv_path=csv_path,
        start=start,
        end=end,
        top_k=top_k,
        portfolio_method=portfolio_method,
        rebalance_every=rebalance_every,
        entry_rank=entry_rank,
        exit_rank=exit_rank,
        max_turnover=max_turnover,
        min_holding_days=min_holding_days,
        objective=objective,
        purge_days=purge_days,
        horizon_days=horizon_days,
        capital=capital,
        slippage_bps=slippage_bps,
        commission_rate=commission_rate,
        tax_rate=tax_rate,
        out_dir=sim_out_dir,
        symbols=symbols,
        min_turnover_ntd=min_turnover_ntd,
        alpha_source=alpha_source,
        alpha_ids=alpha_ids,
        skip_effective_filter=skip_effective_filter,
        trigger_window_days=trigger_window_days,
        trigger_eval_gap_days=trigger_eval_gap_days,
        shadow_warmup_days=shadow_warmup_days,
        train_window_days=train_window_days,
        allow_yfinance=allow_yfinance,
    )

    # --- 0. cost-sweep 模式：外層遍歷成本場景，每場景 N 策略 ---
    if cost_sweep is not None:
        return _run_cost_sweep(
            cost_sweep=list(cost_sweep),
            strategies=strategies,
            run_dir=run_dir,
            run_id=run_id,
            common_sim_kwargs=common_sim_kwargs,
        )

    # --- 1. 跑各組模擬（單一 baseline 成本，預設 None=用三細項） ---
    results: dict[str, dict] = {}
    for strat_key in strategies:
        cfg = dict(DEFAULT_STRATEGIES[strat_key])
        logger.info("ab_strategy_start", strategy=strat_key, config=cfg)

        result = simulate(
            **common_sim_kwargs,
            round_trip_cost_pct=round_trip_cost_pct,
            run_tag=strat_key,
            **cfg,
        )
        results[strat_key] = result
        logger.info(
            "ab_strategy_complete",
            strategy=strat_key,
            sharpe=result["summary_metrics"].get("sharpe"),
            n_retrains=result["summary_metrics"].get("n_retrains"),
        )

    # --- 2. 彙整對照表 ---
    compare_rows = []
    for strat_key, res in results.items():
        m = res["summary_metrics"]
        compare_rows.append({
            "strategy": strat_key,
            "n_retrains": m.get("n_retrains", 0),
            "cumulative_return_pct": m.get("cumulative_return_pct", 0.0),
            "annualized_return_pct": m.get("annualized_return_pct", 0.0),
            "sharpe": m.get("sharpe", 0.0),
            "max_drawdown_pct": m.get("max_drawdown_pct", 0.0),
            "win_rate_pct": m.get("win_rate_pct", 0.0),
            "avg_turnover": m.get("avg_turnover", 0.0),
            "avg_gross_return_bps": m.get("avg_gross_return_bps", 0.0),
            "avg_total_cost_bps": m.get("avg_total_cost_bps", 0.0),
            "avg_net_return_bps": m.get("avg_net_return_bps", 0.0),
            "final_value": m.get("final_value", 0.0),
            "n_pool_reuses": m.get("n_pool_reuses", 0),
            "n_pool_misses": m.get("n_pool_misses", 0),
        })
    benchmark_result = None
    if benchmark == "ew_buy_hold_universe":
        benchmark_result = _run_ew_buy_hold_benchmark(
            csv_path=csv_path,
            start=start,
            end=end,
            capital=capital,
            slippage_bps=slippage_bps,
            commission_rate=commission_rate,
            tax_rate=tax_rate,
            round_trip_cost_pct=round_trip_cost_pct,
            symbols=symbols,
            min_turnover_ntd=min_turnover_ntd,
            run_dir=run_dir,
            allow_yfinance=allow_yfinance,
        )
        bm = benchmark_result["summary_metrics"]
        compare_rows.append({
            "strategy": "ew_buy_hold_universe",
            "n_retrains": 0,
            "cumulative_return_pct": bm.get("cumulative_return_pct", 0.0),
            "annualized_return_pct": bm.get("annualized_return_pct", 0.0),
            "sharpe": bm.get("sharpe", 0.0),
            "max_drawdown_pct": bm.get("max_drawdown_pct", 0.0),
            "win_rate_pct": bm.get("win_rate_pct", 0.0),
            "avg_turnover": bm.get("avg_turnover", 0.0),
            "avg_gross_return_bps": bm.get("avg_gross_return_bps", 0.0),
            "avg_total_cost_bps": bm.get("avg_total_cost_bps", 0.0),
            "avg_net_return_bps": bm.get("avg_net_return_bps", 0.0),
            "final_value": bm.get("final_value", 0.0),
            "n_pool_reuses": 0,
            "n_pool_misses": 0,
        })
    comparison_df = pd.DataFrame(compare_rows).set_index("strategy")
    comparison_path = run_dir / "comparison.csv"
    comparison_df.to_csv(comparison_path)

    # --- 3. 載入各組 daily_pnl + retrain_log 用於繪圖 ---
    loaded: dict[str, dict] = {}
    for strat_key, res in results.items():
        pnl = pd.read_csv(res["daily_pnl_path"], parse_dates=["date"])
        retrains = pd.read_csv(res["retrain_log_path"], parse_dates=["date"])
        loaded[strat_key] = {"pnl": pnl, "retrains": retrains}

    # --- 4. 繪製 4 面板圖 ---
    plot_path = run_dir / "comparison.png"
    _plot_comparison(loaded, plot_path, strategies)

    # --- 5. 寫 experiment_summary.md ---
    summary_md_path = run_dir / "experiment_summary.md"
    _write_experiment_summary(
        summary_md_path, run_id, start, end, top_k, comparison_df, strategies,
        trigger_window_days=trigger_window_days,
        trigger_eval_gap_days=trigger_eval_gap_days,
        shadow_warmup_days=shadow_warmup_days,
        commission_rate=commission_rate,
        tax_rate=tax_rate,
        slippage_bps=slippage_bps,
        round_trip_cost_pct=round_trip_cost_pct,
    )

    # --- 6. 存一份 config 方便回溯 ---
    config_path = run_dir / "config.json"
    config_path.write_text(
        json.dumps({
            "run_id": run_id,
            "start": str(start),
            "end": str(end),
            "top_k": top_k,
            "portfolio_method": portfolio_method,
            "rebalance_every": rebalance_every,
            "entry_rank": entry_rank,
            "exit_rank": exit_rank,
            "max_turnover": max_turnover,
            "min_holding_days": min_holding_days,
            "objective": objective,
            "benchmark": benchmark,
            "purge_days": purge_days,
            "horizon_days": horizon_days,
            "capital": capital,
            "slippage_bps": slippage_bps,
            "commission_rate": commission_rate,
            "tax_rate": tax_rate,
            "round_trip_cost_pct": round_trip_cost_pct,
            "trigger_window_days": trigger_window_days,
            "trigger_eval_gap_days": trigger_eval_gap_days,
            "shadow_warmup_days": shadow_warmup_days,
            "symbols": symbols,
            "min_turnover_ntd": min_turnover_ntd,
            "alpha_source": alpha_source,
            "alpha_ids": alpha_ids,
            "skip_effective_filter": skip_effective_filter,
            "strategies": {k: DEFAULT_STRATEGIES[k] for k in strategies},
            "run_dirs": {k: res["run_dir"] for k, res in results.items()},
            "benchmark_path": benchmark_result["daily_pnl_path"] if benchmark_result else None,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("ab_experiment_complete", run_id=run_id)
    return {
        "run_dir": str(run_dir),
        "comparison_path": str(comparison_path),
        "plot_path": str(plot_path),
        "summary_md_path": str(summary_md_path),
        "comparison_df": comparison_df,
        "results": results,
        "benchmark_result": benchmark_result,
    }


def _run_ew_buy_hold_benchmark(
    *,
    csv_path: str | Path,
    start: date,
    end: date,
    capital: float,
    slippage_bps: float,
    commission_rate: float,
    tax_rate: float,
    round_trip_cost_pct: float | None,
    symbols: list[str] | None,
    min_turnover_ntd: float,
    run_dir: Path,
    allow_yfinance: bool,
) -> dict:
    bars = load_csv_data(csv_path, allow_yfinance=allow_yfinance)
    bars["security_id"] = bars["security_id"].astype(str)
    bars, _ = _filter_universe(bars, symbols, min_turnover_ntd, start)
    days = _trading_days(bars, start, end)
    if not days:
        raise RuntimeError("benchmark window has no trading days")

    first_day = days[0]
    initial = bars[bars["tradetime"] == first_day]["security_id"].astype(str).drop_duplicates()
    if initial.empty:
        raise RuntimeError("benchmark first day has no tradable securities")

    weight = 1.0 / len(initial)
    weights = {sec: weight for sec in initial}
    next_ret = _next_day_returns(bars)
    portfolio_value = capital
    records: list[dict] = []

    for i, t in enumerate(days):
        buys = 1.0 if i == 0 else 0.0
        sells = 0.0
        turnover = max(buys, sells)
        commission_cost, tax_cost, slippage_cost = _compute_costs(
            buys=buys,
            sells=sells,
            commission_rate=commission_rate,
            tax_rate=tax_rate,
            slippage_bps=slippage_bps,
            round_trip_cost_pct=round_trip_cost_pct,
        )
        gross_return = 0.0
        for sec, w in weights.items():
            r = next_ret["next_return"].get((sec, t), np.nan)
            if not np.isnan(r):
                gross_return += w * float(r)
        net_return = gross_return - commission_cost - tax_cost - slippage_cost
        portfolio_value *= (1 + net_return)
        records.append({
            "date": t.strftime("%Y-%m-%d"),
            "n_holdings": len(weights),
            "gross_exposure": sum(weights.values()),
            "turnover": turnover,
            "buys_turnover": buys,
            "sells_turnover": sells,
            "rebalance_flag": i == 0,
            "held_from_prev_count": len(weights) if i > 0 else 0,
            "forced_sells_count": 0,
            "turnover_cap_applied": False,
            "gross_return": gross_return,
            "commission_cost": commission_cost,
            "tax_cost": tax_cost,
            "slippage_cost": slippage_cost,
            "net_return": net_return,
            "cumulative_value": portfolio_value,
            "rolling_ic": np.nan,
            "rolling_sharpe": np.nan,
        })

    bench_dir = run_dir / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    pnl_df = pd.DataFrame(records)
    daily_path = bench_dir / "ew_buy_hold_universe_daily_pnl.csv"
    pnl_df.to_csv(daily_path, index=False)
    summary = _summarize(pnl_df, capital)
    summary_path = bench_dir / "ew_buy_hold_universe_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "daily_pnl_path": str(daily_path),
        "summary_path": str(summary_path),
        "summary_metrics": summary,
    }


def _run_cost_sweep(
    *,
    cost_sweep: list[float],
    strategies: list[str],
    run_dir: Path,
    run_id: str,
    common_sim_kwargs: dict,
) -> dict:
    """成本敏感度 sweep：外層遍歷 cost 場景，每場景跑 N 策略。

    每個 cost 場景把 ``round_trip_cost_pct`` 設為該值（覆寫三細項），讓 net_return
    完全由單一參數控制，方便比較「成本提升 → adaptation 排序變化」的穩健性。

    Outputs（在 ``run_dir`` 底下）：
        * ``cost_sensitivity.csv`` — long-format：strategy × cost_pct × 主要績效欄位
        * ``cost_sensitivity.png`` — 4 panel 圖：Sharpe / 累積報酬 / 最大回撤 / avg_turnover 隨成本走勢
        * ``config.json`` — 紀錄完整 sweep 參數
    """
    sweep_records: list[dict] = []
    sub_runs: dict[str, dict[str, str]] = {}  # cost_label -> {strategy -> sub_run_dir}

    for cost_pct in cost_sweep:
        cost_label = f"cost_{cost_pct:.3f}".rstrip("0").rstrip(".")
        sub_runs[cost_label] = {}
        for strat_key in strategies:
            cfg = dict(DEFAULT_STRATEGIES[strat_key])
            logger.info(
                "ab_sweep_strategy_start",
                cost_pct=cost_pct,
                strategy=strat_key,
                config=cfg,
            )
            result = simulate(
                **common_sim_kwargs,
                round_trip_cost_pct=float(cost_pct),
                run_tag=f"{strat_key}_{cost_label}",
                **cfg,
            )
            m = result["summary_metrics"]
            sub_runs[cost_label][strat_key] = result["run_dir"]
            sweep_records.append({
                "cost_pct": float(cost_pct),
                "strategy": strat_key,
                "n_retrains": int(m.get("n_retrains", 0)),
                "sharpe": float(m.get("sharpe", 0.0)),
                "cumulative_return_pct": float(m.get("cumulative_return_pct", 0.0)),
                "annualized_return_pct": float(m.get("annualized_return_pct", 0.0)),
                "max_drawdown_pct": float(m.get("max_drawdown_pct", 0.0)),
                "win_rate_pct": float(m.get("win_rate_pct", 0.0)),
                "avg_turnover": float(m.get("avg_turnover", 0.0)),
                "avg_gross_return_bps": float(m.get("avg_gross_return_bps", 0.0)),
                "avg_total_cost_bps": float(m.get("avg_total_cost_bps", 0.0)),
                "avg_net_return_bps": float(m.get("avg_net_return_bps", 0.0)),
                "final_value": float(m.get("final_value", 0.0)),
                "n_pool_reuses": int(m.get("n_pool_reuses", 0)),
                "n_pool_misses": int(m.get("n_pool_misses", 0)),
            })

    sweep_df = pd.DataFrame(sweep_records)
    sweep_csv_path = run_dir / "cost_sensitivity.csv"
    sweep_df.to_csv(sweep_csv_path, index=False)

    # 簡單繪圖：Sharpe / cum_return / max_dd / avg_turnover vs cost_pct
    plot_path = run_dir / "cost_sensitivity.png"
    _plot_cost_sweep(sweep_df, plot_path, strategies)

    # 寫 config + 摘要
    config_path = run_dir / "config.json"
    config_path.write_text(
        json.dumps({
            "run_id": run_id,
            "mode": "cost_sweep",
            "cost_sweep": list(cost_sweep),
            "strategies": strategies,
            "sub_runs": sub_runs,
            "common_sim_kwargs": {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in common_sim_kwargs.items() if k != "csv_path"
            } | {"csv_path": str(common_sim_kwargs["csv_path"])},
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    summary_path = run_dir / "experiment_summary.md"
    _write_cost_sweep_summary(summary_path, run_id, sweep_df, strategies, list(cost_sweep))

    logger.info("ab_cost_sweep_complete", run_id=run_id, n_records=len(sweep_records))
    return {
        "run_dir": str(run_dir),
        "cost_sensitivity_path": str(sweep_csv_path),
        "plot_path": str(plot_path),
        "summary_md_path": str(summary_path),
        "sweep_df": sweep_df,
    }


def _plot_cost_sweep(sweep_df: pd.DataFrame, out_path: Path, strategies: list[str]) -> None:
    """4 panel 圖：橫軸 cost_pct，縱軸主要績效欄位，每策略一條線。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    color_cycle = plt.cm.tab10.colors
    colors = {s: color_cycle[i % len(color_cycle)] for i, s in enumerate(strategies)}

    metrics = [
        ("sharpe", "Sharpe Ratio", axes[0, 0]),
        ("cumulative_return_pct", "Cumulative Return (%)", axes[0, 1]),
        ("max_drawdown_pct", "Max Drawdown (%)", axes[1, 0]),
        ("avg_turnover", "Avg Turnover", axes[1, 1]),
    ]
    for col, ylabel, ax in metrics:
        for strat in strategies:
            sub = sweep_df[sweep_df["strategy"] == strat].sort_values("cost_pct")
            ax.plot(sub["cost_pct"], sub[col],
                    marker="o", label=strat, color=colors[strat], linewidth=1.5)
        ax.set_xlabel("Round-trip cost (%)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("Cost Sensitivity Sweep — 5 strategies × N cost scenarios", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _write_cost_sweep_summary(
    path: Path, run_id: str, sweep_df: pd.DataFrame,
    strategies: list[str], cost_sweep: list[float],
) -> None:
    """成本敏感度摘要：列出每個 cost 場景的策略 Sharpe 排序。"""
    lines = [
        f"# Cost Sensitivity Sweep 摘要：{run_id}",
        "",
        f"- Cost scenarios（round-trip %）：{cost_sweep}",
        f"- 對照策略：{', '.join(strategies)}",
        "",
        "## Sharpe 排序（按 cost 場景）",
        "",
    ]
    for cost in cost_sweep:
        sub = sweep_df[sweep_df["cost_pct"] == float(cost)].sort_values("sharpe", ascending=False)
        lines.append(f"### cost = {cost:.3f}%")
        lines.append("")
        lines.append("| 排名 | strategy | sharpe | cum_ret_% | max_dd_% | n_retrains |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for rank, (_, row) in enumerate(sub.iterrows(), start=1):
            lines.append(
                f"| {rank} | `{row['strategy']}` | "
                f"{row['sharpe']:.3f} | "
                f"{row['cumulative_return_pct']:.3f} | "
                f"{row['max_drawdown_pct']:.3f} | "
                f"{int(row['n_retrains'])} |"
            )
        lines.append("")

    lines.extend([
        "## 解讀",
        "若 Sharpe 排序在不同 cost 場景下穩定，代表 adaptation 結論對成本不敏感；",
        "若排序反轉，則「no-adapt 勝出」可能源自高換手策略未扣交易成本。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_comparison(
    loaded: dict[str, dict],
    out_path: Path,
    strategies: list[str],
) -> None:
    """繪製四面板對照圖：累積報酬 / rolling Sharpe / rolling IC + 重訓標記 / 重訓次數條狀圖。"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax_ret, ax_sharpe, ax_ic, ax_bar = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    color_cycle = plt.cm.tab10.colors
    colors = {s: color_cycle[i % len(color_cycle)] for i, s in enumerate(strategies)}

    # Panel A: 累積報酬
    for strat in strategies:
        pnl = loaded[strat]["pnl"]
        cum = pnl["cumulative_value"] / pnl["cumulative_value"].iloc[0]
        ax_ret.plot(pnl["date"], cum, label=strat, color=colors[strat], linewidth=1.3)
    ax_ret.set_title("A. 累積報酬（初始化為 1.0）", fontsize=11)
    ax_ret.set_ylabel("Normalized cumulative value")
    ax_ret.legend(loc="best", fontsize=9)
    ax_ret.grid(True, alpha=0.3)
    ax_ret.axhline(1.0, color="gray", linestyle="--", linewidth=0.7)

    # Panel B: Rolling Sharpe（20 日）
    for strat in strategies:
        pnl = loaded[strat]["pnl"]
        ax_sharpe.plot(pnl["date"], pnl["rolling_sharpe"], label=strat,
                       color=colors[strat], linewidth=1.0, alpha=0.85)
    ax_sharpe.set_title("B. Rolling Sharpe (20 日)", fontsize=11)
    ax_sharpe.set_ylabel("Annualized Sharpe")
    ax_sharpe.legend(loc="best", fontsize=9)
    ax_sharpe.grid(True, alpha=0.3)
    ax_sharpe.axhline(0.0, color="gray", linestyle="--", linewidth=0.7)

    # Panel C: Rolling IC + 重訓時機標記
    for strat in strategies:
        pnl = loaded[strat]["pnl"]
        ax_ic.plot(pnl["date"], pnl["rolling_ic"], label=strat,
                   color=colors[strat], linewidth=1.0, alpha=0.85)
        retrains = loaded[strat]["retrains"]
        if not retrains.empty and strat != "none":
            if strat == "model_pool" and "reason" in retrains.columns:
                # model_pool 依 reason 分色：reused=綠圓 / new=紅三角 / kept=灰叉
                for _, row in retrains.iterrows():
                    reason = str(row.get("reason", ""))
                    if "shadow_selected_reused" in reason:
                        m_marker, m_color, m_s = "o", "green", 40
                    elif "shadow_selected_new_pool" in reason:
                        m_marker, m_color, m_s = "v", "red", 35
                    else:
                        m_marker, m_color, m_s = "x", "gray", 30
                    ax_ic.scatter(row["date"], 0.0, color=m_color,
                                  marker=m_marker, s=m_s, zorder=5)
            else:
                ax_ic.scatter(retrains["date"], [0.0] * len(retrains),
                              color=colors[strat], marker="v", s=30,
                              zorder=5, label=None)
    ax_ic.set_title("C. Rolling IC (20 日) — ▼ 重訓 / ○ pool reuse / × kept", fontsize=11)
    ax_ic.set_ylabel("Information Coefficient")
    ax_ic.legend(loc="best", fontsize=9)
    ax_ic.grid(True, alpha=0.3)
    ax_ic.axhline(0.0, color="gray", linestyle="--", linewidth=0.7)

    # Panel D: 重訓次數 vs Sharpe 散佈條
    n_retrains = [len(loaded[s]["retrains"]) for s in strategies]
    sharpes = [
        loaded[s]["pnl"]["net_return"].mean() / loaded[s]["pnl"]["net_return"].std() * np.sqrt(252)
        if loaded[s]["pnl"]["net_return"].std() > 0 else 0.0
        for s in strategies
    ]
    x = np.arange(len(strategies))
    bars1 = ax_bar.bar(x - 0.2, n_retrains, width=0.4, label="重訓次數",
                       color=[colors[s] for s in strategies], alpha=0.6)
    ax_bar2 = ax_bar.twinx()
    bars2 = ax_bar2.bar(x + 0.2, sharpes, width=0.4, label="Sharpe",
                        color=[colors[s] for s in strategies], alpha=1.0,
                        edgecolor="black", linewidth=1.2)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(strategies, rotation=15)
    ax_bar.set_ylabel("重訓次數", color="gray")
    ax_bar2.set_ylabel("Sharpe Ratio")
    ax_bar.set_title("D. 重訓次數 vs 最終 Sharpe", fontsize=11)
    for b, v in zip(bars1, n_retrains):
        ax_bar.text(b.get_x() + b.get_width() / 2, b.get_height(), str(v),
                    ha="center", va="bottom", fontsize=8, color="gray")
    for b, v in zip(bars2, sharpes):
        ax_bar2.text(b.get_x() + b.get_width() / 2, b.get_height(),
                     f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"Adaptation A/B Experiment — {', '.join(strategies)}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    logger.info("comparison_plot_saved", path=str(out_path))


def _df_to_markdown(df: pd.DataFrame) -> str:
    """無依賴 tabulate 的簡易 DataFrame → Markdown table 轉換。"""
    rounded = df.round(3)
    header = "| " + " | ".join(["strategy"] + list(rounded.columns)) + " |"
    sep = "| " + " | ".join(["---"] * (len(rounded.columns) + 1)) + " |"
    rows = [
        "| " + " | ".join([str(idx)] + [str(v) for v in row]) + " |"
        for idx, row in zip(rounded.index, rounded.values)
    ]
    return "\n".join([header, sep, *rows])


def _write_experiment_summary(
    path: Path,
    run_id: str,
    start: date,
    end: date,
    top_k: int,
    comparison_df: pd.DataFrame,
    strategies: list[str],
    *,
    trigger_window_days: int = 60,
    trigger_eval_gap_days: int = 20,
    shadow_warmup_days: int = 5,
    commission_rate: float = 0.000926,
    tax_rate: float = 0.003,
    slippage_bps: float = 5.0,
    round_trip_cost_pct: float | None = None,
) -> None:
    """寫 Markdown 格式實驗摘要。"""
    if round_trip_cost_pct is not None:
        cost_block = (
            f"- 交易成本：使用單一 round-trip rate "
            f"`{round_trip_cost_pct:.3f}%`（覆寫三細項，cost-sensitivity sweep 場景）。"
        )
    else:
        commission_pct = commission_rate * 100
        tax_pct = tax_rate * 100
        slippage_pct = slippage_bps / 100  # bps → %
        round_trip_total = (commission_pct * 2 + tax_pct + slippage_pct * 2)
        cost_block = (
            f"- 交易成本：commission `{commission_pct:.4f}%`/side、"
            f"tax `{tax_pct:.3f}%`/sell-side、slippage `{slippage_bps:.1f}` bps/side；"
            f"round-trip 等效 ≈ `{round_trip_total:.4f}%`。"
        )

    lines = [
        f"# Adaptation A/B 實驗摘要：{run_id}",
        "",
        "## 實驗設計",
        f"- 期間：{start} → {end}",
        f"- Top-K：{top_k}",
        f"- 對照策略：{', '.join(strategies)}",
        "- 共用條件：同一份資料、同一組 XGBoost 超參數、同一份 effective alpha subset、",
        "  同一 portfolio constructor（equal_weight_topk, long_only）與 risk_manager。",
        "",
        "## 窗口與成本（Phase A 修正）",
        f"- Trigger window（rolling IC / Sharpe）：`signal_time ∈ "
        f"[t-{trigger_window_days}, t-{trigger_eval_gap_days}]`（calendar days）。",
        f"- Shadow eval window（model_pool）：`[t-30, t-10]`（shadow_window=20，與 trigger 不重疊）。",
        f"- Shadow warm-up gap：`{shadow_warmup_days}` 日（新候選訓練 cutoff 額外往前推，避免 IS leakage）。",
        cost_block,
        "",
        "## 對照結果",
        "",
        _df_to_markdown(comparison_df),
        "",
        "## 觀察",
        "",
    ]

    # 基本排名觀察
    ranked = comparison_df.sort_values("sharpe", ascending=False)
    best = ranked.index[0]
    worst = ranked.index[-1]
    lines.append(
        f"- Sharpe 最高：`{best}` ({comparison_df.loc[best, 'sharpe']:.3f})；"
        f"最低：`{worst}` ({comparison_df.loc[worst, 'sharpe']:.3f})。"
    )
    lines.append(
        f"- 重訓次數範圍：{int(comparison_df['n_retrains'].min())} → "
        f"{int(comparison_df['n_retrains'].max())}。"
    )

    if "none" in comparison_df.index and "triggered" in comparison_df.index:
        diff = comparison_df.loc["triggered", "sharpe"] - comparison_df.loc["none", "sharpe"]
        lines.append(
            f"- Triggered vs No-adapt Sharpe 差異：{diff:+.3f}"
            f"（正值代表 adaptation 有幫助）。"
        )

    lines.extend([
        "",
        "## 延伸分析",
        "詳細的 regime-stratified 分析、paired t-test、drift 指標疊圖請見 ",
        "`notebooks/03_adaptation_evaluation.py`。",
        "",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data-source",
        choices=["csv", "tej"],
        default=DEFAULT_DATA_SOURCE,
        help=(
            "tej = TEJ survivorship-correct parquet（預設）; "
            "csv = yfinance demo 路徑（已知 8476 資料污染）"
        ),
    )
    p.add_argument(
        "--csv",
        default=None,
        help="OHLCV 路徑；省略時依 --data-source 取對應預設（tej→tw_stocks_tej.parquet）",
    )
    p.add_argument(
        "--allow-yfinance",
        action="store_true",
        help="明確允許使用已知污染的 yfinance CSV（僅 demo/反例使用）",
    )
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default=None, help="預設：CSV 最後一日")
    p.add_argument(
        "--strategies", nargs="+", default=list(DEFAULT_STRATEGIES.keys()),
        choices=list(DEFAULT_STRATEGIES.keys()),
        help="要跑的策略子集（空格分隔）",
    )
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument(
        "--portfolio-method",
        choices=["equal_weight_topk", "score_proportional", "volatility_scaled", "turnover_aware_topk"],
        default="equal_weight_topk",
        help="Portfolio construction method; turnover_aware_topk enables low-turnover entry/exit buffers.",
    )
    p.add_argument("--rebalance-every", type=int, default=1, help="Portfolio rebalance interval in trading days.")
    p.add_argument("--entry-rank", type=int, default=20, help="turnover_aware_topk entry pool rank cutoff.")
    p.add_argument("--exit-rank", type=int, default=40, help="turnover_aware_topk exit buffer rank cutoff.")
    p.add_argument("--max-turnover", type=float, default=1.0, help="Maximum one-way turnover per rebalance.")
    p.add_argument("--min-holding-days", type=int, default=0, help="Minimum holding age before rank-based selling.")
    p.add_argument(
        "--objective",
        choices=["forward_return", "net_return_proxy"],
        default="forward_return",
        help="Model evaluation objective metadata; net_return_proxy adds cost-aware holdout diagnostics.",
    )
    p.add_argument(
        "--benchmark",
        choices=["none", "ew_buy_hold_universe"],
        default="none",
        help="Optional benchmark row to append to comparison.csv.",
    )
    p.add_argument("--purge-days", type=int, default=5)
    p.add_argument("--horizon-days", type=int, default=5)
    p.add_argument("--capital", type=float, default=10_000_000.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--commission-rate", type=float, default=0.000926,
                   help="per-side 手續費率，預設 0.1425%% × 0.65 折扣後")
    p.add_argument("--tax-rate", type=float, default=0.003,
                   help="sell-side only 證交稅，預設 0.3%%")
    p.add_argument("--round-trip-cost-pct", type=float, default=None,
                   help="若指定，覆寫三細項（單跑模式用），通常透過 --cost-sweep 而非單一值")
    p.add_argument("--cost-sweep", nargs="+", type=float, default=None,
                   help="成本敏感度 sweep：例如 --cost-sweep 0 0.2 0.4 0.6")
    p.add_argument("--trigger-window-days", type=int, default=60,
                   help="trigger 用 IC/Sharpe 計算回看上限（calendar days），預設 [t-60, ...]")
    p.add_argument("--trigger-eval-gap", type=int, default=20,
                   help="trigger 用 IC/Sharpe 排除最近（calendar days），預設 [..., t-20]")
    p.add_argument("--shadow-warmup-days", type=int, default=5,
                   help="model_pool shadow 候選的訓練 cutoff 額外往前推 N 日")
    p.add_argument("--train-window-days", type=int, default=None,
                   help="訓練窗口（calendar days）。None=expanding（預設）；正整數=rolling")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--min-turnover-ntd", type=float, default=0.0)
    p.add_argument("--run-tag", default=None)
    p.add_argument(
        "--alpha-source", choices=["python", "dolphindb"], default="python",
        help="alpha 來源：python=近似 15 個 / dolphindb=真實 WQ101",
    )
    p.add_argument(
        "--alpha-ids", nargs="+", default=None,
        help="限制 alpha 子集（僅 dolphindb 生效）",
    )
    p.add_argument(
        "--skip-effective-filter", action="store_true",
        help="跳過 effective_alphas.json 過濾（跑全 101 建議開）",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    csv_path = args.csv or DATA_SOURCE_DEFAULTS[args.data_source]
    if args.end:
        end = datetime.strptime(args.end, "%Y-%m-%d").date()
    else:
        path_obj = Path(csv_path)
        if path_obj.suffix.lower() == ".parquet":
            bars_dates = pd.read_parquet(path_obj, columns=["datetime"])
            bars_dates["datetime"] = pd.to_datetime(bars_dates["datetime"])
        else:
            bars_dates = pd.read_csv(path_obj, usecols=["datetime"], parse_dates=["datetime"])
        end = bars_dates["datetime"].max().date()

    result = run_ab_experiment(
        csv_path=csv_path,
        start=start,
        end=end,
        strategies=args.strategies,
        top_k=args.top_k,
        portfolio_method=args.portfolio_method,
        rebalance_every=args.rebalance_every,
        entry_rank=args.entry_rank,
        exit_rank=args.exit_rank,
        max_turnover=args.max_turnover,
        min_holding_days=args.min_holding_days,
        objective=args.objective,
        benchmark=args.benchmark,
        purge_days=args.purge_days,
        horizon_days=args.horizon_days,
        capital=args.capital,
        slippage_bps=args.slippage_bps,
        commission_rate=args.commission_rate,
        tax_rate=args.tax_rate,
        round_trip_cost_pct=args.round_trip_cost_pct,
        cost_sweep=args.cost_sweep,
        trigger_window_days=args.trigger_window_days,
        trigger_eval_gap_days=args.trigger_eval_gap,
        shadow_warmup_days=args.shadow_warmup_days,
        train_window_days=args.train_window_days,
        symbols=args.symbols,
        min_turnover_ntd=args.min_turnover_ntd,
        out_dir=args.out_dir,
        run_tag=args.run_tag,
        alpha_source=args.alpha_source,
        alpha_ids=args.alpha_ids,
        skip_effective_filter=args.skip_effective_filter,
        allow_yfinance=args.allow_yfinance,
    )

    print("\n=== A/B 實驗完成 ===")
    print(f"輸出目錄：{result['run_dir']}")
    if "comparison_df" in result:
        print("\n對照表：")
        print(result["comparison_df"].round(3).to_string())
        print(f"\n視覺化：{result['plot_path']}")
        print(f"摘要：{result['summary_md_path']}")
    elif "sweep_df" in result:
        print("\nCost sensitivity sweep：")
        pivot = result["sweep_df"].pivot(index="strategy", columns="cost_pct", values="sharpe")
        print("Sharpe 樞紐表（行=策略，列=cost_pct）：")
        print(pivot.round(3).to_string())
        print(f"\n視覺化：{result['plot_path']}")
        print(f"摘要：{result['summary_md_path']}")


if __name__ == "__main__":
    main()
