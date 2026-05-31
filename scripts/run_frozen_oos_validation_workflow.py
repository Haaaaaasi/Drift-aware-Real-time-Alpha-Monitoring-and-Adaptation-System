"""凍結版 scheduled_20 的 OOS 防守性驗證 workflow。

執行三件事：
1. terminal exposure diagnostic
2. shuffled-signal placebo seeds
3. same-cadence / liquidity-filtered EW benchmark sensitivity
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pipelines.daily_batch_pipeline import load_csv_data
from pipelines.simulate_recent import (
    DATA_SOURCE_DEFAULTS,
    _compute_costs,
    _filter_universe,
    _next_day_returns,
    _summarize,
    _trading_days,
)
from src.config.constants import DEFAULT_DATA_SOURCE


BASE_OUT_DIR = Path("reports/adaptation_ab")
WORKFLOW_DIR = BASE_OUT_DIR / "frozen_oos_validation_20260512"
LOG_DIR = WORKFLOW_DIR / "logs"

START = "2024-07-01"
END = "2026-04-30"

ENTRY_RANK = 20
EXIT_RANK = 60
MAX_TURNOVER = 0.25
MIN_HOLDING_DAYS = 10
TAIL_CLEANUP_WEIGHT = 0.0025

FROZEN_RUN_DIRS = {
    "next_vwap": BASE_OUT_DIR
    / "ab_20240701_20260430_top10_oos_no_indcap_cachealign_tail25_e20x60_t25_h10_nextvwap_20260512",
    "next_open": BASE_OUT_DIR
    / "ab_20240701_20260430_top10_oos_no_indcap_cachealign_tail25_e20x60_t25_h10_nextopen_20260512",
}


def _suffix(execution_price: str) -> str:
    return "nextopen" if execution_price == "next_open" else "nextvwap"


def _placebo_run_tag(execution_price: str, seed: int) -> str:
    return (
        "oos_no_indcap_cachealign_tail25_placebo_shuffle_"
        f"seed{seed}_{_suffix(execution_price)}_20260512"
    )


def _placebo_run_dir(execution_price: str, seed: int) -> Path:
    return BASE_OUT_DIR / f"ab_20240701_20260430_top10_{_placebo_run_tag(execution_price, seed)}"


def _frozen_run_tag(execution_price: str) -> str:
    return f"oos_no_indcap_cachealign_tail25_e20x60_t25_h10_{_suffix(execution_price)}_20260512"


def _frozen_command(execution_price: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pipelines.ab_experiment",
        "--data-source",
        "tej",
        "--start",
        START,
        "--end",
        END,
        "--top-k",
        "10",
        "--strategies",
        "none",
        "scheduled_20",
        "--benchmark",
        "ew_buy_hold_universe",
        "--portfolio-method",
        "turnover_aware_topk",
        "--rebalance-every",
        "10",
        "--entry-rank",
        str(ENTRY_RANK),
        "--exit-rank",
        str(EXIT_RANK),
        "--max-turnover",
        str(MAX_TURNOVER),
        "--min-holding-days",
        str(MIN_HOLDING_DAYS),
        "--objective",
        "net_return_proxy",
        "--train-window-days",
        "500",
        "--horizon-days",
        "5",
        "--execution-price",
        execution_price,
        "--exclude-indclass-cap-alphas",
        "--tail-cleanup-weight",
        str(TAIL_CLEANUP_WEIGHT),
        "--hard-exit-min-holding-days",
        str(MIN_HOLDING_DAYS),
        "--run-tag",
        _frozen_run_tag(execution_price),
    ]


def ensure_frozen_run(execution_price: str, *, force: bool = False) -> None:
    run_dir = FROZEN_RUN_DIRS[execution_price]
    if not force and (run_dir / "comparison.csv").exists():
        print(f"[skip frozen] {execution_price}: {run_dir}")
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_log = LOG_DIR / f"frozen_{execution_price}.out.log"
    err_log = LOG_DIR / f"frozen_{execution_price}.err.log"
    print(f"[run frozen] {execution_price}")
    with out_log.open("w", encoding="utf-8") as out, err_log.open("w", encoding="utf-8") as err:
        completed = subprocess.run(_frozen_command(execution_price), stdout=out, stderr=err, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"frozen run failed: {execution_price}; see {out_log} / {err_log}")


def _placebo_command(execution_price: str, seed: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pipelines.ab_experiment",
        "--data-source",
        "tej",
        "--start",
        START,
        "--end",
        END,
        "--top-k",
        "10",
        "--strategies",
        "scheduled_20",
        "--benchmark",
        "ew_buy_hold_universe",
        "--portfolio-method",
        "turnover_aware_topk",
        "--rebalance-every",
        "10",
        "--entry-rank",
        str(ENTRY_RANK),
        "--exit-rank",
        str(EXIT_RANK),
        "--max-turnover",
        str(MAX_TURNOVER),
        "--min-holding-days",
        str(MIN_HOLDING_DAYS),
        "--objective",
        "net_return_proxy",
        "--train-window-days",
        "500",
        "--horizon-days",
        "5",
        "--execution-price",
        execution_price,
        "--exclude-indclass-cap-alphas",
        "--tail-cleanup-weight",
        str(TAIL_CLEANUP_WEIGHT),
        "--hard-exit-min-holding-days",
        str(MIN_HOLDING_DAYS),
        "--placebo-mode",
        "shuffle_signal",
        "--placebo-seed",
        str(seed),
        "--run-tag",
        _placebo_run_tag(execution_price, seed),
    ]


def _load_scheduled_row(run_dir: Path) -> dict:
    df = pd.read_csv(run_dir / "comparison.csv")
    row = df[df["strategy"] == "scheduled_20"].iloc[0].to_dict()
    row["run_dir"] = str(run_dir)
    return row


def _run_or_load_placebo(execution_price: str, seed: int, *, force: bool = False) -> dict:
    run_dir = _placebo_run_dir(execution_price, seed)
    if not force and (run_dir / "comparison.csv").exists():
        print(f"[skip placebo] {execution_price} seed={seed}")
        row = _load_scheduled_row(run_dir)
    else:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        out_log = LOG_DIR / f"placebo_{execution_price}_seed{seed}.out.log"
        err_log = LOG_DIR / f"placebo_{execution_price}_seed{seed}.err.log"
        print(f"[run placebo] {execution_price} seed={seed}")
        with out_log.open("w", encoding="utf-8") as out, err_log.open("w", encoding="utf-8") as err:
            completed = subprocess.run(_placebo_command(execution_price, seed), stdout=out, stderr=err, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"placebo failed: {execution_price} seed={seed}; see {out_log} / {err_log}")
        row = _load_scheduled_row(run_dir)
    row["execution_price"] = execution_price
    row["placebo_seed"] = seed
    row["placebo_mode"] = "shuffle_signal"
    return row


def run_terminal_diagnostic() -> None:
    out_dir = WORKFLOW_DIR / "terminal_exposure"
    cmd = [
        sys.executable,
        "scripts/diagnose_terminal_exposure.py",
        "--out-dir",
        str(out_dir),
    ]
    for run_dir in FROZEN_RUN_DIRS.values():
        cmd.extend(["--run-dir", str(run_dir)])
    print("[run] terminal exposure diagnostic")
    subprocess.run(cmd, check=True)


def _liquidity_keep(
    bars: pd.DataFrame,
    day: pd.Timestamp,
    *,
    threshold_ntd: float,
    lookback_days: int = 60,
) -> set[str]:
    if threshold_ntd <= 0:
        return set(bars[bars["tradetime"] == day]["security_id"].astype(str))
    lookback = bars[
        (bars["tradetime"] < day)
        & (bars["tradetime"] >= day - pd.Timedelta(days=lookback_days * 2))
    ].copy()
    if lookback.empty:
        return set()
    lookback["turnover_value"] = lookback["vol"] * lookback["close"]
    avg = lookback.groupby("security_id")["turnover_value"].mean()
    return set(avg[avg >= threshold_ntd].index.astype(str))


def _run_equal_weight_rebalance_benchmark(
    *,
    bars: pd.DataFrame,
    execution_price: str,
    benchmark_name: str,
    rebalance_every: int,
    liquidity_threshold_ntd: float,
    capital: float = 10_000_000.0,
    commission_rate: float = 0.000926,
    tax_rate: float = 0.003,
    slippage_bps: float = 5.0,
    round_trip_cost_pct: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    days = _trading_days(bars, pd.to_datetime(START).date(), pd.to_datetime(END).date())
    next_ret = _next_day_returns(bars, execution_price=execution_price)
    prev_weights: dict[str, float] = {}
    current_weights: dict[str, float] = {}
    last_rebalance_idx = -10**6
    portfolio_value = capital
    records: list[dict] = []

    for i, day in enumerate(days):
        tradable = set(bars[bars["tradetime"] == day]["security_id"].astype(str))
        rebalance_due = not current_weights or (i - last_rebalance_idx) >= rebalance_every
        if rebalance_due:
            keep = _liquidity_keep(bars, day, threshold_ntd=liquidity_threshold_ntd)
            universe = sorted(tradable & keep)
            weight = 1.0 / len(universe) if universe else 0.0
            current_weights = {sec: weight for sec in universe}
            last_rebalance_idx = i
        else:
            current_weights = {sec: w for sec, w in current_weights.items() if sec in tradable}

        all_secs = set(prev_weights) | set(current_weights)
        buys = sum(max(0.0, current_weights.get(sec, 0.0) - prev_weights.get(sec, 0.0)) for sec in all_secs)
        sells = sum(max(0.0, prev_weights.get(sec, 0.0) - current_weights.get(sec, 0.0)) for sec in all_secs)
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
        for sec, weight in current_weights.items():
            r = next_ret["next_return"].get((sec, day), np.nan)
            if not np.isnan(r):
                gross_return += weight * float(r)
        net_return = gross_return - commission_cost - tax_cost - slippage_cost
        portfolio_value *= 1.0 + net_return
        records.append({
            "date": day.strftime("%Y-%m-%d"),
            "benchmark": benchmark_name,
            "n_holdings": len(current_weights),
            "gross_exposure": sum(current_weights.values()),
            "execution_price": execution_price,
            "turnover": turnover,
            "buys_turnover": buys,
            "sells_turnover": sells,
            "rebalance_flag": bool(rebalance_due),
            "gross_return": gross_return,
            "commission_cost": commission_cost,
            "tax_cost": tax_cost,
            "slippage_cost": slippage_cost,
            "net_return": net_return,
            "cumulative_value": portfolio_value,
        })
        prev_weights = current_weights

    pnl = pd.DataFrame(records)
    summary = _summarize(pnl, capital)
    summary["benchmark"] = benchmark_name
    summary["execution_price"] = execution_price
    summary["rebalance_every"] = rebalance_every
    summary["liquidity_threshold_ntd"] = liquidity_threshold_ntd
    return pnl, summary


def run_benchmark_sensitivity() -> pd.DataFrame:
    out_dir = WORKFLOW_DIR / "benchmark_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_SOURCE_DEFAULTS[DEFAULT_DATA_SOURCE]
    bars = load_csv_data(csv_path, allow_yfinance=False)
    bars["security_id"] = bars["security_id"].astype(str)
    bars, _ = _filter_universe(bars, None, 0.0, pd.to_datetime(START).date())

    rows: list[dict] = []
    for execution_price, frozen_dir in FROZEN_RUN_DIRS.items():
        real = _load_scheduled_row(frozen_dir)
        rows.append({
            "execution_price": execution_price,
            "benchmark": "scheduled_20_tail25",
            "cumulative_return_pct": float(real["cumulative_return_pct"]),
            "sharpe": float(real["sharpe"]),
            "max_drawdown_pct": float(real["max_drawdown_pct"]),
            "avg_turnover": float(real["avg_turnover"]),
            "avg_total_cost_bps": float(real["avg_total_cost_bps"]),
            "run_dir": str(frozen_dir),
        })
        existing = pd.read_csv(frozen_dir / "comparison.csv")
        ew = existing[existing["strategy"] == "ew_buy_hold_universe"].iloc[0]
        rows.append({
            "execution_price": execution_price,
            "benchmark": "ew_buy_hold_universe",
            "cumulative_return_pct": float(ew["cumulative_return_pct"]),
            "sharpe": float(ew["sharpe"]),
            "max_drawdown_pct": float(ew["max_drawdown_pct"]),
            "avg_turnover": float(ew["avg_turnover"]),
            "avg_total_cost_bps": float(ew["avg_total_cost_bps"]),
            "run_dir": str(frozen_dir),
        })
        for name, threshold in [
            ("ew_same_cadence_universe", 0.0),
            ("ew_same_cadence_liq100m", 100_000_000.0),
        ]:
            pnl, summary = _run_equal_weight_rebalance_benchmark(
                bars=bars,
                execution_price=execution_price,
                benchmark_name=name,
                rebalance_every=10,
                liquidity_threshold_ntd=threshold,
            )
            daily_path = out_dir / f"{execution_price}_{name}_daily_pnl.csv"
            pnl.to_csv(daily_path, index=False, encoding="utf-8-sig")
            rows.append({
                "execution_price": execution_price,
                "benchmark": name,
                "cumulative_return_pct": float(summary["cumulative_return_pct"]),
                "sharpe": float(summary["sharpe"]),
                "max_drawdown_pct": float(summary["max_drawdown_pct"]),
                "avg_turnover": float(summary["avg_turnover"]),
                "avg_total_cost_bps": float(summary["avg_total_cost_bps"]),
                "run_dir": str(daily_path),
            })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "benchmark_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    return summary_df


def run_placebos(*, n_vwap_seeds: int, n_open_seeds: int, force: bool) -> pd.DataFrame:
    rows: list[dict] = []
    for execution_price, n in [("next_vwap", n_vwap_seeds), ("next_open", n_open_seeds)]:
        for seed in range(n):
            rows.append(_run_or_load_placebo(execution_price, seed, force=force))
            pd.DataFrame(rows).to_csv(
                WORKFLOW_DIR / "placebo_progress.csv",
                index=False,
                encoding="utf-8-sig",
            )
    placebo = pd.DataFrame(rows)
    placebo.to_csv(WORKFLOW_DIR / "placebo_results.csv", index=False, encoding="utf-8-sig")
    return placebo


def summarise_placebos(placebo: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for execution_price, group in placebo.groupby("execution_price"):
        real = _load_scheduled_row(FROZEN_RUN_DIRS[execution_price])
        for metric in ["cumulative_return_pct", "sharpe", "max_drawdown_pct"]:
            vals = group[metric].astype(float)
            real_val = float(real[metric])
            if metric == "max_drawdown_pct":
                percentile = float((vals >= real_val).mean() * 100.0)
            else:
                percentile = float((vals <= real_val).mean() * 100.0)
            rows.append({
                "execution_price": execution_price,
                "metric": metric,
                "real_value": real_val,
                "placebo_mean": float(vals.mean()),
                "placebo_p05": float(vals.quantile(0.05)),
                "placebo_p50": float(vals.quantile(0.50)),
                "placebo_p95": float(vals.quantile(0.95)),
                "real_percentile": percentile,
                "n_seeds": int(len(vals)),
            })
    out = pd.DataFrame(rows)
    out.to_csv(WORKFLOW_DIR / "placebo_summary.csv", index=False, encoding="utf-8-sig")
    return out


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        vals = []
        for col in headers:
            val = row[col]
            if pd.isna(val):
                vals.append("")
            elif isinstance(val, (float, np.floating)):
                vals.append(f"{float(val):.3f}".rstrip("0").rstrip("."))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_summary(placebo_summary: pd.DataFrame, benchmark_summary: pd.DataFrame) -> None:
    terminal_path = WORKFLOW_DIR / "terminal_exposure" / "terminal_exposure_summary.md"
    lines = [
        "# Frozen scheduled_20 OOS validation workflow",
        "",
        f"- Period: {START} -> {END}",
        f"- Frozen strategy: scheduled_20 + turnover-aware e{ENTRY_RANK}/x{EXIT_RANK}/t{MAX_TURNOVER}/h{MIN_HOLDING_DAYS} + tail25",
        "",
        "## Terminal Exposure",
        "",
        f"- 詳細結果：`{terminal_path}`",
        "",
        "## Placebo Shuffled Signal",
        "",
        _markdown_table(placebo_summary),
        "",
        "## Benchmark Sensitivity",
        "",
        _markdown_table(benchmark_summary),
        "",
    ]
    (WORKFLOW_DIR / "workflow_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-vwap-seeds", type=int, default=30)
    parser.add_argument("--n-open-seeds", type=int, default=10)
    parser.add_argument("--force-frozen", action="store_true")
    parser.add_argument("--force-placebo", action="store_true")
    parser.add_argument("--skip-frozen", action="store_true")
    parser.add_argument("--skip-terminal", action="store_true")
    parser.add_argument("--skip-placebo", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()

    WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_frozen:
        ensure_frozen_run("next_vwap", force=args.force_frozen)
        ensure_frozen_run("next_open", force=args.force_frozen)

    if not args.skip_terminal:
        run_terminal_diagnostic()

    if args.skip_placebo and (WORKFLOW_DIR / "placebo_results.csv").exists():
        placebo = pd.read_csv(WORKFLOW_DIR / "placebo_results.csv")
    elif args.skip_placebo:
        placebo = pd.DataFrame()
    else:
        placebo = run_placebos(
            n_vwap_seeds=args.n_vwap_seeds,
            n_open_seeds=args.n_open_seeds,
            force=args.force_placebo,
        )
    placebo_summary = summarise_placebos(placebo) if not placebo.empty else pd.DataFrame()

    if args.skip_benchmark and (WORKFLOW_DIR / "benchmark_sensitivity" / "benchmark_sensitivity_summary.csv").exists():
        benchmark_summary = pd.read_csv(WORKFLOW_DIR / "benchmark_sensitivity" / "benchmark_sensitivity_summary.csv")
    elif args.skip_benchmark:
        benchmark_summary = pd.DataFrame()
    else:
        benchmark_summary = run_benchmark_sensitivity()

    manifest = {
        "start": START,
        "end": END,
        "n_vwap_seeds": args.n_vwap_seeds,
        "n_open_seeds": args.n_open_seeds,
        "frozen_run_dirs": {k: str(v) for k, v in FROZEN_RUN_DIRS.items()},
    }
    (WORKFLOW_DIR / "workflow_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary(placebo_summary, benchmark_summary)
    print(f"workflow complete: {WORKFLOW_DIR}")


if __name__ == "__main__":
    main()
