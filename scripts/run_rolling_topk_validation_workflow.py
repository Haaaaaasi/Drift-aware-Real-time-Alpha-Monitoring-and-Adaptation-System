"""rolling_topk20_w126_pen10 的 frozen OOS 防守性驗證 workflow。

本 workflow 補做四件事：

1. shuffled-signal placebo。
2. liquidity-filtered EW benchmark sensitivity。
3. calendar regime 分段。
4. paired t-test 與 circular block bootstrap。
"""

from __future__ import annotations

import argparse
import json
import math
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
WORKFLOW_DIR = BASE_OUT_DIR / "rolling_topk_validation_20260514"
LOG_DIR = WORKFLOW_DIR / "logs"

START = "2024-07-01"
END = "2026-04-30"

ENTRY_RANK = 20
EXIT_RANK = 60
MAX_TURNOVER = 0.25
MIN_HOLDING_DAYS = 10
TAIL_CLEANUP_WEIGHT = 0.0025

SELECTOR_TOP_K = 20
SELECTOR_WINDOW_DAYS = 126
SELECTOR_STABILITY_PENALTY = 0.10
SELECTOR_MIN_COVERAGE = 0.2
SELECTOR_MIN_OBSERVATIONS = 1000

REAL_RUN_DIRS = {
    "next_vwap": BASE_OUT_DIR
    / "rolling_topk_stability_matrix_20260514"
    / "sim_20240701_20260430_top10_sched20_rtop20_w126_pen10_nextvwap",
    "next_open": BASE_OUT_DIR
    / "rolling_topk_best_execution_check_20260514"
    / "sim_20240701_20260430_top10_sched20_rtop20_w126_pen10_nextopen",
}

STATIC_RUN_DIRS = {
    "next_vwap": BASE_OUT_DIR
    / "selector_equivalence_full_20260514"
    / "sim_20240701_20260430_top10_sched20_static_is_nextvwap",
    "next_open": BASE_OUT_DIR
    / "selector_equivalence_full_20260514"
    / "sim_20240701_20260430_top10_sched20_static_is_nextopen",
}


def _suffix(execution_price: str) -> str:
    return "nextopen" if execution_price == "next_open" else "nextvwap"


def _run_tag(execution_price: str) -> str:
    return f"rtop20_w126_pen10_{_suffix(execution_price)}"


def _placebo_run_tag(execution_price: str, seed: int) -> str:
    return f"rtop20_w126_pen10_placebo_seed{seed}_{_suffix(execution_price)}"


def _placebo_run_dir(execution_price: str, seed: int) -> Path:
    return WORKFLOW_DIR / "placebo_runs" / (
        f"sim_20240701_20260430_top10_sched20_{_placebo_run_tag(execution_price, seed)}"
    )


def _simulation_command(
    *,
    execution_price: str,
    run_tag: str,
    out_dir: Path,
    placebo_seed: int | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "pipelines.simulate_recent",
        "--data-source",
        "tej",
        "--start",
        START,
        "--end",
        END,
        "--strategy",
        "scheduled",
        "--selector",
        "rolling_topk",
        "--selector-alpha-top-k",
        str(SELECTOR_TOP_K),
        "--selector-window-days",
        str(SELECTOR_WINDOW_DAYS),
        "--selector-min-coverage",
        str(SELECTOR_MIN_COVERAGE),
        "--selector-min-observations",
        str(SELECTOR_MIN_OBSERVATIONS),
        "--selector-stability-penalty",
        str(SELECTOR_STABILITY_PENALTY),
        "--retrain-every",
        "20",
        "--top-k",
        "10",
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
        "--tail-cleanup-weight",
        str(TAIL_CLEANUP_WEIGHT),
        "--objective",
        "net_return_proxy",
        "--execution-price",
        execution_price,
        "--exclude-indclass-cap-alphas",
        "--out-dir",
        str(out_dir),
        "--run-tag",
        run_tag,
    ]
    if placebo_seed is not None:
        cmd.extend(["--placebo-mode", "shuffle_signal", "--placebo-seed", str(placebo_seed)])
    return cmd


def _summary_from_daily_pnl(path: Path) -> dict:
    pnl = pd.read_csv(path)
    summary = _summarize(pnl, 10_000_000.0)
    return summary


def ensure_real_run(execution_price: str, *, force: bool = False) -> None:
    run_dir = REAL_RUN_DIRS[execution_price]
    if not force and (run_dir / "daily_pnl.csv").exists():
        print(f"[skip real] {execution_price}: {run_dir}")
        return
    out_dir = WORKFLOW_DIR / "real_runs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_log = LOG_DIR / f"real_{execution_price}.out.log"
    err_log = LOG_DIR / f"real_{execution_price}.err.log"
    cmd = _simulation_command(
        execution_price=execution_price,
        run_tag=_run_tag(execution_price),
        out_dir=out_dir,
    )
    print(f"[run real] {execution_price}")
    with out_log.open("w", encoding="utf-8") as out, err_log.open("w", encoding="utf-8") as err:
        completed = subprocess.run(cmd, stdout=out, stderr=err, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"real run failed: {execution_price}; see {out_log} / {err_log}")
    REAL_RUN_DIRS[execution_price] = out_dir / f"sim_20240701_20260430_top10_sched20_{_run_tag(execution_price)}"


def _run_or_load_placebo(execution_price: str, seed: int, *, force: bool = False) -> dict:
    run_dir = _placebo_run_dir(execution_price, seed)
    if not force and (run_dir / "daily_pnl.csv").exists():
        print(f"[skip placebo] {execution_price} seed={seed}")
    else:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        out_log = LOG_DIR / f"placebo_{execution_price}_seed{seed}.out.log"
        err_log = LOG_DIR / f"placebo_{execution_price}_seed{seed}.err.log"
        cmd = _simulation_command(
            execution_price=execution_price,
            run_tag=_placebo_run_tag(execution_price, seed),
            out_dir=WORKFLOW_DIR / "placebo_runs",
            placebo_seed=seed,
        )
        print(f"[run placebo] {execution_price} seed={seed}")
        with out_log.open("w", encoding="utf-8") as out, err_log.open("w", encoding="utf-8") as err:
            completed = subprocess.run(cmd, stdout=out, stderr=err, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"placebo failed: {execution_price} seed={seed}; see {out_log} / {err_log}")

    row = _summary_from_daily_pnl(run_dir / "daily_pnl.csv")
    row["execution_price"] = execution_price
    row["placebo_seed"] = seed
    row["placebo_mode"] = "shuffle_signal"
    row["run_dir"] = str(run_dir)
    return row


def run_placebos(*, n_vwap_seeds: int, n_open_seeds: int, force: bool = False) -> pd.DataFrame:
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
        real = _summary_from_daily_pnl(REAL_RUN_DIRS[execution_price] / "daily_pnl.csv")
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
            round_trip_cost_pct=None,
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
    for execution_price, real_dir in REAL_RUN_DIRS.items():
        real = _summary_from_daily_pnl(real_dir / "daily_pnl.csv")
        rows.append({
            "execution_price": execution_price,
            "benchmark": "rolling_topk20_w126_pen10",
            **{k: real.get(k) for k in [
                "cumulative_return_pct",
                "sharpe",
                "max_drawdown_pct",
                "avg_turnover",
                "avg_total_cost_bps",
            ]},
            "run_dir": str(real_dir),
        })
        static = _summary_from_daily_pnl(STATIC_RUN_DIRS[execution_price] / "daily_pnl.csv")
        rows.append({
            "execution_price": execution_price,
            "benchmark": "static_is_scheduled_20",
            **{k: static.get(k) for k in [
                "cumulative_return_pct",
                "sharpe",
                "max_drawdown_pct",
                "avg_turnover",
                "avg_total_cost_bps",
            ]},
            "run_dir": str(STATIC_RUN_DIRS[execution_price]),
        })
        for name, threshold in [
            ("ew_same_cadence_universe", 0.0),
            ("ew_same_cadence_liq100m", 100_000_000.0),
            ("ew_same_cadence_liq200m", 200_000_000.0),
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


def _load_strategy_pnl(run_dir: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "daily_pnl.csv", parse_dates=["date"])
    df["series"] = label
    return df


def _load_benchmark_pnl(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["series"] = label
    return df


def _regime_segments() -> list[tuple[str, str, str]]:
    return [
        ("2024_H2", "2024-07-01", "2024-12-31"),
        ("2025_H1", "2025-01-01", "2025-06-30"),
        ("2025_H2", "2025-07-01", "2025-12-31"),
        ("2026_YTD", "2026-01-01", "2026-04-30"),
    ]


def _segment_summary(df: pd.DataFrame, *, start: str, end: str, capital: float = 10_000_000.0) -> dict:
    seg = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))].copy()
    if seg.empty:
        return {}
    cumulative = float(np.prod(1.0 + seg["net_return"].astype(float)) - 1.0)
    r = seg["net_return"].astype(float)
    sharpe = 0.0 if float(r.std(ddof=1)) == 0.0 else float(r.mean() / r.std(ddof=1) * math.sqrt(252.0))
    value = capital * (1.0 + r).cumprod()
    dd = float(((value / value.cummax()) - 1.0).min() * 100.0)
    return {
        "n_days": int(len(seg)),
        "cumulative_return_pct": cumulative * 100.0,
        "sharpe": sharpe,
        "max_drawdown_pct": dd,
        "avg_turnover": float(seg["turnover"].mean()) if "turnover" in seg else np.nan,
        "avg_net_return_bps": float(r.mean() * 10_000.0),
    }


def run_regime_breakdown() -> pd.DataFrame:
    out_dir = WORKFLOW_DIR / "regime_breakdown"
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_dir = WORKFLOW_DIR / "benchmark_sensitivity"
    rows: list[dict] = []
    for execution_price in ["next_vwap", "next_open"]:
        series = [
            _load_strategy_pnl(REAL_RUN_DIRS[execution_price], "rolling_topk20_w126_pen10"),
            _load_strategy_pnl(STATIC_RUN_DIRS[execution_price], "static_is_scheduled_20"),
            _load_benchmark_pnl(
                bench_dir / f"{execution_price}_ew_same_cadence_liq100m_daily_pnl.csv",
                "ew_same_cadence_liq100m",
            ),
        ]
        for df in series:
            label = str(df["series"].iloc[0])
            for regime, start, end in _regime_segments():
                summary = _segment_summary(df, start=start, end=end)
                rows.append({
                    "execution_price": execution_price,
                    "series": label,
                    "regime": regime,
                    "start": start,
                    "end": end,
                    **summary,
                })
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "regime_breakdown_summary.csv", index=False, encoding="utf-8-sig")
    return out


def _paired_t(diff: pd.Series) -> tuple[float, float]:
    diff = diff.dropna().astype(float)
    if len(diff) < 3:
        return float("nan"), float("nan")
    mean = float(diff.mean())
    std = float(diff.std(ddof=1))
    if std == 0.0:
        return float("inf") if mean > 0 else float("-inf"), 0.0 if mean > 0 else 1.0
    t_stat = mean / (std / math.sqrt(len(diff)))
    try:
        from scipy import stats

        p = float(1.0 - stats.t.cdf(t_stat, df=len(diff) - 1))
    except Exception:
        p = float("nan")
    return t_stat, p


def _block_bootstrap(diff: pd.Series, *, block_len: int = 20, n_boot: int = 3000, seed: int = 20260514) -> dict:
    x = diff.dropna().astype(float).to_numpy()
    n = len(x)
    if n == 0:
        return {
            "block_len": block_len,
            "n_boot": n_boot,
            "bootstrap_mean_bps": np.nan,
            "bootstrap_p_one_sided": np.nan,
            "bootstrap_ci05_bps": np.nan,
            "bootstrap_ci50_bps": np.nan,
            "bootstrap_ci95_bps": np.nan,
        }
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=math.ceil(n / block_len))
        sample = []
        for s in starts:
            idx = (np.arange(block_len) + s) % n
            sample.extend(x[idx])
            if len(sample) >= n:
                break
        means[i] = np.mean(sample[:n])
    return {
        "block_len": block_len,
        "n_boot": n_boot,
        "bootstrap_mean_bps": float(means.mean() * 10_000.0),
        "bootstrap_p_one_sided": float((means <= 0.0).mean()),
        "bootstrap_ci05_bps": float(np.quantile(means, 0.05) * 10_000.0),
        "bootstrap_ci50_bps": float(np.quantile(means, 0.50) * 10_000.0),
        "bootstrap_ci95_bps": float(np.quantile(means, 0.95) * 10_000.0),
    }


def run_bootstrap(*, n_boot: int, block_len: int) -> pd.DataFrame:
    out_dir = WORKFLOW_DIR / "bootstrap"
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_dir = WORKFLOW_DIR / "benchmark_sensitivity"
    rows: list[dict] = []
    comparisons = [
        ("static_is_scheduled_20", lambda ep: STATIC_RUN_DIRS[ep] / "daily_pnl.csv"),
        (
            "ew_same_cadence_liq100m",
            lambda ep: bench_dir / f"{ep}_ew_same_cadence_liq100m_daily_pnl.csv",
        ),
        (
            "ew_same_cadence_liq200m",
            lambda ep: bench_dir / f"{ep}_ew_same_cadence_liq200m_daily_pnl.csv",
        ),
    ]
    for execution_price in ["next_vwap", "next_open"]:
        real = pd.read_csv(REAL_RUN_DIRS[execution_price] / "daily_pnl.csv", parse_dates=["date"])
        real = real[["date", "net_return"]].rename(columns={"net_return": "real_net_return"})
        for benchmark_name, path_fn in comparisons:
            bench = pd.read_csv(path_fn(execution_price), parse_dates=["date"])
            bench = bench[["date", "net_return"]].rename(columns={"net_return": "benchmark_net_return"})
            merged = real.merge(bench, on="date", how="inner")
            diff = merged["real_net_return"] - merged["benchmark_net_return"]
            t_stat, p_t = _paired_t(diff)
            rows.append({
                "execution_price": execution_price,
                "comparison": f"rolling_topk20_w126_pen10_vs_{benchmark_name}",
                "n_days": int(len(diff)),
                "mean_daily_excess_bps": float(diff.mean() * 10_000.0),
                "paired_t_stat": t_stat,
                "paired_t_p_one_sided": p_t,
                **_block_bootstrap(diff, block_len=block_len, n_boot=n_boot),
            })
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "bootstrap_paired_results.csv", index=False, encoding="utf-8-sig")
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


def write_summary(
    placebo_summary: pd.DataFrame,
    benchmark_summary: pd.DataFrame,
    regime_summary: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
) -> None:
    lines = [
        "# rolling_topk20_w126_pen10 OOS validation workflow",
        "",
        f"- Period: {START} -> {END}",
        "- Selector: rolling_topk20_w126_pen10",
        "- Strategy: scheduled_20",
        "",
        "## Placebo Shuffled Signal",
        "",
        _markdown_table(placebo_summary),
        "",
        "## Benchmark Sensitivity",
        "",
        _markdown_table(benchmark_summary),
        "",
        "## Regime Breakdown",
        "",
        _markdown_table(regime_summary),
        "",
        "## Paired / Block Bootstrap",
        "",
        _markdown_table(bootstrap_summary),
        "",
    ]
    (WORKFLOW_DIR / "workflow_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-vwap-seeds", type=int, default=30)
    parser.add_argument("--n-open-seeds", type=int, default=10)
    parser.add_argument("--n-boot", type=int, default=3000)
    parser.add_argument("--block-len", type=int, default=20)
    parser.add_argument("--force-real", action="store_true")
    parser.add_argument("--force-placebo", action="store_true")
    parser.add_argument("--skip-placebo", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-regime", action="store_true")
    parser.add_argument("--skip-bootstrap", action="store_true")
    args = parser.parse_args()

    WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    ensure_real_run("next_vwap", force=args.force_real)
    ensure_real_run("next_open", force=args.force_real)

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

    if args.skip_regime and (WORKFLOW_DIR / "regime_breakdown" / "regime_breakdown_summary.csv").exists():
        regime_summary = pd.read_csv(WORKFLOW_DIR / "regime_breakdown" / "regime_breakdown_summary.csv")
    elif args.skip_regime:
        regime_summary = pd.DataFrame()
    else:
        regime_summary = run_regime_breakdown()

    if args.skip_bootstrap and (WORKFLOW_DIR / "bootstrap" / "bootstrap_paired_results.csv").exists():
        bootstrap_summary = pd.read_csv(WORKFLOW_DIR / "bootstrap" / "bootstrap_paired_results.csv")
    elif args.skip_bootstrap:
        bootstrap_summary = pd.DataFrame()
    else:
        bootstrap_summary = run_bootstrap(n_boot=args.n_boot, block_len=args.block_len)

    manifest = {
        "start": START,
        "end": END,
        "selector_top_k": SELECTOR_TOP_K,
        "selector_window_days": SELECTOR_WINDOW_DAYS,
        "selector_stability_penalty": SELECTOR_STABILITY_PENALTY,
        "n_vwap_seeds": args.n_vwap_seeds,
        "n_open_seeds": args.n_open_seeds,
        "n_boot": args.n_boot,
        "block_len": args.block_len,
        "real_run_dirs": {k: str(v) for k, v in REAL_RUN_DIRS.items()},
        "static_run_dirs": {k: str(v) for k, v in STATIC_RUN_DIRS.items()},
    }
    (WORKFLOW_DIR / "workflow_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary(placebo_summary, benchmark_summary, regime_summary, bootstrap_summary)
    print(f"workflow complete: {WORKFLOW_DIR}")


if __name__ == "__main__":
    main()
