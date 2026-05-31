"""As-of 2025-12-31 tuning 與 2026 temporal holdout replay。

目的：

1. Phase A 只使用 2024-07-01 至 2025-12-31 做 selector tuning。
2. 用預先宣告的排序規則選出 rolling_topk config。
3. Phase B 將選出的 config 原封不動跑 2026-01-01 至 2026-04-30。

這不是 untouched holdout，因為研究過程已經看過 2026 YTD；正式命名為
temporal holdout replay。真正 prospective holdout 仍需使用 2026-05 之後新增 TEJ 資料。
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
    _next_day_returns,
    _summarize,
    _trading_days,
)
from src.config.constants import DEFAULT_DATA_SOURCE


BASE_OUT_DIR = Path("reports/adaptation_ab")
WORKFLOW_DIR = BASE_OUT_DIR / "temporal_holdout_2026_20260518"
LOG_DIR = WORKFLOW_DIR / "logs"

PHASE_A_START = "2024-07-01"
PHASE_A_END = "2025-12-31"
PHASE_B_START = "2026-01-01"
PHASE_B_END = "2026-04-30"

MATRIX_TOP_K = [20, 30, 40]
MATRIX_WINDOWS = [126, 252, 504]
STABILITY_PENALTY = 0.10

SELECTOR_MIN_COVERAGE = 0.20
SELECTOR_MIN_OBSERVATIONS = 1000

ENTRY_RANK = 20
EXIT_RANK = 60
MAX_TURNOVER = 0.25
MIN_HOLDING_DAYS = 10
TAIL_CLEANUP_WEIGHT = 0.0025
TRAIN_WINDOW_DAYS = 500

CAPITAL = 10_000_000.0


def _suffix(execution_price: str) -> str:
    return "nextopen" if execution_price == "next_open" else "nextvwap"


def _penalty_label(value: float) -> str:
    return f"pen{int(round(value * 100)):02d}"


def _run_tag(prefix: str, *, alpha_top_k: int, window_days: int, execution_price: str) -> str:
    return f"{prefix}_rtop{alpha_top_k}_w{window_days}_{_penalty_label(STABILITY_PENALTY)}_{_suffix(execution_price)}"


def _static_tag(prefix: str, execution_price: str) -> str:
    return f"{prefix}_static_is_{_suffix(execution_price)}"


def _run_dir(out_dir: Path, *, start: str, end: str, tag: str) -> Path:
    return out_dir / f"sim_{start.replace('-', '')}_{end.replace('-', '')}_top10_sched20_{tag}"


def _simulation_command(
    *,
    start: str,
    end: str,
    out_dir: Path,
    run_tag: str,
    selector: str,
    execution_price: str,
    alpha_top_k: int = 20,
    window_days: int = 126,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "pipelines.simulate_recent",
        "--data-source",
        "tej",
        "--start",
        start,
        "--end",
        end,
        "--strategy",
        "scheduled",
        "--selector",
        selector,
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
        "--train-window-days",
        str(TRAIN_WINDOW_DAYS),
        "--out-dir",
        str(out_dir),
        "--run-tag",
        run_tag,
    ]
    if selector == "rolling_topk":
        cmd.extend([
            "--selector-alpha-top-k",
            str(alpha_top_k),
            "--selector-window-days",
            str(window_days),
            "--selector-min-coverage",
            str(SELECTOR_MIN_COVERAGE),
            "--selector-min-observations",
            str(SELECTOR_MIN_OBSERVATIONS),
            "--selector-stability-penalty",
            str(STABILITY_PENALTY),
        ])
    return cmd


def _run_command(cmd: list[str], *, label: str, force: bool, expected_daily_pnl: Path) -> None:
    if expected_daily_pnl.exists() and not force:
        print(f"[skip] {label}: {expected_daily_pnl.parent}")
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    safe_label = label.replace("/", "_").replace(" ", "_")
    out_log = LOG_DIR / f"{safe_label}.out.log"
    err_log = LOG_DIR / f"{safe_label}.err.log"
    print(f"[run] {label}")
    with out_log.open("w", encoding="utf-8") as out, err_log.open("w", encoding="utf-8") as err:
        completed = subprocess.run(cmd, stdout=out, stderr=err, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed; see {out_log} / {err_log}")


def _summary_from_run(run_dir: Path) -> dict:
    pnl = pd.read_csv(run_dir / "daily_pnl.csv")
    row = _summarize(pnl, CAPITAL)
    row["run_dir"] = str(run_dir)
    retrain_log = run_dir / "retrain_log.csv"
    row["n_retrains"] = int(len(pd.read_csv(retrain_log))) if retrain_log.exists() else 0
    return row


def run_phase_a(*, force: bool) -> pd.DataFrame:
    out_dir = WORKFLOW_DIR / "phase_a_runs"
    rows: list[dict] = []
    for alpha_top_k in MATRIX_TOP_K:
        for window_days in MATRIX_WINDOWS:
            tag = _run_tag(
                "asof20251231",
                alpha_top_k=alpha_top_k,
                window_days=window_days,
                execution_price="next_vwap",
            )
            run_dir = _run_dir(out_dir, start=PHASE_A_START, end=PHASE_A_END, tag=tag)
            cmd = _simulation_command(
                start=PHASE_A_START,
                end=PHASE_A_END,
                out_dir=out_dir,
                run_tag=tag,
                selector="rolling_topk",
                execution_price="next_vwap",
                alpha_top_k=alpha_top_k,
                window_days=window_days,
            )
            _run_command(cmd, label=f"phaseA_rtop{alpha_top_k}_w{window_days}", force=force, expected_daily_pnl=run_dir / "daily_pnl.csv")
            row = _summary_from_run(run_dir)
            row.update({
                "phase": "A_calibration",
                "execution_price": "next_vwap",
                "selector": "rolling_topk",
                "selector_alpha_top_k": alpha_top_k,
                "selector_window_days": window_days,
                "selector_stability_penalty": STABILITY_PENALTY,
            })
            rows.append(row)
    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["sharpe", "max_drawdown_pct", "cumulative_return_pct"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    out["selection_rank"] = np.arange(1, len(out) + 1)
    out.to_csv(WORKFLOW_DIR / "phase_a_matrix_summary.csv", index=False, encoding="utf-8-sig")
    return out


def select_config(matrix: pd.DataFrame) -> dict:
    if matrix.empty:
        raise RuntimeError("Phase A matrix is empty")
    best = matrix.iloc[0].to_dict()
    selected = {
        "selected_by": "phase_a_next_vwap_sharpe_then_drawdown_then_return",
        "calibration_start": PHASE_A_START,
        "calibration_end": PHASE_A_END,
        "holdout_start": PHASE_B_START,
        "holdout_end": PHASE_B_END,
        "selector": "rolling_topk",
        "selector_alpha_top_k": int(best["selector_alpha_top_k"]),
        "selector_window_days": int(best["selector_window_days"]),
        "selector_stability_penalty": float(best["selector_stability_penalty"]),
        "phase_a_sharpe": float(best["sharpe"]),
        "phase_a_cumulative_return_pct": float(best["cumulative_return_pct"]),
        "phase_a_max_drawdown_pct": float(best["max_drawdown_pct"]),
        "phase_a_run_dir": best["run_dir"],
    }
    (WORKFLOW_DIR / "selected_config.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return selected


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
    start: str,
    end: str,
    execution_price: str,
    benchmark_name: str,
    liquidity_threshold_ntd: float,
    out_dir: Path,
    rebalance_every: int = 10,
    commission_rate: float = 0.000926,
    tax_rate: float = 0.003,
    slippage_bps: float = 5.0,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{execution_price}_{benchmark_name}_daily_pnl.csv"
    days = _trading_days(bars, pd.to_datetime(start).date(), pd.to_datetime(end).date())
    next_ret = _next_day_returns(bars, execution_price=execution_price)
    prev_weights: dict[str, float] = {}
    current_weights: dict[str, float] = {}
    last_rebalance_idx = -10**6
    portfolio_value = CAPITAL
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
            "execution_price": execution_price,
            "n_holdings": len(current_weights),
            "turnover": turnover,
            "gross_return": gross_return,
            "commission_cost": commission_cost,
            "tax_cost": tax_cost,
            "slippage_cost": slippage_cost,
            "net_return": net_return,
            "cumulative_value": portfolio_value,
        })
        prev_weights = current_weights

    pnl = pd.DataFrame(records)
    pnl.to_csv(output_path, index=False, encoding="utf-8-sig")
    summary = _summarize(pnl, CAPITAL)
    summary.update({
        "phase": "B_holdout",
        "execution_price": execution_price,
        "series": benchmark_name,
        "selector": "benchmark",
        "liquidity_threshold_ntd": liquidity_threshold_ntd,
        "run_dir": str(output_path),
    })
    return summary


def run_phase_b(*, selected: dict, force: bool) -> pd.DataFrame:
    out_dir = WORKFLOW_DIR / "phase_b_runs"
    rows: list[dict] = []
    for execution_price in ["next_vwap", "next_open"]:
        tag = _run_tag(
            "holdout2026_selected",
            alpha_top_k=int(selected["selector_alpha_top_k"]),
            window_days=int(selected["selector_window_days"]),
            execution_price=execution_price,
        )
        run_dir = _run_dir(out_dir, start=PHASE_B_START, end=PHASE_B_END, tag=tag)
        cmd = _simulation_command(
            start=PHASE_B_START,
            end=PHASE_B_END,
            out_dir=out_dir,
            run_tag=tag,
            selector="rolling_topk",
            execution_price=execution_price,
            alpha_top_k=int(selected["selector_alpha_top_k"]),
            window_days=int(selected["selector_window_days"]),
        )
        _run_command(cmd, label=f"phaseB_selected_{execution_price}", force=force, expected_daily_pnl=run_dir / "daily_pnl.csv")
        row = _summary_from_run(run_dir)
        row.update({
            "phase": "B_holdout",
            "execution_price": execution_price,
            "series": f"selected_rtop{selected['selector_alpha_top_k']}_w{selected['selector_window_days']}_pen10",
            "selector": "rolling_topk",
            "selector_alpha_top_k": int(selected["selector_alpha_top_k"]),
            "selector_window_days": int(selected["selector_window_days"]),
            "selector_stability_penalty": STABILITY_PENALTY,
        })
        rows.append(row)

        static_tag = _static_tag("holdout2026", execution_price)
        static_dir = _run_dir(out_dir, start=PHASE_B_START, end=PHASE_B_END, tag=static_tag)
        static_cmd = _simulation_command(
            start=PHASE_B_START,
            end=PHASE_B_END,
            out_dir=out_dir,
            run_tag=static_tag,
            selector="static_is",
            execution_price=execution_price,
        )
        _run_command(static_cmd, label=f"phaseB_static_{execution_price}", force=force, expected_daily_pnl=static_dir / "daily_pnl.csv")
        static_row = _summary_from_run(static_dir)
        static_row.update({
            "phase": "B_holdout",
            "execution_price": execution_price,
            "series": "static_is_scheduled_20",
            "selector": "static_is",
        })
        rows.append(static_row)

    benchmark_out_dir = WORKFLOW_DIR / "phase_b_benchmarks"
    csv_path = DATA_SOURCE_DEFAULTS[DEFAULT_DATA_SOURCE]
    bars = load_csv_data(csv_path, allow_yfinance=False)
    bars["security_id"] = bars["security_id"].astype(str)
    for execution_price in ["next_vwap", "next_open"]:
        for name, threshold in [
            ("ew_same_cadence_universe", 0.0),
            ("ew_same_cadence_liq100m", 100_000_000.0),
            ("ew_same_cadence_liq200m", 200_000_000.0),
        ]:
            rows.append(_run_equal_weight_rebalance_benchmark(
                bars=bars,
                start=PHASE_B_START,
                end=PHASE_B_END,
                execution_price=execution_price,
                benchmark_name=name,
                liquidity_threshold_ntd=threshold,
                out_dir=benchmark_out_dir,
            ))

    out = pd.DataFrame(rows)
    out = out.sort_values(["execution_price", "series"]).reset_index(drop=True)
    out.to_csv(WORKFLOW_DIR / "phase_b_summary.csv", index=False, encoding="utf-8-sig")
    return out


def _paired_t(diff: pd.Series) -> tuple[float, float]:
    diff = diff.dropna().astype(float)
    if len(diff) < 3:
        return float("nan"), float("nan")
    mean = float(diff.mean())
    std = float(diff.std(ddof=1))
    if std == 0.0:
        return (float("inf") if mean > 0 else float("-inf"), 0.0 if mean > 0 else 1.0)
    t_stat = mean / (std / math.sqrt(len(diff)))
    try:
        from scipy import stats

        p = float(1.0 - stats.t.cdf(t_stat, df=len(diff) - 1))
    except Exception:
        p = float("nan")
    return t_stat, p


def _block_bootstrap(diff: pd.Series, *, block_len: int = 10, n_boot: int = 3000, seed: int = 20260518) -> dict:
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
        sample: list[float] = []
        for start in starts:
            idx = (np.arange(block_len) + start) % n
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


def _load_pnl(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["series"] = label
    return df


def run_phase_b_bootstrap(selected: dict) -> pd.DataFrame:
    out_dir = WORKFLOW_DIR / "phase_b_runs"
    bench_dir = WORKFLOW_DIR / "phase_b_benchmarks"
    selected_label = f"selected_rtop{selected['selector_alpha_top_k']}_w{selected['selector_window_days']}_pen10"
    rows: list[dict] = []
    for execution_price in ["next_vwap", "next_open"]:
        selected_tag = _run_tag(
            "holdout2026_selected",
            alpha_top_k=int(selected["selector_alpha_top_k"]),
            window_days=int(selected["selector_window_days"]),
            execution_price=execution_price,
        )
        selected_dir = _run_dir(out_dir, start=PHASE_B_START, end=PHASE_B_END, tag=selected_tag)
        real = _load_pnl(selected_dir / "daily_pnl.csv", selected_label)
        comparisons = [
            ("static_is_scheduled_20", _run_dir(out_dir, start=PHASE_B_START, end=PHASE_B_END, tag=_static_tag("holdout2026", execution_price)) / "daily_pnl.csv"),
            ("ew_same_cadence_liq100m", bench_dir / f"{execution_price}_ew_same_cadence_liq100m_daily_pnl.csv"),
            ("ew_same_cadence_liq200m", bench_dir / f"{execution_price}_ew_same_cadence_liq200m_daily_pnl.csv"),
        ]
        real_ret = real[["date", "net_return"]].rename(columns={"net_return": "real_net_return"})
        for benchmark_name, path in comparisons:
            bench = pd.read_csv(path, parse_dates=["date"])
            bench_ret = bench[["date", "net_return"]].rename(columns={"net_return": "benchmark_net_return"})
            merged = real_ret.merge(bench_ret, on="date", how="inner")
            diff = merged["real_net_return"] - merged["benchmark_net_return"]
            t_stat, p_t = _paired_t(diff)
            rows.append({
                "execution_price": execution_price,
                "comparison": f"{selected_label}_vs_{benchmark_name}",
                "n_days": int(len(diff)),
                "mean_daily_excess_bps": float(diff.mean() * 10_000.0),
                "paired_t_stat": t_stat,
                "paired_t_p_one_sided": p_t,
                **_block_bootstrap(diff),
            })
    out = pd.DataFrame(rows)
    out.to_csv(WORKFLOW_DIR / "phase_b_bootstrap.csv", index=False, encoding="utf-8-sig")
    return out


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(empty)"
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


def write_summary(matrix: pd.DataFrame, selected: dict, phase_b: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    selected_series = f"selected_rtop{selected['selector_alpha_top_k']}_w{selected['selector_window_days']}_pen10"
    phase_b_view = phase_b[
        [
            "execution_price",
            "series",
            "cumulative_return_pct",
            "sharpe",
            "max_drawdown_pct",
            "avg_turnover",
            "avg_total_cost_bps",
        ]
    ].copy()
    phase_a_view = matrix[
        [
            "selection_rank",
            "selector_alpha_top_k",
            "selector_window_days",
            "cumulative_return_pct",
            "sharpe",
            "max_drawdown_pct",
            "avg_turnover",
            "run_dir",
        ]
    ].copy()
    lines = [
        "# 2026 Temporal Holdout Replay",
        "",
        "建立日期：2026-05-18",
        "",
        "## 定義",
        "",
        f"- Phase A calibration：{PHASE_A_START} -> {PHASE_A_END}",
        f"- Phase B temporal holdout replay：{PHASE_B_START} -> {PHASE_B_END}",
        "- Phase A 只看 `next_vwap`。",
        "- 選參規則：Sharpe 由高到低，其次 max drawdown 較淺者，其次 cumulative return 較高者。",
        "- 這不是 untouched holdout；因研究過程已看過 2026 YTD，正式命名為 temporal holdout replay。",
        "",
        "## Phase A Matrix",
        "",
        _markdown_table(phase_a_view),
        "",
        "## Selected Config",
        "",
        "```json",
        json.dumps(selected, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Phase B Result",
        "",
        _markdown_table(phase_b_view),
        "",
        "## Phase B Paired / Block Bootstrap",
        "",
        _markdown_table(bootstrap),
        "",
        "## 初步解讀",
        "",
        f"- Phase B 使用 `{selected_series}`，沒有根據 2026 結果改參數。",
        "- 若 Phase B 仍優於 static_is 與 liquidity-filtered EW，只能說 temporal replay 支持穩健性；仍不可替代 2026-05 之後 prospective holdout。",
        "- 若 Phase B 不穩，表示先前 full validation 高績效很可能受 2026 regime 或參數選擇影響，正式報告應進一步降調 claim。",
        "",
    ]
    (WORKFLOW_DIR / "temporal_holdout_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(selected: dict) -> None:
    manifest = {
        "created_at": "2026-05-18",
        "workflow": "temporal_holdout_2026",
        "phase_a": {
            "start": PHASE_A_START,
            "end": PHASE_A_END,
            "matrix_top_k": MATRIX_TOP_K,
            "matrix_windows": MATRIX_WINDOWS,
            "stability_penalty": STABILITY_PENALTY,
            "selection_rule": "next_vwap Sharpe desc, max_drawdown_pct desc, cumulative_return_pct desc",
        },
        "phase_b": {
            "start": PHASE_B_START,
            "end": PHASE_B_END,
            "status": "TEMPORAL_HOLDOUT_REPLAY_NOT_UNTOUCHED",
        },
        "selected_config": selected,
        "artifacts": {
            "phase_a_matrix_summary": str(WORKFLOW_DIR / "phase_a_matrix_summary.csv"),
            "selected_config": str(WORKFLOW_DIR / "selected_config.json"),
            "phase_b_summary": str(WORKFLOW_DIR / "phase_b_summary.csv"),
            "phase_b_bootstrap": str(WORKFLOW_DIR / "phase_b_bootstrap.csv"),
            "summary": str(WORKFLOW_DIR / "temporal_holdout_summary.md"),
        },
    }
    (WORKFLOW_DIR / "workflow_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="即使 run output 已存在也重新執行。")
    parser.add_argument("--skip-phase-a", action="store_true", help="沿用既有 phase_a_matrix_summary.csv。")
    parser.add_argument("--skip-phase-b", action="store_true", help="只跑 Phase A，不跑 2026 holdout。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    if args.skip_phase_a and (WORKFLOW_DIR / "phase_a_matrix_summary.csv").exists():
        matrix = pd.read_csv(WORKFLOW_DIR / "phase_a_matrix_summary.csv")
    else:
        matrix = run_phase_a(force=args.force)
    selected = select_config(matrix)
    if args.skip_phase_b:
        write_manifest(selected)
        return
    phase_b = run_phase_b(selected=selected, force=args.force)
    bootstrap = run_phase_b_bootstrap(selected)
    write_summary(matrix, selected, phase_b, bootstrap)
    write_manifest(selected)
    print(f"[done] summary: {WORKFLOW_DIR / 'temporal_holdout_summary.md'}")


if __name__ == "__main__":
    main()
