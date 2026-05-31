"""執行 OOS exit discipline 小矩陣。

固定使用上一輪低換手 OOS 表現最佳的 portfolio 參數：
entry_rank=20, exit_rank=60, max_turnover=0.25, min_holding_days=10。

本腳本只改「抱住變差股票」的淘汰規則：
- baseline：沿用上一輪結果，不加 hard exit / tail cleanup
- hard0：持股滿 10 天後 signal_score <= 0 即賣出
- tail25 / tail50：只清殘餘小倉位，不用 signal_score 強砍
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BASE_OUT_DIR = Path("reports/adaptation_ab")
WORKFLOW_DIR = BASE_OUT_DIR / "exit_discipline_oos_20260512"
LOG_DIR = WORKFLOW_DIR / "logs"

START = "2024-07-01"
END = "2026-04-30"

ENTRY_RANK = 20
EXIT_RANK = 60
MAX_TURNOVER = 0.25
MIN_HOLDING_DAYS = 10

BASELINE_RUN_DIRS = {
    "next_vwap": BASE_OUT_DIR
    / "ab_20240701_20260430_top10_oos_no_indcap_turnovermatrix_e20_x60_t0p25_h10_nextvwap_20260512",
    "next_open": BASE_OUT_DIR
    / "ab_20240701_20260430_top10_oos_no_indcap_turnovermatrix_e20_x60_t0p25_h10_nextopen_20260512",
}


@dataclass(frozen=True)
class ExitSpec:
    label: str
    hard_exit_score_threshold: float | None
    tail_cleanup_weight: float
    renormalize_after_exit_cleanup: bool = False

    @property
    def is_baseline(self) -> bool:
        return self.hard_exit_score_threshold is None and self.tail_cleanup_weight <= 0


def _specs() -> list[ExitSpec]:
    return [
        ExitSpec("baseline", None, 0.0),
        ExitSpec("hard0", 0.0, 0.0),
        ExitSpec("tail25", None, 0.0025),
        ExitSpec("tail50", None, 0.0050),
    ]


def _run_tag(execution_price: str, spec: ExitSpec) -> str:
    suffix = "nextopen" if execution_price == "next_open" else "nextvwap"
    return (
        "oos_no_indcap_exitdiscipline_"
        f"e{ENTRY_RANK}x{EXIT_RANK}_t25_h{MIN_HOLDING_DAYS}_{spec.label}_{suffix}_20260512"
    )


def _expected_run_dir(execution_price: str, spec: ExitSpec) -> Path:
    if spec.is_baseline:
        return BASELINE_RUN_DIRS[execution_price]
    return BASE_OUT_DIR / f"ab_20240701_20260430_top10_{_run_tag(execution_price, spec)}"


def _command(execution_price: str, spec: ExitSpec) -> list[str]:
    cmd = [
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
        "--run-tag",
        _run_tag(execution_price, spec),
    ]
    if spec.hard_exit_score_threshold is not None:
        cmd.extend([
            "--hard-exit-score-threshold",
            str(spec.hard_exit_score_threshold),
            "--hard-exit-min-holding-days",
            str(MIN_HOLDING_DAYS),
        ])
    if spec.tail_cleanup_weight > 0:
        cmd.extend(["--tail-cleanup-weight", str(spec.tail_cleanup_weight)])
    if spec.renormalize_after_exit_cleanup:
        cmd.append("--renormalize-after-exit-cleanup")
    return cmd


def _load_comparison(run_dir: Path, execution_price: str, spec: ExitSpec) -> pd.DataFrame:
    path = run_dir / "comparison.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["execution_price"] = execution_price
    df["exit_spec"] = spec.label
    df["hard_exit_score_threshold"] = spec.hard_exit_score_threshold
    df["tail_cleanup_weight"] = spec.tail_cleanup_weight
    df["renormalize_after_exit_cleanup"] = spec.renormalize_after_exit_cleanup
    df["run_dir"] = str(run_dir)
    return df


def _run_or_load(execution_price: str, spec: ExitSpec, *, force: bool = False) -> tuple[pd.DataFrame, Path]:
    run_dir = _expected_run_dir(execution_price, spec)
    if not force and (run_dir / "comparison.csv").exists():
        print(f"[skip] {execution_price} {spec.label}: {run_dir}")
        return _load_comparison(run_dir, execution_price, spec), run_dir

    if spec.is_baseline:
        raise FileNotFoundError(f"baseline run missing: {run_dir}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_log = LOG_DIR / f"{execution_price}_{spec.label}.out.log"
    err_log = LOG_DIR / f"{execution_price}_{spec.label}.err.log"
    print(f"[run] {execution_price} {spec.label}")
    with out_log.open("w", encoding="utf-8") as out, err_log.open("w", encoding="utf-8") as err:
        completed = subprocess.run(_command(execution_price, spec), stdout=out, stderr=err, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"run failed: {execution_price} {spec.label}; see {out_log} / {err_log}")
    return _load_comparison(run_dir, execution_price, spec), run_dir


def _score(matrix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (execution, spec_label), group in matrix.groupby(["execution_price", "exit_spec"]):
        metrics = group.set_index("strategy")
        if "scheduled_20" not in metrics.index:
            continue
        sched = metrics.loc["scheduled_20"]
        none = metrics.loc["none"] if "none" in metrics.index else None
        bench = metrics.loc["ew_buy_hold_universe"] if "ew_buy_hold_universe" in metrics.index else None
        none_sharpe = float(none["sharpe"]) if none is not None else 0.0
        none_ret = float(none["cumulative_return_pct"]) if none is not None else 0.0
        bench_sharpe = float(bench["sharpe"]) if bench is not None else 0.0
        bench_ret = float(bench["cumulative_return_pct"]) if bench is not None else 0.0
        bench_dd = float(bench["max_drawdown_pct"]) if bench is not None else 0.0
        dd_penalty = max(0.0, abs(float(sched["max_drawdown_pct"])) - abs(bench_dd))
        score = (
            2.0 * (float(sched["sharpe"]) - bench_sharpe)
            + 0.03 * (float(sched["cumulative_return_pct"]) - bench_ret)
            + 1.0 * (float(sched["sharpe"]) - none_sharpe)
            + 0.01 * (float(sched["cumulative_return_pct"]) - none_ret)
            - 0.01 * dd_penalty
        )
        rows.append({
            "execution_price": execution,
            "exit_spec": spec_label,
            "hard_exit_score_threshold": sched.get("hard_exit_score_threshold"),
            "tail_cleanup_weight": float(sched.get("tail_cleanup_weight", 0.0)),
            "run_dir": str(sched["run_dir"]),
            "scheduled_cum_ret_pct": float(sched["cumulative_return_pct"]),
            "scheduled_sharpe": float(sched["sharpe"]),
            "scheduled_max_dd_pct": float(sched["max_drawdown_pct"]),
            "scheduled_avg_turnover": float(sched["avg_turnover"]),
            "scheduled_avg_cost_bps": float(sched["avg_total_cost_bps"]),
            "scheduled_avg_hard_exit_count": float(sched.get("avg_hard_exit_count", 0.0)),
            "scheduled_avg_hard_exit_weight": float(sched.get("avg_hard_exit_weight", 0.0)),
            "scheduled_avg_tail_exit_count": float(sched.get("avg_tail_exit_count", 0.0)),
            "scheduled_avg_tail_exit_weight": float(sched.get("avg_tail_exit_weight", 0.0)),
            "scheduled_avg_exit_cleanup_weight": float(sched.get("avg_exit_cleanup_weight", 0.0)),
            "scheduled_avg_negative_score_weight_after": float(
                sched.get("avg_negative_score_weight_after", 0.0)
            ),
            "excess_cum_vs_benchmark": float(sched["cumulative_return_pct"]) - bench_ret,
            "excess_sharpe_vs_benchmark": float(sched["sharpe"]) - bench_sharpe,
            "excess_cum_vs_none": float(sched["cumulative_return_pct"]) - none_ret,
            "excess_sharpe_vs_none": float(sched["sharpe"]) - none_sharpe,
            "score": score,
        })
    return pd.DataFrame(rows).sort_values(["score", "scheduled_sharpe"], ascending=False)


def _daily_returns(run_dir: Path, strategy: str) -> pd.Series:
    if strategy == "ew_buy_hold_universe":
        path = run_dir / "benchmarks" / "ew_buy_hold_universe_daily_pnl.csv"
    else:
        pattern = "*sched20_scheduled_20" if strategy == "scheduled_20" else "*none_none"
        matches = list((run_dir / "simulations").glob(f"{pattern}/daily_pnl.csv"))
        if not matches:
            raise FileNotFoundError(f"missing daily_pnl for {strategy} in {run_dir}")
        path = matches[0]
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("date")["net_return"].astype(float)


def _paired_t(diff: pd.Series) -> tuple[float, float]:
    diff = diff.dropna()
    if len(diff) < 3:
        return math.nan, math.nan
    mean = float(diff.mean())
    std = float(diff.std(ddof=1))
    if std <= 0:
        return math.inf, 0.0
    t_stat = mean / (std / math.sqrt(len(diff)))
    try:
        from scipy import stats

        p_one_sided = float(stats.t.sf(t_stat, df=len(diff) - 1))
    except Exception:
        p_one_sided = math.nan
    return float(t_stat), p_one_sided


def _block_bootstrap(diff: pd.Series, *, block_len: int = 20, n_boot: int = 3000) -> dict:
    values = diff.dropna().to_numpy(dtype=float)
    n = len(values)
    if n == 0:
        return {}
    rng = np.random.default_rng(42)
    boot_means = np.empty(n_boot, dtype=float)
    n_blocks = math.ceil(n / block_len)
    for i in range(n_boot):
        chunks = []
        starts = rng.integers(0, n, size=n_blocks)
        for start in starts:
            idx = np.arange(start, start + block_len) % n
            chunks.append(values[idx])
        boot_means[i] = np.concatenate(chunks)[:n].mean()
    return {
        "mean_daily_excess": float(values.mean()),
        "mean_daily_excess_bps": float(values.mean() * 1e4),
        "ci_low_bps": float(np.quantile(boot_means, 0.025) * 1e4),
        "ci_high_bps": float(np.quantile(boot_means, 0.975) * 1e4),
        "p_one_sided_boot": float((boot_means <= 0).mean()),
        "block_len": block_len,
        "n_boot": n_boot,
        "n_days": int(n),
    }


def _bootstrap_for_run(run_dir: Path, execution_price: str, exit_spec: str, *, n_boot: int) -> pd.DataFrame:
    scheduled = _daily_returns(run_dir, "scheduled_20")
    rows: list[dict] = []
    for comparator in ["none", "ew_buy_hold_universe"]:
        other = _daily_returns(run_dir, comparator)
        common = scheduled.index.intersection(other.index)
        diff = scheduled.loc[common] - other.loc[common]
        t_stat, p_t = _paired_t(diff)
        rows.append({
            "execution_price": execution_price,
            "exit_spec": exit_spec,
            "run_dir": str(run_dir),
            "comparison": f"scheduled_20_vs_{comparator}",
            "paired_t_stat": t_stat,
            "paired_t_p_one_sided": p_t,
            **_block_bootstrap(diff, n_boot=n_boot),
        })
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    def fmt(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.3f}".rstrip("0").rstrip(".")
        return str(value)

    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def _write_summary(scored: pd.DataFrame, confirm: pd.DataFrame, boot: pd.DataFrame, path: Path) -> None:
    cols = [
        "exit_spec",
        "scheduled_cum_ret_pct",
        "scheduled_sharpe",
        "scheduled_max_dd_pct",
        "scheduled_avg_turnover",
        "scheduled_avg_cost_bps",
        "scheduled_avg_hard_exit_count",
        "scheduled_avg_tail_exit_count",
        "scheduled_avg_negative_score_weight_after",
        "excess_cum_vs_benchmark",
        "excess_sharpe_vs_benchmark",
        "score",
    ]
    lines = [
        "# Exit discipline OOS workflow",
        "",
        f"- Period: {START} -> {END}",
        f"- Portfolio: entry={ENTRY_RANK}, exit={EXIT_RANK}, max_turnover={MAX_TURNOVER}, min_holding_days={MIN_HOLDING_DAYS}",
        "- Alpha universe: TEJ effective alphas, exclude indclass/cap placeholders",
        "- Baseline source: turnover_matrix_oos_20260512 best combo",
        "",
        "## next_vwap ranking",
        "",
        _markdown_table(scored[scored["execution_price"] == "next_vwap"][cols]),
        "",
        "## next_open confirmation",
        "",
    ]
    if confirm.empty:
        lines.append("尚未執行 next_open confirmation。")
    else:
        lines.append(_markdown_table(confirm[cols]))
    lines.extend(["", "## Bootstrap / paired test", ""])
    if boot.empty:
        lines.append("尚未執行 bootstrap。")
    else:
        boot_cols = [
            "execution_price",
            "exit_spec",
            "comparison",
            "mean_daily_excess_bps",
            "ci_low_bps",
            "ci_high_bps",
            "p_one_sided_boot",
            "paired_t_p_one_sided",
        ]
        lines.append(_markdown_table(boot[boot_cols]))
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-top-n", type=int, default=2)
    parser.add_argument("--n-boot", type=int, default=3000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    raw_frames: list[pd.DataFrame] = []
    for spec in _specs():
        df, _ = _run_or_load("next_vwap", spec, force=args.force and not spec.is_baseline)
        raw_frames.append(df)
        pd.concat(raw_frames, ignore_index=True).to_csv(
            WORKFLOW_DIR / "next_vwap_raw_progress.csv",
            index=False,
            encoding="utf-8-sig",
        )

    raw = pd.concat(raw_frames, ignore_index=True)
    raw.to_csv(WORKFLOW_DIR / "next_vwap_raw.csv", index=False, encoding="utf-8-sig")
    scored = _score(raw)
    scored.to_csv(WORKFLOW_DIR / "next_vwap_ranked.csv", index=False, encoding="utf-8-sig")

    confirm_frames: list[pd.DataFrame] = []
    selected = scored[scored["execution_price"] == "next_vwap"].head(args.confirm_top_n)
    label_to_spec = {spec.label: spec for spec in _specs()}
    for _, row in selected.iterrows():
        spec = label_to_spec[str(row["exit_spec"])]
        df, _ = _run_or_load("next_open", spec, force=args.force and not spec.is_baseline)
        confirm_frames.append(df)

    confirm_raw = pd.concat(confirm_frames, ignore_index=True) if confirm_frames else pd.DataFrame()
    confirm_raw.to_csv(WORKFLOW_DIR / "next_open_confirm_raw.csv", index=False, encoding="utf-8-sig")
    confirm_scored = _score(confirm_raw) if not confirm_raw.empty else pd.DataFrame()
    confirm_scored.to_csv(WORKFLOW_DIR / "next_open_confirm_ranked.csv", index=False, encoding="utf-8-sig")

    boot_frames: list[pd.DataFrame] = []
    for _, row in selected.iterrows():
        boot_frames.append(
            _bootstrap_for_run(
                Path(str(row["run_dir"])),
                "next_vwap",
                str(row["exit_spec"]),
                n_boot=args.n_boot,
            )
        )
    for _, row in confirm_scored.iterrows():
        boot_frames.append(
            _bootstrap_for_run(
                Path(str(row["run_dir"])),
                "next_open",
                str(row["exit_spec"]),
                n_boot=args.n_boot,
            )
        )
    boot = pd.concat(boot_frames, ignore_index=True) if boot_frames else pd.DataFrame()
    boot.to_csv(WORKFLOW_DIR / "bootstrap_paired_results.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "start": START,
        "end": END,
        "entry_rank": ENTRY_RANK,
        "exit_rank": EXIT_RANK,
        "max_turnover": MAX_TURNOVER,
        "min_holding_days": MIN_HOLDING_DAYS,
        "confirm_top_n": args.confirm_top_n,
        "n_boot": args.n_boot,
        "specs": [asdict(spec) for spec in _specs()],
    }
    (WORKFLOW_DIR / "workflow_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_summary(scored, confirm_scored, boot, WORKFLOW_DIR / "workflow_summary.md")
    print(f"workflow complete: {WORKFLOW_DIR}")


if __name__ == "__main__":
    main()
