"""過夜執行 turnover-aware OOS matrix 與後續統計檢定。

流程：
1. 跑 16 組 next_vwap turnover-aware matrix。
2. 依 scheduled_20 相對 benchmark / none 的表現排序。
3. 挑前 N 組跑 next_open 複驗。
4. 對 selected runs 做 paired t-test 與 circular block bootstrap。
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BASE_OUT_DIR = Path("reports/adaptation_ab")
WORKFLOW_DIR = BASE_OUT_DIR / "turnover_matrix_oos_20260512"
LOG_DIR = WORKFLOW_DIR / "logs"

START = "2024-07-01"
END = "2026-04-30"


@dataclass(frozen=True)
class Combo:
    entry_rank: int
    exit_rank: int
    max_turnover: float
    min_holding_days: int

    @property
    def label(self) -> str:
        turnover = str(self.max_turnover).replace(".", "p")
        return f"e{self.entry_rank}_x{self.exit_rank}_t{turnover}_h{self.min_holding_days}"


def _all_combos() -> list[Combo]:
    return [
        Combo(entry, exit_, turnover, hold)
        for entry, exit_, turnover, hold in itertools.product(
            [20, 30],
            [40, 60],
            [0.25, 0.50],
            [5, 10],
        )
    ]


def _seed_run_dir(execution_price: str, combo: Combo) -> Path | None:
    if combo != Combo(20, 40, 0.25, 5):
        return None
    suffix = "nextopen" if execution_price == "next_open" else "nextvwap"
    path = BASE_OUT_DIR / (
        "ab_20240701_20260430_top10_oos_no_indcap_"
        f"turnoveraware_e20x40_t25_h5_{suffix}_20260511"
    )
    return path if (path / "comparison.csv").exists() else None


def _run_tag(execution_price: str, combo: Combo) -> str:
    suffix = "nextopen" if execution_price == "next_open" else "nextvwap"
    return f"oos_no_indcap_turnovermatrix_{combo.label}_{suffix}_20260512"


def _expected_run_dir(execution_price: str, combo: Combo) -> Path:
    return BASE_OUT_DIR / f"ab_20240701_20260430_top10_{_run_tag(execution_price, combo)}"


def _command(execution_price: str, combo: Combo) -> list[str]:
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
        str(combo.entry_rank),
        "--exit-rank",
        str(combo.exit_rank),
        "--max-turnover",
        str(combo.max_turnover),
        "--min-holding-days",
        str(combo.min_holding_days),
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
        _run_tag(execution_price, combo),
    ]


def _load_comparison(run_dir: Path, execution_price: str, combo: Combo) -> pd.DataFrame:
    path = run_dir / "comparison.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["execution_price"] = execution_price
    df["run_dir"] = str(run_dir)
    for key, value in asdict(combo).items():
        df[key] = value
    df["combo_label"] = combo.label
    return df


def _run_or_load(execution_price: str, combo: Combo, *, force: bool = False) -> tuple[pd.DataFrame, Path]:
    seed = None if force else _seed_run_dir(execution_price, combo)
    run_dir = seed or _expected_run_dir(execution_price, combo)
    if not force and (run_dir / "comparison.csv").exists():
        print(f"[skip] {execution_price} {combo.label}: {run_dir}")
        return _load_comparison(run_dir, execution_price, combo), run_dir

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_log = LOG_DIR / f"{execution_price}_{combo.label}.out.log"
    err_log = LOG_DIR / f"{execution_price}_{combo.label}.err.log"
    cmd = _command(execution_price, combo)
    print(f"[run] {execution_price} {combo.label}")
    with out_log.open("w", encoding="utf-8") as out, err_log.open("w", encoding="utf-8") as err:
        completed = subprocess.run(cmd, stdout=out, stderr=err, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"run failed: {execution_price} {combo.label}; see {out_log} / {err_log}"
        )
    return _load_comparison(run_dir, execution_price, combo), run_dir


def _score_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (execution, combo_label), group in matrix.groupby(["execution_price", "combo_label"]):
        metrics = group.set_index("strategy")
        if "scheduled_20" not in metrics.index:
            continue
        sched = metrics.loc["scheduled_20"]
        none = metrics.loc["none"] if "none" in metrics.index else None
        bench = metrics.loc["ew_buy_hold_universe"] if "ew_buy_hold_universe" in metrics.index else None
        bench_sharpe = float(bench["sharpe"]) if bench is not None else 0.0
        bench_ret = float(bench["cumulative_return_pct"]) if bench is not None else 0.0
        none_sharpe = float(none["sharpe"]) if none is not None else 0.0
        none_ret = float(none["cumulative_return_pct"]) if none is not None else 0.0
        dd_penalty = max(0.0, abs(float(sched["max_drawdown_pct"])) - abs(float(bench["max_drawdown_pct"]))) if bench is not None else 0.0
        score = (
            2.0 * (float(sched["sharpe"]) - bench_sharpe)
            + 0.03 * (float(sched["cumulative_return_pct"]) - bench_ret)
            + 1.0 * (float(sched["sharpe"]) - none_sharpe)
            + 0.01 * (float(sched["cumulative_return_pct"]) - none_ret)
            - 0.01 * dd_penalty
        )
        rows.append({
            "execution_price": execution,
            "combo_label": combo_label,
            "entry_rank": int(sched["entry_rank"]),
            "exit_rank": int(sched["exit_rank"]),
            "max_turnover": float(sched["max_turnover"]),
            "min_holding_days": int(sched["min_holding_days"]),
            "run_dir": str(sched["run_dir"]),
            "scheduled_cum_ret_pct": float(sched["cumulative_return_pct"]),
            "scheduled_sharpe": float(sched["sharpe"]),
            "scheduled_max_dd_pct": float(sched["max_drawdown_pct"]),
            "scheduled_avg_turnover": float(sched["avg_turnover"]),
            "scheduled_avg_cost_bps": float(sched["avg_total_cost_bps"]),
            "excess_cum_vs_benchmark": float(sched["cumulative_return_pct"]) - bench_ret,
            "excess_sharpe_vs_benchmark": float(sched["sharpe"]) - bench_sharpe,
            "excess_cum_vs_none": float(sched["cumulative_return_pct"]) - none_ret,
            "excess_sharpe_vs_none": float(sched["sharpe"]) - none_sharpe,
            "score": score,
        })
    scored = pd.DataFrame(rows)
    return scored.sort_values(["score", "scheduled_sharpe"], ascending=False)


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


def _block_bootstrap(
    diff: pd.Series,
    *,
    block_len: int = 20,
    n_boot: int = 5000,
    seed: int = 42,
) -> dict:
    values = diff.dropna().to_numpy(dtype=float)
    n = len(values)
    if n == 0:
        return {}
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=float)
    n_blocks = math.ceil(n / block_len)
    for i in range(n_boot):
        chunks = []
        starts = rng.integers(0, n, size=n_blocks)
        for start in starts:
            idx = (np.arange(start, start + block_len) % n)
            chunks.append(values[idx])
        sample = np.concatenate(chunks)[:n]
        boot_means[i] = sample.mean()
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


def _bootstrap_for_run(
    run_dir: Path,
    execution_price: str,
    combo: Combo,
    *,
    n_boot: int,
) -> pd.DataFrame:
    scheduled = _daily_returns(run_dir, "scheduled_20")
    rows: list[dict] = []
    for comparator in ["none", "ew_buy_hold_universe"]:
        other = _daily_returns(run_dir, comparator)
        common = scheduled.index.intersection(other.index)
        diff = scheduled.loc[common] - other.loc[common]
        t_stat, p_t = _paired_t(diff)
        boot = _block_bootstrap(diff, n_boot=n_boot)
        rows.append({
            "execution_price": execution_price,
            "combo_label": combo.label,
            "run_dir": str(run_dir),
            "comparison": f"scheduled_20_vs_{comparator}",
            "paired_t_stat": t_stat,
            "paired_t_p_one_sided": p_t,
            **boot,
        })
    return pd.DataFrame(rows)


def _write_summary(scored: pd.DataFrame, confirm_scored: pd.DataFrame, boot: pd.DataFrame, path: Path) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        headers = list(frame.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in frame.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
        return "\n".join(lines)

    lines = [
        "# Turnover-aware OOS matrix workflow",
        "",
        f"- Matrix period: {START} → {END}",
        "- Matrix execution: next_vwap",
        "- Confirmation execution: next_open",
        "",
        "## next_vwap 前 10 名",
        "",
    ]
    top_cols = [
        "combo_label",
        "scheduled_cum_ret_pct",
        "scheduled_sharpe",
        "scheduled_max_dd_pct",
        "excess_cum_vs_benchmark",
        "excess_sharpe_vs_benchmark",
        "score",
    ]
    lines.append(markdown_table(scored.head(10)[top_cols]))
    lines.extend(["", "## next_open 複驗", ""])
    if confirm_scored.empty:
        lines.append("尚未產生 next_open 複驗。")
    else:
        lines.append(markdown_table(confirm_scored[top_cols]))
    lines.extend(["", "## Bootstrap / paired test", ""])
    if boot.empty:
        lines.append("尚未產生 bootstrap 結果。")
    else:
        boot_cols = [
            "execution_price",
            "combo_label",
            "comparison",
            "mean_daily_excess_bps",
            "ci_low_bps",
            "ci_high_bps",
            "p_one_sided_boot",
            "paired_t_p_one_sided",
        ]
        lines.append(markdown_table(boot[boot_cols]))
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-boot", type=int, default=5000)
    args = parser.parse_args()

    WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    all_results: list[pd.DataFrame] = []
    for combo in _all_combos():
        df, _ = _run_or_load("next_vwap", combo, force=args.force)
        all_results.append(df)
        pd.concat(all_results, ignore_index=True).to_csv(
            WORKFLOW_DIR / "matrix_raw_progress.csv",
            index=False,
            encoding="utf-8-sig",
        )

    matrix = pd.concat(all_results, ignore_index=True)
    matrix.to_csv(WORKFLOW_DIR / "matrix_raw.csv", index=False, encoding="utf-8-sig")
    scored = _score_matrix(matrix)
    scored.to_csv(WORKFLOW_DIR / "matrix_ranked.csv", index=False, encoding="utf-8-sig")

    selected = scored.head(args.top_n)
    selected.to_csv(WORKFLOW_DIR / "selected_for_next_open.csv", index=False, encoding="utf-8-sig")

    confirm_results: list[pd.DataFrame] = []
    selected_combos: list[Combo] = []
    for _, row in selected.iterrows():
        combo = Combo(
            int(row["entry_rank"]),
            int(row["exit_rank"]),
            float(row["max_turnover"]),
            int(row["min_holding_days"]),
        )
        selected_combos.append(combo)
        df, _ = _run_or_load("next_open", combo, force=args.force)
        confirm_results.append(df)

    confirm = pd.concat(confirm_results, ignore_index=True) if confirm_results else pd.DataFrame()
    confirm.to_csv(WORKFLOW_DIR / "next_open_confirm_raw.csv", index=False, encoding="utf-8-sig")
    confirm_scored = _score_matrix(confirm) if not confirm.empty else pd.DataFrame()
    confirm_scored.to_csv(WORKFLOW_DIR / "next_open_confirm_ranked.csv", index=False, encoding="utf-8-sig")

    boot_frames: list[pd.DataFrame] = []
    for _, row in selected.iterrows():
        combo = Combo(
            int(row["entry_rank"]),
            int(row["exit_rank"]),
            float(row["max_turnover"]),
            int(row["min_holding_days"]),
        )
        vwap_run = Path(str(row["run_dir"]))
        boot_frames.append(_bootstrap_for_run(vwap_run, "next_vwap", combo, n_boot=args.n_boot))
    for _, row in confirm_scored.iterrows():
        combo = Combo(
            int(row["entry_rank"]),
            int(row["exit_rank"]),
            float(row["max_turnover"]),
            int(row["min_holding_days"]),
        )
        open_run = Path(str(row["run_dir"]))
        boot_frames.append(_bootstrap_for_run(open_run, "next_open", combo, n_boot=args.n_boot))

    boot = pd.concat(boot_frames, ignore_index=True) if boot_frames else pd.DataFrame()
    boot.to_csv(WORKFLOW_DIR / "bootstrap_paired_results.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "start": START,
        "end": END,
        "top_n": args.top_n,
        "n_boot": args.n_boot,
        "combos": [asdict(c) for c in _all_combos()],
    }
    (WORKFLOW_DIR / "workflow_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_summary(scored, confirm_scored, boot, WORKFLOW_DIR / "workflow_summary.md")
    print(f"workflow complete: {WORKFLOW_DIR}")


if __name__ == "__main__":
    main()
