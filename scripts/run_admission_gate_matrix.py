"""執行 all_valid_82 + admission gate OOS 小矩陣。

這個 workflow 只負責跑 selector admission gate 的候選設定，並把已完成 run
彙整成 matrix summary。若中途停止，重新執行會跳過已有 ``daily_pnl.csv`` 的 run。
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BASE_OUT_DIR = Path("reports/adaptation_ab")
WORKFLOW_DIR = BASE_OUT_DIR / "admission_gate_matrix_20260517"
LOG_DIR = WORKFLOW_DIR / "logs"

START = "2024-07-01"
END = "2026-04-30"

PROMOTED_GRID = [2, 4]
MIN_SCORE_GRID = [0.02, 0.03, 0.05]
MAX_CORR_GRID = [0.95, 0.98]


@dataclass(frozen=True)
class AdmissionSpec:
    max_promoted: int
    min_score: float
    max_corr: float
    execution_price: str

    @property
    def tag(self) -> str:
        score = str(self.min_score).replace(".", "p")
        corr = str(self.max_corr).replace(".", "p")
        suffix = "nextopen" if self.execution_price == "next_open" else "nextvwap"
        return f"adgate_p{self.max_promoted}_s{score}_c{corr}_{suffix}"

    @property
    def run_dir(self) -> Path:
        return WORKFLOW_DIR / f"sim_20240701_20260430_top10_sched20_{self.tag}"


def _summary_from_daily_pnl(path: Path) -> dict[str, float | int]:
    pnl = pd.read_csv(path)
    returns = pnl["net_return"].astype(float)
    n_days = int(len(returns))
    cumulative = float(np.prod(1.0 + returns) - 1.0)
    annualized = float((1.0 + cumulative) ** (252.0 / n_days) - 1.0) if n_days else np.nan
    std = float(returns.std(ddof=1))
    sharpe = 0.0 if std == 0.0 else float(returns.mean() / std * math.sqrt(252.0))
    equity = (1.0 + returns).cumprod()
    max_drawdown = float((equity / equity.cummax() - 1.0).min())
    out: dict[str, float | int] = {
        "n_days": n_days,
        "cumulative_return_pct": cumulative * 100.0,
        "annualized_return_pct": annualized * 100.0,
        "sharpe": sharpe,
        "max_drawdown_pct": max_drawdown * 100.0,
        "avg_net_return_bps": float(returns.mean() * 10_000.0),
    }
    for col in ["turnover", "n_holdings"]:
        if col in pnl.columns:
            out[f"avg_{col}"] = float(pnl[col].mean())
    return out


def _admission_counts(run_dir: Path) -> dict[str, float | int | None]:
    scores_path = run_dir / "alpha_scores.csv"
    if not scores_path.exists():
        return {
            "avg_admitted_count": None,
            "avg_quarantine_selected_count": None,
            "avg_quarantine_selected_weight": None,
        }
    scores = pd.read_csv(scores_path)
    admitted = (
        scores.assign(is_admitted=scores["admission_status"].eq("admitted"))
        .groupby("as_of_date")["is_admitted"]
        .sum()
    )
    selected_quarantine = scores[
        scores["selected"].astype(bool) & scores["alpha_pool"].eq("quarantine")
    ]
    if selected_quarantine.empty:
        selected_count = pd.Series(dtype=float)
        selected_weight = pd.Series(dtype=float)
    else:
        selected_count = selected_quarantine.groupby("as_of_date")["alpha_id"].count()
        selected_weight = selected_quarantine.groupby("as_of_date")["weight"].sum()
    return {
        "avg_admitted_count": float(admitted.mean()) if not admitted.empty else 0.0,
        "avg_quarantine_selected_count": (
            float(selected_count.mean()) if not selected_count.empty else 0.0
        ),
        "avg_quarantine_selected_weight": (
            float(selected_weight.mean()) if not selected_weight.empty else 0.0
        ),
    }


def _command(spec: AdmissionSpec) -> list[str]:
    return [
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
        "--skip-effective-filter",
        "--exclude-indclass-cap-alphas",
        "--selector-admission-gate",
        "--selector-alpha-top-k",
        "20",
        "--selector-window-days",
        "126",
        "--selector-min-coverage",
        "0.2",
        "--selector-min-observations",
        "1000",
        "--selector-stability-penalty",
        "0.10",
        "--admission-max-promoted",
        str(spec.max_promoted),
        "--admission-min-score",
        str(spec.min_score),
        "--admission-min-coverage",
        "0.2",
        "--admission-min-observations",
        "1000",
        "--admission-subwindows",
        "3",
        "--admission-min-subwindow-passes",
        "2",
        "--admission-subwindow-min-abs-ic",
        "0.01",
        "--admission-max-abs-corr-to-live",
        str(spec.max_corr),
        "--retrain-every",
        "20",
        "--top-k",
        "10",
        "--portfolio-method",
        "turnover_aware_topk",
        "--rebalance-every",
        "10",
        "--entry-rank",
        "20",
        "--exit-rank",
        "60",
        "--max-turnover",
        "0.25",
        "--min-holding-days",
        "10",
        "--tail-cleanup-weight",
        "0.0025",
        "--objective",
        "net_return_proxy",
        "--execution-price",
        spec.execution_price,
        "--out-dir",
        str(WORKFLOW_DIR),
        "--run-tag",
        spec.tag,
    ]


def _write_summary(specs: list[AdmissionSpec]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        daily = spec.run_dir / "daily_pnl.csv"
        if not daily.exists():
            rows.append(
                {
                    "execution_price": spec.execution_price,
                    "max_promoted": spec.max_promoted,
                    "min_score": spec.min_score,
                    "max_corr": spec.max_corr,
                    "status": "pending",
                    "run_dir": str(spec.run_dir),
                }
            )
            continue
        rows.append(
            {
                "execution_price": spec.execution_price,
                "max_promoted": spec.max_promoted,
                "min_score": spec.min_score,
                "max_corr": spec.max_corr,
                "status": "done",
                **_summary_from_daily_pnl(daily),
                **_admission_counts(spec.run_dir),
                "run_dir": str(spec.run_dir),
            }
        )
    out = pd.DataFrame(rows).sort_values(
        ["execution_price", "status", "sharpe", "cumulative_return_pct"],
        ascending=[True, True, False, False],
        na_position="last",
    )
    out.to_csv(WORKFLOW_DIR / "matrix_summary.csv", index=False, encoding="utf-8-sig")
    return out


def _write_manifest(specs: list[AdmissionSpec], *, executions: list[str]) -> None:
    payload = {
        "start": START,
        "end": END,
        "workflow_dir": str(WORKFLOW_DIR),
        "executions": executions,
        "promoted_grid": PROMOTED_GRID,
        "min_score_grid": MIN_SCORE_GRID,
        "max_corr_grid": MAX_CORR_GRID,
        "n_specs": len(specs),
    }
    (WORKFLOW_DIR / "workflow_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _all_specs(executions: list[str]) -> list[AdmissionSpec]:
    return [
        AdmissionSpec(max_promoted=p, min_score=s, max_corr=c, execution_price=execution)
        for execution in executions
        for p in PROMOTED_GRID
        for s in MIN_SCORE_GRID
        for c in MAX_CORR_GRID
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executions",
        nargs="+",
        choices=["next_vwap", "next_open"],
        default=["next_vwap"],
    )
    parser.add_argument("--max-runs", type=int, default=None, help="最多啟動幾個尚未完成的 run。")
    parser.add_argument("--force", action="store_true", help="忽略既有 daily_pnl.csv 重新跑。")
    parser.add_argument("--summary-only", action="store_true", help="只重建 matrix_summary.csv。")
    args = parser.parse_args()

    WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    specs = _all_specs(args.executions)
    _write_manifest(specs, executions=args.executions)
    if args.summary_only:
        summary = _write_summary(specs)
        print(summary.to_string(index=False))
        return

    launched = 0
    progress_rows = []
    for spec in specs:
        if not args.force and (spec.run_dir / "daily_pnl.csv").exists():
            print(f"[skip] {spec.tag}")
            progress_rows.append({"tag": spec.tag, "status": "skipped_existing"})
            continue
        if args.max_runs is not None and launched >= args.max_runs:
            progress_rows.append({"tag": spec.tag, "status": "pending_max_runs"})
            continue

        out_log = LOG_DIR / f"{spec.tag}.out.log"
        err_log = LOG_DIR / f"{spec.tag}.err.log"
        cmd = _command(spec)
        print(f"[run] {spec.tag}")
        with out_log.open("w", encoding="utf-8") as out, err_log.open("w", encoding="utf-8") as err:
            completed = subprocess.run(cmd, stdout=out, stderr=err, text=True)
        progress_rows.append({"tag": spec.tag, "status": "done" if completed.returncode == 0 else "failed"})
        pd.DataFrame(progress_rows).to_csv(WORKFLOW_DIR / "matrix_progress.csv", index=False)
        _write_summary(specs)
        launched += 1
        if completed.returncode != 0:
            raise RuntimeError(f"{spec.tag} failed; see {out_log} / {err_log}")

    summary = _write_summary(specs)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
