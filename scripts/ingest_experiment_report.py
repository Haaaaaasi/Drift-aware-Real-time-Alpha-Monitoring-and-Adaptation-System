"""將既有實驗報告匯入 PostgreSQL experiment reporting tables。

第一版只讀取 reports/ 內已產出的 CSV / JSON / Markdown，不修改任何實驗
pipeline。寫入策略採「run row upsert + child rows delete/insert」，讓同一個
run_id 可以安全重複匯入。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from psycopg2.extras import Json, execute_values

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VALID_RUN_TYPES = {"ab_experiment", "simulate_recent", "selector_matrix"}
VALID_STATUSES = {"completed", "failed", "partial"}


@dataclass
class IngestOptions:
    report_dir: Path
    run_type: str
    run_name: str | None = None
    data_source: str | None = None
    is_official: bool = False
    status: str = "completed"
    notes: str | None = None
    dry_run: bool = False


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if not isinstance(value, (list, tuple, dict)):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, pd.Timestamp):
        return value.date()
    return value


def _clean_json(value: Any) -> Any:
    value = _clean_scalar(value)
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_json(v) for v in value]
    if isinstance(value, tuple):
        return [_clean_json(v) for v in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "isoformat") and value.__class__.__name__ == "date":
        return value.isoformat()
    return value


def _clean_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: _clean_json(value) for key, value in record.items()}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _path_from_text(raw: str | Path) -> Path:
    return Path(str(raw).replace("\\", os.sep))


def _resolve_existing_path(raw: str | Path | None, report_dir: Path) -> Path | None:
    if raw in (None, ""):
        return None
    path = _path_from_text(raw)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([
            PROJECT_ROOT / path,
            report_dir / path,
            report_dir.parent / path,
            Path.cwd() / path,
        ])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve() if candidates else None


def _display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "strategy" not in df.columns and len(df.columns) > 0:
        first_col = df.columns[0]
        if first_col.startswith("Unnamed") or first_col == "":
            df = df.rename(columns={first_col: "strategy"})
    return df


def _first_existing(report_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        path = report_dir / name
        if path.exists():
            return path
    return None


def find_report_path(report_dir: Path) -> Path | None:
    candidates: list[Path] = [
        report_dir / "experiment_summary.md",
        report_dir.with_suffix(".md"),
    ]
    candidates.extend(sorted(report_dir.glob("model_pool_selector_matrix_*.md")))
    candidates.extend(sorted(report_dir.parent.glob("model_pool_selector_matrix_*.md")))
    candidates.extend(sorted(report_dir.glob("model_pool_failure_diagnosis_*.md")))
    candidates.extend(sorted(report_dir.parent.glob("model_pool_failure_diagnosis_*.md")))
    candidates.extend(sorted(report_dir.glob("*.md")))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _file_time_bounds(report_dir: Path) -> tuple[datetime | None, datetime | None]:
    files = [p for p in report_dir.rglob("*") if p.is_file()]
    if not files:
        return None, None
    mtimes = [p.stat().st_mtime for p in files]
    return (
        datetime.fromtimestamp(min(mtimes), tz=timezone.utc),
        datetime.fromtimestamp(max(mtimes), tz=timezone.utc),
    )


def _git_sha() -> str | None:
    try:
        proc = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={PROJECT_ROOT.as_posix()}",
                "rev-parse",
                "HEAD",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return proc.stdout.strip()
    except Exception:
        return None


def _as_date(value: Any) -> Any:
    value = _clean_scalar(value)
    if value is None:
        return None
    return pd.to_datetime(value).date()


def _get(row: pd.Series | dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row:
            value = row[name]
            value = _clean_scalar(value)
            if value is not None:
                return value
    return default


def _to_float(value: Any) -> float | None:
    value = _clean_scalar(value)
    if value is None or value == "":
        return None
    return float(value)


def _to_int(value: Any, default: int = 0) -> int:
    value = _clean_scalar(value)
    if value is None or value == "":
        return default
    return int(value)


def _cost_label(cost_pct: float) -> str:
    return f"cost_{cost_pct:.3f}".rstrip("0").rstrip(".")


def _cost_scenario_name(cost_pct: float) -> str:
    return f"cost_{cost_pct:g}pct"


def _strategy_config(config: dict[str, Any], strategy: str) -> dict[str, Any]:
    strategies = config.get("strategies") or {}
    cfg = strategies.get(strategy) or {}
    return cfg if isinstance(cfg, dict) else {}


def _selection_metric_for(config: dict[str, Any], strategy: str) -> str | None:
    cfg = _strategy_config(config, strategy)
    metric = cfg.get("model_pool_selection_metric")
    if metric is None and cfg.get("strategy") == "model_pool":
        metric = "ic"
    return metric


def _similarity_threshold_for(config: dict[str, Any], strategy: str) -> float | None:
    cfg = _strategy_config(config, strategy)
    return _to_float(cfg.get("similarity_threshold"))


def _strategy_result_from_row(
    *,
    run_id: str,
    row: pd.Series,
    strategy: str,
    variant_name: str = "",
    scenario_name: str = "baseline",
    round_trip_cost_pct: float | None = None,
    is_matrix_cell: bool = False,
    is_benchmark: bool = False,
    selection_metric: str | None = None,
    similarity_threshold: float | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "strategy": strategy,
        "variant_name": variant_name,
        "scenario_name": scenario_name,
        "round_trip_cost_pct": round_trip_cost_pct,
        "is_matrix_cell": is_matrix_cell,
        "is_benchmark": is_benchmark,
        "selection_metric": selection_metric,
        "similarity_threshold": similarity_threshold,
        "cumulative_return_pct": _to_float(_get(row, "cumulative_return_pct", "cum_return_pct")),
        "annualized_return_pct": _to_float(_get(row, "annualized_return_pct")),
        "sharpe": _to_float(_get(row, "sharpe")),
        "max_drawdown_pct": _to_float(_get(row, "max_drawdown_pct")),
        "win_rate_pct": _to_float(_get(row, "win_rate_pct")),
        "avg_turnover": _to_float(_get(row, "avg_turnover")),
        "avg_gross_return_bps": _to_float(_get(row, "avg_gross_return_bps", "avg_gross_bps")),
        "avg_total_cost_bps": _to_float(_get(row, "avg_total_cost_bps")),
        "avg_net_return_bps": _to_float(_get(row, "avg_net_return_bps", "avg_net_bps")),
        "final_value": _to_float(_get(row, "final_value")),
        "n_retrains": _to_int(_get(row, "n_retrains"), 0),
        "n_pool_reuses": _to_int(_get(row, "n_pool_reuses"), 0),
        "n_pool_misses": _to_int(_get(row, "n_pool_misses"), 0),
        "rank_by_net_return": None,
        "n_days": _to_int(_get(row, "n_days"), 0) or None,
        "avg_holdings": _to_float(_get(row, "avg_holdings")),
        "selected_current": _to_int(_get(row, "selected_current"), 0) if "selected_current" in row else None,
        "selected_new": _to_int(_get(row, "selected_new"), 0) if "selected_new" in row else None,
        "selected_reused": _to_int(_get(row, "selected_reused"), 0) if "selected_reused" in row else None,
        "decision_rows": _to_int(_get(row, "decision_rows"), 0) if "decision_rows" in row else None,
        "trigger_events": _to_int(_get(row, "trigger_events"), 0) if "trigger_events" in row else None,
        "avg_selected_shadow_topk_net": _to_float(_get(row, "avg_selected_shadow_topk_net")),
        "avg_selected_proxy_net": _to_float(_get(row, "avg_selected_proxy_net")),
        "selected_proxy_rank_mean": _to_float(_get(row, "selected_proxy_rank_mean")),
    }


def _summarize_pnl(pnl: pd.DataFrame, capital: float | None = None) -> dict[str, Any]:
    if pnl.empty:
        return {}
    net = pd.to_numeric(pnl.get("net_return", pd.Series(dtype=float)), errors="coerce").dropna()
    gross = pd.to_numeric(pnl.get("gross_return", pd.Series(dtype=float)), errors="coerce")
    costs = (
        pd.to_numeric(pnl.get("commission_cost", 0.0), errors="coerce").fillna(0.0)
        + pd.to_numeric(pnl.get("tax_cost", 0.0), errors="coerce").fillna(0.0)
        + pd.to_numeric(pnl.get("slippage_cost", 0.0), errors="coerce").fillna(0.0)
    )
    cumulative = pd.to_numeric(pnl.get("cumulative_value", pd.Series(dtype=float)), errors="coerce").dropna()
    if capital is None:
        if len(cumulative) and len(net):
            first_net = float(net.iloc[0])
            capital = float(cumulative.iloc[0]) / (1.0 + first_net) if abs(1.0 + first_net) > 1e-12 else None
    final_value = float(cumulative.iloc[-1]) if len(cumulative) else None
    cumulative_return_pct = None
    annualized_return_pct = None
    max_drawdown_pct = None
    if capital and final_value:
        cumulative_return_pct = (final_value / capital - 1.0) * 100.0
        if len(pnl) > 0 and final_value > 0:
            annualized_return_pct = ((final_value / capital) ** (252.0 / len(pnl)) - 1.0) * 100.0
        running_max = cumulative.cummax()
        drawdown = cumulative / running_max - 1.0
        max_drawdown_pct = float(drawdown.min() * 100.0) if len(drawdown) else None
    sharpe = None
    if len(net) > 1 and float(net.std(ddof=1)) > 0:
        sharpe = float(net.mean() / net.std(ddof=1) * math.sqrt(252.0))
    return {
        "cumulative_return_pct": cumulative_return_pct,
        "annualized_return_pct": annualized_return_pct,
        "sharpe": sharpe,
        "max_drawdown_pct": max_drawdown_pct,
        "win_rate_pct": float((net > 0).mean() * 100.0) if len(net) else None,
        "avg_turnover": _to_float(pd.to_numeric(pnl.get("turnover", pd.Series(dtype=float)), errors="coerce").mean()),
        "avg_gross_return_bps": _to_float(gross.mean() * 10000.0),
        "avg_total_cost_bps": _to_float(costs.mean() * 10000.0),
        "avg_net_return_bps": _to_float(net.mean() * 10000.0) if len(net) else None,
        "final_value": final_value,
        "n_days": int(len(pnl)),
        "avg_holdings": _to_float(pd.to_numeric(pnl.get("n_holdings", pd.Series(dtype=float)), errors="coerce").mean()),
    }


def _collect_daily_rows(
    *,
    run_id: str,
    daily_path: Path,
    strategy: str,
    variant_name: str = "",
    scenario_name: str = "baseline",
    is_benchmark: bool = False,
) -> list[dict[str, Any]]:
    if not daily_path.exists():
        return []
    pnl = pd.read_csv(daily_path)
    rows: list[dict[str, Any]] = []
    for _, row in pnl.iterrows():
        rows.append({
            "run_id": run_id,
            "strategy": strategy,
            "variant_name": variant_name,
            "scenario_name": scenario_name,
            "is_benchmark": is_benchmark,
            "trade_date": _as_date(_get(row, "date", "trade_date")),
            "gross_return": _to_float(_get(row, "gross_return")),
            "commission_cost": _to_float(_get(row, "commission_cost")),
            "tax_cost": _to_float(_get(row, "tax_cost")),
            "slippage_cost": _to_float(_get(row, "slippage_cost")),
            "net_return": _to_float(_get(row, "net_return")),
            "cumulative_value": _to_float(_get(row, "cumulative_value")),
            "turnover": _to_float(_get(row, "turnover")),
            "n_holdings": _to_int(_get(row, "n_holdings"), 0) if "n_holdings" in row else None,
        })
    return rows


DECISION_COLUMNS = [
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


def _bool_value(value: Any) -> bool | None:
    value = _clean_scalar(value)
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _collect_decision_rows(
    *,
    run_id: str,
    decisions_path: Path,
    strategy: str = "model_pool",
    variant_name: str = "",
    scenario_name: str = "baseline",
) -> list[dict[str, Any]]:
    if not decisions_path.exists():
        return []
    decisions = pd.read_csv(decisions_path)
    rows: list[dict[str, Any]] = []
    for _, row in decisions.iterrows():
        raw = _clean_record(row.to_dict())
        record: dict[str, Any] = {
            "run_id": run_id,
            "strategy": strategy,
            "variant_name": variant_name,
            "scenario_name": scenario_name,
            "date": _as_date(_get(row, "date")),
            "raw_record": raw,
        }
        for column in DECISION_COLUMNS:
            value = _get(row, column)
            if column in {"selected", "pool_hit"}:
                record[column] = _bool_value(value)
            elif column in {
                "day_idx",
                "n_reused_candidates",
                "shadow_n_samples",
                "shadow_topk_n_days",
                "proxy_n_days",
            }:
                record[column] = _to_int(value, 0) if value is not None else None
            elif column.endswith("_id") or column in {
                "candidate_role",
                "selected_role",
                "decision_reason",
                "selection_metric",
            }:
                record[column] = None if value is None else str(value)
            else:
                record[column] = _to_float(value)
        rows.append(record)
    return rows


def _assign_ranks(strategy_rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    for row in strategy_rows:
        row["rank_by_net_return"] = None
        if row.get("is_benchmark"):
            continue
        key = (row.get("scenario_name", "baseline"), bool(row.get("is_matrix_cell")))
        groups.setdefault(key, []).append(row)
    for rows in groups.values():
        ranked = sorted(
            rows,
            key=lambda r: (
                r.get("cumulative_return_pct")
                if r.get("cumulative_return_pct") is not None
                else r.get("avg_net_return_bps", float("-inf"))
            ),
            reverse=True,
        )
        for rank, row in enumerate(ranked, start=1):
            row["rank_by_net_return"] = rank


def _collect_ab_results(
    *,
    run_id: str,
    report_dir: Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    strategy_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []

    comparison_path = _first_existing(report_dir, ["comparison_consolidated.csv", "comparison.csv"])
    if comparison_path is not None:
        comparison = _read_csv(comparison_path)
        benchmark_name = config.get("benchmark")
        for _, row in comparison.iterrows():
            strategy = str(_get(row, "strategy"))
            is_benchmark = strategy == "ew_buy_hold_universe" or strategy == benchmark_name
            strategy_rows.append(_strategy_result_from_row(
                run_id=run_id,
                row=row,
                strategy=strategy,
                is_benchmark=is_benchmark,
                selection_metric=_selection_metric_for(config, strategy),
                similarity_threshold=_similarity_threshold_for(config, strategy),
                round_trip_cost_pct=_to_float(config.get("round_trip_cost_pct")),
            ))

    for strategy, raw_path in (config.get("run_dirs") or {}).items():
        sub_dir = _resolve_existing_path(raw_path, report_dir)
        if sub_dir is None:
            continue
        daily_rows.extend(_collect_daily_rows(
            run_id=run_id,
            daily_path=sub_dir / "daily_pnl.csv",
            strategy=strategy,
        ))
        decision_rows.extend(_collect_decision_rows(
            run_id=run_id,
            decisions_path=sub_dir / "model_pool_decisions.csv",
            strategy="model_pool" if strategy == "model_pool" else strategy,
        ))

    benchmark_path = _resolve_existing_path(config.get("benchmark_path"), report_dir)
    if benchmark_path is not None and benchmark_path.exists():
        daily_rows.extend(_collect_daily_rows(
            run_id=run_id,
            daily_path=benchmark_path,
            strategy="ew_buy_hold_universe",
            is_benchmark=True,
        ))

    return strategy_rows, daily_rows, decision_rows


def _collect_cost_sweep_results(
    *,
    run_id: str,
    report_dir: Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    path = report_dir / "cost_sensitivity.csv"
    if not path.exists():
        return [], [], []
    sweep = pd.read_csv(path)
    strategy_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    sub_runs = config.get("sub_runs") or {}
    for _, row in sweep.iterrows():
        strategy = str(_get(row, "strategy"))
        cost_pct = float(_get(row, "cost_pct", default=0.0))
        scenario_name = _cost_scenario_name(cost_pct)
        label = _cost_label(cost_pct)
        strategy_rows.append(_strategy_result_from_row(
            run_id=run_id,
            row=row,
            strategy=strategy,
            scenario_name=scenario_name,
            round_trip_cost_pct=cost_pct,
            selection_metric=_selection_metric_for(config, strategy),
            similarity_threshold=_similarity_threshold_for(config, strategy),
        ))
        sub_dir = _resolve_existing_path((sub_runs.get(label) or {}).get(strategy), report_dir)
        if sub_dir is not None:
            daily_rows.extend(_collect_daily_rows(
                run_id=run_id,
                daily_path=sub_dir / "daily_pnl.csv",
                strategy=strategy,
                scenario_name=scenario_name,
            ))
            decision_rows.extend(_collect_decision_rows(
                run_id=run_id,
                decisions_path=sub_dir / "model_pool_decisions.csv",
                strategy="model_pool" if strategy == "model_pool" else strategy,
                scenario_name=scenario_name,
            ))
    return strategy_rows, daily_rows, decision_rows


def _collect_selector_matrix_results(
    *,
    run_id: str,
    report_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    path = report_dir / "selector_matrix_summary.csv"
    if not path.exists():
        return [], [], [], []
    summary = pd.read_csv(path)
    strategy_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    child_configs: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        label = str(_get(row, "label"))
        sub_dir = _resolve_existing_path(_get(row, "run_dir"), report_dir)
        strategy_rows.append(_strategy_result_from_row(
            run_id=run_id,
            row=row,
            strategy="model_pool",
            variant_name=label,
            is_matrix_cell=True,
            selection_metric=str(_get(row, "selection_metric")) if _get(row, "selection_metric") else None,
            similarity_threshold=_to_float(_get(row, "similarity_threshold")),
        ))
        if sub_dir is not None:
            child_config = _read_json(sub_dir / "config.json")
            if child_config:
                child_configs.append({"variant_name": label, "config": child_config})
            daily_rows.extend(_collect_daily_rows(
                run_id=run_id,
                daily_path=sub_dir / "daily_pnl.csv",
                strategy="model_pool",
                variant_name=label,
            ))
            decision_rows.extend(_collect_decision_rows(
                run_id=run_id,
                decisions_path=sub_dir / "model_pool_decisions.csv",
                strategy="model_pool",
                variant_name=label,
            ))
    return strategy_rows, daily_rows, decision_rows, child_configs


def _standalone_strategy_name(config: dict[str, Any]) -> str:
    strategy = str(config.get("strategy") or "unknown")
    if strategy == "scheduled":
        return f"scheduled_{config.get('retrain_every', '')}".rstrip("_")
    return strategy


def _collect_simulate_recent_results(
    *,
    run_id: str,
    report_dir: Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    daily_path = report_dir / "daily_pnl.csv"
    if not daily_path.exists():
        return [], [], []
    strategy = _standalone_strategy_name(config)
    pnl = pd.read_csv(daily_path)
    summary = _summarize_pnl(pnl, _to_float(config.get("capital")))
    retrain_path = report_dir / "retrain_log.csv"
    if retrain_path.exists():
        summary["n_retrains"] = len(pd.read_csv(retrain_path))
    row = pd.Series(summary)
    strategy_rows = [_strategy_result_from_row(
        run_id=run_id,
        row=row,
        strategy=strategy,
        selection_metric=_selection_metric_for(config, strategy),
        similarity_threshold=_similarity_threshold_for(config, strategy),
    )]
    daily_rows = _collect_daily_rows(run_id=run_id, daily_path=daily_path, strategy=strategy)
    decision_rows = _collect_decision_rows(
        run_id=run_id,
        decisions_path=report_dir / "model_pool_decisions.csv",
        strategy=strategy,
    )
    return strategy_rows, daily_rows, decision_rows


def _infer_data_source(config: dict[str, Any], child_configs: list[dict[str, Any]]) -> str:
    candidates: list[str] = []
    for cfg in [config] + [c.get("config", {}) for c in child_configs]:
        if "data_source" in cfg:
            candidates.append(str(cfg["data_source"]))
        if "csv_path" in cfg:
            candidates.append(str(cfg["csv_path"]))
        common = cfg.get("common_sim_kwargs")
        if isinstance(common, dict) and "csv_path" in common:
            candidates.append(str(common["csv_path"]))
    joined = " ".join(candidates).lower()
    if "tej" in joined:
        return "tej"
    if "yfinance" in joined or "ohlcv" in joined or ".csv" in joined:
        return "csv"
    return "unknown"


def _date_bounds(config: dict[str, Any], child_configs: list[dict[str, Any]], daily_rows: list[dict[str, Any]]) -> tuple[Any, Any]:
    starts: list[Any] = []
    ends: list[Any] = []
    for cfg in [config] + [c.get("config", {}) for c in child_configs]:
        common = cfg.get("common_sim_kwargs") if isinstance(cfg.get("common_sim_kwargs"), dict) else {}
        start = cfg.get("start") or common.get("start")
        end = cfg.get("end") or common.get("end")
        if start:
            starts.append(_as_date(start))
        if end:
            ends.append(_as_date(end))
    trade_dates = [r["trade_date"] for r in daily_rows if r.get("trade_date")]
    if trade_dates:
        starts.append(min(trade_dates))
        ends.append(max(trade_dates))
    return (min(starts) if starts else None, max(ends) if ends else None)


def build_payload(options: IngestOptions) -> dict[str, Any]:
    report_dir = options.report_dir.resolve()
    if options.run_type not in VALID_RUN_TYPES:
        raise ValueError(f"run_type 必須為 {sorted(VALID_RUN_TYPES)}")
    if options.status not in VALID_STATUSES:
        raise ValueError(f"status 必須為 {sorted(VALID_STATUSES)}")
    if not report_dir.exists():
        raise FileNotFoundError(f"report_dir 不存在：{report_dir}")
    if options.is_official and not options.data_source:
        raise ValueError("--is-official 匯入必須明確指定 --data-source")

    run_id = report_dir.name
    config = _read_json(report_dir / "config.json")
    strategy_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    child_configs: list[dict[str, Any]] = []

    if options.run_type == "selector_matrix" or (report_dir / "selector_matrix_summary.csv").exists():
        s_rows, d_rows, m_rows, configs = _collect_selector_matrix_results(run_id=run_id, report_dir=report_dir)
        strategy_rows.extend(s_rows)
        daily_rows.extend(d_rows)
        decision_rows.extend(m_rows)
        child_configs.extend(configs)

    if options.run_type == "ab_experiment" or _first_existing(report_dir, ["comparison_consolidated.csv", "comparison.csv"]):
        s_rows, d_rows, m_rows = _collect_ab_results(run_id=run_id, report_dir=report_dir, config=config)
        strategy_rows.extend(s_rows)
        daily_rows.extend(d_rows)
        decision_rows.extend(m_rows)

    s_rows, d_rows, m_rows = _collect_cost_sweep_results(run_id=run_id, report_dir=report_dir, config=config)
    strategy_rows.extend(s_rows)
    daily_rows.extend(d_rows)
    decision_rows.extend(m_rows)

    if not strategy_rows:
        s_rows, d_rows, m_rows = _collect_simulate_recent_results(run_id=run_id, report_dir=report_dir, config=config)
        strategy_rows.extend(s_rows)
        daily_rows.extend(d_rows)
        decision_rows.extend(m_rows)

    if not strategy_rows:
        raise RuntimeError(f"找不到可匯入的實驗結果：{report_dir}")

    _assign_ranks(strategy_rows)
    started_at, completed_at = _file_time_bounds(report_dir)
    start_date, end_date = _date_bounds(config, child_configs, daily_rows)
    report_path = find_report_path(report_dir)
    config_json = dict(config)
    if child_configs:
        config_json["child_configs"] = child_configs
    config_json["source_files"] = {
        "comparison": _display_path(_first_existing(report_dir, ["comparison_consolidated.csv", "comparison.csv"])),
        "cost_sensitivity": _display_path(report_dir / "cost_sensitivity.csv") if (report_dir / "cost_sensitivity.csv").exists() else None,
        "selector_matrix_summary": _display_path(report_dir / "selector_matrix_summary.csv") if (report_dir / "selector_matrix_summary.csv").exists() else None,
    }

    data_source = options.data_source or _infer_data_source(config, child_configs)
    run = {
        "run_id": run_id,
        "run_name": options.run_name or config.get("run_id") or run_id,
        "run_type": options.run_type,
        "is_official": bool(options.is_official),
        "status": options.status,
        "started_at": started_at,
        "completed_at": completed_at,
        "data_source": data_source,
        "start_date": start_date,
        "end_date": end_date,
        "config_json": _clean_json(config_json),
        "report_path": _display_path(report_path),
        "source_report_dir": _display_path(report_dir),
        "git_sha": _git_sha(),
        "notes": options.notes,
    }
    return {
        "run": run,
        "strategy_results": strategy_rows,
        "daily_pnl": daily_rows,
        "model_pool_decisions": decision_rows,
    }


RUN_COLUMNS = [
    "run_id", "run_name", "run_type", "is_official", "status", "started_at",
    "completed_at", "data_source", "start_date", "end_date", "config_json",
    "report_path", "source_report_dir", "git_sha", "notes",
]

STRATEGY_COLUMNS = [
    "run_id", "strategy", "variant_name", "scenario_name", "round_trip_cost_pct",
    "is_matrix_cell", "is_benchmark", "selection_metric", "similarity_threshold",
    "cumulative_return_pct", "annualized_return_pct", "sharpe", "max_drawdown_pct",
    "win_rate_pct", "avg_turnover", "avg_gross_return_bps", "avg_total_cost_bps",
    "avg_net_return_bps", "final_value", "n_retrains", "n_pool_reuses",
    "n_pool_misses", "rank_by_net_return", "n_days", "avg_holdings",
    "selected_current", "selected_new", "selected_reused", "decision_rows",
    "trigger_events", "avg_selected_shadow_topk_net", "avg_selected_proxy_net",
    "selected_proxy_rank_mean",
]

DAILY_COLUMNS = [
    "run_id", "strategy", "variant_name", "scenario_name", "is_benchmark",
    "trade_date", "gross_return", "commission_cost", "tax_cost", "slippage_cost",
    "net_return", "cumulative_value", "turnover", "n_holdings",
]

MODEL_POOL_COLUMNS = [
    "run_id", "strategy", "variant_name", "scenario_name", "date", *DECISION_COLUMNS,
    "raw_record",
]


def _tuple_for(row: dict[str, Any], columns: list[str]) -> tuple[Any, ...]:
    out: list[Any] = []
    for column in columns:
        value = row.get(column)
        value = _clean_scalar(value)
        if column in {"config_json", "raw_record"}:
            out.append(Json(_clean_json(value)))
        else:
            out.append(value)
    return tuple(out)


def _connect():
    from src.common.db import get_pg_connection

    return get_pg_connection()


def write_payload(payload: dict[str, Any], conn: Any | None = None) -> None:
    own_conn = conn is None
    conn = conn or _connect()
    try:
        with conn.cursor() as cur:
            run_values = _tuple_for(payload["run"], RUN_COLUMNS)
            cur.execute(
                f"""
                INSERT INTO experiment_runs ({", ".join(RUN_COLUMNS)})
                VALUES ({", ".join(["%s"] * len(RUN_COLUMNS))})
                ON CONFLICT (run_id) DO UPDATE SET
                    run_name = EXCLUDED.run_name,
                    run_type = EXCLUDED.run_type,
                    is_official = EXCLUDED.is_official,
                    status = EXCLUDED.status,
                    started_at = EXCLUDED.started_at,
                    completed_at = EXCLUDED.completed_at,
                    data_source = EXCLUDED.data_source,
                    start_date = EXCLUDED.start_date,
                    end_date = EXCLUDED.end_date,
                    config_json = EXCLUDED.config_json,
                    report_path = EXCLUDED.report_path,
                    source_report_dir = EXCLUDED.source_report_dir,
                    ingested_at = now(),
                    git_sha = EXCLUDED.git_sha,
                    notes = EXCLUDED.notes
                """,
                run_values,
            )
            run_id = payload["run"]["run_id"]
            for table in [
                "experiment_model_pool_decisions",
                "experiment_daily_pnl",
                "experiment_strategy_results",
            ]:
                cur.execute(f"DELETE FROM {table} WHERE run_id = %s", (run_id,))
            if payload["strategy_results"]:
                execute_values(
                    cur,
                    f"INSERT INTO experiment_strategy_results ({', '.join(STRATEGY_COLUMNS)}) VALUES %s",
                    [_tuple_for(row, STRATEGY_COLUMNS) for row in payload["strategy_results"]],
                    page_size=1000,
                )
            if payload["daily_pnl"]:
                execute_values(
                    cur,
                    f"INSERT INTO experiment_daily_pnl ({', '.join(DAILY_COLUMNS)}) VALUES %s",
                    [_tuple_for(row, DAILY_COLUMNS) for row in payload["daily_pnl"]],
                    page_size=5000,
                )
            if payload["model_pool_decisions"]:
                execute_values(
                    cur,
                    f"INSERT INTO experiment_model_pool_decisions ({', '.join(MODEL_POOL_COLUMNS)}) VALUES %s",
                    [_tuple_for(row, MODEL_POOL_COLUMNS) for row in payload["model_pool_decisions"]],
                    page_size=2000,
                )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        if own_conn:
            conn.close()


def dry_run_summary(payload: dict[str, Any]) -> str:
    run = payload["run"]
    official = "official" if run["is_official"] else "non-official"
    return "\n".join([
        f"Run: {run['run_id']} ({run['run_type']}, {official}, status={run['status']})",
        f"Data source: {run['data_source']}",
        f"Report path: {run.get('report_path') or '(none)'}",
        f"Strategy rows: {len(payload['strategy_results'])}",
        f"Daily PnL rows: {len(payload['daily_pnl'])}",
        f"Model pool decision rows: {len(payload['model_pool_decisions'])}",
    ])


def parse_args(argv: list[str] | None = None) -> IngestOptions:
    parser = argparse.ArgumentParser(description="Ingest DARAMS experiment report outputs into PostgreSQL")
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--run-type", required=True, choices=sorted(VALID_RUN_TYPES))
    parser.add_argument("--run-name")
    parser.add_argument("--data-source")
    parser.add_argument("--is-official", action="store_true")
    parser.add_argument("--status", default="completed", choices=sorted(VALID_STATUSES))
    parser.add_argument("--notes")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.is_official and not args.data_source:
        parser.error("--is-official 匯入必須指定 --data-source")
    return IngestOptions(
        report_dir=args.report_dir,
        run_type=args.run_type,
        run_name=args.run_name,
        data_source=args.data_source,
        is_official=args.is_official,
        status=args.status,
        notes=args.notes,
        dry_run=args.dry_run,
    )


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    payload = build_payload(options)
    if options.dry_run:
        print(dry_run_summary(payload))
        return 0
    write_payload(payload)
    print(dry_run_summary(payload))
    print("Ingest complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
