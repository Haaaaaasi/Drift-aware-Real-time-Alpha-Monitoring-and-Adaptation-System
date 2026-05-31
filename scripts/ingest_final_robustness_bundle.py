"""將 final robustness bundle 匯入 PostgreSQL，供 Grafana 查詢。

這支腳本只讀取已產出的 `manifest.json` 與 `grafana_tables.json`，不重跑任何
實驗。寫入採 bundle upsert + child rows delete/insert，讓同一份 bundle 可以安全
重複匯入。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from psycopg2.extras import Json, execute_values

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class IngestOptions:
    report_dir: Path
    is_official: bool = True
    status: str = "completed"
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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved)


def _as_date(value: Any) -> Any:
    value = _clean_scalar(value)
    if value in (None, ""):
        return None
    return pd.to_datetime(value).date()


def _file_time(path: Path) -> datetime | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone()


def _get(row: dict[str, Any], name: str, default: Any = None) -> Any:
    return _clean_scalar(row.get(name, default))


def _row_with_bundle(bundle_id: str, row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["bundle_id"] = bundle_id
    return out


def build_payload(options: IngestOptions) -> dict[str, Any]:
    report_dir = options.report_dir.resolve()
    if not report_dir.exists():
        raise FileNotFoundError(f"report_dir 不存在：{report_dir}")
    if options.status not in {"completed", "failed", "partial"}:
        raise ValueError("--status 必須為 completed / failed / partial")

    manifest_path = report_dir / "manifest.json"
    tables_path = report_dir / "grafana_tables.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"找不到 manifest.json：{manifest_path}")
    if not tables_path.exists():
        raise FileNotFoundError(f"找不到 grafana_tables.json：{tables_path}")

    manifest = _read_json(manifest_path)
    tables = _read_json(tables_path)
    bundle_cfg = dict(tables.get("bundle", {}))
    bundle_id = str(bundle_cfg.get("bundle_id") or manifest.get("package") or report_dir.name)

    validation = manifest.get("validation_period", {})
    bundle = {
        "bundle_id": bundle_id,
        "title": bundle_cfg.get("title") or bundle_id,
        "is_official": bool(options.is_official),
        "status": options.status,
        "validation_start": _as_date(bundle_cfg.get("validation_start") or validation.get("start")),
        "validation_end": _as_date(bundle_cfg.get("validation_end") or validation.get("end")),
        "validation_status": bundle_cfg.get("validation_status") or validation.get("status") or "unknown",
        "frozen_selector_id": bundle_cfg.get("frozen_selector_id") or manifest.get("frozen_selector_id") or "unknown",
        "official_selector": bundle_cfg.get("official_selector") or manifest.get("decisions", {}).get("official_alpha_selector") or "unknown",
        "official_adaptation": bundle_cfg.get("official_adaptation") or manifest.get("decisions", {}).get("official_adaptation") or "unknown",
        "primary_execution": bundle_cfg.get("primary_execution") or manifest.get("primary_execution") or "next_vwap",
        "secondary_execution": bundle_cfg.get("secondary_execution") or manifest.get("secondary_execution"),
        "summary_report": manifest.get("summary_report") or _display_path(report_dir / "final_robustness_summary.md"),
        "manifest_path": _display_path(manifest_path),
        "frozen_config": manifest.get("frozen_config"),
        "config_json": _clean_json(manifest),
        "created_at": _file_time(manifest_path),
        "notes": bundle_cfg.get("notes"),
    }

    return {
        "bundle": bundle,
        "strategy_results": [_row_with_bundle(bundle_id, row) for row in tables.get("strategy_results", [])],
        "checks": [_row_with_bundle(bundle_id, row) for row in tables.get("checks", [])],
        "regime_results": [_row_with_bundle(bundle_id, row) for row in tables.get("regime_results", [])],
        "decisions": [_row_with_bundle(bundle_id, row) for row in tables.get("decisions", [])],
        "artifacts": [_row_with_bundle(bundle_id, row) for row in tables.get("artifacts", [])],
    }


BUNDLE_COLUMNS = [
    "bundle_id", "title", "is_official", "status", "validation_start",
    "validation_end", "validation_status", "frozen_selector_id",
    "official_selector", "official_adaptation", "primary_execution",
    "secondary_execution", "summary_report", "manifest_path", "frozen_config",
    "config_json", "created_at", "notes",
]

STRATEGY_COLUMNS = [
    "bundle_id", "execution_price", "series", "result_role",
    "is_official_strategy", "is_benchmark", "cumulative_return_pct", "sharpe",
    "max_drawdown_pct", "avg_turnover", "avg_cost_bps", "n_reuses",
    "n_misses", "sort_order",
]

CHECK_COLUMNS = [
    "bundle_id", "check_type", "execution_price", "metric", "comparison",
    "real_value", "reference_value", "p_value", "percentile", "ci05", "ci95",
    "n_samples", "passed", "sort_order",
]

REGIME_COLUMNS = [
    "bundle_id", "execution_price", "regime", "cumulative_return_pct",
    "sharpe", "max_drawdown_pct", "sort_order",
]

DECISION_COLUMNS = [
    "bundle_id", "topic", "decision", "severity", "evidence", "sort_order",
]

ARTIFACT_COLUMNS = [
    "bundle_id", "artifact_type", "label", "path", "sort_order",
]


def _tuple_for(row: dict[str, Any], columns: list[str]) -> tuple[Any, ...]:
    out: list[Any] = []
    for column in columns:
        value = _clean_scalar(row.get(column))
        if column == "config_json":
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
            cur.execute(
                f"""
                INSERT INTO final_robustness_bundles ({", ".join(BUNDLE_COLUMNS)})
                VALUES ({", ".join(["%s"] * len(BUNDLE_COLUMNS))})
                ON CONFLICT (bundle_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    is_official = EXCLUDED.is_official,
                    status = EXCLUDED.status,
                    validation_start = EXCLUDED.validation_start,
                    validation_end = EXCLUDED.validation_end,
                    validation_status = EXCLUDED.validation_status,
                    frozen_selector_id = EXCLUDED.frozen_selector_id,
                    official_selector = EXCLUDED.official_selector,
                    official_adaptation = EXCLUDED.official_adaptation,
                    primary_execution = EXCLUDED.primary_execution,
                    secondary_execution = EXCLUDED.secondary_execution,
                    summary_report = EXCLUDED.summary_report,
                    manifest_path = EXCLUDED.manifest_path,
                    frozen_config = EXCLUDED.frozen_config,
                    config_json = EXCLUDED.config_json,
                    created_at = EXCLUDED.created_at,
                    ingested_at = now(),
                    notes = EXCLUDED.notes
                """,
                _tuple_for(payload["bundle"], BUNDLE_COLUMNS),
            )
            bundle_id = payload["bundle"]["bundle_id"]
            for table in [
                "final_robustness_artifacts",
                "final_robustness_decisions",
                "final_robustness_regime_results",
                "final_robustness_checks",
                "final_robustness_strategy_results",
            ]:
                cur.execute(f"DELETE FROM {table} WHERE bundle_id = %s", (bundle_id,))
            inserts = [
                ("final_robustness_strategy_results", STRATEGY_COLUMNS, payload["strategy_results"]),
                ("final_robustness_checks", CHECK_COLUMNS, payload["checks"]),
                ("final_robustness_regime_results", REGIME_COLUMNS, payload["regime_results"]),
                ("final_robustness_decisions", DECISION_COLUMNS, payload["decisions"]),
                ("final_robustness_artifacts", ARTIFACT_COLUMNS, payload["artifacts"]),
            ]
            for table, columns, rows in inserts:
                if rows:
                    execute_values(
                        cur,
                        f"INSERT INTO {table} ({', '.join(columns)}) VALUES %s",
                        [_tuple_for(row, columns) for row in rows],
                        page_size=1000,
                    )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        if own_conn:
            conn.close()


def dry_run_summary(payload: dict[str, Any]) -> str:
    bundle = payload["bundle"]
    return "\n".join([
        f"Bundle: {bundle['bundle_id']} ({bundle['status']}, official={bundle['is_official']})",
        f"Period: {bundle['validation_start']} -> {bundle['validation_end']} ({bundle['validation_status']})",
        f"Strategy rows: {len(payload['strategy_results'])}",
        f"Check rows: {len(payload['checks'])}",
        f"Regime rows: {len(payload['regime_results'])}",
        f"Decision rows: {len(payload['decisions'])}",
        f"Artifact rows: {len(payload['artifacts'])}",
    ])


def parse_args(argv: list[str] | None = None) -> IngestOptions:
    parser = argparse.ArgumentParser(description="Ingest final robustness bundle into PostgreSQL")
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--status", default="completed", choices=["completed", "failed", "partial"])
    parser.add_argument("--not-official", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return IngestOptions(
        report_dir=args.report_dir,
        is_official=not args.not_official,
        status=args.status,
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
    print("Final robustness ingest complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
