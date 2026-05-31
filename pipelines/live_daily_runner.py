"""Run the daily live operating workflow end to end.

流程：
1. 可選：把每日 TEJ CSV append 到 data/tw_stocks_tej.parquet。
2. 執行 daily_online_pipeline，沿用 production artifact 與前一筆 official holdings。
3. 寫入 live operational tables，供 Grafana / API 顯示。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from pipelines.daily_online_pipeline import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_FROZEN_CONFIG,
    DEFAULT_OUTPUT_DIR,
    run_daily_online,
)
from src.ingestion.tej_daily_append import (
    DEFAULT_BACKUP_DIR,
    DEFAULT_TEJ_OUTPUT,
    DEFAULT_UNIVERSE_OUTPUT,
    append_tej_daily_files,
)


def run_live_daily(
    *,
    tej_input: list[str | Path] | None = None,
    mode: str = "auto",
    as_of: str | None = None,
    frozen_config: str | Path = DEFAULT_FROZEN_CONFIG,
    frozen_execution: str = "primary",
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    production_artifact: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    capital: float = 10_000_000.0,
    force_retrain: bool = False,
    run_purpose: str = "production",
    official: bool = True,
    persist_db: bool = True,
    dry_run_ingest: bool = False,
    skip_online_run: bool = False,
    tej_output: str | Path = DEFAULT_TEJ_OUTPUT,
    universe_output: str | Path = DEFAULT_UNIVERSE_OUTPUT,
    backup_dir: str | Path | None = DEFAULT_BACKUP_DIR,
    encoding: str = "utf-16-le",
    sep: str = "\t",
) -> dict[str, Any]:
    """執行每日 live workflow，回傳 append 與 online run 的摘要。"""

    append_result = None
    resolved_as_of = as_of
    if tej_input:
        append_result = append_tej_daily_files(
            tej_input,
            output_path=tej_output,
            universe_output_path=universe_output,
            backup_dir=backup_dir,
            dry_run=dry_run_ingest,
            encoding=encoding,
            sep=sep,
        )
        if resolved_as_of is None:
            resolved_as_of = append_result.output_max_date

    if dry_run_ingest or skip_online_run:
        return {
            "append": append_result.to_dict() if append_result is not None else None,
            "live_run": None,
            "message": "ingest dry-run completed" if dry_run_ingest else "online run skipped",
        }

    as_of_date = (
        datetime.strptime(resolved_as_of, "%Y-%m-%d").date()
        if resolved_as_of is not None
        else None
    )
    live_result = run_daily_online(
        mode=mode,  # type: ignore[arg-type]
        as_of=as_of_date,
        frozen_config=frozen_config,
        frozen_execution=frozen_execution,
        artifact_root=artifact_root,
        production_artifact=production_artifact,
        output_dir=output_dir,
        capital=capital,
        force_retrain=force_retrain,
        run_purpose=run_purpose,
        is_official=official,
        persist_db=persist_db,
    )
    return {
        "append": append_result.to_dict() if append_result is not None else None,
        "live_run": live_result,
        "message": "daily live workflow completed",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tej-input", nargs="+", help="每日 TEJ OHLCV CSV；省略時只跑現有資料")
    parser.add_argument("--mode", choices=["auto", "predict-only", "train-only"], default="auto")
    parser.add_argument("--as-of", help="指定 live run as-of date；預設使用 append 後最新日期")
    parser.add_argument("--frozen-config", default=str(DEFAULT_FROZEN_CONFIG))
    parser.add_argument("--frozen-execution", default="primary")
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--production-artifact")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--capital", type=float, default=10_000_000.0)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument(
        "--run-purpose",
        choices=["production", "smoke", "backfill"],
        default="production",
    )
    parser.add_argument(
        "--no-official",
        action="store_true",
        help="不要把這次 production run 標成 official",
    )
    parser.add_argument("--no-db", action="store_true", help="不寫 PostgreSQL")
    parser.add_argument("--dry-run-ingest", action="store_true", help="只檢查 append，不跑 live")
    parser.add_argument("--skip-online-run", action="store_true", help="只 append，不跑 live")
    parser.add_argument("--tej-output", default=str(DEFAULT_TEJ_OUTPUT))
    parser.add_argument("--universe-output", default=str(DEFAULT_UNIVERSE_OUTPUT))
    parser.add_argument("--backup-dir", default=str(DEFAULT_BACKUP_DIR))
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--encoding", default="utf-16-le")
    parser.add_argument("--sep", default="\t")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_live_daily(
        tej_input=args.tej_input,
        mode=args.mode,
        as_of=args.as_of,
        frozen_config=args.frozen_config,
        frozen_execution=args.frozen_execution,
        artifact_root=args.artifact_root,
        production_artifact=args.production_artifact,
        output_dir=args.output_dir,
        capital=args.capital,
        force_retrain=args.force_retrain,
        run_purpose=args.run_purpose,
        official=not args.no_official and args.run_purpose == "production",
        persist_db=not args.no_db,
        dry_run_ingest=args.dry_run_ingest,
        skip_online_run=args.skip_online_run,
        tej_output=args.tej_output,
        universe_output=args.universe_output,
        backup_dir=None if args.no_backup else args.backup_dir,
        encoding=args.encoding,
        sep=args.sep,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
