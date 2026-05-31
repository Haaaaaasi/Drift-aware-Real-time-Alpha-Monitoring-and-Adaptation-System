"""把每日 TEJ OHLCV CSV 安全併入既有 TEJ parquet。

範例：

    python scripts/append_tej_daily.py --input TEJ_20260501.csv
    python scripts/append_tej_daily.py --input TEJ_20260501.csv --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ingestion.tej_daily_append import (
    DEFAULT_BACKUP_DIR,
    DEFAULT_TEJ_OUTPUT,
    DEFAULT_UNIVERSE_OUTPUT,
    append_tej_daily_files,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="每日 TEJ OHLCV CSV 檔案")
    parser.add_argument("--output", default=str(DEFAULT_TEJ_OUTPUT), help="輸出的 TEJ bars parquet")
    parser.add_argument(
        "--universe-output",
        default=str(DEFAULT_UNIVERSE_OUTPUT),
        help="輸出的 TEJ universe bounds parquet",
    )
    parser.add_argument("--backup-dir", default=str(DEFAULT_BACKUP_DIR), help="覆寫前備份目錄")
    parser.add_argument("--no-backup", action="store_true", help="不備份既有 parquet")
    parser.add_argument("--dry-run", action="store_true", help="只檢查與回報，不寫檔")
    parser.add_argument("--encoding", default="utf-16-le", help="TEJ CSV encoding")
    parser.add_argument("--sep", default="\t", help="TEJ CSV separator")
    parser.add_argument("--active-threshold-days", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = append_tej_daily_files(
        args.input,
        output_path=Path(args.output),
        universe_output_path=Path(args.universe_output),
        backup_dir=None if args.no_backup else Path(args.backup_dir),
        dry_run=args.dry_run,
        encoding=args.encoding,
        sep=args.sep,
        active_threshold_days=args.active_threshold_days,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
