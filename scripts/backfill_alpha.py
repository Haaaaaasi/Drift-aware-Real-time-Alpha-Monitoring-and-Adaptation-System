"""Backfill historical alpha features via DolphinDB batch computation.

Usage:
    python scripts/backfill_alpha.py --start 2022-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
from datetime import date

from src.alpha_engine.batch_compute import BatchAlphaComputer
from src.common.logging import get_logger, setup_logging

setup_logging()
logger = get_logger("backfill_alpha")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2022-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    computer = BatchAlphaComputer()
    try:
        result = computer.compute(start_date=start, end_date=end)
        logger.info("backfill_complete", rows=len(result))
        print(f"Backfilled {len(result)} alpha feature rows")
    except Exception as e:
        logger.error("backfill_failed", error=str(e))
        print(f"Backfill failed: {e}")
        print("Ensure DolphinDB is running and wq101alpha module is installed.")
    finally:
        computer.close()


if __name__ == "__main__":
    main()
