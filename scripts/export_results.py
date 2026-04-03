"""Export research results for analysis and presentation.

Usage:
    python scripts/export_results.py --output ./data/results/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.db import get_pg_connection
from src.common.logging import get_logger, setup_logging

setup_logging()
logger = get_logger("export_results")


EXPORT_QUERIES = {
    "meta_signals": "SELECT * FROM meta_signals ORDER BY signal_time",
    "monitoring_metrics": "SELECT * FROM monitoring_metrics ORDER BY metric_time",
    "alerts": "SELECT * FROM alerts ORDER BY alert_time",
    "model_registry": "SELECT * FROM model_registry ORDER BY trained_at",
    "labels_outcomes": "SELECT * FROM labels_outcomes ORDER BY signal_time",
    "positions": "SELECT * FROM positions ORDER BY snapshot_time",
}


def export(output_dir: str):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    conn = get_pg_connection()
    try:
        for name, query in EXPORT_QUERIES.items():
            df = pd.read_sql(query, conn)
            path = output / f"{name}.csv"
            df.to_csv(path, index=False)
            logger.info("exported", table=name, rows=len(df), path=str(path))
            print(f"Exported {name}: {len(df)} rows -> {path}")
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="./data/results/")
    args = parser.parse_args()
    export(args.output)
