"""Label Update Pipeline — Generate delayed labels for recently matured signal windows.

Usage:
    python -m pipelines.label_update_pipeline
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from src.common.db import get_pg_connection
from src.common.logging import get_logger, setup_logging
from src.labeling.label_generator import LabelGenerator

setup_logging()
logger = get_logger("label_update_pipeline")


def run_label_update(lookback_days: int = 30) -> dict:
    """Generate labels for signals whose horizon windows have now matured."""
    conn = get_pg_connection()
    try:
        # Find signals that don't yet have labels
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        existing_labels = pd.read_sql(
            "SELECT DISTINCT security_id, signal_time, horizon FROM labels_outcomes",
            conn,
        )

        # Load price data for label computation
        price_data = pd.read_sql(
            "SELECT security_id, event_ts as tradetime, close FROM raw_market_events "
            "WHERE event_type = 'kbar_daily' AND event_ts >= %s "
            "ORDER BY security_id, event_ts",
            conn,
            params=[cutoff],
        )
    finally:
        conn.close()

    if price_data.empty:
        logger.info("no_price_data_for_labels")
        return {"labels_generated": 0}

    gen = LabelGenerator(horizons=[1, 5, 10, 20], bar_type="daily")
    new_labels = gen.generate_labels(price_data)

    if not new_labels.empty and not existing_labels.empty:
        # Filter out already-existing labels
        existing_keys = set(
            zip(existing_labels["security_id"], existing_labels["signal_time"],
                existing_labels["horizon"])
        )
        mask = new_labels.apply(
            lambda r: (r["security_id"], r["signal_time"], r["horizon"]) not in existing_keys,
            axis=1,
        )
        new_labels = new_labels[mask]

    n = gen.persist_labels(new_labels) if not new_labels.empty else 0

    summary = {"timestamp": datetime.utcnow().isoformat(), "labels_generated": n}
    logger.info("label_update_complete", **summary)
    return summary


if __name__ == "__main__":
    result = run_label_update()
    print(f"Label update complete: {result}")
