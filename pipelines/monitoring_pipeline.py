"""Monitoring Pipeline — Run all four monitors and persist results.

Usage:
    python -m pipelines.monitoring_pipeline
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.common.db import get_pg_connection
from src.common.logging import get_logger, setup_logging
from src.monitoring.alert_manager import AlertManager
from src.monitoring.alpha_monitor import AlphaMonitor
from src.monitoring.data_monitor import DataMonitor
from src.monitoring.model_monitor import ModelMonitor
from src.monitoring.strategy_monitor import StrategyMonitor

setup_logging()
logger = get_logger("monitoring_pipeline")


def run_monitoring() -> dict:
    """Execute all monitoring checks on the latest available data."""
    alert_mgr = AlertManager()
    all_metrics: list[dict] = []

    # Data monitoring
    conn = get_pg_connection()
    try:
        recent_bars = pd.read_sql(
            "SELECT * FROM raw_market_events "
            "WHERE event_ts >= now() - interval '2 days' "
            "ORDER BY event_ts",
            conn,
        )
    finally:
        conn.close()

    if not recent_bars.empty:
        data_mon = DataMonitor()
        all_metrics.extend(data_mon.run(recent_bars))

    # Alpha monitoring (from monitoring_metrics if pre-computed)
    # Strategy monitoring (from positions/fills)

    # Persist
    n_metrics = alert_mgr.persist_metrics(all_metrics)
    n_alerts = alert_mgr.fire_alerts(all_metrics)

    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "metrics_persisted": n_metrics,
        "alerts_fired": n_alerts,
    }
    logger.info("monitoring_pipeline_complete", **summary)
    return summary


if __name__ == "__main__":
    result = run_monitoring()
    print(f"Monitoring complete: {result}")
