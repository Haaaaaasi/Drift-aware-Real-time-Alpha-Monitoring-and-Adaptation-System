"""Adaptation Pipeline — Check triggers and execute adaptation policies.

Usage:
    python -m pipelines.adaptation_pipeline
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.adaptation.model_registry import ModelRegistryManager
from src.adaptation.performance_trigger import PerformanceTriggeredAdapter
from src.adaptation.scheduler import ScheduledRetrainer
from src.common.db import get_pg_connection
from src.common.logging import get_logger, setup_logging
from src.monitoring.alert_manager import AlertManager

setup_logging()
logger = get_logger("adaptation_pipeline")


def run_adaptation() -> dict:
    """Check adaptation triggers and execute appropriate policy."""
    now = datetime.utcnow()
    registry = ModelRegistryManager()
    alert_mgr = AlertManager()

    # Policy 1: Scheduled retrain
    scheduler = ScheduledRetrainer(retrain_interval_days=7)
    if scheduler.should_retrain(now):
        logger.info("scheduled_retrain_due")
        # In production, load real alpha_panel and forward_returns
        # For now, log that retrain would execute
        logger.info("scheduled_retrain_skipped_no_data")

    # Policy 2: Performance-triggered
    adapter = PerformanceTriggeredAdapter()
    critical_count = alert_mgr.get_unacknowledged_critical_count()

    # Load rolling metrics from monitoring_metrics
    conn = get_pg_connection()
    try:
        ic_df = pd.read_sql(
            "SELECT metric_time, metric_value FROM monitoring_metrics "
            "WHERE metric_name = 'rolling_ic' "
            "ORDER BY metric_time DESC LIMIT 20",
            conn,
        )
        sharpe_df = pd.read_sql(
            "SELECT metric_time, metric_value FROM monitoring_metrics "
            "WHERE metric_name = 'rolling_sharpe' "
            "ORDER BY metric_time DESC LIMIT 20",
            conn,
        )
    finally:
        conn.close()

    rolling_ic = pd.Series(
        ic_df["metric_value"].values if not ic_df.empty else [],
        dtype=float,
    )
    rolling_sharpe = pd.Series(
        sharpe_df["metric_value"].values if not sharpe_df.empty else [],
        dtype=float,
    )

    should_trigger, reason = adapter.check_trigger(
        rolling_ic, rolling_sharpe, critical_count
    )

    if should_trigger:
        logger.info("performance_trigger_fired", reason=reason)
        # In production, execute adapter.adapt(...)

    # Status
    prod_model = registry.get_production_model()
    summary = {
        "timestamp": now.isoformat(),
        "scheduled_retrain_due": scheduler.should_retrain(now),
        "performance_trigger": should_trigger,
        "trigger_reason": reason,
        "critical_alerts": critical_count,
        "production_model": prod_model.get("model_id") if prod_model else None,
    }
    logger.info("adaptation_pipeline_complete", **summary)
    return summary


if __name__ == "__main__":
    result = run_adaptation()
    print(f"Adaptation check complete: {result}")
