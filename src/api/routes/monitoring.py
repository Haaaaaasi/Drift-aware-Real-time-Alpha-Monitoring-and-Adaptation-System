"""API routes for monitoring metrics and alerts."""

from __future__ import annotations

from fastapi import APIRouter, Query

from src.monitoring.alert_manager import AlertManager

router = APIRouter()


@router.get("/metrics")
def get_recent_metrics(
    monitor_type: str | None = Query(None),
    hours: int = Query(24),
):
    """Retrieve recent monitoring metrics."""
    import pandas as pd
    from src.common.db import get_pg_connection

    conn = get_pg_connection()
    try:
        query = (
            "SELECT * FROM monitoring_metrics "
            "WHERE metric_time >= now() - interval '%s hours'"
        )
        params = [hours]
        if monitor_type:
            query += " AND monitor_type = %s"
            params.append(monitor_type)
        query += " ORDER BY metric_time DESC LIMIT 500"
        df = pd.read_sql(query, conn, params=params)
        return df.to_dict(orient="records")
    finally:
        conn.close()


@router.get("/alerts")
def get_recent_alerts(
    severity: str | None = Query(None),
    hours: int = Query(24),
):
    """Retrieve recent alerts."""
    mgr = AlertManager()
    df = mgr.get_recent_alerts(hours=hours, severity=severity)
    return df.to_dict(orient="records")


@router.get("/alerts/critical/count")
def get_critical_alert_count():
    mgr = AlertManager()
    return {"count": mgr.get_unacknowledged_critical_count()}
