"""Layer 9 — Alert management: persist metrics and fire alerts."""

from __future__ import annotations

import pandas as pd
from psycopg2.extras import execute_batch

from src.common.db import get_pg_connection
from src.common.logging import get_logger

logger = get_logger(__name__)


class AlertManager:
    """Persist monitoring metrics and alerts to PostgreSQL."""

    def persist_metrics(self, metrics: list[dict]) -> int:
        """Write monitoring metrics to the monitoring_metrics table."""
        if not metrics:
            return 0

        conn = get_pg_connection()
        try:
            sql = """
                INSERT INTO monitoring_metrics
                    (metric_time, monitor_type, metric_name, metric_value,
                     dimension, window_size)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            records = [
                (
                    m["metric_time"], m["monitor_type"], m["metric_name"],
                    m["metric_value"], m.get("dimension"), m.get("window_size"),
                )
                for m in metrics
            ]
            with conn.cursor() as cur:
                execute_batch(cur, sql, records, page_size=500)
            conn.commit()
            logger.info("metrics_persisted", count=len(records))
            return len(records)
        finally:
            conn.close()

    def fire_alerts(self, metrics: list[dict]) -> int:
        """Create alert records for metrics that exceeded thresholds."""
        alertable = [m for m in metrics if m.get("severity") is not None]
        if not alertable:
            return 0

        conn = get_pg_connection()
        try:
            sql = """
                INSERT INTO alerts
                    (alert_time, monitor_type, metric_name, severity,
                     current_value, threshold, message)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            records = [
                (
                    m["metric_time"],
                    m["monitor_type"],
                    m["metric_name"],
                    m["severity"],
                    m["metric_value"],
                    0.0,  # threshold stored for reference
                    f"{m['metric_name']} = {m['metric_value']:.4f} "
                    f"[{m['severity']}] dim={m.get('dimension', 'global')}",
                )
                for m in alertable
            ]
            with conn.cursor() as cur:
                execute_batch(cur, sql, records, page_size=500)
            conn.commit()

            for m in alertable:
                logger.warning(
                    "alert_fired",
                    severity=m["severity"],
                    metric=m["metric_name"],
                    value=m["metric_value"],
                )
            return len(records)
        finally:
            conn.close()

    def get_recent_alerts(
        self,
        hours: int = 24,
        severity: str | None = None,
    ) -> pd.DataFrame:
        """Retrieve recent alerts from PostgreSQL."""
        conn = get_pg_connection()
        try:
            query = """
                SELECT * FROM alerts
                WHERE alert_time >= now() - interval '%s hours'
            """
            params = [hours]
            if severity:
                query += " AND severity = %s"
                params.append(severity)
            query += " ORDER BY alert_time DESC"
            return pd.read_sql(query, conn, params=params)
        finally:
            conn.close()

    def get_unacknowledged_critical_count(self) -> int:
        """Count unacknowledged critical alerts."""
        conn = get_pg_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM alerts "
                    "WHERE severity = 'CRITICAL' AND is_acknowledged = FALSE"
                )
                return cur.fetchone()[0]
        finally:
            conn.close()
