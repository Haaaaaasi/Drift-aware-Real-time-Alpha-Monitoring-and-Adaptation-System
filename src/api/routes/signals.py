"""API routes for signal queries."""

from __future__ import annotations

from fastapi import APIRouter, Query

router = APIRouter()


@router.get("/latest")
def get_latest_signals(limit: int = Query(50)):
    """Retrieve the most recent meta signals."""
    import pandas as pd
    from src.common.db import get_pg_connection

    conn = get_pg_connection()
    try:
        df = pd.read_sql(
            "SELECT * FROM meta_signals ORDER BY signal_time DESC LIMIT %s",
            conn,
            params=[limit],
        )
        return df.to_dict(orient="records")
    finally:
        conn.close()


@router.get("/by-security/{security_id}")
def get_signals_by_security(security_id: str, limit: int = Query(100)):
    import pandas as pd
    from src.common.db import get_pg_connection

    conn = get_pg_connection()
    try:
        df = pd.read_sql(
            "SELECT * FROM meta_signals WHERE security_id = %s "
            "ORDER BY signal_time DESC LIMIT %s",
            conn,
            params=[security_id, limit],
        )
        return df.to_dict(orient="records")
    finally:
        conn.close()
