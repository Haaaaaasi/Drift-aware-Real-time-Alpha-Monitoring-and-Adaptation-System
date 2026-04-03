"""Layer 1 — Event replay for research reproducibility."""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Any

import pandas as pd

from src.common.db import get_pg_connection
from src.common.logging import get_logger

logger = get_logger(__name__)


class EventReplayer:
    """Replay raw_market_events in chronological order for backtesting."""

    def __init__(self, on_event: Callable[[dict[str, Any]], None] | None = None) -> None:
        self._conn = get_pg_connection()
        self._on_event = on_event

    def replay(
        self,
        start: datetime,
        end: datetime,
        event_type: str = "kbar_daily",
        security_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        """Replay events from raw_market_events, ordered by event_ts.

        Returns the events as a DataFrame and optionally calls on_event for each row.
        """
        query = """
            SELECT security_id, event_type, event_ts, open, high, low, close,
                   volume, vwap, bid_price, ask_price
            FROM raw_market_events
            WHERE event_ts BETWEEN %s AND %s
              AND event_type = %s
        """
        params: list[Any] = [start, end, event_type]

        if security_ids:
            query += " AND security_id = ANY(%s)"
            params.append(security_ids)

        query += " ORDER BY event_ts, security_id"

        df = pd.read_sql(query, self._conn, params=params)

        if self._on_event and not df.empty:
            for _, row in df.iterrows():
                self._on_event(row.to_dict())

        logger.info("replay_complete", rows=len(df), start=str(start), end=str(end))
        return df

    def close(self) -> None:
        self._conn.close()
