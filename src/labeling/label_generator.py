"""Layer 8 — Delayed label generation for model training and evaluation."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from psycopg2.extras import execute_batch

from src.common.db import get_pg_connection
from src.common.logging import get_logger
from src.common.time_utils import compute_label_available_at
from src.config.constants import LABEL_HORIZONS

logger = get_logger(__name__)


class LabelGenerator:
    """Generate forward-return labels with strict temporal discipline.

    Key invariant: label_available_at > signal_time + horizon + buffer
    to prevent any form of look-ahead bias.
    """

    def __init__(
        self,
        horizons: list[int] | None = None,
        bar_type: str = "daily",
        buffer_bars: int = 1,
    ) -> None:
        self._horizons = horizons or LABEL_HORIZONS
        self._bar_type = bar_type
        self._buffer = buffer_bars

    def generate_labels(
        self,
        price_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute forward returns for all horizons.

        Args:
            price_data: DataFrame with [security_id, tradetime, close], sorted by (security_id, tradetime).

        Returns:
            DataFrame [security_id, signal_time, horizon, forward_return,
                       forward_direction, label_available_at].
        """
        all_labels = []

        for sec_id, group in price_data.groupby("security_id"):
            group = group.sort_values("tradetime").reset_index(drop=True)

            for h in self._horizons:
                group[f"fwd_{h}"] = group["close"].shift(-h) / group["close"] - 1

                for idx, row in group.iterrows():
                    fwd = row.get(f"fwd_{h}")
                    if pd.isna(fwd):
                        continue

                    signal_time = row["tradetime"]
                    avail = compute_label_available_at(
                        signal_time, h, self._bar_type, self._buffer
                    )

                    all_labels.append({
                        "security_id": sec_id,
                        "signal_time": signal_time,
                        "horizon": h,
                        "forward_return": fwd,
                        "forward_direction": 1 if fwd > 0 else -1,
                        "realized_pnl": None,
                        "label_available_at": avail,
                    })

        result = pd.DataFrame(all_labels)
        logger.info(
            "labels_generated",
            total=len(result),
            horizons=self._horizons,
        )
        return result

    def persist_labels(self, labels: pd.DataFrame) -> int:
        """Write labels to PostgreSQL labels_outcomes table."""
        if labels.empty:
            return 0

        conn = get_pg_connection()
        try:
            sql = """
                INSERT INTO labels_outcomes
                    (security_id, signal_time, horizon, forward_return,
                     forward_direction, realized_pnl, label_available_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (security_id, signal_time, horizon)
                DO UPDATE SET forward_return = EXCLUDED.forward_return,
                              forward_direction = EXCLUDED.forward_direction,
                              label_available_at = EXCLUDED.label_available_at
            """
            records = labels[
                ["security_id", "signal_time", "horizon", "forward_return",
                 "forward_direction", "realized_pnl", "label_available_at"]
            ].values.tolist()
            with conn.cursor() as cur:
                execute_batch(cur, sql, records, page_size=1000)
            conn.commit()
            logger.info("labels_persisted", count=len(records))
            return len(records)
        finally:
            conn.close()

    def get_available_labels(
        self,
        as_of: datetime,
        horizon: int = 1,
    ) -> pd.DataFrame:
        """Retrieve only labels that are temporally available as of a given time."""
        conn = get_pg_connection()
        try:
            return pd.read_sql(
                """
                SELECT security_id, signal_time, horizon, forward_return, forward_direction
                FROM labels_outcomes
                WHERE label_available_at <= %s AND horizon = %s
                ORDER BY signal_time
                """,
                conn,
                params=[as_of, horizon],
            )
        finally:
            conn.close()
