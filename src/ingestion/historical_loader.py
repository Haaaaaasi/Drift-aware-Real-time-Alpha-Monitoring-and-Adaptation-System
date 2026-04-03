"""Layer 1 — Historical data ingestion from CSV files and APIs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg2.extras

from src.common.db import get_pg_connection
from src.common.logging import get_logger

logger = get_logger(__name__)


class HistoricalLoader:
    """Load historical OHLCV data from CSV into raw_market_events (PostgreSQL)."""

    REQUIRED_COLUMNS = {"security_id", "datetime", "open", "high", "low", "close", "volume"}

    def __init__(self) -> None:
        self._conn = get_pg_connection()

    def load_csv(
        self,
        path: str | Path,
        event_type: str = "kbar_daily",
        extra_columns: dict[str, str] | None = None,
    ) -> int:
        """Ingest a CSV file into raw_market_events.

        The CSV must have at least: security_id, datetime, open, high, low, close, volume.
        Optional: vwap, bid_price, ask_price, bid_size, ask_size.

        Args:
            path: Path to CSV file.
            event_type: Type of bar (kbar_daily, kbar_5m, etc.).
            extra_columns: Mapping from CSV column names to expected names.

        Returns:
            Number of rows inserted.
        """
        df = pd.read_csv(path, parse_dates=["datetime"])

        if extra_columns:
            df = df.rename(columns=extra_columns)

        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        df["event_type"] = event_type
        df["event_ts"] = df["datetime"]
        df["ingestion_ts"] = datetime.utcnow()

        for col in ["vwap", "bid_price", "ask_price", "bid_size", "ask_size"]:
            if col not in df.columns:
                df[col] = None

        records = df[
            [
                "security_id", "event_type", "event_ts", "ingestion_ts",
                "open", "high", "low", "close", "volume", "vwap",
                "bid_price", "ask_price", "bid_size", "ask_size",
            ]
        ].values.tolist()

        insert_sql = """
            INSERT INTO raw_market_events
                (security_id, event_type, event_ts, ingestion_ts,
                 open, high, low, close, volume, vwap,
                 bid_price, ask_price, bid_size, ask_size)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (security_id, event_ts, event_type) DO NOTHING
        """

        with self._conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, insert_sql, records, page_size=1000)
        self._conn.commit()

        logger.info("historical_load_complete", path=str(path), rows=len(records))
        return len(records)

    def load_dataframe(
        self,
        df: pd.DataFrame,
        event_type: str = "kbar_daily",
    ) -> int:
        """Ingest a pandas DataFrame directly into raw_market_events."""
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        df = df.copy()
        df["event_type"] = event_type
        df["event_ts"] = df["datetime"]
        df["ingestion_ts"] = datetime.utcnow()

        for col in ["vwap", "bid_price", "ask_price", "bid_size", "ask_size"]:
            if col not in df.columns:
                df[col] = None

        records = df[
            [
                "security_id", "event_type", "event_ts", "ingestion_ts",
                "open", "high", "low", "close", "volume", "vwap",
                "bid_price", "ask_price", "bid_size", "ask_size",
            ]
        ].values.tolist()

        insert_sql = """
            INSERT INTO raw_market_events
                (security_id, event_type, event_ts, ingestion_ts,
                 open, high, low, close, volume, vwap,
                 bid_price, ask_price, bid_size, ask_size)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (security_id, event_ts, event_type) DO NOTHING
        """

        with self._conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, insert_sql, records, page_size=1000)
        self._conn.commit()

        logger.info("dataframe_load_complete", rows=len(records))
        return len(records)

    def close(self) -> None:
        self._conn.close()
