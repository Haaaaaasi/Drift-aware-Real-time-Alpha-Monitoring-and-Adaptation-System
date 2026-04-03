"""Layer 2 — Schema mapping and standardization for DolphinDB prepareData format."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

from src.common.db import get_pg_connection, get_dolphindb
from src.common.logging import get_logger
from src.standardization.calendar import TradingCalendar
from src.standardization.quality_check import QualityChecker

logger = get_logger(__name__)


class SchemaMapper:
    """Transform raw_market_events into standardized_bars suitable for DolphinDB wq101alpha."""

    WQ_COLUMNS = [
        "security_id", "tradetime", "bar_type",
        "open", "high", "low", "close", "vol", "vwap",
        "cap", "indclass", "is_tradable", "missing_flags",
    ]

    def __init__(self, bar_type: str = "daily") -> None:
        self._bar_type = bar_type
        self._calendar = TradingCalendar()
        self._quality = QualityChecker()

    def standardize_batch(
        self,
        start_date: date,
        end_date: date,
        security_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load raw events from PostgreSQL, standardize, and return a DataFrame
        ready for DolphinDB ingestion."""
        conn = get_pg_connection()
        try:
            query = """
                SELECT security_id, event_ts as tradetime,
                       open, high, low, close, volume as vol, vwap
                FROM raw_market_events
                WHERE event_ts::date BETWEEN %s AND %s
                  AND event_type = %s
            """
            params = [start_date, end_date, f"kbar_{self._bar_type}"]

            if security_ids:
                query += " AND security_id = ANY(%s)"
                params.append(security_ids)

            query += " ORDER BY security_id, event_ts"
            raw = pd.read_sql(query, conn, params=params)
        finally:
            conn.close()

        if raw.empty:
            logger.warning("no_raw_data", start=str(start_date), end=str(end_date))
            return pd.DataFrame(columns=self.WQ_COLUMNS)

        sec_master = self._load_security_master()

        std = self._map_schema(raw, sec_master)
        std = self._quality.check_and_flag(std)

        logger.info(
            "standardize_batch_complete",
            rows=len(std),
            symbols=std["security_id"].nunique(),
        )
        return std

    def standardize_incremental(self, new_events: pd.DataFrame) -> pd.DataFrame:
        """Standardize a small batch of new events (for online pipeline)."""
        sec_master = self._load_security_master()
        std = self._map_schema(new_events, sec_master)
        std = self._quality.check_and_flag(std)
        return std

    def _map_schema(self, raw: pd.DataFrame, sec_master: pd.DataFrame) -> pd.DataFrame:
        """Core schema mapping logic."""
        df = raw.copy()
        df["bar_type"] = self._bar_type
        df["tradetime"] = pd.to_datetime(df["tradetime"])

        # Fill missing vwap
        mask_no_vwap = df["vwap"].isna()
        df.loc[mask_no_vwap, "vwap"] = (
            df.loc[mask_no_vwap, "high"]
            + df.loc[mask_no_vwap, "low"]
            + df.loc[mask_no_vwap, "close"]
        ) / 3.0

        # Join security master for cap and indclass
        if not sec_master.empty:
            df = df.merge(
                sec_master[["security_id", "industry_code"]],
                on="security_id",
                how="left",
            )
            df["indclass"] = df["industry_code"].fillna(0).astype(int)
            df.drop(columns=["industry_code"], inplace=True)
        else:
            df["indclass"] = 0

        # Cap placeholder: use close * 1e6 as rough proxy until real data is available
        df["cap"] = df["close"] * 1_000_000

        # Tradability
        df["is_tradable"] = (df["vol"] > 0) & df["close"].notna()

        # Missing flags bitmask: 1=vwap, 2=cap, 4=indclass
        df["missing_flags"] = 0
        df.loc[mask_no_vwap, "missing_flags"] += 1
        df.loc[df["indclass"] == 0, "missing_flags"] += 4

        return df[self.WQ_COLUMNS]

    def _load_security_master(self) -> pd.DataFrame:
        conn = get_pg_connection()
        try:
            return pd.read_sql(
                "SELECT security_id, industry_code FROM security_master WHERE is_active = TRUE",
                conn,
            )
        finally:
            conn.close()

    def push_to_dolphindb(self, df: pd.DataFrame) -> None:
        """Write standardized bars into DolphinDB partitioned table."""
        if df.empty:
            return

        session = get_dolphindb()
        session.upload({"std_data": df})
        session.run("""
            db = loadTable("dfs://darams_market", "standardized_bars")
            db.append!(std_data)
        """)
        logger.info("dolphindb_ingestion_complete", rows=len(df))
