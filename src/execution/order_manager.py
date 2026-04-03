"""Layer 7 — Order lifecycle management and persistence."""

from __future__ import annotations

import pandas as pd
from psycopg2.extras import execute_batch

from src.common.db import get_pg_connection
from src.common.logging import get_logger

logger = get_logger(__name__)


class OrderManager:
    """Persist orders and fills to PostgreSQL and manage order lifecycle."""

    def persist_orders(self, orders: pd.DataFrame) -> int:
        if orders.empty:
            return 0
        conn = get_pg_connection()
        try:
            sql = """
                INSERT INTO orders
                    (order_id, security_id, order_time, side, order_type,
                     quantity, limit_price, status, expected_price)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (order_id) DO NOTHING
            """
            records = orders[
                ["order_id", "security_id", "order_time", "side", "order_type",
                 "quantity", "limit_price", "status", "expected_price"]
            ].values.tolist()
            with conn.cursor() as cur:
                execute_batch(cur, sql, records, page_size=500)
            conn.commit()
            logger.info("orders_persisted", count=len(records))
            return len(records)
        finally:
            conn.close()

    def persist_fills(self, fills: pd.DataFrame) -> int:
        if fills.empty:
            return 0
        conn = get_pg_connection()
        try:
            sql = """
                INSERT INTO fills
                    (fill_id, order_id, security_id, fill_time,
                     fill_price, fill_quantity, commission, slippage_bps)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (fill_id) DO NOTHING
            """
            records = fills[
                ["fill_id", "order_id", "security_id", "fill_time",
                 "fill_price", "fill_quantity", "commission", "slippage_bps"]
            ].values.tolist()
            with conn.cursor() as cur:
                execute_batch(cur, sql, records, page_size=500)
            conn.commit()
            logger.info("fills_persisted", count=len(records))
            return len(records)
        finally:
            conn.close()

    def persist_positions(self, positions: pd.DataFrame) -> int:
        if positions.empty:
            return 0
        conn = get_pg_connection()
        try:
            sql = """
                INSERT INTO positions
                    (snapshot_time, security_id, quantity, avg_cost,
                     market_value, unrealized_pnl)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (snapshot_time, security_id)
                DO UPDATE SET quantity = EXCLUDED.quantity,
                              avg_cost = EXCLUDED.avg_cost,
                              market_value = EXCLUDED.market_value,
                              unrealized_pnl = EXCLUDED.unrealized_pnl
            """
            records = positions[
                ["snapshot_time", "security_id", "quantity", "avg_cost",
                 "market_value", "unrealized_pnl"]
            ].values.tolist()
            with conn.cursor() as cur:
                execute_batch(cur, sql, records, page_size=500)
            conn.commit()
            logger.info("positions_persisted", count=len(records))
            return len(records)
        finally:
            conn.close()
