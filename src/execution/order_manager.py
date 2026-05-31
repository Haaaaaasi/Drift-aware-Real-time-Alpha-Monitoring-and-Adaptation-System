"""Layer 7 — Order lifecycle management and persistence."""

from __future__ import annotations

import pandas as pd
from psycopg2.extras import Json, execute_batch

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
            optional_cols = {
                "account_id", "run_id", "recommendation_id", "execution_mode",
                "broker_order_id", "submitted_at", "reject_reason",
                "price_source", "adjustment_mode", "raw_payload",
            }
            if optional_cols.issubset(set(orders.columns)):
                sql = """
                    INSERT INTO orders
                        (order_id, account_id, run_id, recommendation_id, security_id,
                         order_time, side, order_type, quantity, limit_price, status,
                         expected_price, execution_mode, broker_order_id, submitted_at,
                         reject_reason, price_source, adjustment_mode, raw_payload)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (order_id) DO NOTHING
                """
                records = [
                    (
                        row["order_id"], row.get("account_id"), row.get("run_id"),
                        row.get("recommendation_id"), row["security_id"],
                        row["order_time"], row["side"], row["order_type"],
                        row["quantity"], row["limit_price"], row["status"],
                        row["expected_price"], row.get("execution_mode"),
                        row.get("broker_order_id"), row.get("submitted_at"),
                        row.get("reject_reason"), row.get("price_source"),
                        row.get("adjustment_mode"), Json(row.get("raw_payload") or {}),
                    )
                    for _, row in orders.iterrows()
                ]
                with conn.cursor() as cur:
                    execute_batch(cur, sql, records, page_size=500)
                conn.commit()
                logger.info("orders_persisted", count=len(records))
                return len(records)

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
            optional_cols = {
                "account_id", "run_id", "recommendation_id", "broker_fill_id",
                "side", "gross_notional", "tax", "fees_total", "price_source",
                "adjustment_mode", "source_file", "raw_payload",
            }
            if optional_cols.issubset(set(fills.columns)):
                sql = """
                    INSERT INTO fills
                        (fill_id, order_id, account_id, run_id, recommendation_id,
                         broker_fill_id, security_id, side, fill_time, fill_price,
                         fill_quantity, commission, tax, fees_total, gross_notional,
                         slippage_bps, price_source, adjustment_mode, source_file,
                         raw_payload)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (fill_id) DO NOTHING
                """
                records = [
                    (
                        row["fill_id"], row["order_id"], row.get("account_id"),
                        row.get("run_id"), row.get("recommendation_id"),
                        row.get("broker_fill_id"), row["security_id"],
                        row.get("side"), row["fill_time"], row["fill_price"],
                        row["fill_quantity"], row.get("commission", 0.0),
                        row.get("tax", 0.0), row.get("fees_total", 0.0),
                        row.get("gross_notional"), row.get("slippage_bps"),
                        row.get("price_source"), row.get("adjustment_mode"),
                        row.get("source_file"), Json(row.get("raw_payload") or {}),
                    )
                    for _, row in fills.iterrows()
                ]
                with conn.cursor() as cur:
                    execute_batch(cur, sql, records, page_size=500)
                conn.commit()
                logger.info("fills_persisted", count=len(records))
                return len(records)

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
