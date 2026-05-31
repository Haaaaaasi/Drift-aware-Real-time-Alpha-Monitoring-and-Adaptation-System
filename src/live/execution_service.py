"""Live execution and accounting reconciliation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from psycopg2.extras import Json, execute_batch

from src.common.db import get_pg_connection
from src.common.logging import get_logger
from src.live.order_units import split_tw_stock_order_legs

logger = get_logger(__name__)

DEFAULT_ACCOUNT_ID = "paper_main"
DEFAULT_INITIAL_CAPITAL = 10_000_000.0
DEFAULT_PRICE_SOURCE = "paper_next_vwap"
DEFAULT_ADJUSTMENT_MODE = "raw"

ACTION_TO_SIDE = {
    "BUY": "BUY",
    "INCREASE": "BUY",
    "SELL": "SELL",
    "REDUCE": "SELL",
}


@dataclass(frozen=True)
class ReconciliationResult:
    """Account-aware reconciliation outputs for one accounting date."""

    positions: pd.DataFrame
    account_snapshot: dict[str, Any]
    market_prices: pd.DataFrame
    executed_recommendation_ids: list[int]


def build_orders_from_recommendations(
    recommendations: pd.DataFrame,
    *,
    account_id: str = DEFAULT_ACCOUNT_ID,
    execution_mode: str = "paper",
    order_time: datetime | pd.Timestamp | None = None,
    price_source: str = DEFAULT_PRICE_SOURCE,
    adjustment_mode: str = DEFAULT_ADJUSTMENT_MODE,
) -> pd.DataFrame:
    """Convert approved trade recommendations into executable orders."""

    if recommendations.empty:
        return pd.DataFrame()
    order_time = pd.Timestamp(order_time or datetime.utcnow()).to_pydatetime()
    rows: list[dict[str, Any]] = []
    for _, row in recommendations.iterrows():
        status = str(row.get("status", "APPROVED")).upper()
        action = str(row.get("action", "")).upper()
        side = ACTION_TO_SIDE.get(action)
        total_shares = abs(int(row.get("delta_shares") or 0))
        legs = split_tw_stock_order_legs(total_shares)
        if status != "APPROVED" or side is None or not legs:
            continue

        recommendation_id = _clean_int(row.get("recommendation_id"))
        run_id = _clean_str(row.get("run_id"))
        base_order_id = (
            f"ORD-REC-{recommendation_id}"
            if recommendation_id is not None
            else f"ORD-{uuid4().hex[:16]}"
        )
        for leg in legs:
            order_id = base_order_id
            if len(legs) > 1:
                order_id = f"{base_order_id}-{leg.order_lot.upper()}"
            rows.append(
                {
                    "order_id": order_id,
                    "account_id": account_id,
                    "run_id": run_id,
                    "recommendation_id": recommendation_id,
                    "security_id": str(row["security_id"]),
                    "order_time": order_time,
                    "side": side,
                    "order_type": "MARKET",
                    "quantity": leg.share_quantity,
                    "limit_price": None,
                    "status": "SUBMITTED",
                    "expected_price": _clean_float(row.get("last_price")),
                    "execution_mode": execution_mode,
                    "broker_order_id": None,
                    "submitted_at": order_time,
                    "reject_reason": None,
                    "price_source": price_source,
                    "adjustment_mode": adjustment_mode,
                    "raw_payload": {
                        "action": action,
                        "target_weight": _clean_float(row.get("target_weight")),
                        "delta_weight": _clean_float(row.get("delta_weight")),
                        "total_share_quantity": total_shares,
                        "share_quantity": leg.share_quantity,
                        "quantity_unit": "share",
                        "shioaji_order_lot": leg.order_lot,
                        "shioaji_quantity": leg.shioaji_quantity,
                        "shioaji_quantity_unit": leg.quantity_unit,
                    },
                }
            )
    return pd.DataFrame(rows)


def build_paper_fills_from_orders(
    orders: pd.DataFrame,
    *,
    fill_time: datetime | pd.Timestamp | None = None,
    slippage_bps: float = 0.0,
    commission_rate: float = 0.000926,
    tax_rate: float = 0.003,
    price_source: str = DEFAULT_PRICE_SOURCE,
    adjustment_mode: str = DEFAULT_ADJUSTMENT_MODE,
) -> pd.DataFrame:
    """Generate paper fills from submitted orders."""

    if orders.empty:
        return pd.DataFrame()
    fill_time = pd.Timestamp(fill_time or datetime.utcnow()).to_pydatetime()
    rows: list[dict[str, Any]] = []
    for _, row in orders.iterrows():
        side = str(row["side"]).upper()
        expected = _clean_float(row.get("expected_price"))
        quantity = abs(int(row.get("quantity") or 0))
        if expected is None or expected <= 0 or quantity <= 0:
            continue
        slip_mult = 1.0 + (float(slippage_bps) / 10000.0) * (1 if side == "BUY" else -1)
        fill_price = expected * slip_mult
        gross = fill_price * quantity
        commission = gross * float(commission_rate)
        tax = gross * float(tax_rate) if side == "SELL" else 0.0
        rows.append(
            {
                "fill_id": f"FIL-{uuid4().hex[:16]}",
                "order_id": row["order_id"],
                "account_id": row.get("account_id") or DEFAULT_ACCOUNT_ID,
                "run_id": _clean_str(row.get("run_id")),
                "recommendation_id": _clean_int(row.get("recommendation_id")),
                "broker_fill_id": None,
                "security_id": str(row["security_id"]),
                "side": side,
                "fill_time": fill_time,
                "fill_price": fill_price,
                "fill_quantity": quantity,
                "commission": commission,
                "tax": tax,
                "fees_total": commission + tax,
                "gross_notional": gross,
                "slippage_bps": (
                    (fill_price - expected) / expected * 10000.0 if expected else None
                ),
                "price_source": price_source,
                "adjustment_mode": adjustment_mode,
                "source_file": None,
                "raw_payload": {"expected_price": expected, "paper_slippage_bps": slippage_bps},
            }
        )
    return pd.DataFrame(rows)


def normalize_fill_import(
    fills: pd.DataFrame,
    *,
    account_id: str = DEFAULT_ACCOUNT_ID,
    source_file: str | None = None,
    price_source: str = "broker_import",
    adjustment_mode: str = DEFAULT_ADJUSTMENT_MODE,
    recommendations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Normalize manual/broker fill CSV rows into canonical fill records."""

    required = {"run_id", "security_id", "side", "fill_time", "fill_price", "fill_quantity"}
    missing = required - set(fills.columns)
    if missing:
        raise ValueError(f"fill import missing required columns: {sorted(missing)}")

    rec_lookup = _recommendation_lookup(recommendations)
    rows: list[dict[str, Any]] = []
    for row_idx, row in fills.iterrows():
        side = str(row["side"]).upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError(f"unsupported fill side: {side}")
        run_id = str(row["run_id"])
        sec = str(row["security_id"])
        recommendation_id = _clean_int(row.get("recommendation_id"))
        if recommendation_id is None:
            recommendation_id = rec_lookup.get((run_id, sec, side))
        qty = abs(int(row["fill_quantity"]))
        price = float(row["fill_price"])
        gross = price * qty
        commission = float(row.get("commission") or 0.0)
        tax = float(row.get("tax") or 0.0)
        fees_total = float(row.get("fees_total") or (commission + tax))
        broker_fill_id = _clean_str(row.get("broker_fill_id"))
        fill_id = _clean_str(row.get("fill_id")) or (
            f"FIL-BRK-{broker_fill_id}"
            if broker_fill_id
            else _stable_import_fill_id(row, row_idx)
        )
        rows.append(
            {
                "fill_id": fill_id,
                "order_id": _clean_str(row.get("order_id")) or f"ORD-IMP-{fill_id[-16:]}",
                "account_id": _clean_str(row.get("account_id")) or account_id,
                "run_id": run_id,
                "recommendation_id": recommendation_id,
                "broker_fill_id": broker_fill_id,
                "security_id": sec,
                "side": side,
                "fill_time": pd.Timestamp(row["fill_time"]).to_pydatetime(),
                "fill_price": price,
                "fill_quantity": qty,
                "commission": commission,
                "tax": tax,
                "fees_total": fees_total,
                "gross_notional": gross,
                "slippage_bps": _clean_float(row.get("slippage_bps")),
                "price_source": _clean_str(row.get("price_source")) or price_source,
                "adjustment_mode": _clean_str(row.get("adjustment_mode")) or adjustment_mode,
                "source_file": source_file,
                "raw_payload": row.to_dict(),
            }
        )
    return pd.DataFrame(rows)


def reconcile_account(
    *,
    account_id: str,
    as_of_date: date | datetime | pd.Timestamp,
    fills: pd.DataFrame,
    market_prices: pd.DataFrame,
    previous_positions: pd.DataFrame | None = None,
    previous_account: dict[str, Any] | None = None,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    run_id: str | None = None,
    snapshot_time: datetime | pd.Timestamp | None = None,
    price_source: str = DEFAULT_PRICE_SOURCE,
    adjustment_mode: str = DEFAULT_ADJUSTMENT_MODE,
) -> ReconciliationResult:
    """Rebuild account state using average-cost accounting."""

    as_of_day = pd.Timestamp(as_of_date).date()
    snapshot_ts = pd.Timestamp(snapshot_time or datetime.utcnow()).to_pydatetime()
    positions = _positions_from_previous(previous_positions)
    cash = float(
        previous_account.get("cash")
        if previous_account and previous_account.get("cash") is not None
        else initial_capital
    )
    prev_equity = _clean_float((previous_account or {}).get("total_equity"))
    cumulative_realized = float((previous_account or {}).get("realized_pnl") or 0.0)
    daily_realized = 0.0
    executed_recommendation_ids: set[int] = set()

    fills_sorted = fills.copy()
    if not fills_sorted.empty:
        fills_sorted["fill_time"] = pd.to_datetime(fills_sorted["fill_time"])
        fills_sorted = fills_sorted.sort_values("fill_time")
    for _, fill in fills_sorted.iterrows():
        sec = str(fill["security_id"])
        side = str(fill["side"]).upper()
        qty = abs(int(fill["fill_quantity"]))
        price = float(fill["fill_price"])
        gross = price * qty
        fees = float(fill.get("fees_total") or 0.0)
        if fees == 0.0:
            fees = float(fill.get("commission") or 0.0) + float(fill.get("tax") or 0.0)
        pos = positions.setdefault(sec, {"quantity": 0, "avg_cost": 0.0, "realized_pnl": 0.0})
        old_qty = int(pos["quantity"])
        if side == "BUY":
            new_qty = old_qty + qty
            total_cost = float(pos["avg_cost"]) * old_qty + gross + fees
            pos["quantity"] = new_qty
            pos["avg_cost"] = total_cost / new_qty if new_qty else 0.0
            cash -= gross + fees
        elif side == "SELL":
            if qty > old_qty:
                raise ValueError(f"sell quantity exceeds position for {sec}: {qty} > {old_qty}")
            realized = (price - float(pos["avg_cost"])) * qty - fees
            pos["quantity"] = old_qty - qty
            pos["realized_pnl"] = float(pos.get("realized_pnl") or 0.0) + realized
            if pos["quantity"] == 0:
                pos["avg_cost"] = 0.0
            cash += gross - fees
            daily_realized += realized
            cumulative_realized += realized
        else:
            raise ValueError(f"unsupported fill side: {side}")
        rec_id = _clean_int(fill.get("recommendation_id"))
        if rec_id is not None:
            executed_recommendation_ids.add(rec_id)

    price_frame = _normalize_market_prices(
        market_prices,
        account_id=account_id,
        as_of_date=as_of_day,
        run_id=run_id,
        snapshot_time=snapshot_ts,
        price_source=price_source,
        adjustment_mode=adjustment_mode,
    )
    price_map = price_frame.set_index("security_id")["price"].to_dict() if not price_frame.empty else {}

    position_rows: list[dict[str, Any]] = []
    market_value = 0.0
    unrealized = 0.0
    for sec, pos in sorted(positions.items()):
        qty = int(pos.get("quantity") or 0)
        if qty == 0:
            continue
        last_price = _clean_float(price_map.get(sec))
        if last_price is None:
            last_price = float(pos.get("avg_cost") or 0.0)
        sec_mv = qty * last_price
        sec_unrealized = (last_price - float(pos.get("avg_cost") or 0.0)) * qty
        market_value += sec_mv
        unrealized += sec_unrealized
        position_rows.append(
            {
                "account_id": account_id,
                "as_of_date": as_of_day,
                "run_id": run_id,
                "snapshot_time": snapshot_ts,
                "security_id": sec,
                "quantity": qty,
                "avg_cost": float(pos.get("avg_cost") or 0.0),
                "last_price": last_price,
                "market_value": sec_mv,
                "realized_pnl": float(pos.get("realized_pnl") or 0.0),
                "unrealized_pnl": sec_unrealized,
                "price_source": price_source,
                "adjustment_mode": adjustment_mode,
                "metadata": {},
            }
        )

    total_equity = cash + market_value
    daily_return = None if prev_equity in (None, 0.0) else (total_equity / prev_equity - 1.0)
    cumulative_return = total_equity / float(initial_capital) - 1.0
    account_snapshot = {
        "account_id": account_id,
        "as_of_date": as_of_day,
        "run_id": run_id,
        "snapshot_time": snapshot_ts,
        "cash": cash,
        "market_value": market_value,
        "realized_pnl": cumulative_realized,
        "unrealized_pnl": unrealized,
        "total_equity": total_equity,
        "daily_return": daily_return,
        "cumulative_return": cumulative_return,
        "price_source": price_source,
        "adjustment_mode": adjustment_mode,
        "metadata": {
            "daily_realized_pnl": daily_realized,
            "n_fills": int(len(fills_sorted)),
            "n_positions": int(len(position_rows)),
        },
    }
    return ReconciliationResult(
        positions=pd.DataFrame(position_rows),
        account_snapshot=account_snapshot,
        market_prices=price_frame,
        executed_recommendation_ids=sorted(executed_recommendation_ids),
    )


class LiveExecutionService:
    """DB-backed live execution and reconciliation service."""

    def __init__(
        self,
        *,
        account_id: str = DEFAULT_ACCOUNT_ID,
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
        commission_rate: float = 0.000926,
        tax_rate: float = 0.003,
        slippage_bps: float = 0.0,
        emit_monitoring_metrics: bool = True,
    ) -> None:
        self.account_id = account_id
        self.initial_capital = float(initial_capital)
        self.commission_rate = float(commission_rate)
        self.tax_rate = float(tax_rate)
        self.slippage_bps = float(slippage_bps)
        self.emit_monitoring_metrics = bool(emit_monitoring_metrics)

    def paper_fill_run(
        self,
        *,
        run_id: str,
        price_source: str = DEFAULT_PRICE_SOURCE,
        adjustment_mode: str = DEFAULT_ADJUSTMENT_MODE,
    ) -> ReconciliationResult:
        """Execute approved recommendations for a run in paper mode."""

        recs = self.load_recommendations(run_id=run_id, statuses=["APPROVED"])
        orders = build_orders_from_recommendations(
            recs,
            account_id=self.account_id,
            execution_mode="paper",
            price_source=price_source,
            adjustment_mode=adjustment_mode,
        )
        fills = build_paper_fills_from_orders(
            orders,
            slippage_bps=self.slippage_bps,
            commission_rate=self.commission_rate,
            tax_rate=self.tax_rate,
            price_source=price_source,
            adjustment_mode=adjustment_mode,
        )
        self.persist_orders(orders)
        self.persist_fills(fills)
        return self.reconcile_run(
            run_id=run_id,
            fills=fills,
            price_source=price_source,
            adjustment_mode=adjustment_mode,
        )

    def import_fills_csv(
        self,
        *,
        csv_path: str | Path,
        price_source: str = "broker_import",
        adjustment_mode: str = DEFAULT_ADJUSTMENT_MODE,
    ) -> ReconciliationResult:
        """Import broker/manual fills and reconcile the referenced run."""

        path = Path(csv_path)
        raw = pd.read_csv(path)
        run_ids = raw["run_id"].dropna().astype(str).unique().tolist()
        if len(run_ids) != 1:
            raise ValueError("fill CSV must contain exactly one run_id for v1 import")
        run_id = run_ids[0]
        recs = self.load_recommendations(run_id=run_id, statuses=None)
        fills = normalize_fill_import(
            raw,
            account_id=self.account_id,
            source_file=str(path.as_posix()),
            price_source=price_source,
            adjustment_mode=adjustment_mode,
            recommendations=recs,
        )
        self.persist_import_orders_for_fills(fills)
        self.persist_fills(fills)
        return self.reconcile_run(
            run_id=run_id,
            fills=fills,
            price_source=price_source,
            adjustment_mode=adjustment_mode,
        )

    def reconcile_run(
        self,
        *,
        run_id: str,
        fills: pd.DataFrame | None = None,
        price_source: str = DEFAULT_PRICE_SOURCE,
        adjustment_mode: str = DEFAULT_ADJUSTMENT_MODE,
    ) -> ReconciliationResult:
        run = self.load_run(run_id)
        as_of_day = pd.Timestamp(run["as_of_date"]).date()
        previous_positions = self.load_previous_positions(as_of_day)
        previous_account = self.load_previous_account(as_of_day)
        market_prices = self.load_market_prices_from_recommendations(
            run_id=run_id,
            as_of_date=as_of_day,
            price_source=price_source,
            adjustment_mode=adjustment_mode,
        )
        if fills is None:
            fills = self.load_fills(run_id=run_id)
        result = reconcile_account(
            account_id=self.account_id,
            as_of_date=as_of_day,
            fills=fills,
            market_prices=market_prices,
            previous_positions=previous_positions,
            previous_account=previous_account,
            initial_capital=self.initial_capital,
            run_id=run_id,
            price_source=price_source,
            adjustment_mode=adjustment_mode,
        )
        self.persist_market_prices(result.market_prices)
        self.persist_position_snapshots(result.positions)
        self.persist_account_snapshot(result.account_snapshot)
        if result.executed_recommendation_ids:
            self.mark_recommendations_executed(result.executed_recommendation_ids)
        if self.emit_monitoring_metrics:
            self.emit_live_pnl_metrics(run=run, result=result)
        return result

    def emit_live_pnl_metrics(
        self,
        *,
        run: dict[str, Any],
        result: ReconciliationResult,
    ) -> dict[str, int]:
        """Persist live PnL monitoring metrics and alerts after reconciliation."""

        try:
            from src.monitoring.alert_manager import AlertManager
            from src.monitoring.live_pnl_monitor import LivePnLMonitor

            run_id = str(run["run_id"])
            as_of_day = pd.Timestamp(result.account_snapshot["as_of_date"]).date()
            account_snapshots = self.load_account_snapshots_until(as_of_day)
            orders = self.load_orders(run_id=run_id)
            fills = self.load_fills(run_id=run_id)
            recommendations = self.load_recommendations(run_id=run_id, statuses=None)
            metrics = LivePnLMonitor().run(
                account_snapshots=account_snapshots,
                orders=orders,
                fills=fills,
                recommendations=recommendations,
                metric_time=result.account_snapshot["snapshot_time"],
                account_id=self.account_id,
                run_id=run_id,
                model_id=_clean_str(run.get("production_model_id")),
                strategy_id=_clean_str(run.get("frozen_selector_id")) or "live_daily",
            )
            alert_mgr = AlertManager()
            metric_count = alert_mgr.persist_metrics(metrics)
            alert_count = alert_mgr.fire_alerts(metrics)
            logger.info(
                "live_pnl_monitoring_emitted",
                run_id=run_id,
                account_id=self.account_id,
                metrics=metric_count,
                alerts=alert_count,
            )
            return {"metrics": metric_count, "alerts": alert_count}
        except Exception as exc:
            logger.warning(
                "live_pnl_monitoring_emit_failed",
                run_id=run.get("run_id"),
                account_id=self.account_id,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return {"metrics": 0, "alerts": 0}

    def load_run(self, run_id: str) -> dict[str, Any]:
        conn = get_pg_connection()
        try:
            df = pd.read_sql(
                "SELECT * FROM daily_live_runs WHERE run_id = %s",
                conn,
                params=[run_id],
            )
        finally:
            conn.close()
        if df.empty:
            raise ValueError(f"live run not found: {run_id}")
        return df.iloc[0].to_dict()

    def load_recommendations(self, *, run_id: str, statuses: list[str] | None) -> pd.DataFrame:
        conn = get_pg_connection()
        try:
            params: list[Any] = [run_id]
            where = "run_id = %s"
            if statuses is not None:
                where += " AND status = ANY(%s)"
                params.append([s.upper() for s in statuses])
            return pd.read_sql(
                f"SELECT * FROM trade_recommendations WHERE {where}",
                conn,
                params=params,
            )
        finally:
            conn.close()

    def load_previous_positions(self, as_of_date: date) -> pd.DataFrame:
        conn = get_pg_connection()
        try:
            return pd.read_sql(
                """
                SELECT *
                FROM live_position_snapshots
                WHERE account_id = %s
                  AND as_of_date = (
                      SELECT max(as_of_date)
                      FROM live_position_snapshots
                      WHERE account_id = %s AND as_of_date < %s
                  )
                """,
                conn,
                params=[self.account_id, self.account_id, as_of_date],
            )
        finally:
            conn.close()

    def load_previous_account(self, as_of_date: date) -> dict[str, Any] | None:
        conn = get_pg_connection()
        try:
            df = pd.read_sql(
                """
                SELECT *
                FROM live_account_snapshots
                WHERE account_id = %s AND as_of_date < %s
                ORDER BY as_of_date DESC
                LIMIT 1
                """,
                conn,
                params=[self.account_id, as_of_date],
            )
        finally:
            conn.close()
        return None if df.empty else df.iloc[0].to_dict()

    def load_fills(self, *, run_id: str) -> pd.DataFrame:
        conn = get_pg_connection()
        try:
            return pd.read_sql(
                "SELECT * FROM fills WHERE account_id = %s AND run_id = %s",
                conn,
                params=[self.account_id, run_id],
            )
        finally:
            conn.close()

    def load_orders(self, *, run_id: str) -> pd.DataFrame:
        conn = get_pg_connection()
        try:
            return pd.read_sql(
                "SELECT * FROM orders WHERE account_id = %s AND run_id = %s",
                conn,
                params=[self.account_id, run_id],
            )
        finally:
            conn.close()

    def load_account_snapshots_until(self, as_of_date: date) -> pd.DataFrame:
        conn = get_pg_connection()
        try:
            return pd.read_sql(
                """
                SELECT *
                FROM live_account_snapshots
                WHERE account_id = %s
                  AND as_of_date <= %s
                ORDER BY as_of_date
                """,
                conn,
                params=[self.account_id, as_of_date],
            )
        finally:
            conn.close()

    def load_market_prices_from_recommendations(
        self,
        *,
        run_id: str,
        as_of_date: date,
        price_source: str,
        adjustment_mode: str,
    ) -> pd.DataFrame:
        recs = self.load_recommendations(run_id=run_id, statuses=None)
        if recs.empty:
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "account_id": self.account_id,
                "as_of_date": as_of_date,
                "run_id": run_id,
                "security_id": recs["security_id"].astype(str),
                "price_time": pd.Timestamp(as_of_date),
                "price": recs["last_price"].astype(float),
                "price_type": "close",
                "price_source": price_source,
                "adjustment_mode": adjustment_mode,
                "metadata": [{"from": "trade_recommendations.last_price"}] * len(recs),
            }
        ).dropna(subset=["price"])

    def persist_orders(self, orders: pd.DataFrame) -> int:
        if orders.empty:
            return 0
        conn = get_pg_connection()
        try:
            sql = """
                INSERT INTO orders
                    (order_id, account_id, run_id, recommendation_id, security_id,
                     order_time, side, order_type, quantity, limit_price, status,
                     expected_price, execution_mode, broker_order_id, submitted_at,
                     reject_reason, price_source, adjustment_mode, raw_payload)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (order_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    expected_price = EXCLUDED.expected_price,
                    updated_at = now()
            """
            records = [
                (
                    row["order_id"], row["account_id"], row.get("run_id"),
                    row.get("recommendation_id"), row["security_id"], row["order_time"],
                    row["side"], row["order_type"], int(row["quantity"]),
                    row.get("limit_price"), row["status"], row.get("expected_price"),
                    row.get("execution_mode"), row.get("broker_order_id"),
                    row.get("submitted_at"), row.get("reject_reason"),
                    row.get("price_source"), row.get("adjustment_mode"),
                    Json(_jsonable(row.get("raw_payload") or {})),
                )
                for _, row in orders.iterrows()
            ]
            with conn.cursor() as cur:
                execute_batch(cur, sql, records, page_size=500)
            conn.commit()
            return len(records)
        finally:
            conn.close()

    def persist_import_orders_for_fills(self, fills: pd.DataFrame) -> int:
        if fills.empty:
            return 0
        orders = []
        for _, row in fills.iterrows():
            orders.append(
                {
                    "order_id": row["order_id"],
                    "account_id": row["account_id"],
                    "run_id": row["run_id"],
                    "recommendation_id": row.get("recommendation_id"),
                    "security_id": row["security_id"],
                    "order_time": row["fill_time"],
                    "side": row["side"],
                    "order_type": "MARKET",
                    "quantity": int(row["fill_quantity"]),
                    "limit_price": None,
                    "status": "FILLED",
                    "expected_price": row["fill_price"],
                    "execution_mode": "broker_import",
                    "broker_order_id": None,
                    "submitted_at": row["fill_time"],
                    "reject_reason": None,
                    "price_source": row.get("price_source"),
                    "adjustment_mode": row.get("adjustment_mode"),
                    "raw_payload": {"source": "fill_import"},
                }
            )
        return self.persist_orders(pd.DataFrame(orders))

    def persist_fills(self, fills: pd.DataFrame) -> int:
        if fills.empty:
            return 0
        conn = get_pg_connection()
        try:
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
                    row.get("broker_fill_id"), row["security_id"], row["side"],
                    row["fill_time"], float(row["fill_price"]),
                    int(row["fill_quantity"]), float(row.get("commission") or 0.0),
                    float(row.get("tax") or 0.0), float(row.get("fees_total") or 0.0),
                    float(row.get("gross_notional") or 0.0),
                    _clean_float(row.get("slippage_bps")), row.get("price_source"),
                    row.get("adjustment_mode"), row.get("source_file"),
                    Json(_jsonable(row.get("raw_payload") or {})),
                )
                for _, row in fills.iterrows()
            ]
            with conn.cursor() as cur:
                execute_batch(cur, sql, records, page_size=500)
            conn.commit()
            return len(records)
        finally:
            conn.close()

    def persist_market_prices(self, prices: pd.DataFrame) -> int:
        if prices.empty:
            return 0
        conn = get_pg_connection()
        try:
            sql = """
                INSERT INTO live_market_prices
                    (account_id, as_of_date, run_id, security_id, price_time,
                     price, price_type, price_source, adjustment_mode, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (account_id, as_of_date, security_id, price_type, price_source)
                DO UPDATE SET price = EXCLUDED.price,
                              price_time = EXCLUDED.price_time,
                              adjustment_mode = EXCLUDED.adjustment_mode,
                              metadata = EXCLUDED.metadata
            """
            records = [
                (
                    row["account_id"], row["as_of_date"], row.get("run_id"),
                    row["security_id"], row["price_time"], float(row["price"]),
                    row["price_type"], row["price_source"], row["adjustment_mode"],
                    Json(_jsonable(row.get("metadata") or {})),
                )
                for _, row in prices.iterrows()
            ]
            with conn.cursor() as cur:
                execute_batch(cur, sql, records, page_size=1000)
            conn.commit()
            return len(records)
        finally:
            conn.close()

    def persist_position_snapshots(self, positions: pd.DataFrame) -> int:
        if positions.empty:
            return 0
        conn = get_pg_connection()
        try:
            sql = """
                INSERT INTO live_position_snapshots
                    (account_id, as_of_date, run_id, snapshot_time, security_id,
                     quantity, avg_cost, last_price, market_value, realized_pnl,
                     unrealized_pnl, price_source, adjustment_mode, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (account_id, as_of_date, security_id)
                DO UPDATE SET run_id = EXCLUDED.run_id,
                              snapshot_time = EXCLUDED.snapshot_time,
                              quantity = EXCLUDED.quantity,
                              avg_cost = EXCLUDED.avg_cost,
                              last_price = EXCLUDED.last_price,
                              market_value = EXCLUDED.market_value,
                              realized_pnl = EXCLUDED.realized_pnl,
                              unrealized_pnl = EXCLUDED.unrealized_pnl,
                              price_source = EXCLUDED.price_source,
                              adjustment_mode = EXCLUDED.adjustment_mode,
                              metadata = EXCLUDED.metadata,
                              updated_at = now()
            """
            records = [
                (
                    row["account_id"], row["as_of_date"], row.get("run_id"),
                    row["snapshot_time"], row["security_id"], int(row["quantity"]),
                    float(row["avg_cost"]), _clean_float(row.get("last_price")),
                    float(row["market_value"]), float(row["realized_pnl"]),
                    float(row["unrealized_pnl"]), row["price_source"],
                    row["adjustment_mode"], Json(_jsonable(row.get("metadata") or {})),
                )
                for _, row in positions.iterrows()
            ]
            with conn.cursor() as cur:
                execute_batch(cur, sql, records, page_size=1000)
            conn.commit()
            return len(records)
        finally:
            conn.close()

    def persist_account_snapshot(self, snapshot: dict[str, Any]) -> None:
        conn = get_pg_connection()
        try:
            sql = """
                INSERT INTO live_account_snapshots
                    (account_id, as_of_date, run_id, snapshot_time, cash,
                     market_value, realized_pnl, unrealized_pnl, total_equity,
                     daily_return, cumulative_return, price_source,
                     adjustment_mode, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (account_id, as_of_date)
                DO UPDATE SET run_id = EXCLUDED.run_id,
                              snapshot_time = EXCLUDED.snapshot_time,
                              cash = EXCLUDED.cash,
                              market_value = EXCLUDED.market_value,
                              realized_pnl = EXCLUDED.realized_pnl,
                              unrealized_pnl = EXCLUDED.unrealized_pnl,
                              total_equity = EXCLUDED.total_equity,
                              daily_return = EXCLUDED.daily_return,
                              cumulative_return = EXCLUDED.cumulative_return,
                              price_source = EXCLUDED.price_source,
                              adjustment_mode = EXCLUDED.adjustment_mode,
                              metadata = EXCLUDED.metadata,
                              updated_at = now()
            """
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        snapshot["account_id"], snapshot["as_of_date"],
                        snapshot.get("run_id"), snapshot["snapshot_time"],
                        snapshot["cash"], snapshot["market_value"],
                        snapshot["realized_pnl"], snapshot["unrealized_pnl"],
                        snapshot["total_equity"], snapshot.get("daily_return"),
                        snapshot.get("cumulative_return"), snapshot["price_source"],
                        snapshot["adjustment_mode"], Json(_jsonable(snapshot.get("metadata") or {})),
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    def mark_recommendations_executed(self, recommendation_ids: list[int]) -> int:
        if not recommendation_ids:
            return 0
        conn = get_pg_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE trade_recommendations
                    SET status = 'EXECUTED', updated_at = now()
                    WHERE recommendation_id = ANY(%s)
                    """,
                    [recommendation_ids],
                )
                count = cur.rowcount
            conn.commit()
            return count
        finally:
            conn.close()


def _normalize_market_prices(
    market_prices: pd.DataFrame,
    *,
    account_id: str,
    as_of_date: date,
    run_id: str | None,
    snapshot_time: datetime,
    price_source: str,
    adjustment_mode: str,
) -> pd.DataFrame:
    if market_prices.empty:
        return pd.DataFrame()
    out = market_prices.copy()
    if "price" not in out.columns and "last_price" in out.columns:
        out["price"] = out["last_price"]
    if "price" not in out.columns:
        raise ValueError("market_prices requires price or last_price column")
    out["account_id"] = out.get("account_id", account_id)
    out["as_of_date"] = out.get("as_of_date", as_of_date)
    out["run_id"] = out.get("run_id", run_id)
    out["price_time"] = out.get("price_time", snapshot_time)
    out["price_type"] = out.get("price_type", "close")
    out["price_source"] = out.get("price_source", price_source)
    out["adjustment_mode"] = out.get("adjustment_mode", adjustment_mode)
    if "metadata" not in out.columns:
        out["metadata"] = [{} for _ in range(len(out))]
    return out[
        [
            "account_id", "as_of_date", "run_id", "security_id", "price_time",
            "price", "price_type", "price_source", "adjustment_mode", "metadata",
        ]
    ].dropna(subset=["security_id", "price"])


def _positions_from_previous(previous_positions: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if previous_positions is None or previous_positions.empty:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for _, row in previous_positions.iterrows():
        qty = int(row.get("quantity") or 0)
        if qty == 0:
            continue
        out[str(row["security_id"])] = {
            "quantity": qty,
            "avg_cost": float(row.get("avg_cost") or 0.0),
            "realized_pnl": float(row.get("realized_pnl") or 0.0),
        }
    return out


def _recommendation_lookup(recommendations: pd.DataFrame | None) -> dict[tuple[str, str, str], int]:
    if recommendations is None or recommendations.empty:
        return {}
    out: dict[tuple[str, str, str], int] = {}
    for _, row in recommendations.iterrows():
        side = ACTION_TO_SIDE.get(str(row.get("action", "")).upper())
        rec_id = _clean_int(row.get("recommendation_id"))
        if side is None or rec_id is None:
            continue
        out[(str(row["run_id"]), str(row["security_id"]), side)] = rec_id
    return out


def _clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return float(value)


def _clean_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return int(value)


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value)
    return text if text else None


def _stable_import_fill_id(row: pd.Series, row_idx: Any) -> str:
    payload = {
        "row_idx": str(row_idx),
        "run_id": _clean_str(row.get("run_id")),
        "account_id": _clean_str(row.get("account_id")),
        "security_id": _clean_str(row.get("security_id")),
        "side": _clean_str(row.get("side")),
        "fill_time": str(pd.Timestamp(row.get("fill_time"))),
        "fill_price": _clean_float(row.get("fill_price")),
        "fill_quantity": _clean_int(row.get("fill_quantity")),
        "commission": _clean_float(row.get("commission")),
        "tax": _clean_float(row.get("tax")),
        "fees_total": _clean_float(row.get("fees_total")),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:20]
    return f"FIL-IMP-{digest}"


def _jsonable(payload: Any) -> Any:
    return json.loads(json.dumps(payload, ensure_ascii=False, default=str))
