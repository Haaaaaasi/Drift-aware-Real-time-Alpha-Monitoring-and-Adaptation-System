"""Layer 7 — Paper trading engine for simulated execution."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from src.common.db import get_pg_connection
from src.common.logging import get_logger
from src.config.constants import OrderSide, OrderStatus

logger = get_logger(__name__)


class PaperTradingEngine:
    """Simulate order execution with configurable slippage.

    For each rebalance:
    1. Compare target weights with current positions
    2. Generate BUY/SELL orders for the delta
    3. Simulate fills at close price + slippage
    4. Update position snapshots
    """

    def __init__(
        self,
        initial_capital: float = 10_000_000.0,
        slippage_bps: float = 5.0,
        commission_rate: float = 0.000926,
        tax_rate: float = 0.003,
    ) -> None:
        self._capital = initial_capital
        self._slippage_bps = slippage_bps
        self._commission_rate = commission_rate
        self._tax_rate = tax_rate
        self._positions: dict[str, dict[str, Any]] = {}
        self._cash = initial_capital

    def execute_rebalance(
        self,
        targets: pd.DataFrame,
        market_prices: pd.DataFrame,
        rebalance_time: datetime,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Execute a full rebalance cycle.

        Args:
            targets: Risk-adjusted targets [security_id, target_weight].
            market_prices: Current prices [security_id, close].
            rebalance_time: Timestamp of this rebalance.

        Returns:
            Tuple of (orders_df, fills_df).
        """
        portfolio_value = self._calculate_portfolio_value(market_prices)
        orders = []
        fills = []

        for _, row in targets.iterrows():
            sec_id = row["security_id"]
            target_weight = row["target_weight"]

            price_row = market_prices[market_prices["security_id"] == sec_id]
            if price_row.empty:
                continue
            price = float(price_row["close"].iloc[0])
            if price <= 0:
                continue

            target_value = portfolio_value * target_weight
            current_value = self._positions.get(sec_id, {}).get("market_value", 0.0)
            delta_value = target_value - current_value

            if abs(delta_value) < price * 0.5:  # skip if less than half a share
                continue

            quantity = int(delta_value / price)
            if quantity == 0:
                continue

            side = OrderSide.BUY if quantity > 0 else OrderSide.SELL
            order_id = f"ORD-{uuid4().hex[:12]}"

            orders.append({
                "order_id": order_id,
                "security_id": sec_id,
                "order_time": rebalance_time,
                "side": side.value,
                "order_type": "MARKET",
                "quantity": abs(quantity),
                "limit_price": None,
                "status": OrderStatus.FILLED.value,
                "expected_price": price,
            })

            # Simulate fill with slippage
            slip_mult = 1 + self._slippage_bps / 10000 * (1 if side == OrderSide.BUY else -1)
            fill_price = price * slip_mult
            fill_qty = abs(quantity)
            commission = fill_price * fill_qty * self._commission_rate
            tax = fill_price * fill_qty * self._tax_rate if side == OrderSide.SELL else 0.0
            slippage = (fill_price - price) / price * 10000

            fills.append({
                "fill_id": f"FIL-{uuid4().hex[:12]}",
                "order_id": order_id,
                "security_id": sec_id,
                "fill_time": rebalance_time,
                "fill_price": fill_price,
                "fill_quantity": fill_qty,
                "commission": commission,
                "tax": tax,
                "slippage_bps": slippage,
            })

            self._update_position(sec_id, quantity, fill_price, commission + tax)

        orders_df = pd.DataFrame(orders)
        fills_df = pd.DataFrame(fills)

        logger.info(
            "rebalance_executed",
            orders=len(orders),
            fills=len(fills),
            portfolio_value=round(portfolio_value, 2),
        )
        return orders_df, fills_df

    def _update_position(
        self, sec_id: str, quantity: int, price: float, commission: float
    ) -> None:
        """Update internal position state after a fill."""
        pos = self._positions.get(sec_id, {"quantity": 0, "avg_cost": 0.0})
        old_qty = pos["quantity"]
        new_qty = old_qty + quantity

        if new_qty == 0:
            self._positions.pop(sec_id, None)
        else:
            if quantity > 0:  # buying
                total_cost = pos["avg_cost"] * old_qty + price * quantity + commission
                pos["avg_cost"] = total_cost / new_qty if new_qty != 0 else 0
            pos["quantity"] = new_qty
            self._positions[sec_id] = pos

        cost = price * abs(quantity) + commission
        if quantity > 0:
            self._cash -= cost
        else:
            self._cash += price * abs(quantity) - commission

    def _calculate_portfolio_value(self, market_prices: pd.DataFrame) -> float:
        """Total portfolio value = cash + sum of position market values."""
        total = self._cash
        price_map = dict(zip(market_prices["security_id"], market_prices["close"]))
        for sec_id, pos in self._positions.items():
            p = price_map.get(sec_id, pos["avg_cost"])
            pos["market_value"] = p * pos["quantity"]
            total += pos["market_value"]
        return total

    def get_positions_snapshot(self, snapshot_time: datetime) -> pd.DataFrame:
        """Return current positions as a DataFrame."""
        rows = []
        for sec_id, pos in self._positions.items():
            rows.append({
                "snapshot_time": snapshot_time,
                "security_id": sec_id,
                "quantity": pos["quantity"],
                "avg_cost": pos["avg_cost"],
                "market_value": pos.get("market_value", 0.0),
                "unrealized_pnl": pos.get("market_value", 0.0) - pos["avg_cost"] * pos["quantity"],
            })
        return pd.DataFrame(rows)

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def portfolio_value(self) -> float:
        mv = sum(p.get("market_value", 0.0) for p in self._positions.values())
        return self._cash + mv
