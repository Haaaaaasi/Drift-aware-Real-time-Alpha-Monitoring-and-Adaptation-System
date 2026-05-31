"""Taiwan stock order unit conversion helpers for live trading exports.

Internal accounting uses shares. Shioaji stock orders use different quantity
units depending on order_lot: Common uses board lots, IntradayOdd uses shares.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

TW_BOARD_LOT_SHARES = 1000
SHIOAJI_COMMON_LOT = "Common"
SHIOAJI_INTRADAY_ODD_LOT = "IntradayOdd"


@dataclass(frozen=True)
class TwStockOrderLeg:
    """One Shioaji-compatible order leg derived from a desired share quantity."""

    order_lot: str
    shioaji_quantity: int
    share_quantity: int

    @property
    def quantity_unit(self) -> str:
        return "board_lot" if self.order_lot == SHIOAJI_COMMON_LOT else "share"


def split_tw_stock_order_legs(
    share_quantity: int | float,
    *,
    board_lot_shares: int = TW_BOARD_LOT_SHARES,
) -> list[TwStockOrderLeg]:
    """Split a Taiwan stock share quantity into Shioaji order legs.

    Examples:
    - 500 shares -> IntradayOdd quantity=500
    - 1000 shares -> Common quantity=1
    - 2500 shares -> Common quantity=2 + IntradayOdd quantity=500
    """

    shares = abs(_to_int(share_quantity))
    if shares == 0:
        return []
    if board_lot_shares <= 0:
        raise ValueError("board_lot_shares must be positive")

    common_lots, odd_shares = divmod(shares, board_lot_shares)
    legs: list[TwStockOrderLeg] = []
    if common_lots:
        legs.append(
            TwStockOrderLeg(
                order_lot=SHIOAJI_COMMON_LOT,
                shioaji_quantity=common_lots,
                share_quantity=common_lots * board_lot_shares,
            )
        )
    if odd_shares:
        legs.append(
            TwStockOrderLeg(
                order_lot=SHIOAJI_INTRADAY_ODD_LOT,
                shioaji_quantity=odd_shares,
                share_quantity=odd_shares,
            )
        )
    return legs


def order_plan_summary(
    share_quantity: int | float,
    *,
    board_lot_shares: int = TW_BOARD_LOT_SHARES,
) -> str:
    """Return a compact human-readable order lot summary."""

    legs = split_tw_stock_order_legs(
        share_quantity,
        board_lot_shares=board_lot_shares,
    )
    if not legs:
        return "NO_ORDER"
    return " + ".join(
        f"{leg.order_lot} x {leg.shioaji_quantity}" for leg in legs
    )


def add_order_unit_columns(
    frame: pd.DataFrame,
    *,
    quantity_col: str = "delta_shares",
    board_lot_shares: int = TW_BOARD_LOT_SHARES,
) -> pd.DataFrame:
    """Attach share and Shioaji order unit summary columns to a DataFrame."""

    out = frame.copy()
    share_quantities = out.get(quantity_col, pd.Series(dtype="int64")).fillna(0).abs().astype(int)
    out["share_quantity"] = share_quantities
    out["quantity_unit"] = "share"
    out["tw_board_lot_shares"] = board_lot_shares
    out["tw_common_lots"] = share_quantities // board_lot_shares
    out["tw_odd_lot_shares"] = share_quantities % board_lot_shares
    out["shioaji_order_plan"] = [
        order_plan_summary(qty, board_lot_shares=board_lot_shares)
        for qty in share_quantities
    ]
    out["shioaji_order_lot"] = [
        _single_leg_attr(qty, "order_lot", board_lot_shares=board_lot_shares)
        for qty in share_quantities
    ]
    out["shioaji_quantity"] = [
        _single_leg_attr(qty, "shioaji_quantity", board_lot_shares=board_lot_shares)
        for qty in share_quantities
    ]
    return out


def expand_to_shioaji_order_rows(
    frame: pd.DataFrame,
    *,
    quantity_col: str = "delta_shares",
    board_lot_shares: int = TW_BOARD_LOT_SHARES,
) -> pd.DataFrame:
    """Expand recommendation rows into one row per Shioaji order leg."""

    rows: list[dict] = []
    for _, row in frame.iterrows():
        for leg in split_tw_stock_order_legs(
            row.get(quantity_col, 0),
            board_lot_shares=board_lot_shares,
        ):
            record = row.to_dict()
            record["share_quantity"] = leg.share_quantity
            record["quantity_unit"] = "share"
            record["tw_board_lot_shares"] = board_lot_shares
            record["shioaji_order_lot"] = leg.order_lot
            record["shioaji_quantity"] = leg.shioaji_quantity
            record["shioaji_quantity_unit"] = leg.quantity_unit
            record["shioaji_order_plan"] = f"{leg.order_lot} x {leg.shioaji_quantity}"
            rows.append(record)
    return pd.DataFrame(rows)


def _single_leg_attr(
    share_quantity: int,
    attr: str,
    *,
    board_lot_shares: int,
) -> str | int | None:
    legs = split_tw_stock_order_legs(
        share_quantity,
        board_lot_shares=board_lot_shares,
    )
    if len(legs) != 1:
        return None
    return getattr(legs[0], attr)


def _to_int(value: int | float) -> int:
    if pd.isna(value):
        return 0
    return int(round(float(value)))
