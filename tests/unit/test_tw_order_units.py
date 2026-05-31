from __future__ import annotations

import pandas as pd

from src.api.routes.live import _build_export_frame
from src.live.order_units import (
    SHIOAJI_COMMON_LOT,
    SHIOAJI_INTRADAY_ODD_LOT,
    add_order_unit_columns,
    expand_to_shioaji_order_rows,
    order_plan_summary,
    split_tw_stock_order_legs,
)


def test_split_intraday_odd_order_uses_share_quantity() -> None:
    legs = split_tw_stock_order_legs(500)

    assert len(legs) == 1
    assert legs[0].order_lot == SHIOAJI_INTRADAY_ODD_LOT
    assert legs[0].shioaji_quantity == 500
    assert legs[0].share_quantity == 500
    assert legs[0].quantity_unit == "share"


def test_split_common_order_uses_board_lot_quantity() -> None:
    legs = split_tw_stock_order_legs(2_000)

    assert len(legs) == 1
    assert legs[0].order_lot == SHIOAJI_COMMON_LOT
    assert legs[0].shioaji_quantity == 2
    assert legs[0].share_quantity == 2_000
    assert legs[0].quantity_unit == "board_lot"


def test_split_mixed_order_keeps_common_and_odd_legs_separate() -> None:
    legs = split_tw_stock_order_legs(2_500)

    assert [(leg.order_lot, leg.shioaji_quantity, leg.share_quantity) for leg in legs] == [
        (SHIOAJI_COMMON_LOT, 2, 2_000),
        (SHIOAJI_INTRADAY_ODD_LOT, 500, 500),
    ]
    assert order_plan_summary(2_500) == "Common x 2 + IntradayOdd x 500"


def test_add_order_unit_columns_marks_mixed_order_without_single_quantity() -> None:
    frame = pd.DataFrame({"delta_shares": [500, 2_000, 2_500]})

    out = add_order_unit_columns(frame)

    assert out.loc[0, "shioaji_order_lot"] == SHIOAJI_INTRADAY_ODD_LOT
    assert out.loc[0, "shioaji_quantity"] == 500
    assert out.loc[1, "shioaji_order_lot"] == SHIOAJI_COMMON_LOT
    assert out.loc[1, "shioaji_quantity"] == 2
    assert pd.isna(out.loc[2, "shioaji_order_lot"])
    assert pd.isna(out.loc[2, "shioaji_quantity"])
    assert out.loc[2, "shioaji_order_plan"] == "Common x 2 + IntradayOdd x 500"


def test_export_frame_expands_mixed_recommendation_to_shioaji_order_rows() -> None:
    recommendations = pd.DataFrame(
        [
            {
                "recommendation_id": 1,
                "run_id": "11111111-1111-1111-1111-111111111111",
                "as_of_date": "2026-05-01",
                "security_id": "2330",
                "security_name": "台積電",
                "action": "BUY",
                "current_weight": 0.0,
                "target_weight": 0.25,
                "delta_weight": 0.25,
                "current_shares": 0,
                "target_shares": 2500,
                "delta_shares": 2500,
                "last_price": 100.0,
                "signal_score": 0.8,
                "rank": 1,
                "reason": "new_entry",
                "status": "APPROVED",
            }
        ]
    )

    export = _build_export_frame(recommendations)

    assert export["quantity"].tolist() == [2_000, 500]
    assert export["share_quantity"].tolist() == [2_000, 500]
    assert export["quantity_unit"].tolist() == ["share", "share"]
    assert export["shioaji_order_lot"].tolist() == [
        SHIOAJI_COMMON_LOT,
        SHIOAJI_INTRADAY_ODD_LOT,
    ]
    assert export["shioaji_quantity"].tolist() == [2, 500]
    assert export["shioaji_quantity_unit"].tolist() == ["board_lot", "share"]


def test_expand_to_shioaji_order_rows_ignores_zero_share_rows() -> None:
    frame = pd.DataFrame({"delta_shares": [0]})

    assert expand_to_shioaji_order_rows(frame).empty
