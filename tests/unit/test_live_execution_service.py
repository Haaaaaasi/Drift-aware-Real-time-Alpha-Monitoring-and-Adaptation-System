from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pandas as pd

from src.live.execution_service import (
    ReconciliationResult,
    LiveExecutionService,
    build_orders_from_recommendations,
    build_paper_fills_from_orders,
    normalize_fill_import,
    reconcile_account,
)


def _recommendations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "recommendation_id": 1,
                "run_id": "11111111-1111-1111-1111-111111111111",
                "as_of_date": pd.Timestamp("2026-05-01").date(),
                "security_id": "2330",
                "action": "BUY",
                "status": "APPROVED",
                "delta_shares": 100,
                "last_price": 100.0,
                "target_weight": 0.5,
                "delta_weight": 0.5,
            },
            {
                "recommendation_id": 2,
                "run_id": "11111111-1111-1111-1111-111111111111",
                "as_of_date": pd.Timestamp("2026-05-01").date(),
                "security_id": "2317",
                "action": "HOLD",
                "status": "APPROVED",
                "delta_shares": 0,
                "last_price": 50.0,
                "target_weight": 0.0,
                "delta_weight": 0.0,
            },
        ]
    )


def test_build_orders_from_approved_recommendations() -> None:
    orders = build_orders_from_recommendations(_recommendations(), account_id="paper_main")

    assert len(orders) == 1
    row = orders.iloc[0]
    assert row["side"] == "BUY"
    assert row["quantity"] == 100
    assert row["recommendation_id"] == 1
    assert row["price_source"] == "paper_next_vwap"
    assert row["adjustment_mode"] == "raw"
    assert row["raw_payload"]["quantity_unit"] == "share"
    assert row["raw_payload"]["shioaji_order_lot"] == "IntradayOdd"
    assert row["raw_payload"]["shioaji_quantity"] == 100
    assert row["raw_payload"]["shioaji_quantity_unit"] == "share"


def test_build_orders_splits_common_and_intraday_odd_lots() -> None:
    recommendations = _recommendations()
    recommendations.loc[0, "delta_shares"] = 2_500

    orders = build_orders_from_recommendations(recommendations, account_id="paper_main")

    assert len(orders) == 2
    by_lot = {
        row["raw_payload"]["shioaji_order_lot"]: row
        for _, row in orders.iterrows()
    }
    common = by_lot["Common"]
    odd = by_lot["IntradayOdd"]
    assert common["quantity"] == 2_000
    assert common["raw_payload"]["shioaji_quantity"] == 2
    assert common["raw_payload"]["shioaji_quantity_unit"] == "board_lot"
    assert odd["quantity"] == 500
    assert odd["raw_payload"]["shioaji_quantity"] == 500
    assert odd["raw_payload"]["shioaji_quantity_unit"] == "share"


def test_paper_fills_include_cost_and_tax_fields() -> None:
    orders = build_orders_from_recommendations(_recommendations(), account_id="paper_main")
    fills = build_paper_fills_from_orders(
        orders,
        commission_rate=0.001,
        tax_rate=0.003,
        slippage_bps=10,
    )

    assert len(fills) == 1
    row = fills.iloc[0]
    assert row["side"] == "BUY"
    assert row["fill_price"] == 100.1
    assert row["commission"] == 10.01
    assert row["tax"] == 0.0
    assert row["fees_total"] == 10.01


def test_fill_import_maps_recommendation_when_missing_id() -> None:
    raw = pd.DataFrame(
        [
            {
                "run_id": "11111111-1111-1111-1111-111111111111",
                "security_id": "2330",
                "side": "BUY",
                "fill_time": "2026-05-02 09:00:00",
                "fill_price": 101.0,
                "fill_quantity": 100,
            }
        ]
    )

    fills = normalize_fill_import(raw, recommendations=_recommendations())

    assert fills.iloc[0]["recommendation_id"] == 1
    assert fills.iloc[0]["price_source"] == "broker_import"
    assert fills.iloc[0]["adjustment_mode"] == "raw"


def test_fill_import_without_broker_id_uses_stable_fill_id() -> None:
    raw = pd.DataFrame(
        [
            {
                "run_id": "11111111-1111-1111-1111-111111111111",
                "security_id": "2330",
                "side": "BUY",
                "fill_time": "2026-05-02 09:00:00",
                "fill_price": 101.0,
                "fill_quantity": 100,
            }
        ]
    )

    first = normalize_fill_import(raw, recommendations=_recommendations())
    second = normalize_fill_import(raw, recommendations=_recommendations())

    assert first.iloc[0]["fill_id"] == second.iloc[0]["fill_id"]
    assert first.iloc[0]["fill_id"].startswith("FIL-IMP-")


def test_reconcile_account_average_cost_and_sell_realized_pnl() -> None:
    previous_positions = pd.DataFrame(
        [
            {
                "security_id": "2330",
                "quantity": 100,
                "avg_cost": 100.0,
                "realized_pnl": 0.0,
            }
        ]
    )
    fills = pd.DataFrame(
        [
            {
                "security_id": "2330",
                "side": "SELL",
                "fill_time": pd.Timestamp("2026-05-02 09:00"),
                "fill_price": 110.0,
                "fill_quantity": 40,
                "fees_total": 20.0,
                "recommendation_id": 1,
            }
        ]
    )
    prices = pd.DataFrame({"security_id": ["2330"], "price": [120.0]})

    result = reconcile_account(
        account_id="paper_main",
        as_of_date=pd.Timestamp("2026-05-02"),
        fills=fills,
        market_prices=prices,
        previous_positions=previous_positions,
        previous_account={"cash": 0.0, "total_equity": 10_000.0, "realized_pnl": 0.0},
        initial_capital=10_000.0,
        run_id="11111111-1111-1111-1111-111111111111",
    )

    pos = result.positions.set_index("security_id").loc["2330"]
    assert pos["quantity"] == 60
    assert pos["avg_cost"] == 100.0
    assert pos["unrealized_pnl"] == 1200.0
    assert result.account_snapshot["realized_pnl"] == 380.0
    assert result.account_snapshot["cash"] == 4380.0
    assert result.account_snapshot["total_equity"] == 11580.0
    assert math.isclose(result.account_snapshot["cumulative_return"], 0.158)
    assert result.executed_recommendation_ids == [1]


def test_emit_live_pnl_metrics_persists_metrics_and_alerts() -> None:
    service = LiveExecutionService(account_id="paper_main")
    result = ReconciliationResult(
        positions=pd.DataFrame(),
        account_snapshot={
            "account_id": "paper_main",
            "as_of_date": pd.Timestamp("2026-05-02").date(),
            "snapshot_time": pd.Timestamp("2026-05-02 16:00"),
            "cash": 9_400_000.0,
            "market_value": 0.0,
            "realized_pnl": -600_000.0,
            "unrealized_pnl": 0.0,
            "total_equity": 9_400_000.0,
            "daily_return": -0.06,
            "cumulative_return": -0.06,
            "price_source": "paper_next_vwap",
            "adjustment_mode": "raw",
        },
        market_prices=pd.DataFrame(),
        executed_recommendation_ids=[],
    )
    service.load_account_snapshots_until = MagicMock(
        return_value=pd.DataFrame(
            [
                {
                    "account_id": "paper_main",
                    "as_of_date": pd.Timestamp("2026-05-01"),
                    "run_id": "11111111-1111-1111-1111-111111111111",
                    "snapshot_time": pd.Timestamp("2026-05-01 16:00"),
                    "total_equity": 10_000_000.0,
                    "daily_return": 0.0,
                    "cumulative_return": 0.0,
                },
                {
                    "account_id": "paper_main",
                    "as_of_date": pd.Timestamp("2026-05-02"),
                    "run_id": "22222222-2222-2222-2222-222222222222",
                    "snapshot_time": pd.Timestamp("2026-05-02 16:00"),
                    "total_equity": 9_400_000.0,
                    "daily_return": -0.06,
                    "cumulative_return": -0.06,
                },
            ]
        )
    )
    service.load_orders = MagicMock(return_value=pd.DataFrame({"order_id": ["o1"]}))
    service.load_fills = MagicMock(
        return_value=pd.DataFrame(
            {
                "order_id": ["o1"],
                "gross_notional": [100_000.0],
                "fees_total": [50.0],
                "slippage_bps": [2.0],
            }
        )
    )
    service.load_recommendations = MagicMock(return_value=_recommendations())
    alert_mgr = MagicMock()
    alert_mgr.persist_metrics.return_value = 8
    alert_mgr.fire_alerts.return_value = 1

    with patch("src.monitoring.alert_manager.AlertManager", return_value=alert_mgr):
        summary = service.emit_live_pnl_metrics(
            run={
                "run_id": "22222222-2222-2222-2222-222222222222",
                "production_model_id": "ml_xgb_test",
                "frozen_selector_id": "rolling_topk20",
            },
            result=result,
        )

    assert summary == {"metrics": 8, "alerts": 1}
    metrics = alert_mgr.persist_metrics.call_args.args[0]
    assert {m["metric_name"] for m in metrics} >= {
        "daily_return",
        "rolling_sharpe",
        "max_drawdown",
        "fill_rate",
    }
    assert any(m.get("severity") == "CRITICAL" for m in metrics)
    alert_mgr.fire_alerts.assert_called_once_with(metrics)
