from __future__ import annotations

import pandas as pd

from src.monitoring.live_pnl_monitor import LivePnLMonitor


def test_live_pnl_monitor_emits_scoped_metrics() -> None:
    snapshots = pd.DataFrame(
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
    orders = pd.DataFrame({"order_id": ["o1", "o2"]})
    fills = pd.DataFrame(
        {
            "order_id": ["o1"],
            "gross_notional": [100_000.0],
            "fees_total": [50.0],
            "slippage_bps": [2.0],
        }
    )

    metrics = LivePnLMonitor().run(
        account_snapshots=snapshots,
        orders=orders,
        fills=fills,
        account_id="paper_main",
        model_id="ml_xgb_test",
    )

    by_name = {m["metric_name"]: m for m in metrics}
    assert by_name["daily_return"]["severity"] == "CRITICAL"
    assert by_name["daily_return"]["account_id"] == "paper_main"
    assert by_name["daily_return"]["model_id"] == "ml_xgb_test"
    assert by_name["fill_rate"]["metric_value"] == 0.5
    assert by_name["cost_bps"]["metric_value"] == 5.0
