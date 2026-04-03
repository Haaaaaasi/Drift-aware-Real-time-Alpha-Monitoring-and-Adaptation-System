"""Unit tests for Layer 9: Monitoring."""

import numpy as np
import pandas as pd
import pytest

from src.monitoring.data_monitor import DataMonitor
from src.monitoring.alpha_monitor import AlphaMonitor
from src.monitoring.strategy_monitor import StrategyMonitor


class TestDataMonitor:
    def test_missing_ratio_alert(self):
        mon = DataMonitor(missing_warn=0.05, missing_crit=0.15)
        bars = pd.DataFrame({
            "security_id": ["A"] * 80,
            "tradetime": pd.date_range("2024-01-01", periods=80),
            "open": np.random.rand(80) * 100,
            "high": np.random.rand(80) * 100 + 50,
            "low": np.random.rand(80) * 50,
            "close": np.random.rand(80) * 100,
            "vol": np.random.rand(80) * 10000,
        })
        metrics = mon.run(bars, expected_count=100)
        missing_m = [m for m in metrics if m["metric_name"] == "missing_ratio"]
        assert len(missing_m) == 1
        assert missing_m[0]["metric_value"] == pytest.approx(0.2, abs=0.01)
        assert missing_m[0]["severity"] == "CRITICAL"


class TestAlphaMonitor:
    def test_produces_metrics(self):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        secs = ["A", "B", "C"]
        rows = []
        for d in dates:
            for s in secs:
                rows.append({
                    "security_id": s,
                    "tradetime": d,
                    "alpha_id": "wq001",
                    "alpha_value": np.random.randn(),
                })
        panel = pd.DataFrame(rows)
        idx = panel[["security_id", "tradetime"]].drop_duplicates().set_index(
            ["security_id", "tradetime"]
        ).index
        fwd = pd.Series(np.random.randn(len(idx)) * 0.01, index=idx)

        mon = AlphaMonitor(ic_window=10)
        metrics = mon.run(panel, fwd)
        assert len(metrics) > 0
        metric_names = {m["metric_name"] for m in metrics}
        assert "rolling_ic" in metric_names


class TestStrategyMonitor:
    def test_sharpe_alert(self):
        np.random.seed(42)
        returns = pd.Series(np.random.randn(100) * 0.01 - 0.005)
        mon = StrategyMonitor(sharpe_window=60, sharpe_warn=0.5, sharpe_crit=0.0)
        metrics = mon.run(returns)
        sharpe_m = [m for m in metrics if m["metric_name"] == "rolling_sharpe"]
        assert len(sharpe_m) == 1
