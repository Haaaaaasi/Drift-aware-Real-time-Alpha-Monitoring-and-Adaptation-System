"""Integration test for the daily batch pipeline with synthetic data."""

from datetime import date

import pytest


class TestDailyBatchPipeline:
    def test_synthetic_backtest_runs(self):
        """Verify the full pipeline executes without errors on synthetic data."""
        from pipelines.daily_batch_pipeline import run_backtest

        result = run_backtest(
            start=date(2024, 1, 1),
            end=date(2024, 3, 31),
            use_synthetic=True,
        )

        assert result["n_rebalances"] > 0
        assert result["strategy_metrics"]["n_days"] > 0
        assert "sharpe" in result["strategy_metrics"]
        assert "max_drawdown" in result["strategy_metrics"]

    def test_synthetic_data_coverage(self):
        """Verify synthetic data generator produces expected shape."""
        from pipelines.daily_batch_pipeline import generate_synthetic_data

        bars = generate_synthetic_data(date(2024, 1, 1), date(2024, 1, 31), n_symbols=5)
        assert len(bars) > 0
        assert bars["security_id"].nunique() == 5
        assert "close" in bars.columns
        assert "vol" in bars.columns

    def test_synthetic_alphas_coverage(self):
        """Verify synthetic alpha generator covers all MVP alpha IDs."""
        from pipelines.daily_batch_pipeline import (
            generate_synthetic_alphas,
            generate_synthetic_data,
        )
        from src.config.constants import MVP_V1_ALPHA_IDS

        bars = generate_synthetic_data(date(2024, 1, 1), date(2024, 1, 10), n_symbols=3)
        alphas = generate_synthetic_alphas(bars)

        assert set(alphas["alpha_id"].unique()) == set(MVP_V1_ALPHA_IDS)
