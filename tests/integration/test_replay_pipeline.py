"""WP7 — Integration tests for the replay pipeline.

Validates that expanding-window (streaming) alpha computation is consistent
with single-pass batch computation.  No external dependencies required.

Core invariant
--------------
For any rolling-window alpha f and date D:
    f(data[0..D])[-1]  ==  f(data[0..T])  evaluated at row D

This is the correctness property that the DolphinDB streamEngineParser
must satisfy, and that the offline simulation verifies end-to-end.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers: synthetic OHLCV data
# ---------------------------------------------------------------------------

def _make_synthetic_bars(
    n_days: int = 100,
    n_stocks: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a minimal synthetic OHLCV panel for testing."""
    rng = np.random.default_rng(seed)
    start = date(2023, 1, 1)
    dates = pd.bdate_range(start=start, periods=n_days, freq="B")

    rows = []
    for sid in [f"S{i:03d}" for i in range(n_stocks)]:
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days)))
        for i, d in enumerate(dates):
            open_ = close[i] * (1 + rng.normal(0, 0.003))
            high = max(open_, close[i]) * (1 + abs(rng.normal(0, 0.003)))
            low = min(open_, close[i]) * (1 - abs(rng.normal(0, 0.003)))
            volume = float(rng.integers(1_000_000, 5_000_000))
            rows.append({
                "security_id": sid,
                "tradetime": pd.Timestamp(d),
                "open": round(open_, 4),
                "high": round(high, 4),
                "low": round(low, 4),
                "close": round(close[i], 4),
                "vol": volume,           # match load_csv_data output column name
                "vwap": round((open_ + high + low + close[i]) / 4, 4),
                "cap": volume * close[i],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests: offline simulation consistency
# ---------------------------------------------------------------------------

class TestExpandingWindowConsistency:
    """Verify that expanding-window alpha output matches batch output."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.bars = _make_synthetic_bars(n_days=120, n_stocks=5, seed=0)
        self.dates = sorted(self.bars["tradetime"].unique())

    def _batch_alphas(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Run Python batch alphas via daily_batch_pipeline helper."""
        from pipelines.daily_batch_pipeline import compute_python_alphas
        return compute_python_alphas(bars)

    def test_batch_produces_long_format(self):
        panel = self._batch_alphas(self.bars)
        required_cols = {"security_id", "tradetime", "alpha_id", "alpha_value"}
        assert required_cols.issubset(panel.columns), (
            f"Missing columns: {required_cols - set(panel.columns)}"
        )

    def test_batch_covers_all_dates(self):
        panel = self._batch_alphas(self.bars)
        batch_dates = set(panel["tradetime"].unique())
        data_dates = set(self.bars["tradetime"].unique())
        # Every date in bars should appear in at least one alpha row
        assert data_dates.issubset(batch_dates)

    def test_expanding_window_matches_batch_at_checkpoint(self):
        """At checkpoint date D, expanding alpha == batch alpha for date D."""
        panel_full = self._batch_alphas(self.bars)

        # Pick checkpoint at day 90 (enough warm-up for rolling windows)
        checkpoint = self.dates[89]
        history = self.bars[self.bars["tradetime"] <= checkpoint]
        panel_ckpt = self._batch_alphas(history)

        # For each alpha, compare values at checkpoint date
        alpha_ids = panel_full["alpha_id"].unique()
        mismatches = []

        for aid in alpha_ids:
            full_vals = (
                panel_full[
                    (panel_full["alpha_id"] == aid)
                    & (panel_full["tradetime"] == checkpoint)
                ]
                .set_index("security_id")["alpha_value"]
                .dropna()
            )
            ckpt_vals = (
                panel_ckpt[
                    (panel_ckpt["alpha_id"] == aid)
                    & (panel_ckpt["tradetime"] == checkpoint)
                ]
                .set_index("security_id")["alpha_value"]
                .dropna()
            )
            common = full_vals.index.intersection(ckpt_vals.index)
            if len(common) == 0:
                continue
            diff = (full_vals.loc[common] - ckpt_vals.loc[common]).abs()
            if diff.max() > 1e-6:
                mismatches.append((aid, float(diff.max())))

        assert len(mismatches) == 0, (
            f"Batch vs expanding mismatch for alphas: {mismatches}"
        )

    def test_multiple_checkpoints_all_consistent(self):
        """All 3 sampled checkpoints should produce matching values."""
        panel_full = self._batch_alphas(self.bars)
        checkpoints = [self.dates[60], self.dates[90], self.dates[110]]

        for checkpoint in checkpoints:
            history = self.bars[self.bars["tradetime"] <= checkpoint]
            panel_ckpt = self._batch_alphas(history)

            for aid in panel_full["alpha_id"].unique():
                full_vals = (
                    panel_full[
                        (panel_full["alpha_id"] == aid)
                        & (panel_full["tradetime"] == checkpoint)
                    ]
                    .set_index("security_id")["alpha_value"]
                    .dropna()
                )
                ckpt_vals = (
                    panel_ckpt[
                        (panel_ckpt["alpha_id"] == aid)
                        & (panel_ckpt["tradetime"] == checkpoint)
                    ]
                    .set_index("security_id")["alpha_value"]
                    .dropna()
                )
                common = full_vals.index.intersection(ckpt_vals.index)
                if len(common) == 0:
                    continue
                diff = (full_vals.loc[common] - ckpt_vals.loc[common]).abs().max()
                assert diff < 1e-6, (
                    f"Alpha {aid} at {checkpoint}: max diff {diff:.2e}"
                )

    def test_expanding_window_does_not_see_future_data(self):
        """Values at date D must be identical when computed with D+1 history."""
        # This is the no-lookahead property: adding future data must not
        # change past values.
        checkpoint = self.dates[80]
        future_date = self.dates[81]

        history_d = self.bars[self.bars["tradetime"] <= checkpoint]
        history_d1 = self.bars[self.bars["tradetime"] <= future_date]

        panel_d = self._batch_alphas(history_d)
        panel_d1 = self._batch_alphas(history_d1)

        for aid in panel_d["alpha_id"].unique():
            vals_d = (
                panel_d[
                    (panel_d["alpha_id"] == aid)
                    & (panel_d["tradetime"] == checkpoint)
                ]
                .set_index("security_id")["alpha_value"]
                .dropna()
            )
            vals_d1 = (
                panel_d1[
                    (panel_d1["alpha_id"] == aid)
                    & (panel_d1["tradetime"] == checkpoint)
                ]
                .set_index("security_id")["alpha_value"]
                .dropna()
            )
            common = vals_d.index.intersection(vals_d1.index)
            if len(common) == 0:
                continue
            diff = (vals_d.loc[common] - vals_d1.loc[common]).abs().max()
            assert diff < 1e-6, (
                f"Alpha {aid}: adding future bar changed value at {checkpoint} "
                f"(max diff {diff:.2e}) — possible lookahead bias!"
            )


# ---------------------------------------------------------------------------
# Tests: replay pipeline offline mode (end-to-end)
# ---------------------------------------------------------------------------

class TestReplayPipelineOffline:
    """End-to-end test of run_replay_offline using a temp CSV file."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        bars = _make_synthetic_bars(n_days=120, n_stocks=5, seed=7)
        # Write to temp CSV in the format expected by load_csv_data
        # load_csv_data expects: datetime, security_id, open, high, low, close, volume
        # (it renames volume→vol and datetime→tradetime internally)
        csv_path = tmp_path / "test_bars.csv"
        bars_csv = bars.copy()
        bars_csv = bars_csv.rename(columns={"tradetime": "datetime", "vol": "volume"})
        bars_csv["datetime"] = bars_csv["datetime"].dt.strftime("%Y-%m-%d")
        bars_csv["bar_type"] = "daily"
        bars_csv["close_adj"] = bars_csv["close"]
        bars_csv.to_csv(csv_path, index=False)
        self.csv_path = csv_path
        self.bars = bars

    def test_replay_offline_returns_summary(self):
        from pipelines.replay_pipeline import run_replay_offline
        result = run_replay_offline(
            self.csv_path,
            start=date(2023, 1, 1),
            end=date(2024, 12, 31),
            n_checkpoints=3,
            warm_up_days=20,
        )
        assert "overall_match_rate" in result
        assert "per_alpha_match_rate" in result
        assert "checkpoints_evaluated" in result
        assert result["checkpoints_evaluated"] <= 3

    def test_replay_offline_high_match_rate(self):
        """Match rate should be essentially 1.0 for pure rolling-window alphas."""
        from pipelines.replay_pipeline import run_replay_offline
        result = run_replay_offline(
            self.csv_path,
            start=date(2023, 1, 1),
            end=date(2024, 12, 31),
            n_checkpoints=4,
            warm_up_days=25,
        )
        assert result["overall_match_rate"] >= 0.95, (
            f"Expected match rate ≥ 0.95, got {result['overall_match_rate']:.4f}\n"
            f"Per-alpha: {result['per_alpha_match_rate']}"
        )

    def test_replay_offline_streaming_rows_match_expected(self):
        """Streaming panel should have one row per (security, date) per alpha."""
        from pipelines.replay_pipeline import run_replay_offline
        n_stocks = 5
        n_checkpoints = 3
        result = run_replay_offline(
            self.csv_path,
            start=date(2023, 1, 1),
            end=date(2024, 12, 31),
            n_checkpoints=n_checkpoints,
            warm_up_days=20,
        )
        n_alphas = result["alphas_checked"]
        # Each checkpoint × n_stocks × n_alphas rows (some may be NaN-dropped)
        assert result["streaming_rows"] > 0
        assert result["streaming_rows"] <= n_checkpoints * n_stocks * n_alphas

    def test_replay_offline_batch_rows_exceed_streaming(self):
        """Batch runs on the full dataset so has more rows than the checkpoints."""
        from pipelines.replay_pipeline import run_replay_offline
        result = run_replay_offline(
            self.csv_path,
            start=date(2023, 1, 1),
            end=date(2024, 12, 31),
            n_checkpoints=3,
            warm_up_days=20,
        )
        assert result["batch_rows"] > result["streaming_rows"]


# ---------------------------------------------------------------------------
# Tests: StreamAlphaComputer (Python-side, no DolphinDB)
# ---------------------------------------------------------------------------

class TestStreamAlphaComputerInterface:
    """Unit tests for StreamAlphaComputer that don't require DolphinDB."""

    def test_output_buffer_starts_empty(self):
        """Output buffer should be empty before any bars are pushed."""
        from unittest.mock import patch, MagicMock
        with patch("src.alpha_engine.stream_compute.DolphinDBClient"):
            from src.alpha_engine.stream_compute import StreamAlphaComputer
            comp = StreamAlphaComputer()
            assert comp.collect_output() == []

    def test_collect_output_returns_buffer_contents(self):
        """collect_output should drain the internal buffer."""
        from unittest.mock import patch
        with patch("src.alpha_engine.stream_compute.DolphinDBClient"):
            from src.alpha_engine.stream_compute import StreamAlphaComputer
            comp = StreamAlphaComputer()
            comp._output_buffer.append({"alpha_id": "wq002", "value": 0.5})
            comp._output_buffer.append({"alpha_id": "wq003", "value": 0.3})
            result = comp.collect_output()
            assert len(result) == 2
            assert result[0]["alpha_id"] == "wq002"

    def test_clear_output_empties_buffer(self):
        from unittest.mock import patch
        with patch("src.alpha_engine.stream_compute.DolphinDBClient"):
            from src.alpha_engine.stream_compute import StreamAlphaComputer
            comp = StreamAlphaComputer()
            comp._output_buffer.append({"x": 1})
            comp.clear_output()
            assert comp.collect_output() == []

    def test_engine_not_active_on_init(self):
        from unittest.mock import patch
        with patch("src.alpha_engine.stream_compute.DolphinDBClient"):
            from src.alpha_engine.stream_compute import StreamAlphaComputer
            comp = StreamAlphaComputer()
            assert comp._engine_active is False

    def test_stop_engine_when_not_active_is_safe(self):
        """Calling stop_engine before setup_engine should not raise."""
        from unittest.mock import patch
        with patch("src.alpha_engine.stream_compute.DolphinDBClient"):
            from src.alpha_engine.stream_compute import StreamAlphaComputer
            comp = StreamAlphaComputer()
            comp.stop_engine()   # should not raise
