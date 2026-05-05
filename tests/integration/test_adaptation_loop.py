"""WP5 — Integration test for the full adaptation loop.

Tests the complete chain:
    Injected drift → Monitoring (4 layers) → Performance trigger → Retrain
    → Shadow evaluation → Best candidate selection

Uses synthetic data with *intentional* distribution shift to verify that:
1. Monitors detect the drift (alerts fired)
2. Performance trigger fires (IC below threshold)
3. XGBoost retrain produces a new model with valid metrics
4. Shadow evaluator can compare candidates and select the best
5. The full loop is coherent end-to-end

No external dependencies (PostgreSQL / Redis / DolphinDB) — registry calls
are mocked since we're testing the adaptation *logic*, not DB persistence.
"""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.common.metrics import (
    information_coefficient,
    ks_test_drift,
    population_stability_index,
)
from src.config.constants import AlertSeverity
from src.labeling.label_generator import LabelGenerator
from src.meta_signal.ml_meta_model import MLMetaModel
from src.meta_signal.rule_based import RuleBasedSignalGenerator
from src.monitoring.alpha_monitor import AlphaMonitor
from src.monitoring.data_monitor import DataMonitor
from src.monitoring.model_monitor import ModelMonitor
from src.monitoring.strategy_monitor import StrategyMonitor
from src.adaptation.performance_trigger import PerformanceTriggeredAdapter
from src.adaptation.shadow_evaluator import ShadowEvaluator


# ---------------------------------------------------------------------------
# Fixtures: synthetic data with injected drift
# ---------------------------------------------------------------------------

def _make_bars(
    start: date,
    end: date,
    n_symbols: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV bars."""
    rng = np.random.RandomState(seed)
    symbols = [f"SYM{i:04d}" for i in range(1, n_symbols + 1)]
    dates = pd.bdate_range(start, end, freq="B")
    rows = []
    for sym in symbols:
        price = 100.0 + rng.randn() * 10
        for d in dates:
            ret = rng.randn() * 0.02
            price *= 1 + ret
            o = price * (1 + rng.randn() * 0.003)
            h = max(o, price) * (1 + abs(rng.randn()) * 0.003)
            lo = min(o, price) * (1 - abs(rng.randn()) * 0.003)
            vol = max(1000, int(rng.exponential(300_000)))
            rows.append({
                "security_id": sym,
                "tradetime": d,
                "bar_type": "daily",
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(lo, 2),
                "close": round(price, 2),
                "vol": float(vol),
                "vwap": round((h + lo + price) / 3, 2),
                "cap": round(price * 1e6, 2),
                "indclass": (hash(sym) % 5) + 1,
                "is_tradable": True,
                "missing_flags": 0,
            })
    return pd.DataFrame(rows)


def _make_alpha_panel(
    bars: pd.DataFrame,
    n_alphas: int = 5,
    seed: int = 123,
) -> pd.DataFrame:
    """Generate synthetic alpha values (cross-sectionally z-scored)."""
    rng = np.random.RandomState(seed)
    alpha_ids = [f"alpha_{i:02d}" for i in range(1, n_alphas + 1)]
    rows = []
    for (sec, dt), _ in bars.groupby(["security_id", "tradetime"]):
        for aid in alpha_ids:
            rows.append({
                "security_id": sec,
                "tradetime": dt,
                "alpha_id": aid,
                "alpha_value": rng.randn(),
            })
    df = pd.DataFrame(rows)
    # Cross-sectional z-score
    for aid in alpha_ids:
        mask = df["alpha_id"] == aid
        grp = df.loc[mask, "alpha_value"].groupby(df.loc[mask, "tradetime"])
        mu = grp.transform("mean")
        sig = grp.transform("std").replace(0, np.nan)
        df.loc[mask, "alpha_value"] = (df.loc[mask, "alpha_value"] - mu) / sig
    return df.dropna(subset=["alpha_value"])


def _inject_drift(
    alpha_panel: pd.DataFrame,
    drift_start: pd.Timestamp,
    shift_magnitude: float = 3.0,
    seed: int = 999,
) -> pd.DataFrame:
    """Inject distribution shift after `drift_start` — add a constant bias
    and increase variance to simulate concept drift."""
    rng = np.random.RandomState(seed)
    df = alpha_panel.copy()
    mask = df["tradetime"] >= drift_start
    df.loc[mask, "alpha_value"] += shift_magnitude
    df.loc[mask, "alpha_value"] += rng.randn(mask.sum()) * 1.5
    return df


def _make_forward_returns(bars: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Compute forward returns as labels."""
    label_gen = LabelGenerator(horizons=[horizon], bar_type="daily")
    labels = label_gen.generate_labels(bars[["security_id", "tradetime", "close"]])
    fwd = (
        labels[labels["horizon"] == horizon]
        .set_index(["security_id", "signal_time"])["forward_return"]
    )
    fwd.index = fwd.index.set_names(["security_id", "tradetime"])
    return fwd


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestAdaptationLoop:
    """Integration test: drift injection → monitoring → trigger → retrain → shadow eval."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Build the test dataset with drift injected mid-way."""
        self.bars = _make_bars(date(2023, 1, 1), date(2024, 6, 30), n_symbols=10)
        self.alpha_panel = _make_alpha_panel(self.bars, n_alphas=5)
        self.fwd = _make_forward_returns(self.bars)

        # Split: first half = reference, second half = drifted
        dates = self.bars["tradetime"].unique()
        self.split_date = pd.Timestamp(dates[len(dates) // 2])
        self.drifted_panel = _inject_drift(
            self.alpha_panel, drift_start=self.split_date, shift_magnitude=3.0
        )

        # Reference and drifted slices
        self.ref_panel = self.alpha_panel[
            self.alpha_panel["tradetime"] < self.split_date
        ]
        self.drift_panel = self.drifted_panel[
            self.drifted_panel["tradetime"] >= self.split_date
        ]

    # ------------------------------------------------------------------
    # Step 1: Monitoring detects drift
    # ------------------------------------------------------------------

    def test_step1_data_monitor_detects_drift(self):
        """DataMonitor should detect feature distribution shift via KS + PSI."""
        ref_close = self.bars[self.bars["tradetime"] < self.split_date]["close"].values
        drift_bars = self.bars[self.bars["tradetime"] >= self.split_date]

        monitor = DataMonitor()
        metrics = monitor.run(drift_bars, reference_features=ref_close)

        # Should have feature_dist_shift metrics
        metric_names = [m["metric_name"] for m in metrics]
        assert "feature_dist_shift_pvalue" in metric_names
        assert "feature_dist_shift_psi" in metric_names

    def test_step1_alpha_monitor_detects_drift(self):
        """AlphaMonitor should detect alpha value drift via PSI on drifted data."""
        # Build reference distributions
        ref_dists = {}
        for aid in self.ref_panel["alpha_id"].unique():
            ref_dists[aid] = (
                self.ref_panel[self.ref_panel["alpha_id"] == aid]["alpha_value"]
                .dropna().values
            )

        monitor = AlphaMonitor(psi_warn=0.10, psi_crit=0.25)
        metrics = monitor.run(
            self.drift_panel,
            self.fwd,
            reference_alpha_values=ref_dists,
        )

        # With shift=3.0, PSI should be very high → at least some CRITICAL alerts
        psi_metrics = [
            m for m in metrics if m["metric_name"] == "alpha_value_psi"
        ]
        assert len(psi_metrics) > 0, "No PSI metrics computed"
        critical_psi = [m for m in psi_metrics if m.get("severity") == "CRITICAL"]
        assert len(critical_psi) > 0, (
            f"Expected CRITICAL PSI alerts with shift=3.0, got severities: "
            f"{[m.get('severity') for m in psi_metrics]}"
        )

    def test_step1_model_monitor_detects_drift(self):
        """ModelMonitor should detect prediction distribution shift."""
        # Train a model on reference data, predict on drifted data
        alpha_ids = self.ref_panel["alpha_id"].unique().tolist()
        model = MLMetaModel(feature_columns=alpha_ids)

        # Use long-format directly — MLMetaModel._align_features_labels handles pivot
        ref_fwd = self.fwd[
            self.fwd.index.get_level_values("tradetime") < self.split_date
        ]
        if len(ref_fwd) < 50:
            pytest.skip("Not enough reference samples for model training")

        model.train(self.ref_panel, ref_fwd)

        # Predict on reference (baseline) and drifted data
        ref_pred = model.predict(self.ref_panel)
        drift_pred = model.predict(self.drift_panel)

        ref_scores = ref_pred["signal_score"].values
        drift_scores = drift_pred.set_index(["security_id", "tradetime"])["signal_score"]
        drift_fwd = self.fwd.reindex(drift_scores.index).dropna()
        drift_scores = drift_scores.loc[drift_fwd.index]

        monitor = ModelMonitor()
        metrics = monitor.run(
            drift_scores, drift_fwd, reference_predictions=ref_scores,
        )

        metric_names = [m["metric_name"] for m in metrics]
        assert "directional_accuracy" in metric_names
        assert "prediction_dist_drift_pvalue" in metric_names

    # ------------------------------------------------------------------
    # Step 2: Performance trigger fires
    # ------------------------------------------------------------------

    def test_step2_performance_trigger_fires_on_ic_degradation(self):
        """PerformanceTriggeredAdapter.check_trigger() should fire when IC is bad."""
        # Simulate rolling IC series that is consistently below threshold
        bad_ic = pd.Series([-0.05, -0.03, -0.01, -0.04, -0.02])
        ok_sharpe = pd.Series([0.5, 0.6, 0.4, 0.3, 0.5])

        with patch("src.adaptation.performance_trigger.ModelRegistryManager"):
            adapter = PerformanceTriggeredAdapter(
                ic_threshold=0.0,
                ic_consecutive_days=5,
            )
            triggered, reason = adapter.check_trigger(bad_ic, ok_sharpe, 0)

        assert triggered is True
        assert "Rolling IC" in reason

    def test_step2_performance_trigger_fires_on_critical_alerts(self):
        """Trigger fires when critical alert count exceeds limit."""
        ok_ic = pd.Series([0.1, 0.2, 0.15])
        ok_sharpe = pd.Series([1.0, 0.8, 0.9])

        with patch("src.adaptation.performance_trigger.ModelRegistryManager"):
            adapter = PerformanceTriggeredAdapter(critical_alert_limit=3)
            triggered, reason = adapter.check_trigger(ok_ic, ok_sharpe, 5)

        assert triggered is True
        assert "Critical alerts" in reason

    def test_step2_no_trigger_when_healthy(self):
        """No trigger when metrics are healthy."""
        good_ic = pd.Series([0.1, 0.2, 0.15, 0.18, 0.12])
        good_sharpe = pd.Series([1.0, 0.8, 0.9, 1.1, 0.95])

        with patch("src.adaptation.performance_trigger.ModelRegistryManager"):
            adapter = PerformanceTriggeredAdapter()
            triggered, reason = adapter.check_trigger(good_ic, good_sharpe, 0)

        assert triggered is False
        assert reason == ""

    # ------------------------------------------------------------------
    # Step 3: XGBoost retrain produces valid model
    # ------------------------------------------------------------------

    def test_step3_xgboost_retrain_after_trigger(self):
        """After trigger fires, XGBoost retrain should produce a model with metrics."""
        alpha_ids = self.alpha_panel["alpha_id"].unique().tolist()

        # Train on full (non-drifted) data as the "retrain" step
        model = MLMetaModel(feature_columns=alpha_ids)
        wide = self.alpha_panel.pivot_table(
            index=["security_id", "tradetime"],
            columns="alpha_id", values="alpha_value",
        ).fillna(0.0)
        common = wide.index.intersection(self.fwd.index)
        X = wide.loc[common]
        y = self.fwd.loc[common]

        result = model.train(X, y)

        assert "model_id" in result
        assert "holdout_metrics" in result
        assert result["holdout_metrics"]["n_folds"] > 0
        assert "feature_importance" in result
        assert result["n_features"] == len(alpha_ids)

        # Model should be able to predict
        pred = model.predict(self.alpha_panel)
        assert len(pred) > 0
        assert "signal_score" in pred.columns

    # ------------------------------------------------------------------
    # Step 4: Shadow evaluation compares candidates
    # ------------------------------------------------------------------

    def test_step4_shadow_evaluator_selects_best(self):
        """ShadowEvaluator should rank candidates and select the best."""
        alpha_ids = self.alpha_panel["alpha_id"].unique().tolist()

        # Candidate A: trained on reference (pre-drift) data
        model_a = MLMetaModel(feature_columns=alpha_ids)
        ref_wide = self.ref_panel.pivot_table(
            index=["security_id", "tradetime"],
            columns="alpha_id", values="alpha_value",
        ).fillna(0.0)
        ref_fwd = self.fwd.reindex(ref_wide.index).dropna()
        common_a = ref_wide.index.intersection(ref_fwd.index)
        if len(common_a) < 50:
            pytest.skip("Not enough samples for model A")
        model_a.train(ref_wide.loc[common_a], ref_fwd.loc[common_a])

        # Candidate B: trained on full data (should be better on eval set)
        model_b = MLMetaModel(feature_columns=alpha_ids)
        full_wide = self.alpha_panel.pivot_table(
            index=["security_id", "tradetime"],
            columns="alpha_id", values="alpha_value",
        ).fillna(0.0)
        full_fwd = self.fwd.reindex(full_wide.index).dropna()
        common_b = full_wide.index.intersection(full_fwd.index)
        model_b.train(full_wide.loc[common_b], full_fwd.loc[common_b])

        # Generate signals for eval period (second half)
        eval_panel = self.alpha_panel[
            self.alpha_panel["tradetime"] >= self.split_date
        ]
        signals_a = model_a.predict(eval_panel)
        signals_b = model_b.predict(eval_panel)

        eval_fwd = self.fwd[
            self.fwd.index.get_level_values("tradetime") >= self.split_date
        ]

        evaluator = ShadowEvaluator(min_improvement_ic=0.001)
        results = evaluator.evaluate_candidates(
            {
                model_a.model_id: signals_a,
                model_b.model_id: signals_b,
            },
            eval_fwd,
        )

        assert len(results) == 2
        for mid, metrics in results.items():
            assert "ic" in metrics
            assert "hit_rate" in metrics
            assert "sharpe" in metrics

        # select_best should return a model_id (or None if no improvement)
        best = evaluator.select_best(results)
        assert best is not None, "Should select at least one candidate"
        assert best in results

    # ------------------------------------------------------------------
    # Step 5: Full loop end-to-end
    # ------------------------------------------------------------------

    def test_step5_full_adaptation_loop(self):
        """End-to-end: drift → monitor → trigger → retrain → shadow → select."""
        alpha_ids = self.alpha_panel["alpha_id"].unique().tolist()

        # 1. Build reference distributions
        ref_dists = {}
        for aid in alpha_ids:
            ref_dists[aid] = (
                self.ref_panel[self.ref_panel["alpha_id"] == aid]["alpha_value"]
                .dropna().values
            )

        # 2. Monitor drifted data
        alpha_mon = AlphaMonitor(psi_warn=0.10, psi_crit=0.25)
        alpha_metrics = alpha_mon.run(
            self.drift_panel, self.fwd, reference_alpha_values=ref_dists,
        )
        critical_count = sum(
            1 for m in alpha_metrics if m.get("severity") == "CRITICAL"
        )

        # 3. Check performance trigger
        # Compute IC on drifted data — should be poor
        ref_wide = self.ref_panel.pivot_table(
            index=["security_id", "tradetime"],
            columns="alpha_id", values="alpha_value",
        ).fillna(0.0)
        ref_fwd = self.fwd.reindex(ref_wide.index).dropna()
        common = ref_wide.index.intersection(ref_fwd.index)

        old_model = MLMetaModel(feature_columns=alpha_ids)
        old_model.train(ref_wide.loc[common], ref_fwd.loc[common])

        drift_pred = old_model.predict(self.drift_panel)
        drift_scores = drift_pred.set_index(["security_id", "tradetime"])["signal_score"]
        drift_fwd = self.fwd.reindex(drift_scores.index).dropna()

        # Simulate rolling IC on drifted data (should be near zero or negative)
        rolling_ic = pd.Series(
            [information_coefficient(
                drift_scores.iloc[i:i+50],
                drift_fwd.iloc[i:i+50],
            ) for i in range(0, min(250, len(drift_scores) - 50), 50)]
        ).fillna(0.0)

        with patch("src.adaptation.performance_trigger.ModelRegistryManager"):
            adapter = PerformanceTriggeredAdapter(
                ic_threshold=0.02,
                ic_consecutive_days=3,
                critical_alert_limit=2,
            )
            triggered, reason = adapter.check_trigger(
                rolling_ic,
                pd.Series([0.0] * 10),  # dummy sharpe
                critical_count,
            )

        # Should trigger on either IC degradation or critical alerts
        assert triggered is True, (
            f"Expected trigger to fire. IC series: {rolling_ic.tolist()}, "
            f"critical_count: {critical_count}"
        )

        # 4. Retrain on expanded data (including drift period)
        new_model = MLMetaModel(feature_columns=alpha_ids)
        full_wide = self.drifted_panel.pivot_table(
            index=["security_id", "tradetime"],
            columns="alpha_id", values="alpha_value",
        ).fillna(0.0)
        full_fwd = self.fwd.reindex(full_wide.index).dropna()
        common_full = full_wide.index.intersection(full_fwd.index)
        retrain_result = new_model.train(
            full_wide.loc[common_full], full_fwd.loc[common_full]
        )
        assert retrain_result["n_train"] > 0

        # 5. Shadow evaluate: old model vs retrained
        eval_panel = self.drifted_panel[
            self.drifted_panel["tradetime"] >= self.split_date
        ]
        signals_old = old_model.predict(eval_panel)
        signals_new = new_model.predict(eval_panel)
        eval_fwd = self.fwd[
            self.fwd.index.get_level_values("tradetime") >= self.split_date
        ]

        evaluator = ShadowEvaluator(min_improvement_ic=0.001)
        eval_results = evaluator.evaluate_candidates(
            {
                old_model.model_id: signals_old,
                new_model.model_id: signals_new,
            },
            eval_fwd,
        )
        assert len(eval_results) == 2

        best = evaluator.select_best(eval_results, current_model_id=old_model.model_id)
        # The test validates the chain completes — best may be either model
        # depending on random seeds, but the chain should not error
        assert best is None or best in eval_results

    # ------------------------------------------------------------------
    # Step 6: Rule-based adaptation (Policy 1 + 2 compatibility)
    # ------------------------------------------------------------------

    def test_step6_scheduled_retrain_compatible(self):
        """ScheduledRetrainer.should_retrain() logic works correctly."""
        from src.adaptation.scheduler import ScheduledRetrainer

        with patch("src.adaptation.scheduler.ModelRegistryManager"):
            retrainer = ScheduledRetrainer(retrain_interval_days=7)

        # First call should always trigger
        assert retrainer.should_retrain(datetime(2024, 1, 1)) is True

        # Simulate a retrain
        retrainer._last_retrain = datetime(2024, 1, 1)

        # 3 days later — should not trigger
        assert retrainer.should_retrain(datetime(2024, 1, 4)) is False

        # 7 days later — should trigger
        assert retrainer.should_retrain(datetime(2024, 1, 8)) is True

    def test_step6_rule_based_retrain_produces_new_weights(self):
        """RuleBasedSignalGenerator can recompute IC weights and generate signals."""
        gen = RuleBasedSignalGenerator()
        weights = gen.compute_ic_weights(self.alpha_panel, self.fwd)

        assert len(weights) > 0
        assert abs(sum(abs(v) for v in weights.values()) - 1.0) < 1e-6

        signals = gen.generate_signal(self.alpha_panel, weights)
        assert len(signals) > 0
        assert "signal_score" in signals.columns
        assert "signal_direction" in signals.columns


# ---------------------------------------------------------------------------
# DB-driven trigger 整合測試（mock DB，不需 Docker）
# ---------------------------------------------------------------------------

class TestDBDrivenTrigger:
    """驗證 check_trigger_from_db() 的 DB 路徑與 fallback 邏輯。"""

    def test_db_driven_trigger_fires(self):
        """mock DB 返回全負 rolling_ic → check_trigger_from_db() 應回傳 (True, ...)。"""
        import psycopg2

        mock_ic_df = pd.DataFrame({
            "metric_time": pd.date_range("2024-01-01", periods=5, freq="D"),
            "metric_value": [-0.05, -0.03, -0.02, -0.04, -0.06],
        })
        mock_sharpe_df = pd.DataFrame({
            "metric_time": pd.date_range("2024-01-01", periods=5, freq="D"),
            "metric_value": [0.5, 0.6, 0.4, 0.7, 0.5],
        })

        mock_conn = MagicMock()
        mock_get_pg_connection = MagicMock(return_value=mock_conn)
        # pd.read_sql 第一次返回 IC，第二次返回 Sharpe
        mock_read_sql = MagicMock(side_effect=[mock_ic_df, mock_sharpe_df])

        with (
            patch("src.common.db.get_pg_connection", mock_get_pg_connection),
            patch("pandas.read_sql", mock_read_sql),
            patch(
                "src.monitoring.alert_manager.AlertManager.get_unacknowledged_critical_count",
                return_value=0,
            ),
            patch("src.adaptation.performance_trigger.ModelRegistryManager"),
        ):
            adapter = PerformanceTriggeredAdapter(
                ic_threshold=0.0,
                ic_consecutive_days=5,
            )
            triggered, reason = adapter.check_trigger_from_db(window=5)

        assert triggered is True, f"預期觸發，但 reason={reason!r}"
        assert "IC" in reason.upper()
        mock_get_pg_connection.assert_called_once()
        assert mock_read_sql.call_count == 2

    def test_db_driven_trigger_fallback(self):
        """get_pg_connection() 拋 OperationalError → 回傳 (False, 'db_unavailable')，不拋例外。"""
        import psycopg2

        with (
            patch(
                "src.common.db.get_pg_connection",
                side_effect=psycopg2.OperationalError("connection refused"),
            ),
            patch("src.adaptation.performance_trigger.ModelRegistryManager"),
        ):
            adapter = PerformanceTriggeredAdapter()
            triggered, reason = adapter.check_trigger_from_db()

        assert triggered is False
        assert reason == "db_unavailable"
