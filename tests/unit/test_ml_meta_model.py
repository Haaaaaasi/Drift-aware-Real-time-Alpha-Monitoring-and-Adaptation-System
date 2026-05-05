"""Unit tests for WP3 — XGBoost MLMetaModel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.meta_signal.ml_meta_model import MLMetaModel


# ---------------------------------------------------------------------------
# Fixture: synthetic panel where the target is a noisy linear combination of
# two "informative" alphas and three pure-noise alphas. A working tree model
# should place most of its importance on the informative features.
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_panel() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    n_days = 120
    n_sec = 20
    dates = pd.date_range("2023-01-02", periods=n_days, freq="B")
    secs = [f"S{i:02d}" for i in range(n_sec)]
    idx = pd.MultiIndex.from_product([secs, dates], names=["security_id", "tradetime"])

    n = len(idx)
    alpha_strong = rng.normal(0, 1, n)
    alpha_weak = rng.normal(0, 1, n)
    alpha_noise_1 = rng.normal(0, 1, n)
    alpha_noise_2 = rng.normal(0, 1, n)
    alpha_noise_3 = rng.normal(0, 1, n)

    # Target = 0.8 * strong + 0.3 * weak + noise
    y_raw = 0.8 * alpha_strong + 0.3 * alpha_weak + rng.normal(0, 0.5, n)

    features = pd.DataFrame(
        {
            "wq_strong": alpha_strong,
            "wq_weak": alpha_weak,
            "wq_noise_1": alpha_noise_1,
            "wq_noise_2": alpha_noise_2,
            "wq_noise_3": alpha_noise_3,
        },
        index=idx,
    )
    labels = pd.Series(y_raw, index=idx, name="fwd_return")
    return features, labels


class TestTraining:
    def test_train_returns_expected_fields(self, synthetic_panel):
        features, labels = synthetic_panel
        model = MLMetaModel()
        info = model.train(features, labels, purge_days=2, n_splits=3)

        assert set(info) >= {"model_id", "holdout_metrics", "feature_importance", "n_train"}
        assert info["model_id"].startswith("ml_xgb_")
        assert info["n_train"] > 0
        assert info["n_features"] == 5

    def test_holdout_metrics_are_positive_on_informative_data(self, synthetic_panel):
        features, labels = synthetic_panel
        model = MLMetaModel()
        info = model.train(features, labels, purge_days=2, n_splits=3)
        metrics = info["holdout_metrics"]

        assert metrics["n_folds"] >= 1
        # Signal is intentionally strong — IC should clearly be positive.
        assert metrics["ic"] > 0.2, f"IC too low: {metrics['ic']}"
        assert metrics["gross_ic"] == metrics["ic"]
        assert "net_ic_proxy" in metrics
        assert "long_only_topk_net_return" in metrics
        assert "turnover_proxy" in metrics
        assert metrics["dir_accuracy"] > 0.55

    def test_informative_features_get_higher_importance(self, synthetic_panel):
        features, labels = synthetic_panel
        model = MLMetaModel()
        model.train(features, labels, purge_days=2, n_splits=3)
        fi = model.feature_importance

        assert fi["wq_strong"] > fi["wq_noise_1"]
        assert fi["wq_strong"] > fi["wq_noise_2"]
        assert fi["wq_strong"] > fi["wq_noise_3"]

    def test_raises_when_not_enough_samples(self):
        idx = pd.MultiIndex.from_tuples(
            [("A", pd.Timestamp("2024-01-01"))], names=["security_id", "tradetime"]
        )
        X = pd.DataFrame({"a": [1.0]}, index=idx)
        y = pd.Series([0.1], index=idx)
        model = MLMetaModel()
        with pytest.raises(ValueError):
            model.train(X, y)


class TestPrediction:
    def test_predict_schema_matches_rule_based(self, synthetic_panel):
        features, labels = synthetic_panel
        model = MLMetaModel()
        model.train(features, labels, purge_days=2, n_splits=3)
        signals = model.predict(features)

        assert set(signals.columns) >= {
            "security_id",
            "tradetime",
            "signal_score",
            "signal_direction",
            "confidence",
        }
        assert len(signals) == len(features)
        assert signals["signal_direction"].isin([-1, 1]).all()
        assert (signals["confidence"] >= 0).all()

    def test_predict_raises_if_untrained(self, synthetic_panel):
        features, _ = synthetic_panel
        model = MLMetaModel()
        with pytest.raises(RuntimeError):
            model.predict(features)

    def test_predict_is_deterministic(self, synthetic_panel):
        features, labels = synthetic_panel
        model = MLMetaModel()
        model.train(features, labels, purge_days=2, n_splits=3)
        a = model.predict(features)["signal_score"].to_numpy()
        b = model.predict(features)["signal_score"].to_numpy()
        np.testing.assert_array_equal(a, b)


class TestFeatureColumnsFilter:
    def test_feature_columns_restricts_training(self, synthetic_panel):
        features, labels = synthetic_panel
        keep = ["wq_strong", "wq_weak"]
        model = MLMetaModel(feature_columns=keep)
        info = model.train(features, labels, purge_days=2, n_splits=3)
        assert info["n_features"] == 2
        assert set(model.feature_importance.keys()) == set(keep)

    def test_long_format_input_is_pivoted_internally(self, synthetic_panel):
        features, labels = synthetic_panel
        long = (
            features.stack()
            .rename("alpha_value")
            .reset_index()
            .rename(columns={"level_2": "alpha_id"})
        )
        model = MLMetaModel()
        info = model.train(long, labels, purge_days=2, n_splits=3)
        assert info["n_features"] == 5
