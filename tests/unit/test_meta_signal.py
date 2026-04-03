"""Unit tests for Layer 4: Meta Signal Generation."""

import numpy as np
import pandas as pd
import pytest

from src.meta_signal.rule_based import RuleBasedSignalGenerator


class TestRuleBasedSignal:
    @pytest.fixture
    def sample_alpha_panel(self):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        secs = ["A", "B", "C", "D", "E"]
        alphas = ["wq001", "wq002", "wq003"]
        rows = []
        for d in dates:
            for s in secs:
                for a in alphas:
                    rows.append({
                        "security_id": s,
                        "tradetime": d,
                        "alpha_id": a,
                        "alpha_value": np.random.randn(),
                    })
        return pd.DataFrame(rows)

    @pytest.fixture
    def sample_returns(self, sample_alpha_panel):
        np.random.seed(42)
        idx = sample_alpha_panel[["security_id", "tradetime"]].drop_duplicates()
        idx = idx.set_index(["security_id", "tradetime"]).index
        return pd.Series(np.random.randn(len(idx)) * 0.02, index=idx)

    def test_compute_ic_weights(self, sample_alpha_panel, sample_returns):
        gen = RuleBasedSignalGenerator()
        weights = gen.compute_ic_weights(sample_alpha_panel, sample_returns)
        assert len(weights) == 3
        assert abs(sum(abs(v) for v in weights.values()) - 1.0) < 0.01

    def test_generate_signal_shape(self, sample_alpha_panel):
        gen = RuleBasedSignalGenerator()
        weights = {"wq001": 0.4, "wq002": 0.3, "wq003": 0.3}
        signals = gen.generate_signal(sample_alpha_panel, weights)
        assert "signal_score" in signals.columns
        assert "signal_direction" in signals.columns
        assert len(signals) > 0

    def test_signal_direction_matches_score(self, sample_alpha_panel):
        gen = RuleBasedSignalGenerator()
        weights = {"wq001": 0.5, "wq002": 0.25, "wq003": 0.25}
        signals = gen.generate_signal(sample_alpha_panel, weights)
        for _, row in signals.iterrows():
            assert row["signal_direction"] == int(np.sign(row["signal_score"]))
