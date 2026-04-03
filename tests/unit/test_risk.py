"""Unit tests for Layer 6: Risk Management."""

import pandas as pd
import pytest

from src.risk.risk_manager import RiskManager


class TestRiskManager:
    @pytest.fixture
    def sample_targets(self):
        return pd.DataFrame({
            "rebalance_time": pd.Timestamp("2024-01-02"),
            "security_id": ["A", "B", "C", "D"],
            "target_weight": [0.30, 0.30, 0.25, 0.15],
            "target_shares": [0, 0, 0, 0],
            "construction_method": "equal_weight_topk",
            "pre_risk": True,
        })

    def test_position_cap(self, sample_targets):
        rm = RiskManager(max_position_weight=0.10)
        result = rm.apply_constraints(sample_targets)
        assert result["target_weight"].max() <= 0.10 + 1e-9

    def test_exposure_normalization(self, sample_targets):
        rm = RiskManager(max_position_weight=1.0, max_gross_exposure=0.50)
        result = rm.apply_constraints(sample_targets)
        assert result["target_weight"].abs().sum() <= 0.50 + 1e-9

    def test_drawdown_halt(self, sample_targets):
        rm = RiskManager(max_drawdown_halt=0.15)
        result = rm.apply_constraints(sample_targets, cumulative_drawdown=-0.20)
        assert result["target_weight"].sum() == 0.0

    def test_pre_risk_flag_set_to_false(self, sample_targets):
        rm = RiskManager()
        result = rm.apply_constraints(sample_targets)
        assert not result["pre_risk"].any()
