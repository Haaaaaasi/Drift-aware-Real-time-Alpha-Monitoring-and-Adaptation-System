from __future__ import annotations

import math

import pandas as pd

from src.portfolio.constructor import PortfolioConstructor
from src.risk.risk_manager import RiskManager


def _signals(scores: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "security_id": list(scores),
            "signal_time": pd.Timestamp("2024-01-02"),
            "signal_score": list(scores.values()),
            "signal_direction": [1] * len(scores),
        }
    )


class TestTurnoverAwareTopK:
    def test_keeps_previous_holdings_inside_exit_buffer(self) -> None:
        signals = _signals({"A": 0.90, "C": 0.80, "D": 0.70, "E": 0.60, "B": 0.10})
        portfolio = PortfolioConstructor(
            method="turnover_aware_topk",
            top_k=3,
            entry_rank=3,
            exit_rank=5,
        )

        targets = portfolio.construct(
            signals,
            previous_weights={"A": 0.5, "B": 0.5},
            holding_days={"A": 10, "B": 10},
        )

        assert list(targets["security_id"]) == ["A", "B", "C"]
        assert targets.attrs["held_from_prev_count"] == 2
        assert targets.attrs["forced_sells_count"] == 0

    def test_new_names_only_come_from_entry_pool(self) -> None:
        signals = _signals({"A": 0.90, "B": 0.80, "C": 0.70, "D": 0.60})
        portfolio = PortfolioConstructor(
            method="turnover_aware_topk",
            top_k=3,
            entry_rank=2,
            exit_rank=4,
        )

        targets = portfolio.construct(signals)

        assert list(targets["security_id"]) == ["A", "B"]

    def test_min_holding_days_prevents_rank_based_sell(self) -> None:
        signals = _signals({"B": 0.90, "C": 0.80, "D": 0.70, "E": 0.60, "A": 0.10})
        portfolio = PortfolioConstructor(
            method="turnover_aware_topk",
            top_k=2,
            entry_rank=2,
            exit_rank=3,
            min_holding_days=5,
        )

        targets = portfolio.construct(
            signals,
            previous_weights={"A": 0.5},
            holding_days={"A": 2},
        )

        assert "A" in set(targets["security_id"])
        assert bool(targets.loc[targets["security_id"] == "A", "held_from_prev"].iloc[0])

    def test_empty_universe_returns_empty_targets(self) -> None:
        signals = pd.DataFrame(
            columns=["security_id", "signal_time", "signal_score", "signal_direction"]
        )
        portfolio = PortfolioConstructor(method="turnover_aware_topk", top_k=3)

        targets = portfolio.construct(signals, previous_weights={"A": 1.0})

        assert targets.empty


class TestTurnoverCap:
    def test_max_turnover_blends_from_previous_weights(self) -> None:
        targets = pd.DataFrame(
            {
                "rebalance_time": [pd.Timestamp("2024-01-02")] * 2,
                "security_id": ["C", "D"],
                "target_weight": [0.5, 0.5],
                "target_shares": [0, 0],
                "construction_method": ["turnover_aware_topk"] * 2,
                "pre_risk": [True, True],
            }
        )
        risk = RiskManager(max_position_weight=1.0, max_gross_exposure=1.0, max_turnover=0.25)

        adjusted = risk.apply_constraints(
            targets,
            previous_weights={"A": 0.5, "B": 0.5},
        )
        weights = adjusted.set_index("security_id")["target_weight"].to_dict()

        assert adjusted.attrs["turnover_cap_applied"] is True
        assert math.isclose(weights["A"], 0.375, rel_tol=1e-9)
        assert math.isclose(weights["B"], 0.375, rel_tol=1e-9)
        assert math.isclose(weights["C"], 0.125, rel_tol=1e-9)
        assert math.isclose(weights["D"], 0.125, rel_tol=1e-9)
