from __future__ import annotations

import math

import pandas as pd

from src.portfolio.live_service import LivePortfolioConfig, build_live_portfolio


def _signals(scores: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "security_id": list(scores),
            "signal_time": pd.Timestamp("2026-05-18"),
            "signal_score": list(scores.values()),
            "signal_direction": [1] * len(scores),
            "confidence": [abs(v) for v in scores.values()],
        }
    )


def test_live_portfolio_generates_buy_sell_and_holds_previous_buffer() -> None:
    signals = _signals({"A": 0.9, "C": 0.8, "D": 0.7, "E": 0.6, "B": 0.1})

    result = build_live_portfolio(
        signals=signals,
        as_of_date=pd.Timestamp("2026-05-18"),
        previous_weights={"A": 0.5, "B": 0.5},
        previous_shares={"A": 500, "B": 500},
        holding_days={"A": 20, "B": 20},
        last_prices={"A": 100.0, "B": 50.0, "C": 25.0, "D": 20.0, "E": 10.0},
        capital=100_000,
        config=LivePortfolioConfig(
            top_k=3,
            entry_rank=3,
            exit_rank=5,
            max_turnover=1.0,
            min_holding_days=10,
            tail_cleanup_weight=0.0,
            max_position_weight=1.0,
        ),
    )

    recs = result.recommendations.set_index("security_id")
    assert set(recs.index) == {"A", "B", "C"}
    assert recs.loc["C", "action"] == "BUY"
    assert recs.loc["A", "action"] == "REDUCE"
    assert recs.loc["B", "action"] == "REDUCE"
    assert result.metrics["held_from_prev_count"] == 2
    assert math.isclose(result.targets["target_weight"].sum(), 1.0, rel_tol=1e-9)


def test_live_portfolio_tail_cleanup_creates_sell_recommendation() -> None:
    signals = _signals({"A": 0.9, "B": 0.8})

    result = build_live_portfolio(
        signals=signals,
        as_of_date=pd.Timestamp("2026-05-18"),
        previous_weights={"A": 0.999, "Z": 0.001},
        previous_shares={"A": 999, "Z": 1},
        holding_days={"A": 20, "Z": 20},
        last_prices={"A": 100.0, "B": 50.0, "Z": 10.0},
        capital=100_000,
        config=LivePortfolioConfig(
            top_k=2,
            entry_rank=2,
            exit_rank=2,
            max_turnover=0.25,
            min_holding_days=10,
            tail_cleanup_weight=0.0025,
            max_position_weight=1.0,
        ),
    )

    recs = result.recommendations.set_index("security_id")
    assert recs.loc["Z", "action"] == "SELL"
    assert recs.loc["Z", "reason"] in {"tail_cleanup", "exit_rank_or_untradable"}
    assert result.metrics["exit_cleanup_count"] >= 1
