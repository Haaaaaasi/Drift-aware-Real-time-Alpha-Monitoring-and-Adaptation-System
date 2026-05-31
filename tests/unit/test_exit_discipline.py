import pandas as pd

from pipelines.simulate_recent import _apply_exit_discipline


def test_hard_exit_sells_mature_negative_score_holding_only():
    weights, metrics = _apply_exit_discipline(
        current_weights={"2330": 0.10, "2317": 0.08, "2454": 0.07},
        signal_scores=pd.Series({"2330": -0.01, "2317": -0.02, "2454": 0.03}),
        holding_days={"2330": 10, "2317": 2, "2454": 10},
        hard_exit_score_threshold=0.0,
        hard_exit_min_holding_days=5,
        tail_cleanup_weight=0.0,
        renormalize=False,
    )

    assert set(weights) == {"2317", "2454"}
    assert metrics["hard_exit_count"] == 1
    assert metrics["hard_exit_weight"] == 0.10
    assert metrics["negative_score_weight_after"] == 0.08


def test_tail_cleanup_removes_mature_residual_weight():
    weights, metrics = _apply_exit_discipline(
        current_weights={"2330": 0.10, "2317": 0.004, "2454": 0.006},
        signal_scores=pd.Series({"2330": 0.02, "2317": 0.01, "2454": 0.01}),
        holding_days={"2330": 10, "2317": 10, "2454": 2},
        hard_exit_score_threshold=None,
        hard_exit_min_holding_days=5,
        tail_cleanup_weight=0.005,
        renormalize=False,
    )

    assert set(weights) == {"2330", "2454"}
    assert metrics["tail_exit_count"] == 1
    assert metrics["tail_exit_weight"] == 0.004


def test_exit_cleanup_can_renormalize_remaining_gross_exposure():
    weights, metrics = _apply_exit_discipline(
        current_weights={"2330": 0.60, "2317": 0.40},
        signal_scores=pd.Series({"2330": 0.02, "2317": -0.01}),
        holding_days={"2330": 10, "2317": 10},
        hard_exit_score_threshold=0.0,
        hard_exit_min_holding_days=5,
        tail_cleanup_weight=0.0,
        renormalize=True,
    )

    assert weights == {"2330": 1.0}
    assert metrics["exit_cleanup_renormalized"] is True
    assert metrics["exit_cleanup_gross_after"] == 1.0
