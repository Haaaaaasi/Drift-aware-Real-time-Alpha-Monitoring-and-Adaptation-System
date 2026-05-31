"""ShadowEvaluator 的成本感知 top-k proxy 測試。"""

from __future__ import annotations

import pandas as pd

from src.adaptation.shadow_evaluator import ShadowEvaluator


def _signals(model_shift: float = 0.0) -> pd.DataFrame:
    rows = []
    for dt in pd.date_range("2024-01-01", periods=4, freq="D"):
        rows.extend([
            {"security_id": "A", "tradetime": dt, "signal_score": 4.0 + model_shift},
            {"security_id": "B", "tradetime": dt, "signal_score": 3.0 + model_shift},
            {"security_id": "C", "tradetime": dt, "signal_score": 2.0},
            {"security_id": "D", "tradetime": dt, "signal_score": 1.0},
        ])
    return pd.DataFrame(rows)


def _bad_signals() -> pd.DataFrame:
    rows = []
    for dt in pd.date_range("2024-01-01", periods=4, freq="D"):
        rows.extend([
            {"security_id": "A", "tradetime": dt, "signal_score": 1.0},
            {"security_id": "B", "tradetime": dt, "signal_score": 2.0},
            {"security_id": "C", "tradetime": dt, "signal_score": 3.0},
            {"security_id": "D", "tradetime": dt, "signal_score": 4.0},
        ])
    return pd.DataFrame(rows)


def _forward_returns() -> pd.Series:
    rows = []
    for dt in pd.date_range("2024-01-01", periods=4, freq="D"):
        rows.extend([
            ("A", dt, 0.03),
            ("B", dt, 0.02),
            ("C", dt, -0.02),
            ("D", dt, -0.03),
        ])
    df = pd.DataFrame(rows, columns=["security_id", "tradetime", "forward_return"])
    return df.set_index(["security_id", "tradetime"])["forward_return"]


def test_evaluate_candidates_adds_topk_net_proxy() -> None:
    evaluator = ShadowEvaluator()
    results = evaluator.evaluate_candidates(
        {"good": _signals(), "bad": _bad_signals()},
        _forward_returns(),
        proxy_top_k=2,
        round_trip_cost_pct=0.0,
    )

    assert results["good"]["topk_net_return"] > 0
    assert results["bad"]["topk_net_return"] < 0
    assert results["good"]["topk_net_return"] > results["bad"]["topk_net_return"]
    assert results["good"]["topk_n_days"] == 4


def test_select_best_can_use_topk_net_return_instead_of_ic() -> None:
    evaluator = ShadowEvaluator(min_improvement_ic=0.0)
    results = {
        "ic_best": {"ic": 0.10, "topk_net_return": -0.02},
        "proxy_best": {"ic": 0.01, "topk_net_return": 0.04},
    }

    assert evaluator.select_best(results, metric="ic") == "ic_best"
    assert evaluator.select_best(results, metric="topk_net_return") == "proxy_best"
