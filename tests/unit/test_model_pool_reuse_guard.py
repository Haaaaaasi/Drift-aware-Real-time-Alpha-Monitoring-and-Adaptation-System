"""model_pool reused candidate selector guard tests."""

from __future__ import annotations

from src.adaptation.model_pool_strategy import ModelPoolController


def _controller(**kwargs) -> ModelPoolController:
    return ModelPoolController(
        selection_metric="topk_net_return",
        min_improvement_ic=0.005,
        **kwargs,
    )


def test_reuse_guard_rejects_reused_when_margin_is_too_small() -> None:
    ctrl = _controller(reuse_margin=0.005)

    selected, info = ctrl._apply_reuse_guard(
        best_id="reuse",
        eval_results={
            "current": {"topk_net_return": 0.010},
            "new": {"topk_net_return": 0.020},
            "reuse": {"topk_net_return": 0.021},
        },
        regime_by_model={"reuse": "regime_001"},
        current_model_id="current",
    )

    assert selected == "new"
    assert info["reuse_guard_passed"] is False
    assert info["reuse_guard_reason"] == "below_non_reused_margin"
    assert info["raw_best_candidate_model_id"] == "reuse"
    assert info["best_non_reused_model_id"] == "new"


def test_reuse_guard_allows_reused_when_it_clears_margin() -> None:
    ctrl = _controller(reuse_min_score=0.0, reuse_margin=0.005)

    selected, info = ctrl._apply_reuse_guard(
        best_id="reuse",
        eval_results={
            "current": {"topk_net_return": 0.010},
            "new": {"topk_net_return": 0.020},
            "reuse": {"topk_net_return": 0.030},
        },
        regime_by_model={"reuse": "regime_001"},
        current_model_id="current",
    )

    assert selected == "reuse"
    assert info["reuse_guard_passed"] is True
    assert info["reuse_guard_reason"] == "passed"


def test_reuse_guard_rejects_reused_below_min_score() -> None:
    ctrl = _controller(reuse_min_score=0.0)

    selected, info = ctrl._apply_reuse_guard(
        best_id="reuse",
        eval_results={
            "current": {"topk_net_return": -0.030},
            "new": {"topk_net_return": -0.020},
            "reuse": {"topk_net_return": -0.010},
        },
        regime_by_model={"reuse": "regime_001"},
        current_model_id="current",
    )

    assert selected == "new"
    assert info["reuse_guard_passed"] is False
    assert info["reuse_guard_reason"] == "below_min_score"
