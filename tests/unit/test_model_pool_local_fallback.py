"""model_pool 本地 fallback pool 測試。"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from src.adaptation.model_pool_strategy import ModelPoolController


class _FakePool:
    def compute_regime_fingerprint(self, bars, alpha_ic_stats=None):
        return {
            "volatility": 0.01,
            "autocorrelation": 0.0,
            "avg_cross_correlation": 0.0,
            "trend_strength": 0.0,
            "volume_ratio": 1.0,
        }

    def add_to_pool(self, *args, **kwargs):
        raise RuntimeError("db unavailable")


def test_try_add_to_pool_falls_back_to_process_local_entry() -> None:
    ctrl = ModelPoolController(similarity_threshold=0.5)
    ctrl._backend = "postgres"
    ctrl._pool = _FakePool()

    model = MagicMock()
    model.register_to_registry.return_value = False
    ctrl._models_by_id["model_001"] = model

    bars = pd.DataFrame(
        {
            "security_id": ["A", "A", "A"],
            "tradetime": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "close": [100.0, 101.0, 102.0],
            "vol": [1000, 1100, 1200],
        }
    )
    train_info = {
        "model_id": "model_001",
        "feature_importance": {},
        "holdout_metrics": {"rank_ic": 0.01},
    }

    ctrl._try_add_to_pool(
        bars=bars,
        t=pd.Timestamp("2024-01-10"),
        model_id="model_001",
        train_info=train_info,
    )

    assert ctrl._backend == "postgres_with_local_fallback"
    assert len(ctrl._local_entries) == 1
    assert ctrl._local_entries[0]["associated_model_id"] == "model_001"

    candidates, best_seen = ctrl._find_local_candidates(
        ctrl._local_entries[0]["fingerprint"],
        top_k=1,
        now=pd.Timestamp("2024-01-10"),
    )

    assert candidates == [(ctrl._local_entries[0]["regime_id"], best_seen)]
    assert best_seen >= 0.5


def test_topk_selection_metric_does_not_rank_ic_gate_local_candidates() -> None:
    ctrl = ModelPoolController(
        similarity_threshold=0.5,
        selection_metric="topk_net_return",
    )
    ctrl._models_by_id["model_001"] = MagicMock()
    regime_id = ctrl._add_local_entry(
        fingerprint={"volatility": 0.01, "volume_ratio": 1.0},
        model_id="model_001",
        train_info={
            "model_id": "model_001",
            "holdout_metrics": {
                "rank_ic": -0.05,
                "long_only_topk_net_return": 0.01,
            },
        },
        detected_at=pd.Timestamp("2024-01-10"),
    )

    candidates, best_seen = ctrl._find_local_candidates(
        {"volatility": 0.01, "volume_ratio": 1.0},
        top_k=1,
        now=pd.Timestamp("2024-01-10"),
    )

    assert candidates == [(regime_id, best_seen)]


def test_ic_selection_metric_keeps_rank_ic_gate_for_local_candidates() -> None:
    ctrl = ModelPoolController(similarity_threshold=0.5, selection_metric="ic")
    ctrl._models_by_id["model_001"] = MagicMock()
    ctrl._add_local_entry(
        fingerprint={"volatility": 0.01, "volume_ratio": 1.0},
        model_id="model_001",
        train_info={
            "model_id": "model_001",
            "holdout_metrics": {"rank_ic": -0.05},
        },
        detected_at=pd.Timestamp("2024-01-10"),
    )

    candidates, best_seen = ctrl._find_local_candidates(
        {"volatility": 0.01, "volume_ratio": 1.0},
        top_k=1,
        now=pd.Timestamp("2024-01-10"),
    )

    assert candidates == []
    assert best_seen == 0.0
