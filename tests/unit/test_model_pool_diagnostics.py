"""model_pool diagnostics helper tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipelines.simulate_recent import _attach_model_pool_candidate_proxies


class _FakeModel:
    def __init__(self, scores: dict[str, float]):
        self._scores = scores

    def predict(self, panel: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for sec in panel["security_id"].astype(str).unique():
            rows.append({
                "security_id": sec,
                "tradetime": panel["tradetime"].iloc[0],
                "signal_score": self._scores.get(sec, 0.0),
            })
        return pd.DataFrame(rows)


def _alpha_panel(day: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for sec in ["A", "B", "C"]:
        rows.append({
            "security_id": sec,
            "tradetime": day,
            "alpha_id": "wq001",
            "alpha_value": 1.0,
        })
    return pd.DataFrame(rows)


def _next_ret(days: list[pd.Timestamp]) -> pd.DataFrame:
    rows = []
    rets = {"A": 0.02, "B": -0.01, "C": 0.01}
    for d in days:
        for sec, ret in rets.items():
            rows.append({"security_id": sec, "tradetime": d, "next_return": ret})
    return pd.DataFrame(rows).set_index(["security_id", "tradetime"])


def test_attach_candidate_proxies_preserves_new_live_id_mismatch() -> None:
    day = pd.Timestamp("2024-01-02")
    sim_days = list(pd.bdate_range(day, periods=3))
    records = [
        {
            "date": "2024-01-02",
            "day_idx": None,
            "current_model_id": "cur",
            "shadow_new_model_id": "shadow_new",
            "live_model_id": "live_new",
            "selected_candidate_model_id": "shadow_new",
            "applied_model_id": "live_new",
            "candidate_model_id": "shadow_new",
            "candidate_role": "new",
            "selected": True,
            "selected_role": "new",
            "decision_reason": "shadow_selected_new_pool_hit_sim_0.7",
            "pool_hit": True,
            "candidate_similarity": np.nan,
            "selected_similarity": np.nan,
            "best_seen_similarity": 0.7,
            "n_reused_candidates": 1,
            "shadow_ic": 0.05,
            "shadow_hit_rate": 0.55,
            "shadow_sharpe": 1.2,
            "shadow_n_samples": 100,
        }
    ]
    out = _attach_model_pool_candidate_proxies(
        records=records,
        candidate_models={"shadow_new": _FakeModel({"A": 3.0, "B": 2.0, "C": 1.0})},
        t=day,
        day_idx=0,
        sim_days=sim_days,
        alpha_panel=_alpha_panel(day),
        next_ret=_next_ret(sim_days),
        prev_weights={"C": 1.0},
        top_k=2,
        rebalance_every=3,
        commission_rate=0.0,
        tax_rate=0.0,
        slippage_bps=0.0,
        round_trip_cost_pct=0.0,
    )

    assert len(out) == 1
    row = out[0]
    assert row["day_idx"] == 0
    assert row["selected_candidate_model_id"] == "shadow_new"
    assert row["applied_model_id"] == "live_new"
    assert row["proxy_n_days"] == 3
    assert row["proxy_rank_by_net"] == 1
    assert row["proxy_turnover"] > 0


def test_attach_candidate_proxies_ranks_by_proxy_net_return() -> None:
    day = pd.Timestamp("2024-01-02")
    sim_days = list(pd.bdate_range(day, periods=2))
    base = {
        "date": "2024-01-02",
        "day_idx": None,
        "current_model_id": "cur",
        "shadow_new_model_id": "new",
        "live_model_id": "cur",
        "selected_candidate_model_id": "cur",
        "applied_model_id": "cur",
        "selected": False,
        "selected_role": "current",
        "decision_reason": "shadow_kept_current",
        "pool_hit": True,
        "selected_similarity": np.nan,
        "best_seen_similarity": 0.8,
        "n_reused_candidates": 1,
        "shadow_ic": 0.0,
        "shadow_hit_rate": 0.0,
        "shadow_sharpe": 0.0,
        "shadow_n_samples": 100,
    }
    records = [
        {**base, "candidate_model_id": "cur", "candidate_role": "current"},
        {
            **base,
            "candidate_model_id": "reuse",
            "candidate_role": "reused",
            "candidate_similarity": 0.75,
        },
    ]
    out = _attach_model_pool_candidate_proxies(
        records=records,
        candidate_models={
            "cur": _FakeModel({"A": 1.0, "B": 2.0, "C": 3.0}),
            "reuse": _FakeModel({"A": 3.0, "B": 2.0, "C": 1.0}),
        },
        t=day,
        day_idx=0,
        sim_days=sim_days,
        alpha_panel=_alpha_panel(day),
        next_ret=_next_ret(sim_days),
        prev_weights={},
        top_k=1,
        rebalance_every=2,
        commission_rate=0.0,
        tax_rate=0.0,
        slippage_bps=0.0,
        round_trip_cost_pct=0.0,
    )

    ranks = {row["candidate_model_id"]: row["proxy_rank_by_net"] for row in out}
    assert ranks["reuse"] == 1
    assert ranks["cur"] == 2
    reused = [row for row in out if row["candidate_model_id"] == "reuse"][0]
    assert reused["candidate_similarity"] == 0.75
