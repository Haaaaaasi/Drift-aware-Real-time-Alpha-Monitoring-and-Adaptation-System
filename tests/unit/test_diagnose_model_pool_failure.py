"""Tests for scripts.diagnose_model_pool_failure."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.diagnose_model_pool_failure import run_diagnosis


def _write_pnl(path: Path, returns: list[float]) -> None:
    dates = pd.bdate_range("2024-01-02", periods=len(returns))
    rows = []
    value = 10_000_000.0
    for d, ret in zip(dates, returns):
        value *= (1.0 + ret)
        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "gross_return": ret,
            "net_return": ret,
            "cumulative_value": value,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def test_run_diagnosis_writes_expected_outputs(tmp_path: Path) -> None:
    ab_dir = tmp_path / "ab"
    sim_dir = ab_dir / "simulations"
    mp_dir = tmp_path / "model_pool_diag"
    out_dir = tmp_path / "diag_out"
    sim_dir.mkdir(parents=True)
    mp_dir.mkdir(parents=True)

    run_dirs = {}
    for strat, ret in {
        "none": [0.0, 0.0, 0.0],
        "scheduled_20": [0.01, 0.01, 0.01],
        "triggered": [-0.01, 0.0, 0.0],
    }.items():
        d = sim_dir / strat
        d.mkdir()
        _write_pnl(d / "daily_pnl.csv", ret)
        run_dirs[strat] = str(d)

    _write_pnl(mp_dir / "daily_pnl.csv", [-0.02, -0.01, 0.0])
    pd.DataFrame([
        {
            "date": "2024-01-02",
            "day_idx": 0,
            "current_model_id": "cur",
            "shadow_new_model_id": "new",
            "live_model_id": "reuse",
            "selected_candidate_model_id": "reuse",
            "applied_model_id": "reuse",
            "candidate_model_id": "reuse",
            "candidate_role": "reused",
            "selected": True,
            "selected_role": "reused",
            "decision_reason": "shadow_selected_reused_sim_0.7",
            "pool_hit": True,
            "selection_metric": "topk_net_return",
            "selection_score": 0.02,
            "shadow_rank_by_selection_metric": 1,
            "shadow_rank_by_topk_net_return": 2,
            "raw_best_candidate_model_id": "reuse",
            "raw_best_role": "reused",
            "raw_best_score": 0.02,
            "best_non_reused_model_id": "new",
            "best_non_reused_score": 0.018,
            "reuse_score_margin_vs_best_non_reused": 0.002,
            "reuse_guard_min_score": None,
            "reuse_guard_margin": 0.0,
            "reuse_guard_passed": True,
            "reuse_guard_reason": "passed",
            "candidate_similarity": 0.7,
            "selected_similarity": 0.7,
            "best_seen_similarity": 0.7,
            "n_reused_candidates": 1,
            "shadow_ic": 0.05,
            "shadow_hit_rate": 0.55,
            "shadow_sharpe": 1.0,
            "shadow_n_samples": 100,
            "proxy_n_days": 3,
            "proxy_gross_return": -0.02,
            "proxy_net_return": -0.02,
            "proxy_turnover": 0.5,
            "proxy_cost": 0.001,
            "proxy_rank_by_net": 2,
        }
    ]).to_csv(mp_dir / "model_pool_decisions.csv", index=False)

    (ab_dir / "config.json").write_text(
        json.dumps({"run_dirs": run_dirs | {"model_pool": str(mp_dir)}}),
        encoding="utf-8",
    )

    result = run_diagnosis(
        ab_run_dir=ab_dir,
        model_pool_run_dir=mp_dir,
        out_dir=out_dir,
        n_days=3,
    )

    assert result["event_path"].exists()
    assert result["candidate_path"].exists()
    assert result["fig_path"].exists()
    assert result["summary_path"].exists()
    event = pd.read_csv(result["event_path"])
    assert "excess_vs_scheduled_20" in event.columns
    assert "reuse_guard_reason" in event.columns
    assert event["n_post_days"].iloc[0] == 3
    summary = Path(result["summary_path"]).read_text(encoding="utf-8")
    assert "Reuse Guard / Selector Audit" in summary
