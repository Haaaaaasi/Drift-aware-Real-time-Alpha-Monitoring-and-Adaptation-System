from __future__ import annotations

import pandas as pd

from src.alpha_selection import RollingTopKSelector, SelectorContext, hash_universe


def _context(
    as_of: str = "2024-01-08",
    train_start: str = "2024-01-01",
    train_end: str = "2024-01-07",
) -> SelectorContext:
    return SelectorContext(
        as_of_date=pd.Timestamp(as_of),
        label_cutoff=pd.Timestamp(as_of),
        train_window_start=pd.Timestamp(train_start),
        train_window_end=pd.Timestamp(train_end),
        label_horizon_days=2,
        purge_days=1,
        label_available_rule="label_available_at <= as_of_date",
        selector_config_hash="rolling-cfg",
        feature_store_version="feature-store-v1",
        bars_snapshot_hash="bars-v1",
        universe_hash=hash_universe(["A", "B", "C"]),
        alpha_engine_version="python_wq101_v1",
        git_commit="abc123",
    )


def _toy_panel() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    securities = ["A", "B", "C"]
    rows = []
    labels = {}
    available = {}
    for day_idx, d in enumerate(dates):
        for sec_idx, sec in enumerate(securities):
            idx = (sec, d)
            if d <= pd.Timestamp("2024-01-05"):
                y = float(sec_idx + 1)
                good_signal = float(sec_idx + 1)
                future_signal = 1.0
                label_available_at = pd.Timestamp("2024-01-08")
            else:
                good_signal = float(10 - sec_idx)
                future_signal = float(sec_idx + 1)
                y = future_signal * 100.0
                label_available_at = pd.Timestamp("2024-01-20")
            labels[idx] = y
            available[idx] = label_available_at
            rows.extend(
                [
                    {
                        "security_id": sec,
                        "tradetime": d,
                        "alpha_id": "wq_good",
                        "alpha_value": good_signal,
                    },
                    {
                        "security_id": sec,
                        "tradetime": d,
                        "alpha_id": "wq_future",
                        "alpha_value": future_signal,
                    },
                    {
                        "security_id": sec,
                        "tradetime": d,
                        "alpha_id": "wq_flat",
                        "alpha_value": float(day_idx),
                    },
                ]
            )

    index = pd.MultiIndex.from_tuples(labels.keys(), names=["security_id", "tradetime"])
    return (
        pd.DataFrame(rows),
        pd.Series(labels, index=index, name="forward_return"),
        pd.Series(available, index=index, name="label_available_at"),
    )


def test_rolling_topk_uses_only_mature_labels_for_ranking() -> None:
    alpha_panel, labels, label_available_at = _toy_panel()
    selector = RollingTopKSelector(
        candidate_alpha_ids=("wq_future", "wq_good", "wq_flat"),
        top_k=1,
        window_days=30,
        min_coverage=0.1,
        min_observations=3,
    )

    snapshot = selector.select(
        _context(),
        alpha_panel=alpha_panel,
        labels=labels,
        label_available_at=label_available_at,
    )

    assert snapshot.selected_alphas == ["wq_good"]
    future_row = snapshot.scores[snapshot.scores["alpha_id"] == "wq_future"].iloc[0]
    assert bool(future_row["selected"]) is False
    assert future_row["excluded_reason"] == "insufficient_variance"
    assert int(future_row["n_observations"]) == 15


def test_rolling_topk_does_not_select_future_alpha_before_label_matures() -> None:
    alpha_panel, labels, label_available_at = _toy_panel()
    selector = RollingTopKSelector(
        candidate_alpha_ids=("wq_future", "wq_good"),
        top_k=1,
        window_days=30,
        min_coverage=0.1,
        min_observations=3,
    )

    before_mature = selector.select(
        _context("2024-01-08"),
        alpha_panel=alpha_panel,
        labels=labels,
        label_available_at=label_available_at,
    )
    after_mature = selector.select(
        _context("2024-01-20", train_start="2024-01-06", train_end="2024-01-08"),
        alpha_panel=alpha_panel,
        labels=labels,
        label_available_at=label_available_at,
    )

    assert before_mature.selected_alphas == ["wq_good"]
    assert after_mature.selected_alphas == ["wq_future"]


def test_rolling_topk_stability_penalty_prefers_previous_alpha_when_scores_are_close() -> None:
    date = pd.Timestamp("2024-02-01")
    securities = ["A", "B", "C", "D"]
    y_values = [1.0, 2.0, 3.0, 4.0]
    old_values = [1.0, 2.0, 2.0, 4.0]
    new_values = [1.0, 2.0, 3.0, 4.0]
    rows = []
    labels = {}
    available = {}
    for sec, y, old, new in zip(securities, y_values, old_values, new_values):
        idx = (sec, date)
        labels[idx] = y
        available[idx] = pd.Timestamp("2024-02-02")
        rows.extend(
            [
                {"security_id": sec, "tradetime": date, "alpha_id": "wq_old", "alpha_value": old},
                {"security_id": sec, "tradetime": date, "alpha_id": "wq_new", "alpha_value": new},
            ]
        )
    index = pd.MultiIndex.from_tuples(labels.keys(), names=["security_id", "tradetime"])
    alpha_panel = pd.DataFrame(rows)
    labels_s = pd.Series(labels, index=index, name="forward_return")
    available_s = pd.Series(available, index=index, name="label_available_at")

    no_penalty = RollingTopKSelector(
        candidate_alpha_ids=("wq_new", "wq_old"),
        top_k=1,
        window_days=30,
        min_coverage=0.1,
        min_observations=4,
        stability_penalty=0.0,
    ).select(
        _context("2024-02-02", train_start="2024-02-01", train_end="2024-02-01"),
        alpha_panel=alpha_panel,
        labels=labels_s,
        label_available_at=available_s,
        previous_selected_alpha_ids=["wq_old"],
    )
    with_penalty = RollingTopKSelector(
        candidate_alpha_ids=("wq_new", "wq_old"),
        top_k=1,
        window_days=30,
        min_coverage=0.1,
        min_observations=4,
        stability_penalty=0.20,
    ).select(
        _context("2024-02-02", train_start="2024-02-01", train_end="2024-02-01"),
        alpha_panel=alpha_panel,
        labels=labels_s,
        label_available_at=available_s,
        previous_selected_alpha_ids=["wq_old"],
    )

    assert no_penalty.selected_alphas == ["wq_new"]
    assert with_penalty.selected_alphas == ["wq_old"]
    new_row = with_penalty.scores[with_penalty.scores["alpha_id"] == "wq_new"].iloc[0]
    assert new_row["stability"] == 0.0
    assert new_row["turnover_penalty"] > 0.0


def _admission_panel() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    dates = pd.date_range("2024-03-01", periods=9, freq="D")
    securities = ["A", "B", "C", "D"]
    rows = []
    labels = {}
    available = {}
    for day_idx, d in enumerate(dates):
        for sec_idx, sec in enumerate(securities):
            y = float(sec_idx + 1)
            idx = (sec, d)
            labels[idx] = y
            available[idx] = d + pd.Timedelta(days=1)
            unstable_value = y if day_idx >= 6 else 1.0
            rows.extend(
                [
                    {"security_id": sec, "tradetime": d, "alpha_id": "wq_live", "alpha_value": y},
                    {"security_id": sec, "tradetime": d, "alpha_id": "wq_q_stable", "alpha_value": y * 2.0},
                    {"security_id": sec, "tradetime": d, "alpha_id": "wq_q_unstable", "alpha_value": unstable_value},
                    {"security_id": sec, "tradetime": d, "alpha_id": "wq_q_redundant", "alpha_value": y},
                ]
            )
    index = pd.MultiIndex.from_tuples(labels.keys(), names=["security_id", "tradetime"])
    return (
        pd.DataFrame(rows),
        pd.Series(labels, index=index, name="forward_return"),
        pd.Series(available, index=index, name="label_available_at"),
    )


def test_admission_gate_blocks_unstable_quarantine_alpha() -> None:
    alpha_panel, labels, label_available_at = _admission_panel()
    selector = RollingTopKSelector(
        candidate_alpha_ids=("wq_q_unstable", "wq_live"),
        base_alpha_ids=("wq_live",),
        top_k=2,
        window_days=30,
        min_coverage=0.1,
        min_observations=4,
        admission_enabled=True,
        admission_max_promoted=1,
        admission_min_score=0.01,
        admission_min_coverage=0.1,
        admission_min_observations=4,
        admission_subwindows=3,
        admission_min_subwindow_passes=2,
        admission_subwindow_min_abs_ic=0.01,
        admission_max_abs_corr_to_live=1.0,
    )

    snapshot = selector.select(
        _context("2024-03-11", train_start="2024-03-01", train_end="2024-03-09"),
        alpha_panel=alpha_panel,
        labels=labels,
        label_available_at=label_available_at,
    )

    assert snapshot.selected_alphas == ["wq_live"]
    q_row = snapshot.scores[snapshot.scores["alpha_id"] == "wq_q_unstable"].iloc[0]
    assert q_row["alpha_pool"] == "quarantine"
    assert q_row["admission_status"] == "quarantine"
    assert "admission_unstable_subwindows" in q_row["admission_reason"]
    assert q_row["excluded_reason"] == q_row["admission_reason"]


def test_admission_gate_admits_stable_quarantine_alpha() -> None:
    alpha_panel, labels, label_available_at = _admission_panel()
    selector = RollingTopKSelector(
        candidate_alpha_ids=("wq_q_stable", "wq_live"),
        base_alpha_ids=("wq_live",),
        top_k=2,
        window_days=30,
        min_coverage=0.1,
        min_observations=4,
        admission_enabled=True,
        admission_max_promoted=1,
        admission_min_score=0.01,
        admission_min_coverage=0.1,
        admission_min_observations=4,
        admission_subwindows=3,
        admission_min_subwindow_passes=2,
        admission_subwindow_min_abs_ic=0.01,
        admission_max_abs_corr_to_live=1.0,
    )

    snapshot = selector.select(
        _context("2024-03-11", train_start="2024-03-01", train_end="2024-03-09"),
        alpha_panel=alpha_panel,
        labels=labels,
        label_available_at=label_available_at,
    )

    assert set(snapshot.selected_alphas) == {"wq_q_stable", "wq_live"}
    q_row = snapshot.scores[snapshot.scores["alpha_id"] == "wq_q_stable"].iloc[0]
    assert q_row["admission_status"] == "admitted"
    assert q_row["admission_reason"] == "passed_admission_gate"
    assert int(q_row["admission_subwindow_pass_count"]) == 3


def test_admission_gate_blocks_redundant_quarantine_family() -> None:
    alpha_panel, labels, label_available_at = _admission_panel()
    selector = RollingTopKSelector(
        candidate_alpha_ids=("wq_q_redundant", "wq_live"),
        base_alpha_ids=("wq_live",),
        top_k=2,
        window_days=30,
        min_coverage=0.1,
        min_observations=4,
        admission_enabled=True,
        admission_max_promoted=1,
        admission_min_score=0.01,
        admission_min_coverage=0.1,
        admission_min_observations=4,
        admission_subwindows=3,
        admission_min_subwindow_passes=2,
        admission_subwindow_min_abs_ic=0.01,
        admission_max_abs_corr_to_live=0.95,
    )

    snapshot = selector.select(
        _context("2024-03-11", train_start="2024-03-01", train_end="2024-03-09"),
        alpha_panel=alpha_panel,
        labels=labels,
        label_available_at=label_available_at,
    )

    assert snapshot.selected_alphas == ["wq_live"]
    q_row = snapshot.scores[snapshot.scores["alpha_id"] == "wq_q_redundant"].iloc[0]
    assert q_row["admission_status"] == "quarantine"
    assert "admission_redundant_family" in q_row["admission_reason"]
    assert float(q_row["max_abs_corr_to_live"]) == 1.0


def test_admission_gate_caps_number_of_promoted_quarantine_alphas() -> None:
    alpha_panel, labels, label_available_at = _admission_panel()
    selector = RollingTopKSelector(
        candidate_alpha_ids=("wq_q_stable", "wq_q_redundant", "wq_live"),
        base_alpha_ids=("wq_live",),
        top_k=3,
        window_days=30,
        min_coverage=0.1,
        min_observations=4,
        admission_enabled=True,
        admission_max_promoted=1,
        admission_min_score=0.01,
        admission_min_coverage=0.1,
        admission_min_observations=4,
        admission_subwindows=3,
        admission_min_subwindow_passes=2,
        admission_subwindow_min_abs_ic=0.01,
        admission_max_abs_corr_to_live=1.0,
    )

    snapshot = selector.select(
        _context("2024-03-11", train_start="2024-03-01", train_end="2024-03-09"),
        alpha_panel=alpha_panel,
        labels=labels,
        label_available_at=label_available_at,
    )

    q_rows = snapshot.scores[snapshot.scores["alpha_pool"] == "quarantine"]
    assert int((q_rows["admission_status"] == "admitted").sum()) == 1
    assert int((q_rows["admission_reason"] == "admission_capacity").sum()) == 1
