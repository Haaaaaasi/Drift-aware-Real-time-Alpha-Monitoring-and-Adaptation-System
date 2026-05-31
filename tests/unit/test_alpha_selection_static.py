from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.alpha_selection import SelectorContext, StaticISSelector, hash_alpha_ids, hash_universe
from src.alpha_selection.snapshot import write_selection_artifacts
from src.config.constants import WQ101_INDCLASS_OR_CAP_ALPHA_IDS


def _write_effective(path: Path, alphas: list[str]) -> None:
    path.write_text(json.dumps({"effective_alphas": alphas}), encoding="utf-8")


def _context() -> SelectorContext:
    return SelectorContext(
        as_of_date=pd.Timestamp("2024-07-01"),
        label_cutoff=pd.Timestamp("2024-07-01"),
        train_window_start=pd.Timestamp("2023-02-17"),
        train_window_end=pd.Timestamp("2024-06-26"),
        label_horizon_days=5,
        purge_days=5,
        label_available_rule="label_available_at <= as_of_date",
        selector_config_hash="cfg",
        feature_store_version="feature-store-v1",
        bars_snapshot_hash="bars-v1",
        universe_hash=hash_universe(["2330", "2317"]),
        alpha_engine_version="python_wq101_v1",
        git_commit="abc123",
    )


def test_static_is_matches_effective_list_and_excludes_placeholder_alpha(tmp_path: Path) -> None:
    effective_path = tmp_path / "effective_alphas.json"
    blocked = WQ101_INDCLASS_OR_CAP_ALPHA_IDS[0]
    _write_effective(effective_path, ["wq001", blocked, "wq002"])

    selector = StaticISSelector(
        effective_alphas_path=effective_path,
        exclude_indclass_cap=True,
    )

    assert selector.selected_alpha_ids() == ["wq001", "wq002"]


def test_static_is_snapshot_has_event_and_per_alpha_scores(tmp_path: Path) -> None:
    effective_path = tmp_path / "effective_alphas.json"
    blocked = WQ101_INDCLASS_OR_CAP_ALPHA_IDS[0]
    _write_effective(effective_path, ["wq001", blocked])
    selector = StaticISSelector(
        effective_alphas_path=effective_path,
        exclude_indclass_cap=True,
    )

    snapshot = selector.select(_context())

    assert snapshot.event["selector_name"] == "static_is"
    assert snapshot.event["n_selected_alphas"] == 1
    assert snapshot.event["feature_columns_hash"] == hash_alpha_ids(["wq001"])
    blocked_row = snapshot.scores[snapshot.scores["alpha_id"] == blocked].iloc[0]
    assert bool(blocked_row["selected"]) is False
    assert blocked_row["excluded_reason"] == "requires_indclass_or_cap"


def test_static_is_snapshot_preserves_legacy_feature_order(tmp_path: Path) -> None:
    effective_path = tmp_path / "effective_alphas.json"
    _write_effective(effective_path, ["wq019", "wq001", "wq010"])

    snapshot = StaticISSelector(effective_alphas_path=effective_path).select(_context())

    assert snapshot.selected_alphas == ["wq019", "wq001", "wq010"]
    assert snapshot.event["feature_columns_hash"] == hash_alpha_ids(["wq019", "wq001", "wq010"])
    assert hash_alpha_ids(["wq019", "wq001"]) != hash_alpha_ids(["wq001", "wq019"])


def test_write_selection_artifacts_splits_metadata_from_scores(tmp_path: Path) -> None:
    effective_path = tmp_path / "effective_alphas.json"
    _write_effective(effective_path, ["wq001", "wq002"])
    snapshot = StaticISSelector(effective_alphas_path=effective_path).select(_context())

    paths = write_selection_artifacts([snapshot], tmp_path)

    events = pd.read_csv(paths["snapshots_path"])
    scores = pd.read_csv(paths["scores_path"])
    weights = pd.read_csv(paths["weights_path"])
    assert len(events) == 1
    assert set(["snapshot_hash", "feature_store_version", "feature_columns_hash"]).issubset(events.columns)
    assert set(["alpha_id", "selected", "weight", "excluded_reason"]).issubset(scores.columns)
    assert len(weights) == 2
