"""Alpha selector 共用資料結構與 hash helper。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import pandas as pd


def stable_hash(payload: object) -> str:
    """以穩定 JSON 序列化產生 SHA256，供 snapshot 與 schema fingerprint 使用。"""
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def hash_alpha_ids(alpha_ids: list[str] | None) -> str:
    """Hash feature columns in model input order.

    XGBoost receives numpy arrays after columns are reindexed, so column order is
    part of the feature schema and must be reflected in the fingerprint.
    """
    return stable_hash({"alpha_ids": [str(a) for a in (alpha_ids or [])]})


def hash_universe(security_ids: list[str] | pd.Series | pd.Index) -> str:
    values = sorted({str(s) for s in list(security_ids)})
    return stable_hash({"security_ids": values})


@dataclass(frozen=True)
class SelectorContext:
    """一次 point-in-time selection event 的上下文。"""

    as_of_date: pd.Timestamp
    label_cutoff: pd.Timestamp
    train_window_start: pd.Timestamp | None
    train_window_end: pd.Timestamp | None
    label_horizon_days: int
    purge_days: int
    label_available_rule: str
    selector_config_hash: str
    feature_store_version: str
    bars_snapshot_hash: str | None
    universe_hash: str
    alpha_engine_version: str
    git_commit: str | None = None


@dataclass(frozen=True)
class AlphaSelectionSnapshot:
    """Selector 的 event-level metadata 與 per-alpha score table。"""

    event: dict[str, Any]
    scores: pd.DataFrame

    @property
    def snapshot_hash(self) -> str:
        return str(self.event["snapshot_hash"])

    @property
    def selected_alphas(self) -> list[str]:
        if self.scores.empty:
            return []
        selected = self.scores[self.scores["selected"]]
        return selected["alpha_id"].astype(str).tolist()

    @property
    def feature_columns_hash(self) -> str:
        return str(self.event["feature_columns_hash"])


def build_snapshot(
    *,
    context: SelectorContext,
    selector_name: str,
    selector_version: str,
    scores: pd.DataFrame,
) -> AlphaSelectionSnapshot:
    """把 selector score table 收斂成兩層 snapshot。"""
    score_cols = [
        "alpha_id",
        "selected",
        "weight",
        "raw_score",
        "score",
        "n_observations",
        "coverage",
        "rolling_rank_ic",
        "stability",
        "drift_score",
        "turnover_penalty",
        "alpha_pool",
        "admission_status",
        "admission_score",
        "admission_reason",
        "admission_subwindow_pass_count",
        "max_abs_corr_to_live",
        "excluded_reason",
    ]
    scores = scores.copy()
    for col in score_cols:
        if col not in scores.columns:
            scores[col] = None
    scores = scores[score_cols].reset_index(drop=True)

    selected_alphas = scores.loc[scores["selected"], "alpha_id"].astype(str).tolist()
    feature_columns_hash = hash_alpha_ids(selected_alphas)
    event_without_hash: dict[str, Any] = {
        "as_of_date": context.as_of_date.strftime("%Y-%m-%d"),
        "label_cutoff": context.label_cutoff.strftime("%Y-%m-%d"),
        "train_window_start": (
            context.train_window_start.strftime("%Y-%m-%d")
            if context.train_window_start is not None
            else None
        ),
        "train_window_end": (
            context.train_window_end.strftime("%Y-%m-%d")
            if context.train_window_end is not None
            else None
        ),
        "label_horizon_days": context.label_horizon_days,
        "purge_days": context.purge_days,
        "label_available_rule": context.label_available_rule,
        "selector_name": selector_name,
        "selector_version": selector_version,
        "selector_config_hash": context.selector_config_hash,
        "feature_store_version": context.feature_store_version,
        "bars_snapshot_hash": context.bars_snapshot_hash,
        "universe_hash": context.universe_hash,
        "alpha_engine_version": context.alpha_engine_version,
        "git_commit": context.git_commit,
        "n_candidate_alphas": int(len(scores)),
        "n_selected_alphas": int(len(selected_alphas)),
        "feature_columns_hash": feature_columns_hash,
    }
    snapshot_hash = stable_hash(
        {
            "event": event_without_hash,
            "scores": scores.to_dict("records"),
        }
    )
    event = {**event_without_hash, "snapshot_hash": snapshot_hash}
    scores.insert(0, "snapshot_hash", snapshot_hash)
    scores.insert(0, "as_of_date", event["as_of_date"])
    return AlphaSelectionSnapshot(event=event, scores=scores)
