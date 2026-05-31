"""Daily online operating pipeline for DARAMS.

這條 pipeline 是日常 EOD 使用路徑：載入 production model artifact、計算當日
alpha、產生明日 target holdings 與 trade recommendations。除非沒有 production
model、到 scheduled retrain 週期，或指定 ``--force-retrain``，否則不重新訓練。
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import pandas as pd

from pipelines.daily_batch_pipeline import load_csv_data
from pipelines.simulate_recent import (
    _bars_snapshot_hash,
    _git_metadata,
    _make_static_selector,
)
from src.adaptation.model_registry import ModelRegistryManager
from src.alpha_engine.alpha_cache import cache_path_for_data_path, read_cache_manifest
from src.alpha_engine.feature_store import ALPHA_ENGINE_VERSION, FeatureStore
from src.alpha_selection import (
    AlphaSelectionSnapshot,
    RollingTopKSelector,
    SelectorContext,
    hash_alpha_ids,
    hash_universe,
    stable_hash,
)
from src.alpha_selection.snapshot import write_selection_artifacts
from src.common.logging import get_logger, setup_logging
from src.config.frozen_alpha_selector import load_frozen_alpha_selector
from src.labeling.label_generator import LabelGenerator
from src.live import LiveOperationalStore
from src.meta_signal.ml_meta_model import MLMetaModel
from src.portfolio.live_service import LivePortfolioConfig, build_live_portfolio

setup_logging()
logger = get_logger("daily_online_pipeline")

DEFAULT_FROZEN_CONFIG = Path("configs/frozen_alpha_selector_20260517.yaml")
DEFAULT_ARTIFACT_ROOT = Path("artifacts/models")
DEFAULT_OUTPUT_DIR = Path("reports/live")
PRODUCTION_POINTER = "production.json"

Mode = Literal["auto", "predict-only", "train-only"]


def run_daily_online(
    *,
    mode: Mode = "auto",
    as_of: date | None = None,
    frozen_config: str | Path = DEFAULT_FROZEN_CONFIG,
    frozen_execution: str = "primary",
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    production_artifact: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    capital: float = 10_000_000.0,
    force_retrain: bool = False,
    run_purpose: str = "production",
    is_official: bool = False,
    persist_db: bool = True,
) -> dict[str, Any]:
    frozen_spec = load_frozen_alpha_selector(frozen_config)
    overrides = frozen_spec.simulation_overrides(frozen_execution)
    frozen_meta = frozen_spec.metadata(frozen_execution)
    csv_path = Path(overrides["csv_path"])
    data_source = str(overrides["data_source"])
    allow_yfinance = bool(overrides["allow_yfinance"])
    horizon_days = int(overrides["horizon_days"])
    purge_days = int(overrides["purge_days"])
    train_window_days = int(overrides["train_window_days"])
    retrain_every = int(overrides.get("retrain_every", 20))

    run_id = str(uuid4())
    run_started_at = datetime.utcnow()
    artifact_root = Path(artifact_root)
    output_root = Path(output_dir)
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    run_purpose = str(run_purpose).lower()
    if run_purpose not in {"production", "smoke", "backfill"}:
        raise ValueError("run_purpose must be production, smoke, or backfill")
    is_official = bool(is_official and run_purpose == "production")

    bars = load_csv_data(csv_path, allow_yfinance=allow_yfinance)
    data_max_ts = pd.to_datetime(bars["tradetime"]).dt.normalize().max()
    as_of_ts = _resolve_as_of(bars, as_of)
    freshness = _data_freshness(
        as_of_ts=as_of_ts,
        data_max_ts=data_max_ts,
        requested_as_of=as_of,
    )
    bars = bars[bars["tradetime"] <= as_of_ts].reset_index(drop=True)
    bars_snapshot_hash = _bars_snapshot_hash(csv_path, bars)
    universe_hash = hash_universe(bars["security_id"].unique())
    feature_store = FeatureStore.for_data_path(csv_path)
    feature_store_version = feature_store.version
    cache_path = cache_path_for_data_path(csv_path)
    cache_manifest_hash = stable_hash(read_cache_manifest(cache_path))
    git_meta = _git_metadata()

    static_selector = _make_static_selector(
        alpha_ids=None,
        skip_effective_filter=False,
        exclude_indclass_cap_alphas=bool(overrides["exclude_indclass_cap_alphas"]),
    )
    candidate_alpha_ids = static_selector.selected_alpha_ids()
    if not candidate_alpha_ids:
        raise RuntimeError("Frozen config resolved no candidate alpha ids.")

    alpha_panel = feature_store.load_alpha_panel(bars, alpha_ids=candidate_alpha_ids)
    labels, label_available_at = _build_training_labels(
        bars=bars,
        horizon_days=horizon_days,
    )
    diagnostic_snapshot = _select_rolling_topk_snapshot(
        as_of_ts=as_of_ts,
        train_window_days=train_window_days,
        purge_days=purge_days,
        horizon_days=horizon_days,
        candidate_alpha_ids=candidate_alpha_ids,
        alpha_panel=alpha_panel,
        labels=labels,
        label_available_at=label_available_at,
        overrides=overrides,
        feature_store_version=feature_store_version,
        bars_snapshot_hash=bars_snapshot_hash,
        universe_hash=universe_hash,
        git_sha=git_meta.get("git_sha"),
    )

    model, artifact_path, model_manifest, retrain_action, production_snapshot = _resolve_model(
        mode=mode,
        as_of_ts=as_of_ts,
        artifact_root=artifact_root,
        production_artifact=production_artifact,
        force_retrain=force_retrain,
        retrain_every=retrain_every,
        diagnostic_snapshot=diagnostic_snapshot,
        alpha_panel=alpha_panel,
        labels=labels,
        label_available_at=label_available_at,
        overrides=overrides,
        frozen_meta=frozen_meta,
        feature_store_version=feature_store_version,
        bars_snapshot_hash=bars_snapshot_hash,
        cache_manifest_hash=cache_manifest_hash,
        git_meta=git_meta,
        persist_db=persist_db,
    )

    if mode == "train-only":
        signals = pd.DataFrame()
        portfolio_result = None
    else:
        signals = _predict_today(model, alpha_panel, as_of_ts)
        signals["method"] = "ml_meta"
        signals["model_version_id"] = model.model_id
        signals["bar_type"] = "daily"

        previous_weights, previous_shares, holding_days = _load_previous_state(
            persist_db=persist_db,
            as_of_ts=as_of_ts,
        )
        last_prices = (
            bars[bars["tradetime"] == as_of_ts][["security_id", "close"]]
            .set_index("security_id")["close"]
        )
        portfolio_result = build_live_portfolio(
            signals=signals,
            as_of_date=as_of_ts,
            previous_weights=previous_weights,
            previous_shares=previous_shares,
            holding_days=holding_days,
            last_prices=last_prices,
            capital=capital,
            config=LivePortfolioConfig(
                method=str(overrides["portfolio_method"]),
                top_k=int(overrides["top_k"]),
                entry_rank=int(overrides["entry_rank"]),
                exit_rank=int(overrides["exit_rank"]),
                max_turnover=float(overrides["max_turnover"]),
                min_holding_days=int(overrides["min_holding_days"]),
                tail_cleanup_weight=float(overrides["tail_cleanup_weight"]),
            ),
        )

    production_feature_snapshot = _production_feature_snapshot(
        as_of_ts=as_of_ts,
        model=model,
        manifest=model_manifest,
        fallback_snapshot=production_snapshot,
    )
    _write_run_artifacts(
        run_dir=run_dir,
        signals=signals,
        portfolio_result=portfolio_result,
        production_snapshot=production_feature_snapshot,
        diagnostic_snapshot=diagnostic_snapshot,
    )

    run_record = {
        "run_id": run_id,
        "as_of_date": as_of_ts.date(),
        "run_started_at": run_started_at,
        "run_finished_at": datetime.utcnow(),
        "mode": mode,
        "run_purpose": run_purpose,
        "is_official": is_official,
        "status": "COMPLETED",
        "data_source": data_source,
        "data_max_date": data_max_ts.date(),
        "data_lag_days": freshness["data_lag_days"],
        "data_freshness_status": freshness["data_freshness_status"],
        "bars_path": str(csv_path.as_posix()),
        "bars_snapshot_hash": bars_snapshot_hash,
        "alpha_cache_path": str(cache_path.as_posix()),
        "alpha_cache_manifest_hash": cache_manifest_hash,
        "feature_store_version": feature_store_version,
        "frozen_config_path": frozen_meta["frozen_config_path"],
        "frozen_config_hash": frozen_meta["frozen_config_hash"],
        "frozen_selector_id": frozen_meta["frozen_selector_id"],
        "selector_snapshot_hash": model_manifest.get("selector_snapshot_hash"),
        "diagnostic_selector_snapshot_hash": diagnostic_snapshot.snapshot_hash,
        "production_model_id": model.model_id,
        "production_model_artifact_path": str(Path(artifact_path).as_posix()),
        "feature_columns_hash": hash_alpha_ids(model.feature_columns),
        "n_feature_alphas": len(model.feature_columns),
        "retrain_action": retrain_action,
        "message": "daily online run completed",
        "metadata": {
            **frozen_meta,
            **git_meta,
            "capital": capital,
            "data_freshness": freshness,
            "portfolio_metrics": (
                portfolio_result.metrics if portfolio_result is not None else {}
            ),
            "run_dir": str(run_dir.as_posix()),
        },
    }
    if persist_db:
        _persist_run_outputs(
            run_record=run_record,
            production_snapshot=production_feature_snapshot,
            diagnostic_snapshot=diagnostic_snapshot,
            signals=signals,
            portfolio_result=portfolio_result,
        )

    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(run_record, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "as_of_date": str(as_of_ts.date()),
        "mode": mode,
        "run_purpose": run_purpose,
        "is_official": is_official,
        "data_freshness_status": freshness["data_freshness_status"],
        "data_lag_days": freshness["data_lag_days"],
        "retrain_action": retrain_action,
        "model_id": model.model_id,
        "artifact_path": str(artifact_path),
        "feature_columns_hash": hash_alpha_ids(model.feature_columns),
        "n_feature_alphas": len(model.feature_columns),
        "n_recommendations": (
            0 if portfolio_result is None else int(len(portfolio_result.recommendations))
        ),
    }


def _resolve_model(
    *,
    mode: Mode,
    as_of_ts: pd.Timestamp,
    artifact_root: Path,
    production_artifact: str | Path | None,
    force_retrain: bool,
    retrain_every: int,
    diagnostic_snapshot: AlphaSelectionSnapshot,
    alpha_panel: pd.DataFrame,
    labels: pd.Series,
    label_available_at: pd.Series,
    overrides: dict[str, Any],
    frozen_meta: dict[str, Any],
    feature_store_version: str,
    bars_snapshot_hash: str,
    cache_manifest_hash: str,
    git_meta: dict[str, Any],
    persist_db: bool,
) -> tuple[MLMetaModel, Path, dict[str, Any], str, AlphaSelectionSnapshot | None]:
    existing_artifact = (
        Path(production_artifact)
        if production_artifact is not None
        else _find_production_artifact(artifact_root, persist_db=persist_db)
    )
    if mode == "predict-only" and existing_artifact is None:
        raise RuntimeError(
            "predict-only requires a production artifact. Run auto/train-only first "
            "or pass --production-artifact."
        )

    if existing_artifact is not None and not force_retrain and mode != "train-only":
        manifest = _read_model_manifest(existing_artifact)
        due = _scheduled_due(
            manifest=manifest,
            as_of_ts=as_of_ts,
            retrain_every=retrain_every,
        )
        if mode == "predict-only" or not due:
            model = MLMetaModel.load_artifact(existing_artifact)
            return model, existing_artifact, manifest, "predict_only", None

    model, train_info, production_snapshot = _train_model_for_asof(
        as_of_ts=as_of_ts,
        alpha_panel=alpha_panel,
        labels=labels,
        label_available_at=label_available_at,
        selected_snapshot=diagnostic_snapshot,
        overrides=overrides,
    )
    artifact_dir = artifact_root / model.model_id
    extra_manifest = {
        **frozen_meta,
        **git_meta,
        "trained_as_of": as_of_ts.strftime("%Y-%m-%d"),
        "selector_snapshot_hash": production_snapshot.snapshot_hash,
        "selector_event": production_snapshot.event,
        "feature_store_version": feature_store_version,
        "bars_snapshot_hash": bars_snapshot_hash,
        "alpha_cache_manifest_hash": cache_manifest_hash,
        "label_horizon_days": int(overrides["horizon_days"]),
        "purge_days": int(overrides["purge_days"]),
        "train_info": train_info,
    }
    artifact_path = model.save_artifact(artifact_dir, extra_manifest=extra_manifest)
    manifest = _read_model_manifest(artifact_path)
    _write_production_pointer(artifact_root, artifact_path, manifest)
    if persist_db:
        _register_and_promote_model(
            model=model,
            train_info=train_info,
            artifact_path=artifact_path,
            manifest=manifest,
        )
    action = "forced_retrain" if force_retrain else "scheduled_or_initial_retrain"
    return model, artifact_path, manifest, action, production_snapshot


def _train_model_for_asof(
    *,
    as_of_ts: pd.Timestamp,
    alpha_panel: pd.DataFrame,
    labels: pd.Series,
    label_available_at: pd.Series,
    selected_snapshot: AlphaSelectionSnapshot,
    overrides: dict[str, Any],
) -> tuple[MLMetaModel, dict[str, Any], AlphaSelectionSnapshot]:
    active_alpha_ids = selected_snapshot.selected_alphas
    if not active_alpha_ids:
        raise RuntimeError("Rolling selector selected no alpha for production training.")
    purge_days = int(overrides["purge_days"])
    train_window_days = int(overrides["train_window_days"])
    purge_cutoff = as_of_ts - pd.Timedelta(days=purge_days)
    window_start = purge_cutoff - pd.Timedelta(days=train_window_days)
    train_panel = alpha_panel[
        (alpha_panel["tradetime"] >= window_start)
        & (alpha_panel["tradetime"] <= purge_cutoff)
        & (alpha_panel["alpha_id"].isin(active_alpha_ids))
    ]
    label_dates = labels.index.get_level_values("tradetime")
    train_labels = labels[
        (label_available_at <= as_of_ts)
        & (label_dates >= window_start)
        & (label_dates <= purge_cutoff)
    ]
    if len(train_labels) < 100:
        raise RuntimeError(f"Not enough mature labels for live training: {len(train_labels)}")

    model = MLMetaModel(
        feature_columns=active_alpha_ids,
        objective=str(overrides["objective"]),
        proxy_top_k=int(overrides["top_k"]),
    )
    train_info = model.train(train_panel, train_labels, purge_days=purge_days)
    return model, train_info, selected_snapshot


def _select_rolling_topk_snapshot(
    *,
    as_of_ts: pd.Timestamp,
    train_window_days: int,
    purge_days: int,
    horizon_days: int,
    candidate_alpha_ids: list[str],
    alpha_panel: pd.DataFrame,
    labels: pd.Series,
    label_available_at: pd.Series,
    overrides: dict[str, Any],
    feature_store_version: str,
    bars_snapshot_hash: str,
    universe_hash: str,
    git_sha: str | None,
) -> AlphaSelectionSnapshot:
    purge_cutoff = as_of_ts - pd.Timedelta(days=purge_days)
    window_start = purge_cutoff - pd.Timedelta(days=train_window_days)
    selector = RollingTopKSelector(
        candidate_alpha_ids=tuple(candidate_alpha_ids),
        top_k=int(overrides["selector_alpha_top_k"]),
        window_days=int(overrides["selector_window_days"]),
        min_coverage=float(overrides["selector_min_coverage"]),
        min_observations=int(overrides["selector_min_observations"]),
        stability_penalty=float(overrides["selector_stability_penalty"]),
    )
    context = SelectorContext(
        as_of_date=as_of_ts,
        label_cutoff=as_of_ts,
        train_window_start=window_start,
        train_window_end=purge_cutoff,
        label_horizon_days=horizon_days,
        purge_days=purge_days,
        label_available_rule="label_available_at <= as_of_date",
        selector_config_hash=selector.config_hash,
        feature_store_version=feature_store_version,
        bars_snapshot_hash=bars_snapshot_hash,
        universe_hash=universe_hash,
        alpha_engine_version=ALPHA_ENGINE_VERSION,
        git_commit=git_sha,
    )
    train_panel = alpha_panel[
        (alpha_panel["tradetime"] >= window_start)
        & (alpha_panel["tradetime"] <= purge_cutoff)
    ]
    return selector.select(
        context,
        alpha_panel=train_panel,
        labels=labels,
        label_available_at=label_available_at,
    )


def _build_training_labels(
    *,
    bars: pd.DataFrame,
    horizon_days: int,
) -> tuple[pd.Series, pd.Series]:
    label_gen = LabelGenerator(horizons=[horizon_days], bar_type="daily")
    labels_df = label_gen.generate_labels(bars[["security_id", "tradetime", "close"]])
    labels_h = (
        labels_df[labels_df["horizon"] == horizon_days]
        .dropna(subset=["forward_return"])
        .set_index(["security_id", "signal_time"])
        .rename_axis(index=["security_id", "tradetime"])
    )
    return labels_h["forward_return"], labels_h["label_available_at"]


def _predict_today(
    model: MLMetaModel,
    alpha_panel: pd.DataFrame,
    as_of_ts: pd.Timestamp,
) -> pd.DataFrame:
    todays_panel = alpha_panel[
        (alpha_panel["tradetime"] == as_of_ts)
        & (alpha_panel["alpha_id"].isin(model.feature_columns))
    ]
    if todays_panel.empty:
        raise RuntimeError(f"No alpha values for as-of date {as_of_ts.date()}")
    return model.predict(todays_panel).rename(columns={"tradetime": "signal_time"})


def _resolve_as_of(bars: pd.DataFrame, requested: date | None) -> pd.Timestamp:
    days = pd.to_datetime(bars["tradetime"]).dt.normalize().drop_duplicates().sort_values()
    if requested is None:
        return days.iloc[-1]
    requested_ts = pd.Timestamp(requested)
    if requested_ts in set(days):
        return requested_ts
    fallback = days[days <= requested_ts]
    if fallback.empty:
        raise ValueError(f"No bars on or before {requested}; data starts {days.iloc[0].date()}")
    chosen = fallback.iloc[-1]
    logger.warning("as_of_not_in_bars", requested=str(requested), chosen=str(chosen.date()))
    return chosen


def _data_freshness(
    *,
    as_of_ts: pd.Timestamp,
    data_max_ts: pd.Timestamp,
    requested_as_of: date | None,
) -> dict[str, Any]:
    today = pd.Timestamp(date.today()).normalize()
    as_of_day = pd.Timestamp(as_of_ts).normalize()
    max_day = pd.Timestamp(data_max_ts).normalize()
    lag_days = max(0, int((today - as_of_day).days))
    if requested_as_of is not None and as_of_day < max_day:
        status = "BACKDATED"
    elif lag_days <= 3:
        status = "FRESH"
    else:
        status = "STALE"
    return {
        "data_max_date": max_day.strftime("%Y-%m-%d"),
        "as_of_date": as_of_day.strftime("%Y-%m-%d"),
        "data_lag_days": lag_days,
        "data_freshness_status": status,
    }


def _scheduled_due(
    *,
    manifest: dict[str, Any],
    as_of_ts: pd.Timestamp,
    retrain_every: int,
) -> bool:
    trained_as_of = manifest.get("trained_as_of")
    if not trained_as_of:
        return False
    trained_ts = pd.Timestamp(trained_as_of)
    if as_of_ts <= trained_ts:
        return False
    # 若沒有完整交易日曆，先用 business days 作保守近似；實際訓練仍用成熟 label gate。
    elapsed = len(pd.bdate_range(trained_ts + pd.Timedelta(days=1), as_of_ts))
    return elapsed >= retrain_every


def _find_production_artifact(
    artifact_root: Path,
    *,
    persist_db: bool,
) -> Path | None:
    if persist_db:
        try:
            model = ModelRegistryManager().get_production_model()
            if model and model.get("artifact_path"):
                path = Path(str(model["artifact_path"]))
                if path.exists():
                    return path
        except Exception as exc:
            logger.warning("production_model_registry_lookup_failed", error=str(exc))
    pointer = artifact_root / PRODUCTION_POINTER
    if pointer.exists():
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        path = Path(payload["artifact_path"])
        if path.exists():
            return path
    return None


def _read_model_manifest(artifact_path: str | Path) -> dict[str, Any]:
    return json.loads((Path(artifact_path) / "manifest.json").read_text(encoding="utf-8"))


def _write_production_pointer(
    artifact_root: Path,
    artifact_path: Path,
    manifest: dict[str, Any],
) -> None:
    artifact_root.mkdir(parents=True, exist_ok=True)
    pointer = {
        "model_id": manifest.get("model_id"),
        "artifact_path": str(artifact_path.as_posix()),
        "feature_columns_hash": manifest.get("feature_columns_hash"),
        "selector_snapshot_hash": manifest.get("selector_snapshot_hash"),
        "updated_at": datetime.utcnow().isoformat(),
    }
    (artifact_root / PRODUCTION_POINTER).write_text(
        json.dumps(pointer, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _register_and_promote_model(
    *,
    model: MLMetaModel,
    train_info: dict[str, Any],
    artifact_path: Path,
    manifest: dict[str, Any],
) -> None:
    if model.trained_at is None or model.training_window is None:
        return
    try:
        mgr = ModelRegistryManager()
        metrics = {
            **model.holdout_metrics,
            "selector_snapshot_hash": manifest.get("selector_snapshot_hash"),
            "frozen_config_hash": manifest.get("frozen_config_hash"),
            "feature_columns_hash": train_info.get("feature_columns_hash"),
        }
        mgr.register_model(
            model_id=model.model_id,
            model_type="xgboost_regressor",
            trained_at=model.trained_at,
            training_window=model.training_window,
            features_used=model.feature_columns,
            hyperparams=model.hyperparams,
            holdout_metrics=metrics,
            artifact_path=str(artifact_path.as_posix()),
        )
        mgr.promote_model(model.model_id)
    except Exception as exc:
        logger.warning("live_model_registry_promote_failed", model_id=model.model_id, error=str(exc))


def _load_previous_state(
    *,
    persist_db: bool,
    as_of_ts: pd.Timestamp,
) -> tuple[dict[str, float], dict[str, int], dict[str, int]]:
    if not persist_db:
        return {}, {}, {}
    try:
        return LiveOperationalStore().load_previous_portfolio_state(as_of_ts)
    except Exception as exc:
        logger.warning("previous_live_state_load_failed", error=str(exc))
        return {}, {}, {}


def _production_feature_snapshot(
    *,
    as_of_ts: pd.Timestamp,
    model: MLMetaModel,
    manifest: dict[str, Any],
    fallback_snapshot: AlphaSelectionSnapshot | None,
) -> AlphaSelectionSnapshot:
    if fallback_snapshot is not None:
        return fallback_snapshot
    rows = [
        {
            "alpha_id": alpha_id,
            "selected": True,
            "weight": 1.0 / max(len(model.feature_columns), 1),
            "score": None,
            "excluded_reason": None,
        }
        for alpha_id in model.feature_columns
    ]
    event = {
        "as_of_date": as_of_ts.strftime("%Y-%m-%d"),
        "label_cutoff": manifest.get("trained_as_of"),
        "train_window_start": manifest.get("training_window_start"),
        "train_window_end": manifest.get("training_window_end"),
        "label_horizon_days": manifest.get("label_horizon_days"),
        "purge_days": manifest.get("purge_days"),
        "label_available_rule": "production artifact feature_columns",
        "selector_name": "production_artifact_features",
        "selector_version": "production_artifact_v1",
        "selector_config_hash": None,
        "feature_store_version": manifest.get("feature_store_version"),
        "bars_snapshot_hash": manifest.get("bars_snapshot_hash"),
        "universe_hash": None,
        "alpha_engine_version": ALPHA_ENGINE_VERSION,
        "git_commit": manifest.get("git_sha"),
        "n_candidate_alphas": len(rows),
        "n_selected_alphas": len(rows),
        "feature_columns_hash": hash_alpha_ids(model.feature_columns),
        "snapshot_hash": manifest.get("selector_snapshot_hash")
        or stable_hash({"model_id": model.model_id, "features": model.feature_columns}),
    }
    scores = pd.DataFrame(rows)
    scores.insert(0, "snapshot_hash", event["snapshot_hash"])
    scores.insert(0, "as_of_date", event["as_of_date"])
    return AlphaSelectionSnapshot(event=event, scores=scores)


def _write_run_artifacts(
    *,
    run_dir: Path,
    signals: pd.DataFrame,
    portfolio_result: Any,
    production_snapshot: AlphaSelectionSnapshot,
    diagnostic_snapshot: AlphaSelectionSnapshot,
) -> None:
    signals.to_csv(run_dir / "signals.csv", index=False)
    if portfolio_result is not None:
        portfolio_result.targets.to_csv(run_dir / "portfolio_targets.csv", index=False)
        portfolio_result.recommendations.to_csv(
            run_dir / "trade_recommendations.csv",
            index=False,
        )
        portfolio_result.snapshot.to_csv(run_dir / "portfolio_snapshot.csv", index=False)
    production_dir = run_dir / "production_alpha"
    diagnostic_dir = run_dir / "diagnostic_alpha"
    production_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    write_selection_artifacts([production_snapshot], production_dir)
    write_selection_artifacts([diagnostic_snapshot], diagnostic_dir)


def _persist_run_outputs(
    *,
    run_record: dict[str, Any],
    production_snapshot: AlphaSelectionSnapshot,
    diagnostic_snapshot: AlphaSelectionSnapshot,
    signals: pd.DataFrame,
    portfolio_result: Any,
) -> None:
    store = LiveOperationalStore()
    try:
        store.upsert_run(run_record)
        store.persist_alpha_snapshot(run_record["run_id"], production_snapshot, role="production")
        store.persist_alpha_snapshot(run_record["run_id"], diagnostic_snapshot, role="diagnostic")
        if not signals.empty:
            store.persist_meta_signals(run_record["run_id"], signals)
        if portfolio_result is not None:
            store.persist_portfolio_targets(run_record["run_id"], portfolio_result.targets)
            store.persist_portfolio_snapshots(run_record["run_id"], portfolio_result.snapshot)
            store.persist_trade_recommendations(
                run_record["run_id"],
                portfolio_result.recommendations,
            )
    except Exception as exc:
        logger.warning("live_operational_persist_failed", run_id=run_record["run_id"], error=str(exc))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["auto", "predict-only", "train-only"], default="auto")
    parser.add_argument("--as-of", help="As-of date YYYY-MM-DD; default latest bar date")
    parser.add_argument("--frozen-config", default=str(DEFAULT_FROZEN_CONFIG))
    parser.add_argument("--frozen-execution", default="primary")
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--production-artifact")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--capital", type=float, default=10_000_000.0)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument(
        "--run-purpose",
        choices=["production", "smoke", "backfill"],
        default="production",
        help="Label this live run for UI filtering.",
    )
    parser.add_argument(
        "--official",
        action="store_true",
        help="Mark this production run as official. Ignored for smoke/backfill.",
    )
    parser.add_argument("--no-db", action="store_true", help="Do not persist live state to PostgreSQL")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    as_of = datetime.strptime(args.as_of, "%Y-%m-%d").date() if args.as_of else None
    result = run_daily_online(
        mode=args.mode,
        as_of=as_of,
        frozen_config=args.frozen_config,
        frozen_execution=args.frozen_execution,
        artifact_root=args.artifact_root,
        production_artifact=args.production_artifact,
        output_dir=args.output_dir,
        capital=args.capital,
        force_retrain=args.force_retrain,
        run_purpose=args.run_purpose,
        is_official=args.official,
        persist_db=not args.no_db,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
