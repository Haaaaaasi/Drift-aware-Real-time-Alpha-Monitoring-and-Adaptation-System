"""PostgreSQL persistence for the live daily operating layer."""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any
from uuid import UUID

import pandas as pd
from psycopg2.extras import Json, execute_batch

from src.common.db import get_pg_connection
from src.common.logging import get_logger
from src.alpha_selection.base import AlphaSelectionSnapshot

logger = get_logger(__name__)


def _clean(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return value


def _jsonable(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, ensure_ascii=False, default=str))


class LiveOperationalStore:
    """Persist and query daily live state keyed by ``run_id``."""

    def upsert_run(self, run: dict[str, Any]) -> None:
        conn = get_pg_connection()
        try:
            sql = """
                INSERT INTO daily_live_runs
                    (run_id, as_of_date, run_started_at, run_finished_at,
                     mode, run_purpose, is_official, status, data_source,
                     data_max_date, data_lag_days, data_freshness_status,
                     bars_path, bars_snapshot_hash,
                     alpha_cache_path, alpha_cache_manifest_hash, feature_store_version,
                     frozen_config_path, frozen_config_hash, frozen_selector_id,
                     selector_snapshot_hash, diagnostic_selector_snapshot_hash,
                     production_model_id, production_model_artifact_path,
                     feature_columns_hash, n_feature_alphas, retrain_action,
                     message, metadata)
                VALUES
                    (%(run_id)s, %(as_of_date)s, %(run_started_at)s, %(run_finished_at)s,
                     %(mode)s, %(run_purpose)s, %(is_official)s,
                     %(status)s, %(data_source)s, %(data_max_date)s,
                     %(data_lag_days)s, %(data_freshness_status)s, %(bars_path)s,
                     %(bars_snapshot_hash)s, %(alpha_cache_path)s,
                     %(alpha_cache_manifest_hash)s, %(feature_store_version)s,
                     %(frozen_config_path)s, %(frozen_config_hash)s,
                     %(frozen_selector_id)s, %(selector_snapshot_hash)s,
                     %(diagnostic_selector_snapshot_hash)s, %(production_model_id)s,
                     %(production_model_artifact_path)s, %(feature_columns_hash)s,
                     %(n_feature_alphas)s, %(retrain_action)s, %(message)s,
                     %(metadata)s)
                ON CONFLICT (run_id) DO UPDATE SET
                    run_finished_at = EXCLUDED.run_finished_at,
                    run_purpose = EXCLUDED.run_purpose,
                    is_official = EXCLUDED.is_official,
                    status = EXCLUDED.status,
                    data_max_date = EXCLUDED.data_max_date,
                    data_lag_days = EXCLUDED.data_lag_days,
                    data_freshness_status = EXCLUDED.data_freshness_status,
                    selector_snapshot_hash = EXCLUDED.selector_snapshot_hash,
                    diagnostic_selector_snapshot_hash = EXCLUDED.diagnostic_selector_snapshot_hash,
                    production_model_id = EXCLUDED.production_model_id,
                    production_model_artifact_path = EXCLUDED.production_model_artifact_path,
                    feature_columns_hash = EXCLUDED.feature_columns_hash,
                    n_feature_alphas = EXCLUDED.n_feature_alphas,
                    retrain_action = EXCLUDED.retrain_action,
                    message = EXCLUDED.message,
                    metadata = EXCLUDED.metadata,
                    updated_at = now()
            """
            payload = {**run, "metadata": Json(_jsonable(run.get("metadata") or {}))}
            with conn.cursor() as cur:
                cur.execute(sql, payload)
            conn.commit()
            logger.info("daily_live_run_upserted", run_id=str(run.get("run_id")))
        finally:
            conn.close()

    def persist_alpha_snapshot(
        self,
        run_id: UUID | str,
        snapshot: AlphaSelectionSnapshot | None,
        *,
        role: str,
    ) -> int:
        if snapshot is None:
            return 0
        conn = get_pg_connection()
        try:
            event = snapshot.event
            event_sql = """
                INSERT INTO alpha_selection_snapshots
                    (run_id, snapshot_hash, snapshot_role, as_of_date,
                     selector_name, selector_version, selector_config_hash,
                     feature_columns_hash, n_candidate_alphas, n_selected_alphas,
                     event_metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id, snapshot_hash, snapshot_role) DO UPDATE SET
                    event_metadata = EXCLUDED.event_metadata,
                    n_candidate_alphas = EXCLUDED.n_candidate_alphas,
                    n_selected_alphas = EXCLUDED.n_selected_alphas
            """
            score_sql = """
                INSERT INTO alpha_selection_scores
                    (run_id, snapshot_hash, snapshot_role, as_of_date, alpha_id,
                     selected, weight, raw_score, score, n_observations, coverage,
                     rolling_rank_ic, stability, drift_score, turnover_penalty,
                     alpha_pool, admission_status, admission_score,
                     admission_reason, excluded_reason)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id, snapshot_hash, snapshot_role, alpha_id)
                DO UPDATE SET
                    selected = EXCLUDED.selected,
                    weight = EXCLUDED.weight,
                    raw_score = EXCLUDED.raw_score,
                    score = EXCLUDED.score,
                    n_observations = EXCLUDED.n_observations,
                    coverage = EXCLUDED.coverage,
                    rolling_rank_ic = EXCLUDED.rolling_rank_ic,
                    stability = EXCLUDED.stability,
                    drift_score = EXCLUDED.drift_score,
                    turnover_penalty = EXCLUDED.turnover_penalty,
                    alpha_pool = EXCLUDED.alpha_pool,
                    admission_status = EXCLUDED.admission_status,
                    admission_score = EXCLUDED.admission_score,
                    admission_reason = EXCLUDED.admission_reason,
                    excluded_reason = EXCLUDED.excluded_reason
            """
            score_rows = []
            scores = snapshot.scores.copy()
            for _, row in scores.iterrows():
                score_rows.append(
                    (
                        str(run_id),
                        snapshot.snapshot_hash,
                        role,
                        pd.Timestamp(row["as_of_date"]).date(),
                        str(row["alpha_id"]),
                        bool(row["selected"]),
                        _clean(row.get("weight")),
                        _clean(row.get("raw_score")),
                        _clean(row.get("score")),
                        _clean(row.get("n_observations")),
                        _clean(row.get("coverage")),
                        _clean(row.get("rolling_rank_ic")),
                        _clean(row.get("stability")),
                        _clean(row.get("drift_score")),
                        _clean(row.get("turnover_penalty")),
                        _clean(row.get("alpha_pool")),
                        _clean(row.get("admission_status")),
                        _clean(row.get("admission_score")),
                        _clean(row.get("admission_reason")),
                        _clean(row.get("excluded_reason")),
                    )
                )
            with conn.cursor() as cur:
                cur.execute(
                    event_sql,
                    (
                        str(run_id),
                        snapshot.snapshot_hash,
                        role,
                        pd.Timestamp(event["as_of_date"]).date(),
                        event["selector_name"],
                        event.get("selector_version"),
                        event.get("selector_config_hash"),
                        event.get("feature_columns_hash"),
                        event.get("n_candidate_alphas"),
                        event.get("n_selected_alphas"),
                        Json(_jsonable(event)),
                    ),
                )
                execute_batch(cur, score_sql, score_rows, page_size=500)
            conn.commit()
            logger.info(
                "alpha_selection_snapshot_persisted",
                run_id=str(run_id),
                role=role,
                rows=len(score_rows),
            )
            return len(score_rows)
        finally:
            conn.close()

    def persist_meta_signals(self, run_id: UUID | str, signals: pd.DataFrame) -> int:
        if signals.empty:
            return 0
        conn = get_pg_connection()
        try:
            sql = """
                INSERT INTO meta_signals
                    (run_id, security_id, signal_time, bar_type, signal_score,
                     signal_direction, confidence, method, model_version_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            records = [
                (
                    str(run_id),
                    row["security_id"],
                    row["signal_time"],
                    row.get("bar_type", "daily"),
                    float(row["signal_score"]),
                    int(row["signal_direction"]),
                    _clean(row.get("confidence")),
                    row.get("method", "ml_meta"),
                    row.get("model_version_id"),
                )
                for _, row in signals.iterrows()
            ]
            with conn.cursor() as cur:
                execute_batch(cur, sql, records, page_size=1000)
            conn.commit()
            return len(records)
        finally:
            conn.close()

    def persist_portfolio_targets(self, run_id: UUID | str, targets: pd.DataFrame) -> int:
        if targets.empty:
            return 0
        conn = get_pg_connection()
        try:
            sql = """
                INSERT INTO portfolio_targets
                    (run_id, rebalance_time, security_id, target_weight,
                     target_shares, construction_method, pre_risk)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            records = [
                (
                    str(run_id),
                    row["rebalance_time"],
                    row["security_id"],
                    float(row["target_weight"]),
                    int(row.get("target_shares") or 0),
                    row.get("construction_method", "turnover_aware_topk"),
                    bool(row.get("pre_risk", False)),
                )
                for _, row in targets.iterrows()
            ]
            with conn.cursor() as cur:
                execute_batch(cur, sql, records, page_size=1000)
            conn.commit()
            return len(records)
        finally:
            conn.close()

    def persist_portfolio_snapshots(
        self,
        run_id: UUID | str,
        snapshots: pd.DataFrame,
    ) -> int:
        if snapshots.empty:
            return 0
        conn = get_pg_connection()
        try:
            sql = """
                INSERT INTO portfolio_snapshots
                    (run_id, as_of_date, snapshot_time, security_id,
                     current_weight, target_weight, target_shares, last_price,
                     market_value, unrealized_pnl, signal_score, rank,
                     holding_days, reason)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id, security_id) DO UPDATE SET
                    current_weight = EXCLUDED.current_weight,
                    target_weight = EXCLUDED.target_weight,
                    target_shares = EXCLUDED.target_shares,
                    last_price = EXCLUDED.last_price,
                    market_value = EXCLUDED.market_value,
                    unrealized_pnl = EXCLUDED.unrealized_pnl,
                    signal_score = EXCLUDED.signal_score,
                    rank = EXCLUDED.rank,
                    holding_days = EXCLUDED.holding_days,
                    reason = EXCLUDED.reason
            """
            records = [
                (
                    str(run_id),
                    row["as_of_date"],
                    row["snapshot_time"],
                    row["security_id"],
                    float(row.get("current_weight") or 0.0),
                    float(row.get("target_weight") or 0.0),
                    _clean(row.get("target_shares")),
                    _clean(row.get("last_price")),
                    _clean(row.get("market_value")),
                    _clean(row.get("unrealized_pnl")),
                    _clean(row.get("signal_score")),
                    _clean(row.get("rank")),
                    _clean(row.get("holding_days")),
                    _clean(row.get("reason")),
                )
                for _, row in snapshots.iterrows()
            ]
            with conn.cursor() as cur:
                execute_batch(cur, sql, records, page_size=1000)
            conn.commit()
            return len(records)
        finally:
            conn.close()

    def persist_trade_recommendations(
        self,
        run_id: UUID | str,
        recommendations: pd.DataFrame,
    ) -> int:
        if recommendations.empty:
            return 0
        conn = get_pg_connection()
        try:
            sql = """
                INSERT INTO trade_recommendations
                    (run_id, as_of_date, security_id, action, current_weight,
                     target_weight, delta_weight, current_shares, target_shares,
                     delta_shares, last_price, signal_score, rank, reason, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id, security_id) DO UPDATE SET
                    action = EXCLUDED.action,
                    current_weight = EXCLUDED.current_weight,
                    target_weight = EXCLUDED.target_weight,
                    delta_weight = EXCLUDED.delta_weight,
                    current_shares = EXCLUDED.current_shares,
                    target_shares = EXCLUDED.target_shares,
                    delta_shares = EXCLUDED.delta_shares,
                    last_price = EXCLUDED.last_price,
                    signal_score = EXCLUDED.signal_score,
                    rank = EXCLUDED.rank,
                    reason = EXCLUDED.reason,
                    status = EXCLUDED.status,
                    updated_at = now()
            """
            records = [
                (
                    str(run_id),
                    row["as_of_date"],
                    row["security_id"],
                    row["action"],
                    float(row.get("current_weight") or 0.0),
                    float(row.get("target_weight") or 0.0),
                    float(row.get("delta_weight") or 0.0),
                    _clean(row.get("current_shares")),
                    _clean(row.get("target_shares")),
                    _clean(row.get("delta_shares")),
                    _clean(row.get("last_price")),
                    _clean(row.get("signal_score")),
                    _clean(row.get("rank")),
                    _clean(row.get("reason")),
                    row.get("status", "PENDING"),
                )
                for _, row in recommendations.iterrows()
            ]
            with conn.cursor() as cur:
                execute_batch(cur, sql, records, page_size=1000)
            conn.commit()
            return len(records)
        finally:
            conn.close()

    def load_previous_portfolio_state(
        self,
        as_of_date: date | datetime | pd.Timestamp,
        *,
        official_only: bool = True,
    ) -> tuple[dict[str, float], dict[str, int], dict[str, int]]:
        """讀取最近一次 live target 作為今日 current state 的預設值。"""
        conn = get_pg_connection()
        try:
            official_filter = "AND r.is_official = true" if official_only else ""
            df = pd.read_sql(
                f"""
                SELECT ps.security_id, ps.target_weight, ps.target_shares, ps.holding_days
                FROM portfolio_snapshots ps
                JOIN daily_live_runs r ON r.run_id = ps.run_id
                WHERE ps.as_of_date = (
                    SELECT max(ps2.as_of_date)
                    FROM portfolio_snapshots ps2
                    JOIN daily_live_runs r2 ON r2.run_id = ps2.run_id
                    WHERE ps2.as_of_date < %s
                      {'AND r2.is_official = true' if official_only else ''}
                )
                {official_filter}
                """,
                conn,
                params=[pd.Timestamp(as_of_date).date()],
            )
            if df.empty:
                return {}, {}, {}
            weights = {
                str(row["security_id"]): float(row["target_weight"])
                for _, row in df.iterrows()
                if abs(float(row["target_weight"])) > 1e-12
            }
            shares = {
                str(row["security_id"]): int(row["target_shares"])
                for _, row in df.dropna(subset=["target_shares"]).iterrows()
            }
            holding_days = {
                str(row["security_id"]): int(row["holding_days"])
                for _, row in df.dropna(subset=["holding_days"]).iterrows()
            }
            return weights, shares, holding_days
        finally:
            conn.close()
