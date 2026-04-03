"""Layer 10 — Model registry: version management and lifecycle."""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
from psycopg2.extras import execute_batch

from src.common.db import get_pg_connection
from src.common.logging import get_logger
from src.config.constants import ModelStatus

logger = get_logger(__name__)


class ModelRegistryManager:
    """Manage model_registry table: register, promote, retire models."""

    def register_model(
        self,
        model_id: str,
        model_type: str,
        trained_at: datetime,
        training_window: tuple[datetime, datetime],
        features_used: list[str],
        hyperparams: dict,
        holdout_metrics: dict,
        artifact_path: str = "",
        regime_fingerprint: dict | None = None,
        parent_model_id: str | None = None,
    ) -> None:
        """Register a new model version in shadow status."""
        conn = get_pg_connection()
        try:
            sql = """
                INSERT INTO model_registry
                    (model_id, model_type, trained_at,
                     training_window_start, training_window_end,
                     features_used, hyperparams, holdout_metrics,
                     status, regime_fingerprint, parent_model_id, artifact_path)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (model_id) DO UPDATE SET
                    holdout_metrics = EXCLUDED.holdout_metrics,
                    status = EXCLUDED.status
            """
            with conn.cursor() as cur:
                cur.execute(sql, (
                    model_id, model_type, trained_at,
                    training_window[0], training_window[1],
                    json.dumps(features_used), json.dumps(hyperparams),
                    json.dumps(holdout_metrics),
                    ModelStatus.SHADOW.value,
                    json.dumps(regime_fingerprint) if regime_fingerprint else None,
                    parent_model_id, artifact_path,
                ))
            conn.commit()
            logger.info("model_registered", model_id=model_id, status="shadow")
        finally:
            conn.close()

    def promote_model(self, model_id: str) -> None:
        """Promote a shadow model to production; retire current production model."""
        conn = get_pg_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE model_registry SET status = %s WHERE status = %s",
                    (ModelStatus.RETIRED.value, ModelStatus.PRODUCTION.value),
                )
                cur.execute(
                    "UPDATE model_registry SET status = %s WHERE model_id = %s",
                    (ModelStatus.PRODUCTION.value, model_id),
                )
            conn.commit()
            logger.info("model_promoted", model_id=model_id)
        finally:
            conn.close()

    def retire_model(self, model_id: str) -> None:
        conn = get_pg_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE model_registry SET status = %s WHERE model_id = %s",
                    (ModelStatus.RETIRED.value, model_id),
                )
            conn.commit()
        finally:
            conn.close()

    def get_production_model(self) -> dict | None:
        conn = get_pg_connection()
        try:
            df = pd.read_sql(
                "SELECT * FROM model_registry WHERE status = %s",
                conn,
                params=[ModelStatus.PRODUCTION.value],
            )
            if df.empty:
                return None
            row = df.iloc[0]
            return row.to_dict()
        finally:
            conn.close()

    def get_all_models(self, status: str | None = None) -> pd.DataFrame:
        conn = get_pg_connection()
        try:
            query = "SELECT * FROM model_registry"
            params = []
            if status:
                query += " WHERE status = %s"
                params.append(status)
            query += " ORDER BY trained_at DESC"
            return pd.read_sql(query, conn, params=params or None)
        finally:
            conn.close()
