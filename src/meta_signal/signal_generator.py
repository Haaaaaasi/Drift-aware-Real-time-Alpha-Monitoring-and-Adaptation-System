"""Layer 4 — Unified signal generation interface.

Routes to the appropriate signal method (rule-based, ML, regime-ensemble).
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.common.db import get_pg_connection
from src.common.logging import get_logger
from src.config.constants import MetaSignalMethod

logger = get_logger(__name__)


class SignalGenerator:
    """Facade that delegates to the selected meta-signal strategy."""

    def __init__(
        self,
        method: MetaSignalMethod = MetaSignalMethod.RULE_BASED,
        feature_columns: list[str] | None = None,
    ) -> None:
        self._method = method
        self._feature_columns = feature_columns

    def generate(
        self,
        alpha_panel: pd.DataFrame,
        forward_returns: pd.Series | None = None,
        ic_weights: dict[str, float] | None = None,
        model_version_id: str | None = None,
    ) -> pd.DataFrame:
        """Generate meta signals from alpha features.

        Returns DataFrame with: security_id, signal_time, signal_score,
                                signal_direction, confidence, method, model_version_id
        """
        if self._method == MetaSignalMethod.RULE_BASED:
            from src.meta_signal.rule_based import RuleBasedSignalGenerator
            gen = RuleBasedSignalGenerator()
            if ic_weights is None and forward_returns is not None:
                ic_weights = gen.compute_ic_weights(alpha_panel, forward_returns)
            elif ic_weights is None:
                # Equal weight fallback
                alpha_ids = alpha_panel["alpha_id"].unique()
                ic_weights = {a: 1.0 / len(alpha_ids) for a in alpha_ids}
            signals = gen.generate_signal(alpha_panel, ic_weights)

        elif self._method == MetaSignalMethod.ML_META:
            from src.meta_signal.ml_meta_model import MLMetaModel
            if forward_returns is None:
                raise ValueError("ML meta model requires forward_returns for training")
            # If caller did not pre-filter alphas, restrict to effective subset.
            panel_for_ml = alpha_panel
            if self._feature_columns:
                panel_for_ml = alpha_panel[
                    alpha_panel["alpha_id"].isin(self._feature_columns)
                ]
            model = MLMetaModel(feature_columns=self._feature_columns)
            wide = panel_for_ml.pivot_table(
                index=["security_id", "tradetime"],
                columns="alpha_id",
                values="alpha_value",
            )
            train_info = model.train(wide, forward_returns)
            signals = model.predict(wide)
            model_version_id = model_version_id or train_info["model_id"]

        else:
            raise NotImplementedError(f"Method {self._method} not yet implemented")

        signals = signals.rename(columns={"tradetime": "signal_time"})
        signals["method"] = self._method.value
        signals["model_version_id"] = model_version_id or "default"
        signals["bar_type"] = "daily"

        return signals

    def persist_signals(self, signals: pd.DataFrame) -> int:
        """Write signals to PostgreSQL meta_signals table."""
        conn = get_pg_connection()
        try:
            records = signals[
                ["security_id", "signal_time", "bar_type", "signal_score",
                 "signal_direction", "confidence", "method", "model_version_id"]
            ].values.tolist()

            insert_sql = """
                INSERT INTO meta_signals
                    (security_id, signal_time, bar_type, signal_score,
                     signal_direction, confidence, method, model_version_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            with conn.cursor() as cur:
                from psycopg2.extras import execute_batch
                execute_batch(cur, insert_sql, records, page_size=1000)
            conn.commit()
            logger.info("signals_persisted", count=len(records))
            return len(records)
        finally:
            conn.close()
