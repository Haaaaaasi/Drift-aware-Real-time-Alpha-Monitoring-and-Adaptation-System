"""Layer 10 — Policy 3: Recurring concept pool and ECPF-like model reuse.

When drift is detected, instead of always retraining from scratch, search a pool of
historical regime-model pairs for a similar past concept and reuse that model.
"""

from __future__ import annotations

import json
from datetime import datetime
from uuid import uuid4

import numpy as np
import pandas as pd

from src.adaptation.model_registry import ModelRegistryManager
from src.common.db import get_pg_connection
from src.common.logging import get_logger
from src.config.constants import AdaptationPolicy

logger = get_logger(__name__)


class RecurringConceptPool:
    """Manage a pool of (regime_fingerprint, model) pairs for concept reuse.

    Regime fingerprint = vector of market features:
    (volatility_level, return_autocorr, avg_corr, trend_strength, volume_ratio)
    """

    def __init__(self, similarity_threshold: float = 0.8) -> None:
        self._threshold = similarity_threshold
        self._registry = ModelRegistryManager()

    def compute_regime_fingerprint(self, market_data: pd.DataFrame) -> dict[str, float]:
        """Compute a regime fingerprint from recent market data.

        Args:
            market_data: Recent standardized_bars [security_id, tradetime, close, vol, ...].
        """
        returns = market_data.groupby("security_id")["close"].pct_change()
        vol = float(returns.std())
        autocorr = float(returns.autocorr(lag=1)) if len(returns) > 10 else 0.0
        avg_return = float(returns.mean())
        volume_ratio = float(
            market_data["vol"].tail(5).mean()
            / max(market_data["vol"].mean(), 1e-8)
        )

        # Cross-asset correlation
        pivot_ret = market_data.pivot_table(
            index="tradetime", columns="security_id", values="close"
        ).pct_change()
        avg_corr = float(pivot_ret.corr().values[np.triu_indices_from(
            pivot_ret.corr().values, k=1
        )].mean()) if pivot_ret.shape[1] > 1 else 0.0

        fingerprint = {
            "volatility": vol,
            "autocorrelation": autocorr,
            "avg_cross_correlation": avg_corr,
            "trend_strength": avg_return,
            "volume_ratio": volume_ratio,
        }
        return fingerprint

    def find_similar_regime(self, current_fp: dict[str, float]) -> tuple[str | None, float]:
        """Search the regime pool for the most similar historical regime.

        Returns:
            (regime_id, similarity_score) or (None, 0.0) if no match.
        """
        conn = get_pg_connection()
        try:
            pool = pd.read_sql("SELECT * FROM regime_pool", conn)
        finally:
            conn.close()

        if pool.empty:
            return None, 0.0

        best_id = None
        best_sim = 0.0
        current_vec = np.array(list(current_fp.values()))

        for _, row in pool.iterrows():
            hist_fp = json.loads(row["fingerprint"]) if isinstance(row["fingerprint"], str) else row["fingerprint"]
            hist_vec = np.array([hist_fp.get(k, 0.0) for k in current_fp.keys()])

            # Cosine similarity
            dot = np.dot(current_vec, hist_vec)
            norm = np.linalg.norm(current_vec) * np.linalg.norm(hist_vec)
            sim = float(dot / max(norm, 1e-10))

            if sim > best_sim:
                best_sim = sim
                best_id = row["regime_id"]

        if best_sim >= self._threshold:
            logger.info("similar_regime_found", regime_id=best_id, similarity=best_sim)
            return best_id, best_sim
        return None, best_sim

    def add_to_pool(
        self,
        fingerprint: dict[str, float],
        model_id: str,
        alpha_weights: dict[str, float],
        performance_summary: dict[str, float],
    ) -> str:
        """Add a new regime-model pair to the pool."""
        regime_id = f"regime_{uuid4().hex[:8]}"
        conn = get_pg_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO regime_pool
                        (regime_id, detected_at, fingerprint, associated_model_id,
                         associated_alpha_weights, performance_summary)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        regime_id, datetime.utcnow(),
                        json.dumps(fingerprint), model_id,
                        json.dumps(alpha_weights), json.dumps(performance_summary),
                    ),
                )
            conn.commit()
            logger.info("regime_added_to_pool", regime_id=regime_id, model_id=model_id)
            return regime_id
        finally:
            conn.close()

    def get_regime_model(self, regime_id: str) -> dict | None:
        """Retrieve the model/weights associated with a regime."""
        conn = get_pg_connection()
        try:
            df = pd.read_sql(
                "SELECT * FROM regime_pool WHERE regime_id = %s",
                conn,
                params=[regime_id],
            )
            return df.iloc[0].to_dict() if not df.empty else None
        finally:
            conn.close()

    def record_reuse(self, regime_id: str) -> None:
        """Increment the reuse counter for a regime."""
        conn = get_pg_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE regime_pool SET times_reused = times_reused + 1, "
                    "last_reused_at = %s WHERE regime_id = %s",
                    (datetime.utcnow(), regime_id),
                )
            conn.commit()
        finally:
            conn.close()
