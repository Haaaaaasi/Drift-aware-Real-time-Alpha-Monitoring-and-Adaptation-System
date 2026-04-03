"""Layer 5 — Portfolio construction from meta signals."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from src.common.db import get_pg_connection
from src.common.logging import get_logger

logger = get_logger(__name__)


class PortfolioConstructor:
    """Convert meta signals into target portfolio weights.

    Supported methods:
    - equal_weight_topk: top-k by signal score, equal weight
    - score_proportional: weight proportional to signal score
    - volatility_scaled: weight proportional to signal / realized_vol (MVP v2)
    """

    def __init__(
        self,
        method: str = "equal_weight_topk",
        top_k: int = 10,
        long_only: bool = True,
    ) -> None:
        self._method = method
        self._top_k = top_k
        self._long_only = long_only

    def construct(
        self,
        signals: pd.DataFrame,
        volatilities: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Build portfolio targets from signals.

        Args:
            signals: DataFrame with [security_id, signal_time, signal_score, signal_direction].
            volatilities: Optional per-security realized volatility for vol-scaling.

        Returns:
            DataFrame [rebalance_time, security_id, target_weight, target_shares,
                       construction_method, pre_risk].
        """
        all_targets = []

        for signal_time, group in signals.groupby("signal_time"):
            if self._long_only:
                group = group[group["signal_direction"] >= 0]

            group = group.sort_values("signal_score", ascending=False)

            if self._method == "equal_weight_topk":
                selected = group.head(self._top_k)
                n = len(selected)
                if n == 0:
                    continue
                weight = 1.0 / n
                for _, row in selected.iterrows():
                    all_targets.append({
                        "rebalance_time": signal_time,
                        "security_id": row["security_id"],
                        "target_weight": weight,
                        "target_shares": 0,  # computed later with capital
                        "construction_method": self._method,
                        "pre_risk": True,
                    })

            elif self._method == "score_proportional":
                selected = group.head(self._top_k)
                scores = selected["signal_score"].clip(lower=0)
                total_score = scores.sum()
                if total_score <= 0:
                    continue
                for _, row in selected.iterrows():
                    w = max(row["signal_score"], 0) / total_score
                    all_targets.append({
                        "rebalance_time": signal_time,
                        "security_id": row["security_id"],
                        "target_weight": w,
                        "target_shares": 0,
                        "construction_method": self._method,
                        "pre_risk": True,
                    })

            elif self._method == "volatility_scaled":
                if volatilities is None:
                    raise ValueError("volatility_scaled requires volatilities parameter")
                selected = group.head(self._top_k)
                merged = selected.merge(
                    volatilities.rename("vol"),
                    left_on="security_id",
                    right_index=True,
                    how="left",
                )
                merged["vol"] = merged["vol"].fillna(merged["vol"].median())
                merged["raw_weight"] = merged["signal_score"] / merged["vol"].clip(lower=1e-6)
                total = merged["raw_weight"].abs().sum()
                if total <= 0:
                    continue
                for _, row in merged.iterrows():
                    all_targets.append({
                        "rebalance_time": signal_time,
                        "security_id": row["security_id"],
                        "target_weight": row["raw_weight"] / total,
                        "target_shares": 0,
                        "construction_method": self._method,
                        "pre_risk": True,
                    })

        result = pd.DataFrame(all_targets)
        logger.info("portfolio_constructed", method=self._method, rows=len(result))
        return result

    def persist_targets(self, targets: pd.DataFrame) -> int:
        """Write portfolio targets to PostgreSQL."""
        if targets.empty:
            return 0
        conn = get_pg_connection()
        try:
            records = targets[
                ["rebalance_time", "security_id", "target_weight",
                 "target_shares", "construction_method", "pre_risk"]
            ].values.tolist()

            sql = """
                INSERT INTO portfolio_targets
                    (rebalance_time, security_id, target_weight,
                     target_shares, construction_method, pre_risk)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            with conn.cursor() as cur:
                from psycopg2.extras import execute_batch
                execute_batch(cur, sql, records, page_size=1000)
            conn.commit()
            return len(records)
        finally:
            conn.close()
