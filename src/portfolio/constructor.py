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
        entry_rank: int | None = None,
        exit_rank: int | None = None,
        min_holding_days: int = 0,
    ) -> None:
        self._method = method
        self._top_k = top_k
        self._long_only = long_only
        self._entry_rank = entry_rank or top_k
        self._exit_rank = exit_rank or max(self._entry_rank, top_k)
        self._min_holding_days = min_holding_days

    def construct(
        self,
        signals: pd.DataFrame,
        volatilities: pd.Series | None = None,
        previous_weights: dict[str, float] | pd.Series | None = None,
        holding_days: dict[str, int] | None = None,
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
        previous = self._normalize_weights(previous_weights)
        holding_days = holding_days or {}

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

            elif self._method == "turnover_aware_topk":
                selected, meta = self._select_turnover_aware(group, previous, holding_days)
                n = len(selected)
                if n == 0:
                    continue
                weight = 1.0 / n
                for _, row in selected.iterrows():
                    all_targets.append({
                        "rebalance_time": signal_time,
                        "security_id": row["security_id"],
                        "target_weight": weight,
                        "target_shares": 0,
                        "construction_method": self._method,
                        "pre_risk": True,
                        "rank": int(row["_rank"]),
                        "held_from_prev": bool(row["_held_from_prev"]),
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
        if self._method == "turnover_aware_topk" and "meta" in locals():
            result.attrs.update(meta)
        else:
            result.attrs.update({
                "held_from_prev_count": 0,
                "forced_sells_count": 0,
            })
        logger.info("portfolio_constructed", method=self._method, rows=len(result))
        return result

    @staticmethod
    def _normalize_weights(
        weights: dict[str, float] | pd.Series | None,
    ) -> dict[str, float]:
        if weights is None:
            return {}
        if isinstance(weights, pd.Series):
            return {str(k): float(v) for k, v in weights.items() if abs(float(v)) > 1e-12}
        return {str(k): float(v) for k, v in weights.items() if abs(float(v)) > 1e-12}

    def _select_turnover_aware(
        self,
        group: pd.DataFrame,
        previous: dict[str, float],
        holding_days: dict[str, int],
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        ranked = group.copy()
        if ranked.empty:
            return ranked, {
                "held_from_prev_count": 0,
                "forced_sells_count": len(previous),
            }

        ranked["_rank"] = np.arange(1, len(ranked) + 1)
        ranked["_held_from_prev"] = False
        rank_by_sec = {
            str(row["security_id"]): int(row["_rank"])
            for _, row in ranked.iterrows()
        }
        current_secs = set(ranked["security_id"].astype(str))

        kept: list[str] = []
        forced_sells = 0
        for sec in previous:
            if sec not in current_secs:
                forced_sells += 1
                continue
            rank = int(rank_by_sec[sec])
            too_young = int(holding_days.get(sec, 0)) < self._min_holding_days
            if rank <= self._exit_rank or too_young:
                kept.append(sec)
            else:
                forced_sells += 1

        kept = sorted(kept, key=lambda sec: int(rank_by_sec.get(sec, 10**9)))[: self._top_k]
        selected_secs = list(kept)
        open_slots = max(0, self._top_k - len(selected_secs))
        if open_slots:
            entry_pool = ranked[
                (ranked["_rank"] <= self._entry_rank)
                & (~ranked["security_id"].astype(str).isin(selected_secs))
            ]
            selected_secs.extend(entry_pool.head(open_slots)["security_id"].astype(str).tolist())

        selected = ranked[ranked["security_id"].astype(str).isin(selected_secs)].copy()
        selected["_held_from_prev"] = selected["security_id"].astype(str).isin(kept)
        selected["_selection_order"] = selected["security_id"].astype(str).map(
            {sec: i for i, sec in enumerate(selected_secs)}
        )
        selected = selected.sort_values("_selection_order")
        meta = {
            "held_from_prev_count": int(selected["_held_from_prev"].sum()),
            "forced_sells_count": int(forced_sells),
        }
        return selected, meta

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
