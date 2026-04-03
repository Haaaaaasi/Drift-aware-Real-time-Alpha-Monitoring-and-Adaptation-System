"""Layer 7 — Position reconciliation between target and actual."""

from __future__ import annotations

import pandas as pd

from src.common.logging import get_logger

logger = get_logger(__name__)


class Reconciler:
    """Compare target positions with actual fills and detect discrepancies."""

    def reconcile(
        self,
        targets: pd.DataFrame,
        actual_positions: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compare target weights with actual position state.

        Returns a DataFrame showing discrepancies.
        """
        if targets.empty:
            return pd.DataFrame(columns=["security_id", "target_weight", "actual_weight", "diff"])

        target_summary = targets.groupby("security_id")["target_weight"].last().reset_index()

        if actual_positions.empty:
            target_summary["actual_weight"] = 0.0
        else:
            total_mv = actual_positions["market_value"].sum()
            if total_mv > 0:
                actual_summary = actual_positions.copy()
                actual_summary["actual_weight"] = actual_summary["market_value"] / total_mv
                actual_summary = actual_summary[["security_id", "actual_weight"]]
            else:
                actual_summary = pd.DataFrame(columns=["security_id", "actual_weight"])

            target_summary = target_summary.merge(
                actual_summary, on="security_id", how="outer"
            ).fillna(0)

        target_summary["diff"] = (
            target_summary["target_weight"] - target_summary.get("actual_weight", 0)
        )

        significant = target_summary[target_summary["diff"].abs() > 0.01]
        if not significant.empty:
            logger.warning(
                "reconciliation_discrepancy",
                discrepancies=len(significant),
            )

        return target_summary
