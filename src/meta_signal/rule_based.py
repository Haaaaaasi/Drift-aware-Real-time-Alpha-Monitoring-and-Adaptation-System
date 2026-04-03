"""Layer 4 — Plan A: Rule-based IC-weighted alpha composite signal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logging import get_logger
from src.common.metrics import (
    cross_sectional_zscore,
    rank_information_coefficient,
    winsorize,
)

logger = get_logger(__name__)


class RuleBasedSignalGenerator:
    """Generate composite trading signals via IC-weighted alpha aggregation.

    Pipeline:
    1. Cross-sectional z-score each alpha
    2. Winsorize outliers at ±3σ
    3. Weight by rolling rank-IC over a lookback window
    4. Composite signal = Σ(IC_weight_i × z_alpha_i)
    """

    def __init__(
        self,
        ic_lookback: int = 60,
        winsorize_sigma: float = 3.0,
        min_ic_weight: float = 0.0,
    ) -> None:
        self._ic_lookback = ic_lookback
        self._winsorize_sigma = winsorize_sigma
        self._min_ic_weight = min_ic_weight

    def compute_ic_weights(
        self,
        alpha_panel: pd.DataFrame,
        forward_returns: pd.Series,
    ) -> dict[str, float]:
        """Compute IC-based weights for each alpha.

        Args:
            alpha_panel: DataFrame with columns [security_id, tradetime, alpha_id, alpha_value]
            forward_returns: Series indexed by (security_id, tradetime) with forward returns.

        Returns:
            Dictionary mapping alpha_id -> IC weight.
        """
        alpha_ids = alpha_panel["alpha_id"].unique()
        ic_scores: dict[str, float] = {}

        for aid in alpha_ids:
            alpha_slice = alpha_panel[alpha_panel["alpha_id"] == aid].set_index(
                ["security_id", "tradetime"]
            )["alpha_value"]

            common_idx = alpha_slice.index.intersection(forward_returns.index)
            if len(common_idx) < 10:
                ic_scores[aid] = 0.0
                continue

            ic = rank_information_coefficient(
                alpha_slice.loc[common_idx],
                forward_returns.loc[common_idx],
            )
            ic_scores[aid] = max(ic, self._min_ic_weight) if not np.isnan(ic) else 0.0

        # Normalize weights to sum to 1
        total = sum(abs(v) for v in ic_scores.values())
        if total > 0:
            ic_scores = {k: v / total for k, v in ic_scores.items()}

        logger.info("ic_weights_computed", weights=ic_scores)
        return ic_scores

    def generate_signal(
        self,
        alpha_panel: pd.DataFrame,
        ic_weights: dict[str, float],
    ) -> pd.DataFrame:
        """Generate composite signal from alpha panel and IC weights.

        Args:
            alpha_panel: Long-format DataFrame [security_id, tradetime, alpha_id, alpha_value].
            ic_weights: Mapping alpha_id -> weight.

        Returns:
            DataFrame [security_id, tradetime, signal_score, signal_direction, confidence].
        """
        results = []

        for tradetime, group in alpha_panel.groupby("tradetime"):
            pivot = group.pivot(
                index="security_id", columns="alpha_id", values="alpha_value"
            )

            # Z-score and winsorize each alpha cross-sectionally
            for col in pivot.columns:
                mu, sigma = pivot[col].mean(), pivot[col].std()
                if sigma > 0:
                    pivot[col] = (pivot[col] - mu) / sigma
                    pivot[col] = pivot[col].clip(
                        lower=-self._winsorize_sigma, upper=self._winsorize_sigma
                    )
                else:
                    pivot[col] = 0.0

            # Weighted composite
            composite = pd.Series(0.0, index=pivot.index)
            for alpha_id, weight in ic_weights.items():
                if alpha_id in pivot.columns:
                    composite += weight * pivot[alpha_id].fillna(0)

            for sec_id, score in composite.items():
                results.append({
                    "security_id": sec_id,
                    "tradetime": tradetime,
                    "signal_score": score,
                    "signal_direction": int(np.sign(score)),
                    "confidence": abs(score),
                })

        df = pd.DataFrame(results)
        logger.info("rule_based_signal_generated", rows=len(df))
        return df
