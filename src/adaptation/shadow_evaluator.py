"""Layer 10 — Shadow / canary evaluation for candidate models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logging import get_logger
from src.common.metrics import information_coefficient, hit_rate, sharpe_ratio

logger = get_logger(__name__)


class ShadowEvaluator:
    """Evaluate candidate models in shadow mode before promoting to production.

    Compares up to 3 candidates:
    - Current production model
    - Freshly retrained model
    - Reused historical model (from recurring concept pool)
    """

    def __init__(
        self,
        min_improvement_ic: float = 0.005,
        min_evaluation_days: int = 5,
    ) -> None:
        self._min_improvement = min_improvement_ic
        self._min_days = min_evaluation_days

    def evaluate_candidates(
        self,
        candidates: dict[str, pd.DataFrame],
        forward_returns: pd.Series,
    ) -> dict[str, dict[str, float]]:
        """Evaluate multiple candidate signal sets.

        Args:
            candidates: Mapping model_id -> signals DataFrame [security_id, tradetime, signal_score].
            forward_returns: Indexed by (security_id, tradetime).

        Returns:
            Mapping model_id -> evaluation metrics.
        """
        results = {}

        for model_id, signals in candidates.items():
            if signals.empty:
                results[model_id] = {"ic": 0.0, "hit_rate": 0.0, "sharpe": 0.0}
                continue

            sig = signals.set_index(["security_id", "tradetime"])["signal_score"]
            common = sig.index.intersection(forward_returns.index)

            if len(common) < 10:
                results[model_id] = {"ic": 0.0, "hit_rate": 0.0, "sharpe": 0.0}
                continue

            ic = information_coefficient(sig.loc[common], forward_returns.loc[common])
            hr = hit_rate(sig.loc[common], forward_returns.loc[common])

            # Approximate Sharpe from signal-weighted returns
            weighted_ret = sig.loc[common] * forward_returns.loc[common]
            sr = sharpe_ratio(weighted_ret) if len(weighted_ret) > 5 else 0.0

            results[model_id] = {
                "ic": float(ic) if not np.isnan(ic) else 0.0,
                "hit_rate": float(hr) if not np.isnan(hr) else 0.0,
                "sharpe": float(sr) if not np.isnan(sr) else 0.0,
                "n_samples": len(common),
            }

        logger.info("shadow_evaluation_complete", candidates=list(results.keys()))
        return results

    def select_best(
        self,
        evaluation_results: dict[str, dict[str, float]],
        current_model_id: str | None = None,
    ) -> str | None:
        """Select the best candidate model.

        If a current model exists, the new model must improve IC by min_improvement.
        """
        if not evaluation_results:
            return None

        # Rank by IC
        ranked = sorted(
            evaluation_results.items(),
            key=lambda x: x[1].get("ic", 0.0),
            reverse=True,
        )

        best_id, best_metrics = ranked[0]

        if current_model_id and current_model_id in evaluation_results:
            current_ic = evaluation_results[current_model_id].get("ic", 0.0)
            if best_metrics["ic"] - current_ic < self._min_improvement:
                logger.info(
                    "shadow_no_improvement",
                    best_id=best_id,
                    best_ic=best_metrics["ic"],
                    current_ic=current_ic,
                )
                return None

        logger.info("shadow_best_selected", model_id=best_id, metrics=best_metrics)
        return best_id
