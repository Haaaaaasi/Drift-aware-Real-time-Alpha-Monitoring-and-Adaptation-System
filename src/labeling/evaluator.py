"""Layer 8 — Three-level evaluation: alpha, model, and strategy metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logging import get_logger
from src.common.metrics import (
    hit_rate,
    information_coefficient,
    max_drawdown,
    profit_factor,
    rank_information_coefficient,
    sharpe_ratio,
    turnover,
)

logger = get_logger(__name__)


class Evaluator:
    """Compute evaluation metrics at alpha, model, and strategy levels."""

    # ----- Alpha-Level -----

    def evaluate_alpha(
        self,
        alpha_values: pd.Series,
        forward_returns: pd.Series,
    ) -> dict[str, float]:
        """Per-alpha evaluation metrics."""
        ic = information_coefficient(alpha_values, forward_returns)
        rank_ic = rank_information_coefficient(alpha_values, forward_returns)

        return {
            "ic": ic,
            "rank_ic": rank_ic,
            "alpha_mean": float(alpha_values.mean()),
            "alpha_std": float(alpha_values.std()),
            "coverage": float(alpha_values.notna().mean()),
        }

    def evaluate_all_alphas(
        self,
        alpha_panel: pd.DataFrame,
        forward_returns: pd.Series,
    ) -> pd.DataFrame:
        """Evaluate all alphas in the panel.

        Args:
            alpha_panel: Long-format [security_id, tradetime, alpha_id, alpha_value].
            forward_returns: Indexed by (security_id, tradetime).
        """
        results = []
        for alpha_id, group in alpha_panel.groupby("alpha_id"):
            vals = group.set_index(["security_id", "tradetime"])["alpha_value"]
            common = vals.index.intersection(forward_returns.index)
            if len(common) < 10:
                continue
            metrics = self.evaluate_alpha(vals.loc[common], forward_returns.loc[common])
            metrics["alpha_id"] = alpha_id
            results.append(metrics)

        df = pd.DataFrame(results)
        logger.info("alpha_evaluation_complete", alphas=len(df))
        return df

    # ----- Model-Level -----

    def evaluate_model(
        self,
        predictions: pd.Series,
        actuals: pd.Series,
    ) -> dict[str, float]:
        """Model-level evaluation metrics."""
        hr = hit_rate(predictions, actuals)
        ic = information_coefficient(predictions, actuals)

        return {
            "hit_rate": hr,
            "ic": ic,
            "prediction_mean": float(predictions.mean()),
            "prediction_std": float(predictions.std()),
            "n_samples": int(predictions.notna().sum()),
        }

    # ----- Strategy-Level -----

    def evaluate_strategy(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series | None = None,
    ) -> dict[str, float]:
        """Strategy-level performance metrics."""
        cumulative = (1 + portfolio_returns).cumprod()

        metrics = {
            "total_return": float(cumulative.iloc[-1] - 1) if len(cumulative) > 0 else 0.0,
            "annualized_return": float(
                (cumulative.iloc[-1]) ** (252 / max(len(cumulative), 1)) - 1
            ) if len(cumulative) > 0 else 0.0,
            "sharpe": sharpe_ratio(portfolio_returns),
            "max_drawdown": max_drawdown(cumulative),
            "volatility": float(portfolio_returns.std() * np.sqrt(252)),
            "win_rate": float((portfolio_returns > 0).mean()),
            "profit_factor": profit_factor(portfolio_returns),
            "n_days": len(portfolio_returns),
        }

        if benchmark_returns is not None:
            excess = portfolio_returns - benchmark_returns
            metrics["excess_sharpe"] = sharpe_ratio(excess)
            metrics["tracking_error"] = float(excess.std() * np.sqrt(252))

        logger.info("strategy_evaluation_complete", metrics=metrics)
        return metrics
