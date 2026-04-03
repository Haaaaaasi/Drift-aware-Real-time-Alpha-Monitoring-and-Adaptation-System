"""Layer 10 — Policy 1: Scheduled retrain on a fixed cadence."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import pandas as pd

from src.adaptation.model_registry import ModelRegistryManager
from src.common.logging import get_logger
from src.config.constants import AdaptationPolicy

logger = get_logger(__name__)


class ScheduledRetrainer:
    """Policy 1: Retrain meta model or recalculate IC weights on a fixed schedule.

    Typically runs weekly (Friday close) or monthly.
    """

    def __init__(self, retrain_interval_days: int = 7) -> None:
        self._interval = retrain_interval_days
        self._last_retrain: datetime | None = None
        self._registry = ModelRegistryManager()

    def should_retrain(self, current_time: datetime) -> bool:
        """Check if it's time for a scheduled retrain."""
        if self._last_retrain is None:
            return True
        delta = (current_time - self._last_retrain).days
        return delta >= self._interval

    def retrain(
        self,
        alpha_panel: pd.DataFrame,
        forward_returns: pd.Series,
        current_time: datetime,
    ) -> dict[str, float]:
        """Execute scheduled retrain: recompute IC weights and register new model.

        Returns new IC weights.
        """
        from src.meta_signal.rule_based import RuleBasedSignalGenerator

        generator = RuleBasedSignalGenerator()
        new_weights = generator.compute_ic_weights(alpha_panel, forward_returns)

        model_id = f"sched_{current_time.strftime('%Y%m%d')}_{uuid4().hex[:6]}"

        # Evaluate on recent data as holdout proxy
        recent_signals = generator.generate_signal(alpha_panel, new_weights)
        from src.common.metrics import information_coefficient
        if not recent_signals.empty and not forward_returns.empty:
            sig = recent_signals.set_index(["security_id", "tradetime"])["signal_score"]
            common = sig.index.intersection(forward_returns.index)
            ic = information_coefficient(sig.loc[common], forward_returns.loc[common])
        else:
            ic = 0.0

        dates = alpha_panel["tradetime"].agg(["min", "max"])
        self._registry.register_model(
            model_id=model_id,
            model_type="rule_based",
            trained_at=current_time,
            training_window=(dates["min"], dates["max"]),
            features_used=list(new_weights.keys()),
            hyperparams={"ic_lookback": 60, "winsorize_sigma": 3.0},
            holdout_metrics={"ic": float(ic) if not pd.isna(ic) else 0.0},
            artifact_path="",
        )

        self._last_retrain = current_time
        logger.info(
            "scheduled_retrain_complete",
            model_id=model_id,
            policy=AdaptationPolicy.SCHEDULED.value,
            ic=ic,
        )
        return new_weights
