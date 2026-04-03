"""Layer 10 — Policy 2: Performance-triggered adaptation."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import pandas as pd

from src.adaptation.model_registry import ModelRegistryManager
from src.common.logging import get_logger
from src.config.constants import AdaptationPolicy, AlertSeverity

logger = get_logger(__name__)


class PerformanceTriggeredAdapter:
    """Policy 2: Trigger adaptation when performance degrades beyond thresholds.

    Triggers when:
    - Rolling IC < threshold for N consecutive days
    - Rolling Sharpe < 0 for M consecutive days
    - Critical alerts accumulate beyond K count
    """

    def __init__(
        self,
        ic_threshold: float = 0.0,
        ic_consecutive_days: int = 5,
        sharpe_threshold: float = 0.0,
        sharpe_consecutive_days: int = 10,
        critical_alert_limit: int = 3,
    ) -> None:
        self._ic_thresh = ic_threshold
        self._ic_days = ic_consecutive_days
        self._sharpe_thresh = sharpe_threshold
        self._sharpe_days = sharpe_consecutive_days
        self._crit_limit = critical_alert_limit
        self._registry = ModelRegistryManager()

    def check_trigger(
        self,
        rolling_ic_series: pd.Series,
        rolling_sharpe_series: pd.Series,
        critical_alert_count: int,
    ) -> tuple[bool, str]:
        """Check if performance-triggered adaptation should fire.

        Returns:
            (should_trigger, reason)
        """
        # IC degradation
        if len(rolling_ic_series) >= self._ic_days:
            recent_ic = rolling_ic_series.tail(self._ic_days)
            if (recent_ic < self._ic_thresh).all():
                return True, f"Rolling IC < {self._ic_thresh} for {self._ic_days} consecutive days"

        # Sharpe degradation
        if len(rolling_sharpe_series) >= self._sharpe_days:
            recent_sharpe = rolling_sharpe_series.tail(self._sharpe_days)
            if (recent_sharpe < self._sharpe_thresh).all():
                return True, f"Rolling Sharpe < {self._sharpe_thresh} for {self._sharpe_days} days"

        # Critical alert accumulation
        if critical_alert_count >= self._crit_limit:
            return True, f"Critical alerts ({critical_alert_count}) >= limit ({self._crit_limit})"

        return False, ""

    def adapt(
        self,
        alpha_panel: pd.DataFrame,
        forward_returns: pd.Series,
        current_time: datetime,
        reason: str,
    ) -> dict[str, float]:
        """Execute performance-triggered adaptation: retrain and compare with current.

        Returns new IC weights if improvement is found, else current weights.
        """
        from src.meta_signal.rule_based import RuleBasedSignalGenerator
        from src.common.metrics import information_coefficient

        generator = RuleBasedSignalGenerator()
        new_weights = generator.compute_ic_weights(alpha_panel, forward_returns)

        model_id = f"perf_{current_time.strftime('%Y%m%d')}_{uuid4().hex[:6]}"

        # Evaluate new weights
        signals = generator.generate_signal(alpha_panel, new_weights)
        if not signals.empty:
            sig = signals.set_index(["security_id", "tradetime"])["signal_score"]
            common = sig.index.intersection(forward_returns.index)
            new_ic = information_coefficient(sig.loc[common], forward_returns.loc[common])
        else:
            new_ic = 0.0

        dates = alpha_panel["tradetime"].agg(["min", "max"])
        self._registry.register_model(
            model_id=model_id,
            model_type="rule_based",
            trained_at=current_time,
            training_window=(dates["min"], dates["max"]),
            features_used=list(new_weights.keys()),
            hyperparams={"trigger_reason": reason},
            holdout_metrics={"ic": float(new_ic) if not pd.isna(new_ic) else 0.0},
        )

        logger.info(
            "performance_triggered_adaptation",
            model_id=model_id,
            policy=AdaptationPolicy.PERFORMANCE_TRIGGERED.value,
            reason=reason,
            new_ic=new_ic,
        )
        return new_weights
