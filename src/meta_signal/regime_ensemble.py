"""Layer 4 — Plan C: Regime-aware ensemble signal generation.

TODO (MVP v3): Full HMM-based regime identification and ensemble switching.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logging import get_logger

logger = get_logger(__name__)


class RegimeIdentifier:
    """Identify market regimes from price/volume/volatility features.

    MVP v3: Uses Hidden Markov Model on (realized_vol, return_autocorr, cross_corr).
    """

    def __init__(self, n_regimes: int = 3) -> None:
        self._n_regimes = n_regimes

    def identify(self, market_data: pd.DataFrame) -> pd.Series:
        """Return a regime label per tradetime.

        Placeholder: classify by rolling volatility quantile.
        """
        returns = market_data.groupby("tradetime")["close"].pct_change()
        vol = returns.rolling(20).std()

        thresholds = vol.quantile([1 / 3, 2 / 3])
        regime = pd.Series("medium", index=vol.index)
        regime[vol <= thresholds.iloc[0]] = "low_vol"
        regime[vol >= thresholds.iloc[1]] = "high_vol"

        logger.info("regime_identified", unique_regimes=regime.nunique())
        return regime


class RegimeEnsemble:
    """Regime-aware ensemble that switches alpha weights / models per regime.

    Each regime maps to a different set of alpha weights or model.
    """

    def __init__(self) -> None:
        self._regime_models: dict[str, dict[str, float]] = {}

    def register_regime_weights(self, regime: str, weights: dict[str, float]) -> None:
        self._regime_models[regime] = weights

    def generate_signal(
        self,
        alpha_panel: pd.DataFrame,
        current_regime: str,
    ) -> pd.DataFrame:
        """Generate signal using weights for the current regime."""
        weights = self._regime_models.get(current_regime, {})
        if not weights:
            logger.warning("no_weights_for_regime", regime=current_regime)
            return pd.DataFrame()

        # Placeholder: same logic as rule-based but with regime-specific weights
        from src.meta_signal.rule_based import RuleBasedSignalGenerator
        gen = RuleBasedSignalGenerator()
        return gen.generate_signal(alpha_panel, weights)
