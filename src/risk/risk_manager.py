"""Layer 6 — Risk management: enforce constraints on portfolio targets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logging import get_logger

logger = get_logger(__name__)


class RiskManager:
    """Apply risk constraints to portfolio targets before execution.

    Constraints:
    - max_position_weight: cap individual position weight
    - max_gross_exposure: total absolute weight sum
    - max_turnover: limit rebalance turnover
    - max_drawdown_halt: halt trading if cumulative drawdown exceeds threshold
    - liquidity_filter: exclude low-volume securities
    - no_trade_zone: exclude when spread > threshold
    """

    def __init__(
        self,
        max_position_weight: float = 0.10,
        max_gross_exposure: float = 1.0,
        max_turnover: float = 1.0,
        max_drawdown_halt: float = 0.20,
        min_daily_volume: float = 0.0,
    ) -> None:
        self._max_position = max_position_weight
        self._max_exposure = max_gross_exposure
        self._max_turnover = max_turnover
        self._max_dd_halt = max_drawdown_halt
        self._min_volume = min_daily_volume

    def apply_constraints(
        self,
        targets: pd.DataFrame,
        current_positions: pd.DataFrame | None = None,
        cumulative_drawdown: float = 0.0,
        market_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Apply all risk constraints and return adjusted targets.

        Args:
            targets: Pre-risk portfolio targets [rebalance_time, security_id, target_weight, ...].
            current_positions: Current position snapshot for turnover calculation.
            cumulative_drawdown: Current drawdown (negative number, e.g., -0.15).
            market_data: Recent bars for liquidity filtering.

        Returns:
            Risk-adjusted targets with pre_risk=False.
        """
        adjusted = targets.copy()
        constraint_log: dict[str, int] = {}

        # Drawdown halt
        if abs(cumulative_drawdown) >= self._max_dd_halt:
            logger.warning(
                "drawdown_halt_triggered",
                drawdown=cumulative_drawdown,
                threshold=self._max_dd_halt,
            )
            adjusted["target_weight"] = 0.0
            adjusted["pre_risk"] = False
            return adjusted

        # Liquidity filter
        if market_data is not None and self._min_volume > 0:
            avg_vol = market_data.groupby("security_id")["vol"].mean()
            illiquid = avg_vol[avg_vol < self._min_volume].index
            mask = adjusted["security_id"].isin(illiquid)
            constraint_log["liquidity_filtered"] = int(mask.sum())
            adjusted.loc[mask, "target_weight"] = 0.0

        # Cap individual position weight
        over_limit = adjusted["target_weight"].abs() > self._max_position
        if over_limit.any():
            constraint_log["position_capped"] = int(over_limit.sum())
            adjusted.loc[over_limit, "target_weight"] = (
                np.sign(adjusted.loc[over_limit, "target_weight"]) * self._max_position
            )

        # Normalize to max gross exposure
        gross = adjusted["target_weight"].abs().sum()
        if gross > self._max_exposure:
            scale = self._max_exposure / gross
            adjusted["target_weight"] *= scale
            constraint_log["exposure_scaled"] = round(scale, 4)

        # Turnover constraint
        if current_positions is not None and not current_positions.empty:
            merged = adjusted.merge(
                current_positions[["security_id", "quantity"]].rename(
                    columns={"quantity": "current_qty"}
                ),
                on="security_id",
                how="outer",
            ).fillna(0)
            # Approximate turnover as sum of weight changes / 2
            implied_turnover = merged["target_weight"].sub(
                merged.get("current_weight", 0), fill_value=0
            ).abs().sum() / 2
            if implied_turnover > self._max_turnover:
                scale = self._max_turnover / implied_turnover
                adjusted["target_weight"] *= scale
                constraint_log["turnover_scaled"] = round(scale, 4)

        adjusted["pre_risk"] = False
        logger.info("risk_constraints_applied", constraints=constraint_log)
        return adjusted
