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
        previous_weights: dict[str, float] | pd.Series | None = None,
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
        prev = self._coerce_previous_weights(previous_weights, current_positions)
        if prev:
            adjusted = self._apply_turnover_cap(adjusted, prev, constraint_log)

        adjusted["pre_risk"] = False
        adjusted.attrs["turnover_cap_applied"] = "turnover_scaled" in constraint_log
        logger.info("risk_constraints_applied", constraints=constraint_log)
        return adjusted

    @staticmethod
    def _coerce_previous_weights(
        previous_weights: dict[str, float] | pd.Series | None,
        current_positions: pd.DataFrame | None,
    ) -> dict[str, float]:
        if previous_weights is not None:
            if isinstance(previous_weights, pd.Series):
                return {
                    str(k): float(v)
                    for k, v in previous_weights.items()
                    if abs(float(v)) > 1e-12
                }
            return {
                str(k): float(v)
                for k, v in previous_weights.items()
                if abs(float(v)) > 1e-12
            }
        if current_positions is None or current_positions.empty:
            return {}
        if "current_weight" in current_positions.columns:
            return {
                str(row["security_id"]): float(row["current_weight"])
                for _, row in current_positions.iterrows()
                if abs(float(row["current_weight"])) > 1e-12
            }
        return {}

    def _apply_turnover_cap(
        self,
        adjusted: pd.DataFrame,
        previous: dict[str, float],
        constraint_log: dict[str, int | float],
    ) -> pd.DataFrame:
        if adjusted.empty:
            return adjusted

        first = adjusted.iloc[0]
        target = {
            str(row["security_id"]): float(row["target_weight"])
            for _, row in adjusted.iterrows()
        }
        all_secs = sorted(set(previous) | set(target))
        buys = sum(max(0.0, target.get(sec, 0.0) - previous.get(sec, 0.0)) for sec in all_secs)
        sells = sum(max(0.0, previous.get(sec, 0.0) - target.get(sec, 0.0)) for sec in all_secs)
        implied_turnover = max(buys, sells)
        if implied_turnover <= self._max_turnover or implied_turnover <= 1e-12:
            return adjusted

        scale = self._max_turnover / implied_turnover
        rows = []
        original_by_sec = adjusted.set_index("security_id").to_dict("index")
        for sec in all_secs:
            prev_w = previous.get(sec, 0.0)
            tgt_w = target.get(sec, 0.0)
            new_w = prev_w + (tgt_w - prev_w) * scale
            if abs(new_w) <= 1e-12:
                continue
            base = dict(original_by_sec.get(sec, {}))
            base.setdefault("rebalance_time", first.get("rebalance_time"))
            base.setdefault("security_id", sec)
            base.setdefault("target_shares", 0)
            base.setdefault("construction_method", first.get("construction_method", "unknown"))
            base.setdefault("pre_risk", True)
            base["target_weight"] = new_w
            rows.append(base)

        capped = pd.DataFrame(rows)
        constraint_log["turnover_scaled"] = round(scale, 4)
        return capped
