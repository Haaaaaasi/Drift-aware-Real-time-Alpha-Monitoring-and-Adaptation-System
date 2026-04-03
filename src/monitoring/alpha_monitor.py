"""Layer 9B — Alpha factor monitoring: IC, decay, turnover, correlation drift."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logging import get_logger
from src.common.metrics import information_coefficient, rank_information_coefficient
from src.config.constants import AlertSeverity, MonitorType

logger = get_logger(__name__)


class AlphaMonitor:
    """Monitor alpha factor health: rolling IC, decay, turnover, correlation stability."""

    def __init__(
        self,
        ic_window: int = 20,
        ic_warn: float = 0.02,
        ic_crit: float = 0.0,
        turnover_warn: float = 0.80,
        turnover_crit: float = 0.95,
        corr_drift_warn: float = 0.2,
        corr_drift_crit: float = 0.4,
    ) -> None:
        self._ic_window = ic_window
        self._ic_warn = ic_warn
        self._ic_crit = ic_crit
        self._turnover_warn = turnover_warn
        self._turnover_crit = turnover_crit
        self._corr_drift_warn = corr_drift_warn
        self._corr_drift_crit = corr_drift_crit

    def run(
        self,
        alpha_panel: pd.DataFrame,
        forward_returns: pd.Series,
        baseline_corr_matrix: pd.DataFrame | None = None,
    ) -> list[dict]:
        """Run all alpha monitoring checks.

        Args:
            alpha_panel: Long format [security_id, tradetime, alpha_id, alpha_value].
            forward_returns: Series indexed by (security_id, tradetime).
            baseline_corr_matrix: Historical pairwise alpha correlation for drift detection.
        """
        metrics = []
        now = pd.Timestamp.utcnow()
        alpha_ids = alpha_panel["alpha_id"].unique()

        for aid in alpha_ids:
            slice_df = alpha_panel[alpha_panel["alpha_id"] == aid]
            vals = slice_df.set_index(["security_id", "tradetime"])["alpha_value"]
            common = vals.index.intersection(forward_returns.index)

            if len(common) < self._ic_window:
                continue

            # Rolling IC (use latest window)
            recent_vals = vals.loc[common].tail(self._ic_window * 50)
            recent_ret = forward_returns.loc[recent_vals.index]

            ic = information_coefficient(recent_vals, recent_ret)
            rank_ic = rank_information_coefficient(recent_vals, recent_ret)

            sev_ic = self._classify_ic(ic)
            sev_ric = self._classify_ic(rank_ic)

            metrics.append(self._metric(now, "rolling_ic", ic, aid, self._ic_window, sev_ic))
            metrics.append(self._metric(now, "rolling_rank_ic", rank_ic, aid, self._ic_window, sev_ric))

            # Alpha turnover: fraction of rank changes between periods
            dates = slice_df["tradetime"].unique()
            if len(dates) >= 2:
                last_two = sorted(dates)[-2:]
                rank_prev = (
                    slice_df[slice_df["tradetime"] == last_two[0]]
                    .set_index("security_id")["alpha_value"].rank()
                )
                rank_curr = (
                    slice_df[slice_df["tradetime"] == last_two[1]]
                    .set_index("security_id")["alpha_value"].rank()
                )
                common_secs = rank_prev.index.intersection(rank_curr.index)
                if len(common_secs) > 0:
                    changes = (rank_prev.loc[common_secs] != rank_curr.loc[common_secs]).mean()
                    sev_turn = None
                    if changes >= self._turnover_crit:
                        sev_turn = AlertSeverity.CRITICAL
                    elif changes >= self._turnover_warn:
                        sev_turn = AlertSeverity.WARNING
                    metrics.append(self._metric(
                        now, "alpha_turnover", float(changes), aid, severity=sev_turn
                    ))

        # Pairwise correlation drift
        if baseline_corr_matrix is not None and len(alpha_ids) >= 2:
            pivot = alpha_panel.pivot_table(
                index=["security_id", "tradetime"],
                columns="alpha_id",
                values="alpha_value",
            )
            current_corr = pivot.corr()
            drift = (current_corr - baseline_corr_matrix).abs().mean().mean()
            sev_corr = None
            if drift >= self._corr_drift_crit:
                sev_corr = AlertSeverity.CRITICAL
            elif drift >= self._corr_drift_warn:
                sev_corr = AlertSeverity.WARNING
            metrics.append(self._metric(
                now, "alpha_correlation_drift", float(drift), severity=sev_corr
            ))

        logger.info("alpha_monitor_complete", metrics_count=len(metrics))
        return metrics

    def _classify_ic(self, ic: float) -> AlertSeverity | None:
        if np.isnan(ic):
            return None
        if ic < self._ic_crit:
            return AlertSeverity.CRITICAL
        if ic < self._ic_warn:
            return AlertSeverity.WARNING
        return None

    def _metric(self, time, name, value, dimension=None, window=None, severity=None):
        return {
            "metric_time": time,
            "monitor_type": MonitorType.ALPHA.value,
            "metric_name": name,
            "metric_value": float(value) if not np.isnan(value) else 0.0,
            "dimension": dimension,
            "window_size": window or self._ic_window,
            "severity": severity.value if severity else None,
        }
