"""Layer 9D — Strategy-level performance monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logging import get_logger
from src.common.metrics import max_drawdown, sharpe_ratio
from src.config.constants import AlertSeverity, MonitorType

logger = get_logger(__name__)


class StrategyMonitor:
    """Monitor live strategy health: Sharpe, drawdown, realized-vs-expected, turnover."""

    def __init__(
        self,
        sharpe_window: int = 60,
        sharpe_warn: float = 0.5,
        sharpe_crit: float = 0.0,
        dd_warn: float = 0.10,
        dd_crit: float = 0.20,
        rve_warn: float = 0.5,
        rve_crit: float = 0.2,
        turnover_warn: float = 1.0,
        turnover_crit: float = 2.0,
    ) -> None:
        self._sharpe_window = sharpe_window
        self._sharpe_warn = sharpe_warn
        self._sharpe_crit = sharpe_crit
        self._dd_warn = dd_warn
        self._dd_crit = dd_crit
        self._rve_warn = rve_warn
        self._rve_crit = rve_crit
        self._turnover_warn = turnover_warn
        self._turnover_crit = turnover_crit

    def run(
        self,
        portfolio_returns: pd.Series,
        backtest_returns: pd.Series | None = None,
        daily_turnover: pd.Series | None = None,
    ) -> list[dict]:
        """Run strategy monitoring checks."""
        metrics = []
        now = pd.Timestamp.utcnow()

        # Rolling Sharpe
        if len(portfolio_returns) >= self._sharpe_window:
            recent = portfolio_returns.tail(self._sharpe_window)
            sr = sharpe_ratio(recent)
            sev = None
            if sr < self._sharpe_crit:
                sev = AlertSeverity.CRITICAL
            elif sr < self._sharpe_warn:
                sev = AlertSeverity.WARNING
            metrics.append(self._metric(
                now, "rolling_sharpe", sr, window=self._sharpe_window, severity=sev
            ))

        # Max drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        dd = max_drawdown(cumulative)
        sev_dd = None
        if abs(dd) >= self._dd_crit:
            sev_dd = AlertSeverity.CRITICAL
        elif abs(dd) >= self._dd_warn:
            sev_dd = AlertSeverity.WARNING
        metrics.append(self._metric(now, "max_drawdown", abs(dd), severity=sev_dd))

        # Realized vs expected
        if backtest_returns is not None and len(portfolio_returns) >= 20:
            recent_real = portfolio_returns.tail(20).sum()
            recent_bt = backtest_returns.tail(20).sum()
            ratio = recent_real / recent_bt if abs(recent_bt) > 1e-8 else 1.0
            sev_rve = None
            if ratio < self._rve_crit:
                sev_rve = AlertSeverity.CRITICAL
            elif ratio < self._rve_warn:
                sev_rve = AlertSeverity.WARNING
            metrics.append(self._metric(now, "realized_vs_expected", ratio, severity=sev_rve))

        # Turnover
        if daily_turnover is not None and len(daily_turnover) > 0:
            avg_turnover = daily_turnover.mean()
            sev_to = None
            if avg_turnover >= self._turnover_crit:
                sev_to = AlertSeverity.CRITICAL
            elif avg_turnover >= self._turnover_warn:
                sev_to = AlertSeverity.WARNING
            metrics.append(self._metric(now, "avg_turnover", avg_turnover, severity=sev_to))

        logger.info("strategy_monitor_complete", metrics_count=len(metrics))
        return metrics

    def _metric(self, time, name, value, dimension=None, window=None, severity=None):
        return {
            "metric_time": time,
            "monitor_type": MonitorType.STRATEGY.value,
            "metric_name": name,
            "metric_value": float(value) if not np.isnan(value) else 0.0,
            "dimension": dimension,
            "window_size": window or self._sharpe_window,
            "severity": severity.value if severity else None,
        }
