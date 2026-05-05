"""Layer 9A — Data quality monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logging import get_logger
from src.common.metrics import ks_test_drift, population_stability_index
from src.config.constants import AlertSeverity, MonitorType

logger = get_logger(__name__)


class DataMonitor:
    """Monitor data quality: missing bars, extreme values, stale prices, distribution shifts."""

    def __init__(
        self,
        missing_warn: float = 0.05,
        missing_crit: float = 0.15,
        abnormal_warn: float = 0.01,
        abnormal_crit: float = 0.05,
        stale_warn: int = 5,
        stale_crit: int = 10,
        price_sigma: float = 10.0,
        psi_warn: float = 0.10,
        psi_crit: float = 0.25,
    ) -> None:
        self._missing_warn = missing_warn
        self._missing_crit = missing_crit
        self._abnormal_warn = abnormal_warn
        self._abnormal_crit = abnormal_crit
        self._stale_warn = stale_warn
        self._stale_crit = stale_crit
        self._price_sigma = price_sigma
        self._psi_warn = psi_warn
        self._psi_crit = psi_crit

    def run(
        self,
        bars: pd.DataFrame,
        expected_count: int | None = None,
        reference_features: np.ndarray | None = None,
    ) -> list[dict]:
        """Run all data quality checks.

        Args:
            bars: Recent standardized_bars.
            expected_count: Expected number of bars (for missing ratio).
            reference_features: Historical feature distribution for drift detection.

        Returns:
            List of metric dicts: {metric_time, monitor_type, metric_name, metric_value,
                                    dimension, window_size, severity (if alert)}.
        """
        metrics = []
        now = pd.Timestamp.utcnow()

        # Missing ratio
        actual_count = len(bars)
        if expected_count and expected_count > 0:
            missing_ratio = 1 - actual_count / expected_count
            sev = self._classify(missing_ratio, self._missing_warn, self._missing_crit)
            metrics.append(self._metric(now, "missing_ratio", missing_ratio, severity=sev))

        # Abnormal value ratio
        for col in ["open", "high", "low", "close"]:
            if col in bars.columns and bars[col].std() > 0:
                z = (bars[col] - bars[col].mean()) / bars[col].std()
                abnormal = (z.abs() > self._price_sigma).mean()
                sev = self._classify(abnormal, self._abnormal_warn, self._abnormal_crit)
                metrics.append(self._metric(
                    now, f"abnormal_{col}_ratio", abnormal, dimension=col, severity=sev
                ))

        # Stale price per security
        for sec_id, group in bars.groupby("security_id"):
            closes = group.sort_values("tradetime")["close"]
            if len(closes) < 2:
                continue
            max_stale = (closes == closes.shift(1)).astype(int)
            stale_streak = max_stale.groupby((max_stale != max_stale.shift()).cumsum()).cumsum().max()
            sev = self._classify(stale_streak, self._stale_warn, self._stale_crit)
            if sev:
                metrics.append(self._metric(
                    now, "stale_price_streak", float(stale_streak),
                    dimension=sec_id, severity=sev,
                ))

        # Feature distribution shift (KS-test + PSI, complementary indicators)
        if reference_features is not None and "close" in bars.columns:
            current = bars["close"].dropna().values
            stat, p_val = ks_test_drift(reference_features, current)
            sev = None
            if p_val < 0.01:
                sev = AlertSeverity.CRITICAL
            elif p_val < 0.05:
                sev = AlertSeverity.WARNING
            metrics.append(self._metric(
                now, "feature_dist_shift_pvalue", p_val, severity=sev
            ))

            psi = population_stability_index(reference_features, current)
            sev_psi = None
            if psi >= self._psi_crit:
                sev_psi = AlertSeverity.CRITICAL
            elif psi >= self._psi_warn:
                sev_psi = AlertSeverity.WARNING
            metrics.append(self._metric(
                now, "feature_dist_shift_psi", psi, severity=sev_psi
            ))

        logger.info("data_monitor_complete", metrics_count=len(metrics))
        return metrics

    def _classify(self, value: float, warn: float, crit: float) -> AlertSeverity | None:
        if value >= crit:
            return AlertSeverity.CRITICAL
        if value >= warn:
            return AlertSeverity.WARNING
        return None

    def _metric(
        self, time, name, value, dimension=None, window_size=1, severity=None,
    ) -> dict:
        return {
            "metric_time": time,
            "monitor_type": MonitorType.DATA.value,
            "metric_name": name,
            "metric_value": float(value),
            "dimension": dimension,
            "window_size": window_size,
            "severity": severity.value if severity else None,
        }
