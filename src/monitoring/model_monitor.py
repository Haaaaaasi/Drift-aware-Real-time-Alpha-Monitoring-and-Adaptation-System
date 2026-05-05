"""Layer 9C — Model prediction monitoring: distribution drift, accuracy, calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logging import get_logger
from src.common.metrics import calibration_error, hit_rate, ks_test_drift
from src.config.constants import AlertSeverity, MonitorType

logger = get_logger(__name__)


class ModelMonitor:
    """Monitor model health: prediction distribution shift, directional accuracy, AUC."""

    def __init__(
        self,
        accuracy_window: int = 20,
        accuracy_warn: float = 0.52,
        accuracy_crit: float = 0.48,
        dist_drift_warn: float = 0.05,
        dist_drift_crit: float = 0.01,
        calibration_warn: float = 0.10,
        calibration_crit: float = 0.20,
    ) -> None:
        self._window = accuracy_window
        self._acc_warn = accuracy_warn
        self._acc_crit = accuracy_crit
        self._dist_warn = dist_drift_warn
        self._dist_crit = dist_drift_crit
        self._cal_warn = calibration_warn
        self._cal_crit = calibration_crit

    def run(
        self,
        predictions: pd.Series,
        actuals: pd.Series,
        reference_predictions: np.ndarray | None = None,
    ) -> list[dict]:
        """Run model monitoring checks.

        Args:
            predictions: Model predictions (signal scores).
            actuals: Actual forward returns or directions.
            reference_predictions: Historical prediction distribution for drift detection.
        """
        metrics = []
        now = pd.Timestamp.utcnow()

        # Directional accuracy
        hr = hit_rate(predictions, actuals)
        sev_acc = None
        if not np.isnan(hr):
            if hr < self._acc_crit:
                sev_acc = AlertSeverity.CRITICAL
            elif hr < self._acc_warn:
                sev_acc = AlertSeverity.WARNING
        metrics.append(self._metric(now, "directional_accuracy", hr, severity=sev_acc))

        # Prediction distribution drift
        if reference_predictions is not None:
            current = predictions.dropna().values
            stat, p_val = ks_test_drift(reference_predictions, current)
            sev_dist = None
            if p_val < self._dist_crit:
                sev_dist = AlertSeverity.CRITICAL
            elif p_val < self._dist_warn:
                sev_dist = AlertSeverity.WARNING
            metrics.append(self._metric(
                now, "prediction_dist_drift_pvalue", p_val, severity=sev_dist
            ))

        # Expected Calibration Error (ECE) — centralized in common.metrics
        ece = calibration_error(predictions, actuals)
        sev_cal = None
        if ece > self._cal_crit:
            sev_cal = AlertSeverity.CRITICAL
        elif ece > self._cal_warn:
            sev_cal = AlertSeverity.WARNING
        metrics.append(self._metric(now, "calibration_ece", ece, severity=sev_cal))

        logger.info("model_monitor_complete", metrics_count=len(metrics))
        return metrics

    def _metric(self, time, name, value, dimension=None, severity=None):
        return {
            "metric_time": time,
            "monitor_type": MonitorType.MODEL.value,
            "metric_name": name,
            "metric_value": float(value) if not np.isnan(value) else 0.0,
            "dimension": dimension,
            "window_size": self._window,
            "severity": severity.value if severity else None,
        }
