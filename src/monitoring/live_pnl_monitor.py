"""Live PnL monitor for account-aware execution feedback."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.common.metrics import sharpe_ratio


class LivePnLMonitor:
    """Produce strategy-level monitoring metrics from live accounting tables."""

    def __init__(
        self,
        *,
        drawdown_critical: float = -0.20,
        sharpe_critical: float = 0.0,
        daily_return_critical: float = -0.05,
    ) -> None:
        self._drawdown_critical = float(drawdown_critical)
        self._sharpe_critical = float(sharpe_critical)
        self._daily_return_critical = float(daily_return_critical)

    def run(
        self,
        *,
        account_snapshots: pd.DataFrame,
        orders: pd.DataFrame | None = None,
        fills: pd.DataFrame | None = None,
        recommendations: pd.DataFrame | None = None,
        metric_time: datetime | pd.Timestamp | None = None,
        account_id: str | None = None,
        run_id: str | None = None,
        model_id: str | None = None,
        strategy_id: str = "live_daily",
        window_size: int = 20,
    ) -> list[dict[str, Any]]:
        if account_snapshots.empty:
            return []
        snaps = account_snapshots.copy()
        snaps["as_of_date"] = pd.to_datetime(snaps["as_of_date"])
        snaps = snaps.sort_values("as_of_date")
        latest = snaps.iloc[-1]
        metric_ts = pd.Timestamp(metric_time or latest.get("snapshot_time") or datetime.utcnow())
        account_id = account_id or str(latest.get("account_id") or "")
        run_id = run_id or _clean_str(latest.get("run_id"))

        daily_returns = snaps["daily_return"].dropna().astype(float)
        equity = snaps["total_equity"].dropna().astype(float)
        cumulative = _clean_float(latest.get("cumulative_return"))
        daily = _clean_float(latest.get("daily_return"))
        max_dd = _max_drawdown(equity)
        rolling_sharpe = (
            sharpe_ratio(daily_returns.tail(window_size))
            if len(daily_returns.tail(window_size)) >= 2
            else np.nan
        )
        fill_rate = _fill_rate(orders, fills)
        slippage_bps = _mean_or_nan(fills, "slippage_bps")
        cost_bps = _cost_bps(fills)
        tracking_error = _tracking_error(recommendations)

        return [
            self._metric(
                metric_ts, "daily_return", daily, account_id, run_id, model_id,
                strategy_id, window_size, severity="CRITICAL"
                if daily is not None and daily <= self._daily_return_critical else None,
            ),
            self._metric(
                metric_ts, "cumulative_return", cumulative, account_id, run_id,
                model_id, strategy_id, window_size,
            ),
            self._metric(
                metric_ts, "rolling_sharpe", rolling_sharpe, account_id, run_id,
                model_id, strategy_id, window_size, severity="CRITICAL"
                if pd.notna(rolling_sharpe) and rolling_sharpe < self._sharpe_critical else None,
            ),
            self._metric(
                metric_ts, "max_drawdown", max_dd, account_id, run_id, model_id,
                strategy_id, window_size, severity="CRITICAL"
                if pd.notna(max_dd) and max_dd <= self._drawdown_critical else None,
            ),
            self._metric(
                metric_ts, "fill_rate", fill_rate, account_id, run_id, model_id,
                strategy_id, window_size,
            ),
            self._metric(
                metric_ts, "slippage_bps", slippage_bps, account_id, run_id,
                model_id, strategy_id, window_size,
            ),
            self._metric(
                metric_ts, "cost_bps", cost_bps, account_id, run_id, model_id,
                strategy_id, window_size,
            ),
            self._metric(
                metric_ts, "target_vs_actual_tracking_error", tracking_error,
                account_id, run_id, model_id, strategy_id, window_size,
            ),
        ]

    @staticmethod
    def _metric(
        metric_time: pd.Timestamp,
        metric_name: str,
        value: float | None,
        account_id: str,
        run_id: str | None,
        model_id: str | None,
        strategy_id: str,
        window_size: int,
        *,
        severity: str | None = None,
    ) -> dict[str, Any]:
        clean_value = 0.0 if value is None or pd.isna(value) else float(value)
        return {
            "metric_time": metric_time.to_pydatetime(),
            "monitor_type": "strategy",
            "metric_name": metric_name,
            "metric_value": clean_value,
            "dimension": account_id,
            "dimension_type": "account_id",
            "window_size": window_size,
            "run_id": run_id,
            "account_id": account_id,
            "model_id": model_id,
            "strategy_id": strategy_id,
            "metadata": {"source": "live_pnl_monitor"},
            "severity": severity,
            "threshold": 0.0,
        }


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())


def _fill_rate(orders: pd.DataFrame | None, fills: pd.DataFrame | None) -> float:
    if orders is None or orders.empty:
        return np.nan
    n_orders = len(orders)
    if fills is None or fills.empty:
        return 0.0
    return float(fills["order_id"].nunique() / n_orders)


def _cost_bps(fills: pd.DataFrame | None) -> float:
    if fills is None or fills.empty or "gross_notional" not in fills.columns:
        return np.nan
    gross = fills["gross_notional"].fillna(0.0).astype(float).abs().sum()
    if gross <= 0:
        return np.nan
    fees = fills.get("fees_total", pd.Series(dtype=float)).fillna(0.0).astype(float).sum()
    return float(fees / gross * 10000.0)


def _tracking_error(recommendations: pd.DataFrame | None) -> float:
    if recommendations is None or recommendations.empty:
        return np.nan
    if {"target_weight", "current_weight"}.issubset(recommendations.columns):
        diff = recommendations["target_weight"].astype(float) - recommendations["current_weight"].astype(float)
        return float(np.sqrt(np.mean(np.square(diff))))
    return np.nan


def _mean_or_nan(df: pd.DataFrame | None, column: str) -> float:
    if df is None or df.empty or column not in df.columns:
        return np.nan
    values = df[column].dropna().astype(float)
    return float(values.mean()) if not values.empty else np.nan


def _clean_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _clean_str(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)
