"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class MetricResponse(BaseModel):
    metric_time: datetime
    monitor_type: str
    metric_name: str
    metric_value: float
    dimension: str | None = None


class AlertResponse(BaseModel):
    alert_id: int
    alert_time: datetime
    monitor_type: str
    metric_name: str
    severity: str
    current_value: float
    message: str | None = None


class SignalResponse(BaseModel):
    security_id: str
    signal_time: datetime
    signal_score: float
    signal_direction: int
    confidence: float | None = None
    method: str


class BacktestRequest(BaseModel):
    start_date: str
    end_date: str
    alpha_ids: list[str] | None = None
    top_k: int = 10
    rebalance_frequency: str = "daily"


class BacktestSummary(BaseModel):
    total_return: float
    annualized_return: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    n_trades: int


class AdaptationStatus(BaseModel):
    policy: str
    last_triggered: datetime | None = None
    current_model_id: str | None = None
    pool_size: int = 0
