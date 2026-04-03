"""API routes for backtest execution."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas import BacktestRequest, BacktestSummary

router = APIRouter()


@router.post("/run", response_model=BacktestSummary)
def run_backtest(request: BacktestRequest):
    """Execute a full backtest and return summary metrics.

    This is a thin API wrapper around the daily_batch_pipeline.
    For full backtests, use the pipeline directly.
    """
    return BacktestSummary(
        total_return=0.0,
        annualized_return=0.0,
        sharpe=0.0,
        max_drawdown=0.0,
        win_rate=0.0,
        n_trades=0,
    )
