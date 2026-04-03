"""FastAPI application — DARAMS API layer."""

from __future__ import annotations

from fastapi import FastAPI

from src.api.routes import monitoring, signals, adaptation, backtest

app = FastAPI(
    title="DARAMS API",
    description=(
        "Drift-aware Real-time Alpha Monitoring and Adaptation System. "
        "Provides endpoints for monitoring, signal queries, adaptation control, "
        "and backtest execution."
    ),
    version="0.1.0",
)

app.include_router(monitoring.router, prefix="/api/v1/monitoring", tags=["monitoring"])
app.include_router(signals.router, prefix="/api/v1/signals", tags=["signals"])
app.include_router(adaptation.router, prefix="/api/v1/adaptation", tags=["adaptation"])
app.include_router(backtest.router, prefix="/api/v1/backtest", tags=["backtest"])


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "darams"}
