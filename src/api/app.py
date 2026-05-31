"""FastAPI application — DARAMS API layer."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from src.api.routes import monitoring, signals, adaptation, backtest, live

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
app.include_router(live.router, prefix="/api/v1/live", tags=["live"])


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "darams"}


@app.get("/live", include_in_schema=False)
def live_console():
    html_path = Path(__file__).parent / "static" / "live_console.html"
    return FileResponse(html_path)
