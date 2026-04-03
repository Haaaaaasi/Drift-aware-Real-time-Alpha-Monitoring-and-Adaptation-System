"""Layer 1 — Shioaji streaming market data ingestion.

This module is a placeholder for MVP v2 when streaming is introduced.
For MVP v1, use HistoricalLoader with CSV replay instead.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

import pandas as pd

from src.common.logging import get_logger

logger = get_logger(__name__)


class ShioajiStream:
    """Streaming data ingestion via Shioaji API.

    TODO (MVP v2): Implement actual Shioaji connection.
    Currently provides a mock interface for architecture validation.
    """

    def __init__(
        self,
        api_key: str = "",
        secret_key: str = "",
        on_bar: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._api_key = api_key
        self._secret_key = secret_key
        self._on_bar = on_bar
        self._connected = False

    def connect(self) -> None:
        logger.info("shioaji_connect", status="mock")
        self._connected = True

    def subscribe_kbar(self, security_ids: list[str], bar_type: str = "1min") -> None:
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")
        logger.info("shioaji_subscribe", symbols=len(security_ids), bar_type=bar_type)

    def _handle_bar(self, raw_event: dict[str, Any]) -> None:
        """Internal handler that normalizes and forwards bar events."""
        normalized = {
            "security_id": raw_event.get("code", ""),
            "datetime": raw_event.get("ts", datetime.utcnow()),
            "open": raw_event.get("Open", 0.0),
            "high": raw_event.get("High", 0.0),
            "low": raw_event.get("Low", 0.0),
            "close": raw_event.get("Close", 0.0),
            "volume": raw_event.get("Volume", 0),
        }
        if self._on_bar:
            self._on_bar(normalized)

    def disconnect(self) -> None:
        self._connected = False
        logger.info("shioaji_disconnect")
