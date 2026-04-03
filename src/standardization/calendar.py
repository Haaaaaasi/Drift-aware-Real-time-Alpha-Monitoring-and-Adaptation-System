"""Layer 2 — Trading calendar management for TWSE."""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.common.logging import get_logger

logger = get_logger(__name__)


class TradingCalendar:
    """TWSE trading calendar.

    For MVP v1, uses pandas business day calendar (weekdays only).
    TODO (MVP v2): Integrate official TWSE holiday schedule.
    """

    def __init__(self, holidays: list[date] | None = None) -> None:
        self._holidays = set(holidays or [])

    def get_trading_days(self, start: date, end: date) -> pd.DatetimeIndex:
        """Return trading days between start and end (inclusive)."""
        all_bdays = pd.bdate_range(start=start, end=end, freq="B")
        if self._holidays:
            mask = ~all_bdays.normalize().isin(
                pd.DatetimeIndex(list(self._holidays))
            )
            return all_bdays[mask]
        return all_bdays

    def is_trading_day(self, d: date) -> bool:
        """Check if a given date is a trading day."""
        if d.weekday() >= 5:
            return False
        return d not in self._holidays

    def next_trading_day(self, d: date) -> date:
        """Return the next trading day after d."""
        from datetime import timedelta
        candidate = d + timedelta(days=1)
        while not self.is_trading_day(candidate):
            candidate += timedelta(days=1)
        return candidate

    def previous_trading_day(self, d: date) -> date:
        """Return the previous trading day before d."""
        from datetime import timedelta
        candidate = d - timedelta(days=1)
        while not self.is_trading_day(candidate):
            candidate -= timedelta(days=1)
        return candidate
