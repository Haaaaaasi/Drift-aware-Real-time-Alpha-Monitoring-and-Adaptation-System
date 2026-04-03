"""Unit tests for Layer 2: Data Standardization."""

import pandas as pd
import pytest

from src.standardization.calendar import TradingCalendar
from src.standardization.quality_check import QualityChecker


class TestTradingCalendar:
    def test_weekdays_only(self):
        from datetime import date
        cal = TradingCalendar()
        days = cal.get_trading_days(date(2024, 1, 1), date(2024, 1, 7))
        for d in days:
            assert d.weekday() < 5

    def test_is_trading_day_weekend(self):
        from datetime import date
        cal = TradingCalendar()
        assert not cal.is_trading_day(date(2024, 1, 6))  # Saturday
        assert cal.is_trading_day(date(2024, 1, 8))       # Monday

    def test_custom_holiday(self):
        from datetime import date
        cal = TradingCalendar(holidays=[date(2024, 1, 8)])
        assert not cal.is_trading_day(date(2024, 1, 8))

    def test_next_trading_day(self):
        from datetime import date
        cal = TradingCalendar()
        assert cal.next_trading_day(date(2024, 1, 5)).weekday() == 0  # Friday -> Monday


class TestQualityChecker:
    def test_invalid_ohlc_flagged(self):
        checker = QualityChecker()
        df = pd.DataFrame({
            "security_id": ["A", "B"],
            "tradetime": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "bar_type": ["daily", "daily"],
            "open": [100.0, 100.0],
            "high": [90.0, 110.0],   # A: high < open (invalid)
            "low": [95.0, 95.0],
            "close": [98.0, 105.0],
            "vol": [1000.0, 1000.0],
            "vwap": [97.0, 102.0],
            "cap": [1e9, 1e9],
            "indclass": [1, 1],
            "is_tradable": [True, True],
            "missing_flags": [0, 0],
        })
        result = checker.check_and_flag(df)
        assert not result.loc[result["security_id"] == "A", "is_tradable"].values[0]
        assert result.loc[result["security_id"] == "B", "is_tradable"].values[0]

    def test_zero_volume_untradable(self):
        checker = QualityChecker()
        df = pd.DataFrame({
            "security_id": ["A"],
            "tradetime": pd.to_datetime(["2024-01-02"]),
            "bar_type": ["daily"],
            "open": [100.0], "high": [110.0], "low": [95.0], "close": [105.0],
            "vol": [0.0],
            "vwap": [102.0], "cap": [1e9], "indclass": [1],
            "is_tradable": [True], "missing_flags": [0],
        })
        result = checker.check_and_flag(df)
        assert not result["is_tradable"].values[0]
