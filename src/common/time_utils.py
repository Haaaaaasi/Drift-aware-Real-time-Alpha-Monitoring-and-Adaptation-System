"""Time and trading calendar utilities."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Sequence

import numpy as np
import pandas as pd


def get_twse_trading_calendar(start: date, end: date) -> pd.DatetimeIndex:
    """Return TWSE trading days between start and end (inclusive).

    Uses pandas business-day calendar, excluding weekends.
    For production use, integrate an official TWSE holiday calendar.
    """
    all_days = pd.bdate_range(start=start, end=end, freq="B")
    return all_days


def align_to_calendar(
    df: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    time_col: str = "tradetime",
    id_col: str = "security_id",
) -> pd.DataFrame:
    """Align a DataFrame to trading calendar, marking gaps."""
    all_ids = df[id_col].unique()
    full_index = pd.MultiIndex.from_product(
        [all_ids, calendar], names=[id_col, time_col]
    )
    aligned = df.set_index([id_col, time_col]).reindex(full_index)
    aligned["is_missing"] = aligned.isnull().any(axis=1)
    return aligned.reset_index()


def ensure_no_lookahead(
    signal_time: datetime | pd.Timestamp,
    label_time: datetime | pd.Timestamp,
) -> bool:
    """Verify that label_time is strictly after signal_time."""
    return label_time > signal_time


def compute_label_available_at(
    signal_time: datetime,
    horizon_bars: int,
    bar_type: str = "daily",
    buffer_bars: int = 1,
) -> datetime:
    """Compute when a label becomes available (signal_time + horizon + buffer)."""
    if bar_type == "daily":
        delta = timedelta(days=horizon_bars + buffer_bars)
    elif bar_type == "30min":
        delta = timedelta(minutes=30 * (horizon_bars + buffer_bars))
    elif bar_type == "5min":
        delta = timedelta(minutes=5 * (horizon_bars + buffer_bars))
    else:
        delta = timedelta(days=horizon_bars + buffer_bars)
    return signal_time + delta


def generate_bar_timestamps(
    start: datetime,
    end: datetime,
    bar_type: str = "daily",
) -> pd.DatetimeIndex:
    """Generate bar-aligned timestamps for a date range."""
    freq_map = {"daily": "B", "30min": "30min", "5min": "5min", "1min": "1min"}
    freq = freq_map.get(bar_type, "B")
    return pd.date_range(start=start, end=end, freq=freq)
