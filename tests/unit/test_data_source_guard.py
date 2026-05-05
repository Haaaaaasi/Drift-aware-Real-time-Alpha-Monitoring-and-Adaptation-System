"""資料源 guardrail 測試。"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.data_sources import (
    assert_yfinance_allowed,
    infer_data_source_from_path,
    is_known_yfinance_path,
)


def test_known_yfinance_path_is_blocked_by_default():
    path = Path("data/tw_stocks_ohlcv.csv")
    assert is_known_yfinance_path(path)
    with pytest.raises(ValueError, match="已知污染"):
        assert_yfinance_allowed(path)


def test_known_yfinance_path_can_be_explicitly_allowed():
    assert_yfinance_allowed("data/tw_stocks_ohlcv.csv", allow_yfinance=True)


def test_synthetic_csv_is_custom_not_blocked():
    path = Path("tmp/synthetic_ohlcv.csv")
    assert infer_data_source_from_path(path) == "custom"
    assert_yfinance_allowed(path)


def test_tej_parquet_is_tej():
    assert infer_data_source_from_path("data/tw_stocks_tej.parquet") == "tej"
