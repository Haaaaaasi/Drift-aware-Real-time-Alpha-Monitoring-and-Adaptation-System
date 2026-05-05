"""Data-source helpers and guardrails for research pipelines."""

from __future__ import annotations

from pathlib import Path

from src.config.constants import DATA_SOURCE_DEFAULT_PATHS, DEFAULT_DATA_SOURCE


YFINANCE_DEFAULT_PATH = Path(DATA_SOURCE_DEFAULT_PATHS["csv"])
TEJ_DEFAULT_PATH = Path(DATA_SOURCE_DEFAULT_PATHS["tej"])


def normalize_path(path: str | Path) -> Path:
    """Return a normalized path without requiring it to exist."""
    return Path(path).expanduser()


def infer_data_source_from_path(path: str | Path) -> str:
    """Infer data source from an OHLCV file path.

    ``.parquet`` and filenames containing ``tej`` are treated as TEJ.  The known
    yfinance file is treated as ``csv``. Other CSV files, including synthetic
    test fixtures, are classified as ``custom`` to avoid blocking tests.
    """
    path = normalize_path(path)
    name = path.name.lower()
    if path.suffix.lower() == ".parquet" or "tej" in name:
        return "tej"
    if name == YFINANCE_DEFAULT_PATH.name:
        return "csv"
    return "custom"


def is_known_yfinance_path(path: str | Path) -> bool:
    """Return True for the project yfinance OHLCV artifact."""
    path = normalize_path(path)
    return path.name.lower() == YFINANCE_DEFAULT_PATH.name.lower()


def assert_yfinance_allowed(path: str | Path, allow_yfinance: bool = False) -> None:
    """Fail fast when code tries to use the known polluted yfinance dataset."""
    if is_known_yfinance_path(path) and not allow_yfinance:
        raise ValueError(
            "data/tw_stocks_ohlcv.csv 是已知污染的 yfinance 資料源 "
            "(stock 8476 split-adjustment error)。正式研究請使用 "
            "data/tw_stocks_tej.parquet；若只是 demo/反例，請明確加 "
            "--allow-yfinance。"
        )


def default_data_path(data_source: str = DEFAULT_DATA_SOURCE) -> str:
    """Return the configured default path for a named data source."""
    return DATA_SOURCE_DEFAULT_PATHS[data_source]
