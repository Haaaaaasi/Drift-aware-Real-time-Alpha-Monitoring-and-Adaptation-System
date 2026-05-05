"""Unit tests for the alpha parquet cache."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.alpha_engine.alpha_cache import (
    compute_with_cache,
    read_cache,
    read_cache_manifest,
    write_cache,
)


@pytest.fixture
def tiny_bars() -> pd.DataFrame:
    """Minimal bars fixture (3 stocks × 30 days)."""
    np.random.seed(42)
    tickers = ["AA", "BB", "CC"]
    dates = pd.bdate_range("2024-01-01", periods=30)
    rows = []
    for t in tickers:
        price = 50.0
        for d in dates:
            price = max(price * (1 + np.random.randn() * 0.01), 1.0)
            h = price * 1.005
            lo = price * 0.995
            rows.append({
                "security_id": t,
                "tradetime": d,
                "bar_type": "daily",
                "open": price,
                "high": h,
                "low": lo,
                "close": price,
                "vol": float(np.random.exponential(5e4)),
                "vwap": (h + lo + price) / 3,
                "cap": price * 1e6,
                "indclass": (hash(t) % 3) + 1,
                "is_tradable": True,
                "missing_flags": 0,
            })
    return pd.DataFrame(rows)


def test_cache_roundtrip(tmp_path, tiny_bars):
    path = tmp_path / "alpha_cache.parquet"
    df = pd.DataFrame({
        "security_id": ["AA", "BB"],
        "tradetime": pd.to_datetime(["2024-01-02", "2024-01-02"]),
        "alpha_id": ["wq001", "wq001"],
        "alpha_value": [0.5, -0.5],
    })
    write_cache(df, path, data_source="tej")
    loaded = read_cache(path, expected_data_source="tej")
    assert loaded is not None
    assert len(loaded) == len(df)
    assert list(loaded.columns) == ["security_id", "tradetime", "alpha_id", "alpha_value"]
    manifest = read_cache_manifest(path)
    assert manifest["data_source"] == "tej"
    assert manifest["rows"] == len(df)


def test_cold_start_writes_cache(tmp_path, tiny_bars):
    path = tmp_path / "alpha_cache.parquet"
    assert not path.exists()
    result = compute_with_cache(
        tiny_bars,
        alpha_ids=["wq001", "wq012"],
        cache_path=path,
        data_source="custom",
    )
    assert path.exists()
    assert (tmp_path / "alpha_cache.parquet.manifest.json").exists()
    assert len(result) > 0
    assert set(result["alpha_id"].unique()) == {"wq001", "wq012"}


def test_incremental_update(tmp_path, tiny_bars):
    """After caching 20 days, extending to 30 days should only recompute new dates."""
    path = tmp_path / "alpha_cache.parquet"
    early_bars = tiny_bars[tiny_bars["tradetime"] <= tiny_bars["tradetime"].sort_values().unique()[19]]
    full_result = compute_with_cache(
        early_bars,
        alpha_ids=["wq001"],
        cache_path=path,
        data_source="custom",
    )
    first_cache_rows = len(full_result)

    # Now compute with all 30 days — should append new dates
    updated = compute_with_cache(
        tiny_bars,
        alpha_ids=["wq001"],
        cache_path=path,
        data_source="custom",
    )
    assert len(updated) >= first_cache_rows
    # The cache file should cover more dates than the initial cache
    cached = read_cache(path, expected_data_source="custom")
    assert cached["tradetime"].nunique() > early_bars["tradetime"].nunique()


def test_force_recompute(tmp_path, tiny_bars):
    path = tmp_path / "alpha_cache.parquet"
    first = compute_with_cache(tiny_bars, alpha_ids=["wq001"], cache_path=path, data_source="custom")
    second = compute_with_cache(
        tiny_bars,
        alpha_ids=["wq001"],
        cache_path=path,
        force_recompute=True,
        data_source="custom",
    )
    assert len(second) > 0


def test_alpha_id_filter(tmp_path, tiny_bars):
    """compute_with_cache with alpha_ids filter should return only those alphas."""
    path = tmp_path / "alpha_cache.parquet"
    result = compute_with_cache(
        tiny_bars,
        alpha_ids=["wq001", "wq003"],
        cache_path=path,
        data_source="custom",
    )
    assert set(result["alpha_id"].unique()).issubset({"wq001", "wq003"})


def test_manifest_mismatch_raises(tmp_path):
    path = tmp_path / "alpha_cache.parquet"
    df = pd.DataFrame({
        "security_id": ["AA"],
        "tradetime": pd.to_datetime(["2024-01-02"]),
        "alpha_id": ["wq001"],
        "alpha_value": [0.5],
    })
    write_cache(df, path, data_source="csv")
    with pytest.raises(RuntimeError, match="來源不符"):
        read_cache(path, expected_data_source="tej")


def test_missing_manifest_raises_when_expected(tmp_path):
    path = tmp_path / "legacy_cache.parquet"
    df = pd.DataFrame({
        "security_id": ["AA"],
        "tradetime": pd.to_datetime(["2024-01-02"]),
        "alpha_id": ["wq001"],
        "alpha_value": [0.5],
    })
    df.to_parquet(path, index=False)
    with pytest.raises(RuntimeError, match="缺少來源 manifest"):
        read_cache(path, expected_data_source="tej")
