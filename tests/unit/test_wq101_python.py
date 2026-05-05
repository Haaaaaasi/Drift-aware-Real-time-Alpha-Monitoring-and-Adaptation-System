"""Unit tests for the Python WQ101 alpha engine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.alpha_engine.wq101_python import (
    _rank,
    _delta,
    _delay,
    _ts_rank,
    _ts_argmax,
    _correlation,
    _decay_linear,
    _indneutralize,
    _cs_zscore,
    compute_wq101_alphas,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_panel() -> pd.DataFrame:
    """5 stocks × 60 trading days of synthetic OHLCV bars."""
    np.random.seed(0)
    n_stocks = 5
    n_days = 60
    tickers = [f"T{i:04d}" for i in range(n_stocks)]
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rows = []
    for t in tickers:
        price = 100.0
        for d in dates:
            ret = np.random.randn() * 0.015
            price = max(price * (1 + ret), 1.0)
            o = price * (1 + np.random.randn() * 0.003)
            h = max(o, price) * (1 + abs(np.random.randn()) * 0.003)
            lo = min(o, price) * (1 - abs(np.random.randn()) * 0.003)
            v = max(1000.0, float(np.random.exponential(1e5)))
            rows.append({
                "security_id": t,
                "tradetime": d,
                "bar_type": "daily",
                "open": round(o, 4),
                "high": round(h, 4),
                "low": round(lo, 4),
                "close": round(price, 4),
                "vol": v,
                "vwap": (h + lo + price) / 3,
                "cap": price * 1e6,
                "indclass": (hash(t) % 3) + 1,
                "is_tradable": True,
                "missing_flags": 0,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def wide_close(simple_panel) -> pd.DataFrame:
    return simple_panel.pivot_table(index="tradetime", columns="security_id", values="close")


@pytest.fixture
def wide_vol(simple_panel) -> pd.DataFrame:
    return simple_panel.pivot_table(index="tradetime", columns="security_id", values="vol")


# ---------------------------------------------------------------------------
# Operator tests
# ---------------------------------------------------------------------------

def test_rank_percentile(wide_close):
    r = _rank(wide_close)
    assert r.notna().any().any()
    assert (r.dropna() >= 0).all().all()
    assert (r.dropna() <= 1).all().all()


def test_delta_first_row_nan(wide_close):
    d = _delta(wide_close, 1)
    assert d.iloc[0].isna().all()
    expected = wide_close.iloc[1] - wide_close.iloc[0]
    pd.testing.assert_series_equal(d.iloc[1], expected, check_names=False)


def test_delay_shifts(wide_close):
    dl = _delay(wide_close, 3)
    # Row i of delayed panel should equal row i-3 of original
    np.testing.assert_array_almost_equal(
        dl.iloc[3:].values, wide_close.iloc[:-3].values
    )


def test_ts_rank_bounds(wide_close):
    r = _ts_rank(wide_close, 5)
    valid = r.dropna()
    assert (valid >= 0).all().all()
    assert (valid <= 1).all().all()


def test_ts_argmax_today_is_max():
    """If today's value is the running max, ts_argmax should be 0."""
    data = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0, 4.0, 5.0]},
        index=pd.date_range("2024-01-01", periods=5),
    )
    result = _ts_argmax(data, 3)
    assert result.iloc[-1]["A"] == 0.0


def test_correlation_returns_correct_shape(wide_close, wide_vol):
    c = _correlation(wide_close, wide_vol, 10)
    assert c.shape == wide_close.shape


def test_decay_linear_weights_sum_to_one():
    """For a constant panel, decay_linear should return that constant."""
    data = pd.DataFrame(
        {"A": [2.0] * 20, "B": [3.0] * 20},
        index=pd.date_range("2024-01-01", periods=20),
    )
    result = _decay_linear(data, 5)
    pd.testing.assert_frame_equal(result.iloc[5:], data.iloc[5:], atol=1e-9)


def test_indneutralize_zero_mean_per_group():
    close = pd.DataFrame(
        {"A": [1.0, 2.0], "B": [3.0, 4.0], "C": [5.0, 6.0]},
        index=pd.date_range("2024-01-01", periods=2),
    )
    ind = pd.DataFrame(
        {"A": [1, 1], "B": [1, 2], "C": [2, 2]},
        index=close.index,
    )
    neu = _indneutralize(close, ind)
    # For date 0: group 1 = {A=1, B=3}, mean=2; group 2 = {C=5}
    assert abs(neu.iloc[0]["A"] - (1 - 2)) < 1e-9  # -1
    assert abs(neu.iloc[0]["C"] - 0.0) < 1e-9      # only member


def test_cs_zscore_zero_mean(wide_close):
    z = _cs_zscore(wide_close.iloc[10:])
    row_means = z.mean(axis=1)
    assert (row_means.abs() < 1e-9).all()


# ---------------------------------------------------------------------------
# compute_wq101_alphas integration tests
# ---------------------------------------------------------------------------

def test_output_schema(simple_panel):
    out = compute_wq101_alphas(simple_panel, alpha_ids=["wq001", "wq012"])
    assert list(out.columns) == ["security_id", "tradetime", "alpha_id", "alpha_value"]
    assert pd.api.types.is_float_dtype(out["alpha_value"])
    assert pd.api.types.is_datetime64_any_dtype(out["tradetime"])


def test_compute_subset(simple_panel):
    ids = ["wq001", "wq040", "wq061"]
    out = compute_wq101_alphas(simple_panel, alpha_ids=ids)
    assert set(out["alpha_id"].unique()) == set(ids)


def test_smoke_all_101(simple_panel):
    """All 101 alphas should produce at least some non-NaN values."""
    out = compute_wq101_alphas(simple_panel, alpha_ids=None)
    computed = out["alpha_id"].unique()
    # Expect most alphas to succeed (allow a handful of failures on tiny data)
    assert len(computed) >= 80, f"Only {len(computed)} alphas succeeded"
    # Values should be finite after z-scoring
    assert np.isfinite(out["alpha_value"]).all()


def test_no_lookahead(simple_panel):
    """Alpha values on date T must be identical whether computed with
    full history or truncated at T+5."""
    full = compute_wq101_alphas(simple_panel, alpha_ids=["wq001", "wq003"])
    t_cutoff = simple_panel["tradetime"].sort_values().unique()[30]
    truncated = compute_wq101_alphas(
        simple_panel[simple_panel["tradetime"] <= t_cutoff],
        alpha_ids=["wq001", "wq003"],
    )
    full_t = full[full["tradetime"] == t_cutoff].set_index(["security_id", "alpha_id"])["alpha_value"]
    trunc_t = truncated[truncated["tradetime"] == t_cutoff].set_index(["security_id", "alpha_id"])["alpha_value"]
    common = full_t.index.intersection(trunc_t.index)
    assert len(common) > 0
    pd.testing.assert_series_equal(
        full_t.loc[common].sort_index(),
        trunc_t.loc[common].sort_index(),
        atol=1e-6,
    )


def test_unknown_alpha_id_raises(simple_panel):
    with pytest.raises(ValueError, match="Unknown alpha IDs"):
        compute_wq101_alphas(simple_panel, alpha_ids=["wq999"])
