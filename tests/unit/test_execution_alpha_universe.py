from __future__ import annotations

import math

import pandas as pd
import pytest

from pipelines.simulate_recent import _next_day_returns, _resolve_alpha_ids_for_run
from src.config.alpha_selection import exclude_indclass_cap_alpha_ids
from src.config.constants import WQ101_INDCLASS_OR_CAP_ALPHA_IDS, WQ101_PURE_PRICE_ALPHA_IDS


def _sample_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "security_id": ["2330"] * 4,
            "tradetime": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            "open": [100.0, 110.0, 121.0, 133.1],
            "vwap": [100.0, 105.0, 115.5, 127.05],
            "close": [10.0, 11.0, 12.0, 13.0],
        }
    )


def test_next_day_returns_close_keeps_legacy_close_to_close_proxy():
    result = _next_day_returns(_sample_bars(), execution_price="close")
    first = result.loc[("2330", pd.Timestamp("2024-01-02")), "next_return"]
    second = result.loc[("2330", pd.Timestamp("2024-01-03")), "next_return"]

    assert math.isclose(first, 0.10, rel_tol=1e-9)
    assert math.isclose(second, 12.0 / 11.0 - 1.0, rel_tol=1e-9)


def test_next_day_returns_next_open_enters_and_exits_one_bar_later():
    result = _next_day_returns(_sample_bars(), execution_price="next_open")
    first = result.loc[("2330", pd.Timestamp("2024-01-02")), "next_return"]
    second = result.loc[("2330", pd.Timestamp("2024-01-03")), "next_return"]

    assert math.isclose(first, 121.0 / 110.0 - 1.0, rel_tol=1e-9)
    assert math.isclose(second, 133.1 / 121.0 - 1.0, rel_tol=1e-9)


def test_next_day_returns_next_vwap_enters_and_exits_one_bar_later():
    result = _next_day_returns(_sample_bars(), execution_price="next_vwap")
    first = result.loc[("2330", pd.Timestamp("2024-01-02")), "next_return"]
    second = result.loc[("2330", pd.Timestamp("2024-01-03")), "next_return"]

    assert math.isclose(first, 115.5 / 105.0 - 1.0, rel_tol=1e-9)
    assert math.isclose(second, 127.05 / 115.5 - 1.0, rel_tol=1e-9)


def test_next_day_returns_rejects_unknown_execution_price():
    with pytest.raises(ValueError, match="execution_price"):
        _next_day_returns(_sample_bars(), execution_price="today_open")  # type: ignore[arg-type]


def test_exclude_indclass_cap_alpha_ids_filters_placeholder_inputs():
    selected = exclude_indclass_cap_alpha_ids(["wq001", "wq048", "wq056", "wq100"])

    assert selected == ["wq001"]


def test_exclude_indclass_cap_alpha_ids_defaults_to_pure_price_universe():
    selected = exclude_indclass_cap_alpha_ids(None)

    assert selected == WQ101_PURE_PRICE_ALPHA_IDS
    assert not (set(selected or []) & set(WQ101_INDCLASS_OR_CAP_ALPHA_IDS))


def test_resolve_alpha_ids_returns_empty_when_manual_subset_is_fully_blocked():
    selected = _resolve_alpha_ids_for_run(
        alpha_ids=["wq048", "wq056"],
        skip_effective_filter=True,
        exclude_indclass_cap=True,
    )

    assert selected == []
