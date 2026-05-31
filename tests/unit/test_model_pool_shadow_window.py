"""Model pool shadow window 的交易日切窗與 label 成熟檢查。"""

import pandas as pd

from src.adaptation.model_pool_strategy import ModelPoolController


def test_shadow_cutoffs_use_actual_trading_days() -> None:
    bars = pd.DataFrame({
        "tradetime": pd.to_datetime([
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
            "2024-01-10",
            "2024-01-11",
            "2024-01-12",
            "2024-01-16",
        ])
    })

    start, end, train = ModelPoolController._compute_shadow_cutoffs(
        bars=bars,
        t=pd.Timestamp("2024-01-16"),
        shadow_window=4,
        maturity_gap=3,
        warmup_days=2,
    )

    assert start == pd.Timestamp("2024-01-04")
    assert end == pd.Timestamp("2024-01-10")
    assert train == pd.Timestamp("2024-01-02")


def test_shadow_forward_returns_require_mature_labels() -> None:
    idx = pd.MultiIndex.from_tuples(
        [
            ("A", pd.Timestamp("2024-01-05")),
            ("A", pd.Timestamp("2024-01-08")),
            ("A", pd.Timestamp("2024-01-09")),
            ("A", pd.Timestamp("2024-01-10")),
        ],
        names=["security_id", "tradetime"],
    )
    fwd = pd.Series([0.01, 0.02, 0.03, 0.04], index=idx)
    label_available_at = pd.Series(
        [
            pd.Timestamp("2024-01-11"),
            pd.Timestamp("2024-01-12"),
            pd.Timestamp("2024-01-17"),
            pd.Timestamp("2024-01-18"),
        ],
        index=idx,
    )

    sliced = ModelPoolController._slice_shadow_forward_returns(
        fwd_returns=fwd,
        label_available_at=label_available_at,
        shadow_cutoff_start=pd.Timestamp("2024-01-04"),
        shadow_cutoff_end=pd.Timestamp("2024-01-10"),
        as_of=pd.Timestamp("2024-01-16"),
    )

    assert list(sliced.index.get_level_values("tradetime")) == [
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-01-08"),
    ]
