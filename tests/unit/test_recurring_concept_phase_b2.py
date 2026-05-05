"""Unit tests for Phase B-2 fingerprint expansion (15 dims)。

涵蓋：
* 新增 10 維（cross-sectional 4 + temporal 3 + alpha-side 3）都出現在 fingerprint
* 所有維度都是 finite（NaN/Inf 應 fallback 0.0），含病態 input
* ``alpha_ic_stats`` 注入正確、未提供時 alpha-side 三維為 0.0
* SCALE_PRIORS 涵蓋所有 15 個 keys（避免新維度被噪聲放大）
* ``compute_alpha_ic_stats`` helper：成熟標籤 gate + 視窗切片 + 統計輸出
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from src.adaptation.recurring_concept import (
    SCALE_PRIORS,
    RecurringConceptPool,
    compute_alpha_ic_stats,
)


_EXPECTED_KEYS = [
    # base
    "volatility", "autocorrelation", "avg_cross_correlation",
    "trend_strength", "volume_ratio",
    # cross-sectional
    "cs_return_std", "cs_return_skew", "cs_return_kurt", "cs_tail_spread",
    # temporal
    "vol_of_vol_5d", "vol_of_vol_20d", "cvar_5pct",
    # alpha-side
    "alpha_ic_mean", "alpha_ic_std", "alpha_ic_pos_fraction",
]


def _build_bars(symbols: list[str], n_days: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    rows = []
    for sym in symbols:
        price = 100.0
        for d in dates:
            price *= 1 + rng.randn() * 0.01
            rows.append({
                "security_id": sym,
                "tradetime": d,
                "close": float(price),
                "vol": float(max(1000, rng.exponential(1e6))),
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def pool() -> RecurringConceptPool:
    return RecurringConceptPool(similarity_threshold=0.5)


class TestFingerprintFifteenDims:
    def test_fingerprint_has_all_fifteen_keys(self, pool: RecurringConceptPool) -> None:
        bars = _build_bars(["A", "B", "C"], n_days=60, seed=11)
        fp = pool.compute_regime_fingerprint(bars)
        for k in _EXPECTED_KEYS:
            assert k in fp, f"missing dim: {k}"
        assert set(fp.keys()) == set(_EXPECTED_KEYS), (
            f"unexpected keys: {set(fp.keys()) ^ set(_EXPECTED_KEYS)}"
        )

    def test_all_dims_finite_on_normal_input(self, pool: RecurringConceptPool) -> None:
        bars = _build_bars(["A", "B", "C", "D"], n_days=80, seed=22)
        fp = pool.compute_regime_fingerprint(bars)
        for k, v in fp.items():
            assert isinstance(v, float)
            assert math.isfinite(v), f"{k}={v!r}"

    def test_all_dims_finite_on_constant_prices(self, pool: RecurringConceptPool) -> None:
        """病態 input：所有 close 恆定 → returns/skew/kurt/std 都會爆 NaN。"""
        dates = pd.bdate_range("2024-01-02", periods=30, freq="B")
        bars = pd.DataFrame([
            {"security_id": s, "tradetime": d, "close": 100.0, "vol": 1e6}
            for s in ["X", "Y"]
            for d in dates
        ])
        fp = pool.compute_regime_fingerprint(bars)
        for k, v in fp.items():
            assert math.isfinite(v), f"{k}={v!r}"

    def test_all_dims_finite_on_single_symbol(self, pool: RecurringConceptPool) -> None:
        """單一標的：cs_* 與 cross_corr 都不該有意義，必須 fallback 0。"""
        bars = _build_bars(["ONLY"], n_days=30, seed=3)
        fp = pool.compute_regime_fingerprint(bars)
        for k, v in fp.items():
            assert math.isfinite(v), f"{k}={v!r}"
        assert fp["cs_return_std"] == 0.0
        assert fp["cs_tail_spread"] == 0.0


class TestAlphaSideInjection:
    def test_alpha_side_zero_without_stats(self, pool: RecurringConceptPool) -> None:
        bars = _build_bars(["A", "B"], n_days=40, seed=42)
        fp = pool.compute_regime_fingerprint(bars, alpha_ic_stats=None)
        assert fp["alpha_ic_mean"] == 0.0
        assert fp["alpha_ic_std"] == 0.0
        assert fp["alpha_ic_pos_fraction"] == 0.0

    def test_alpha_side_picked_up_from_stats(self, pool: RecurringConceptPool) -> None:
        bars = _build_bars(["A", "B"], n_days=40, seed=43)
        stats = {
            "alpha_ic_mean": 0.025,
            "alpha_ic_std": 0.04,
            "alpha_ic_pos_fraction": 0.7,
        }
        fp = pool.compute_regime_fingerprint(bars, alpha_ic_stats=stats)
        assert fp["alpha_ic_mean"] == pytest.approx(0.025)
        assert fp["alpha_ic_std"] == pytest.approx(0.04)
        assert fp["alpha_ic_pos_fraction"] == pytest.approx(0.7)

    def test_alpha_side_nan_inf_safely_handled(self, pool: RecurringConceptPool) -> None:
        bars = _build_bars(["A", "B"], n_days=20, seed=4)
        # 模擬一個髒 stats：NaN/Inf 都要被歸 0
        stats = {
            "alpha_ic_mean": float("nan"),
            "alpha_ic_std": float("inf"),
            "alpha_ic_pos_fraction": float("-inf"),
        }
        fp = pool.compute_regime_fingerprint(bars, alpha_ic_stats=stats)
        for k in ["alpha_ic_mean", "alpha_ic_std", "alpha_ic_pos_fraction"]:
            assert fp[k] == 0.0


class TestScalePriorsCoverage:
    def test_priors_cover_all_fingerprint_dims(self) -> None:
        """SCALE_PRIORS 必須涵蓋所有 fingerprint key（少了就讓某維度被 default 1.0 放大）。"""
        for k in _EXPECTED_KEYS:
            assert k in SCALE_PRIORS, f"SCALE_PRIORS missing {k}"

    def test_priors_are_positive_finite(self) -> None:
        for k, v in SCALE_PRIORS.items():
            assert v > 0
            assert math.isfinite(v)


# ---------------------------------------------------------------------------
# compute_alpha_ic_stats helper
# ---------------------------------------------------------------------------

def _make_alpha_panel(
    symbols: list[str],
    dates: pd.DatetimeIndex,
    alpha_ids: list[str],
    rng: np.random.RandomState,
) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        for d in dates:
            for aid in alpha_ids:
                rows.append({
                    "security_id": sym,
                    "tradetime": d,
                    "alpha_id": aid,
                    "alpha_value": float(rng.randn()),
                })
    return pd.DataFrame(rows)


class TestComputeAlphaIcStats:
    def test_empty_input_returns_zero(self) -> None:
        out = compute_alpha_ic_stats(
            alpha_panel=pd.DataFrame(),
            fwd_returns=pd.Series(dtype=float),
            label_available_at=pd.Series(dtype="datetime64[ns]"),
            t=pd.Timestamp("2024-06-30"),
        )
        assert out["alpha_ic_mean"] == 0.0
        assert out["alpha_ic_std"] == 0.0
        assert out["alpha_ic_pos_fraction"] == 0.0
        assert out["n_alphas"] == 0

    def test_basic_window_filtering_uses_matured_labels(self) -> None:
        """label_available_at > t 的列必須被排除（不可有 look-ahead）。"""
        rng = np.random.RandomState(123)
        dates = pd.bdate_range("2024-01-02", periods=80, freq="B")
        symbols = [f"S{i}" for i in range(20)]
        alpha_ids = ["wq001", "wq002", "wq003"]
        ap = _make_alpha_panel(symbols, dates, alpha_ids, rng)

        # forward return 用「signal_time + 5 個交易日」的累積 return（簡化版）
        idx = pd.MultiIndex.from_product([symbols, dates], names=["security_id", "tradetime"])
        fwd = pd.Series(rng.randn(len(idx)) * 0.01, index=idx)
        # label 成熟日 = signal_time + 5 BD
        label_avail = pd.Series(
            [d + pd.Timedelta(days=5) for (_, d) in idx], index=idx,
        )

        # t=中間某日，後段標籤未成熟 → 視窗內可用標籤受限
        t = pd.Timestamp("2024-04-01")
        out = compute_alpha_ic_stats(
            alpha_panel=ap, fwd_returns=fwd, label_available_at=label_avail, t=t,
            window_days=60, purge_days=5, horizon_days=5,
        )
        # 至少要算出某些 IC
        assert out["n_alphas"] >= 1
        assert math.isfinite(out["alpha_ic_mean"])
        assert math.isfinite(out["alpha_ic_std"])
        assert 0.0 <= out["alpha_ic_pos_fraction"] <= 1.0

    def test_no_matured_labels_returns_zero(self) -> None:
        """若視窗內全部標籤都未成熟（label_avail > t），輸出 0。"""
        rng = np.random.RandomState(7)
        dates = pd.bdate_range("2024-04-01", periods=20, freq="B")
        symbols = ["A", "B"]
        ap = _make_alpha_panel(symbols, dates, ["wq001"], rng)
        idx = pd.MultiIndex.from_product([symbols, dates], names=["security_id", "tradetime"])
        fwd = pd.Series(rng.randn(len(idx)) * 0.01, index=idx)
        # 故意把成熟日壓在 t 之後
        t = pd.Timestamp("2024-04-10")
        label_avail = pd.Series([t + pd.Timedelta(days=30)] * len(idx), index=idx)
        out = compute_alpha_ic_stats(
            alpha_panel=ap, fwd_returns=fwd, label_available_at=label_avail, t=t,
            window_days=30, purge_days=5, horizon_days=5,
        )
        assert out["n_alphas"] == 0
        assert out["alpha_ic_mean"] == 0.0


class TestFingerprintJSONSafety:
    def test_fifteen_dim_fingerprint_is_jsonb_safe(self, pool: RecurringConceptPool) -> None:
        """新增的 10 維也要能 JSONB 序列化（PostgreSQL 拒 NaN 字面值）。"""
        bars = _build_bars(["A", "B", "C"], n_days=30, seed=99)
        fp = pool.compute_regime_fingerprint(
            bars,
            alpha_ic_stats={"alpha_ic_mean": 0.01, "alpha_ic_std": 0.02, "alpha_ic_pos_fraction": 0.5},
        )
        # allow_nan=False 模擬 PostgreSQL JSONB 嚴格性
        json.dumps(fp, allow_nan=False)
