"""Unit tests for ``RecurringConceptPool.compute_regime_fingerprint``.

針對「fingerprint 出現 NaN/Inf 時，PostgreSQL JSONB 會拒絕 + cosine 相似度
退回 0」這個 silent miss bug 加保護。

Bug 來源：
- 單一標的時 cross-correlation 沒有 upper-triangle pair → ``mean()`` 為 NaN
- 標的報酬完全恆定（autocorr 分母為 0）→ NaN
- 成交量 tail/mean 比值在分母極小時可能變 Inf
所有情況都應 fallback 到 0.0 而非 NaN/Inf。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.adaptation.recurring_concept import RecurringConceptPool


def _build_bars(symbols: list[str], n_days: int = 30, seed: int = 0) -> pd.DataFrame:
    """產一段 ``n_days`` 個交易日的合成 bars，每檔報酬隨機。"""
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
    return RecurringConceptPool(similarity_threshold=0.8)


class TestFingerprintFiniteness:
    """所有 fingerprint 欄位都必須是 finite float（非 NaN、非 Inf）。"""

    def test_normal_multi_security_data_all_finite(self, pool: RecurringConceptPool) -> None:
        bars = _build_bars(["A", "B", "C"], n_days=40, seed=42)
        fp = pool.compute_regime_fingerprint(bars)
        for k, v in fp.items():
            assert isinstance(v, float), f"{k} 應為 float，實際 {type(v).__name__}"
            assert math.isfinite(v), f"{k}={v!r} 不是 finite"

    def test_single_security_yields_zero_cross_correlation(self, pool: RecurringConceptPool) -> None:
        """只有 1 檔股票時 cross-correlation 沒有 upper-triangle → 應 fallback 0.0。"""
        bars = _build_bars(["ONLY"], n_days=30, seed=1)
        fp = pool.compute_regime_fingerprint(bars)
        assert fp["avg_cross_correlation"] == 0.0
        for k, v in fp.items():
            assert math.isfinite(v), f"{k}={v!r}"

    def test_constant_prices_no_nan(self, pool: RecurringConceptPool) -> None:
        """所有標的所有日期 close 完全恆定 → returns 全 0 → autocorr/std 容易爆 NaN。

        這是 silent pool miss bug 的最常見觸發條件。
        """
        dates = pd.bdate_range("2024-01-02", periods=30, freq="B")
        rows = []
        for sym in ["FLAT_A", "FLAT_B"]:
            for d in dates:
                rows.append({
                    "security_id": sym,
                    "tradetime": d,
                    "close": 50.0,
                    "vol": 1_000_000.0,
                })
        bars = pd.DataFrame(rows)
        fp = pool.compute_regime_fingerprint(bars)
        for k, v in fp.items():
            assert math.isfinite(v), f"{k}={v!r} 應 fallback 為 finite"

    def test_single_security_constant_prices_no_nan(self, pool: RecurringConceptPool) -> None:
        """同時觸發 single-security + constant prices → 各個分母都會崩。"""
        dates = pd.bdate_range("2024-01-02", periods=30, freq="B")
        bars = pd.DataFrame([
            {"security_id": "X", "tradetime": d, "close": 100.0, "vol": 50_000.0}
            for d in dates
        ])
        fp = pool.compute_regime_fingerprint(bars)
        for k, v in fp.items():
            assert math.isfinite(v), f"{k}={v!r}"

    def test_zero_volume_no_inf(self, pool: RecurringConceptPool) -> None:
        """vol 全為 0 時 ratio 分母由 max(.., 1e-8) 兜底，不該爆 Inf。"""
        dates = pd.bdate_range("2024-01-02", periods=20, freq="B")
        rng = np.random.RandomState(7)
        rows = []
        for sym in ["A", "B"]:
            price = 100.0
            for d in dates:
                price *= 1 + rng.randn() * 0.005
                rows.append({
                    "security_id": sym,
                    "tradetime": d,
                    "close": float(price),
                    "vol": 0.0,
                })
        bars = pd.DataFrame(rows)
        fp = pool.compute_regime_fingerprint(bars)
        for k, v in fp.items():
            assert math.isfinite(v), f"{k}={v!r}"


class TestFingerprintIsJSONSerializable:
    """fingerprint 必須能直接 ``json.dumps`` 後丟進 PostgreSQL JSONB（不接受 NaN 字面值）。"""

    def test_json_dumps_with_strict_allow_nan_false(self, pool: RecurringConceptPool) -> None:
        """``json.dumps(allow_nan=False)`` 模擬 PostgreSQL JSONB 的嚴格性。

        若 fingerprint 任一欄位為 NaN，dumps 會 raise ``ValueError: Out of range float
        values are not JSON compliant``——這正是 production 端 INSERT 失敗的原因。
        """
        import json

        # 用各種容易 NaN 的場景組合
        scenarios = [
            _build_bars(["A"], n_days=15, seed=1),                    # single security
            _build_bars(["A", "B", "C", "D"], n_days=40, seed=2),     # normal
            pd.DataFrame([                                             # constant prices
                {"security_id": s, "tradetime": d, "close": 10.0, "vol": 1.0}
                for s in ["X", "Y"]
                for d in pd.bdate_range("2024-01-02", periods=20, freq="B")
            ]),
        ]
        for bars in scenarios:
            fp = pool.compute_regime_fingerprint(bars)
            # 若 NaN 漏出 → 此 dumps 會 raise，失敗訊息會直接指出有問題的 key
            json.dumps(fp, allow_nan=False)
