"""Unit tests for Phase B-1 similarity scoring helpers in ``recurring_concept``.

針對 raw cosine 在 5 維 fingerprint 下幾乎永遠 > 0.98 的問題，重構為
z-scored Euclidean → exp decay → staleness × perf_gate 三段相乘。

本檔涵蓋五個 helper 的數學性質：
* ``_compute_pool_scales``：cold-start 用 prior、足量 entries 用 pool std
* ``_standardized_distance``：相同 fp 距離為 0；維度差會被 scale 抵消
* ``_distance_to_similarity``：嚴格遞減、d=0 → 1
* ``_staleness_factor``：今日為 1、tau 後 ~0.37、極老為 ~0
* ``_performance_gate``：低於門檻歸 0、無資料維持 1
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.adaptation.recurring_concept import (
    SCALE_PRIORS,
    _compute_pool_scales,
    _distance_to_similarity,
    _performance_gate,
    _staleness_factor,
    _standardized_distance,
)


_FP_KEYS = list(SCALE_PRIORS.keys())


def _make_pool(fps: list[dict[str, float]]) -> pd.DataFrame:
    """模擬 ``SELECT * FROM regime_pool`` 的 DataFrame，只放本檔需要的欄位。"""
    return pd.DataFrame(
        [{"fingerprint": json.dumps(fp)} for fp in fps]
    )


class TestComputePoolScales:
    def test_uses_prior_when_pool_too_small(self) -> None:
        pool = _make_pool([
            {k: 0.5 for k in _FP_KEYS},
            {k: 1.5 for k in _FP_KEYS},
        ])
        scales = _compute_pool_scales(pool, _FP_KEYS, min_entries=3)
        # 兩筆 < 3 → 全用 prior
        for k in _FP_KEYS:
            assert scales[k] == pytest.approx(SCALE_PRIORS[k]), f"{k}"

    def test_uses_pool_std_when_enough_entries(self) -> None:
        # 構造已知 std 的 pool（3 筆，每維度 std 都不同）
        rows = []
        for v in [0.0, 1.0, 2.0]:
            rows.append({k: v for k in _FP_KEYS})
        pool = _make_pool(rows)
        scales = _compute_pool_scales(pool, _FP_KEYS, min_entries=3)
        # ddof=1 std of [0,1,2] = 1.0
        for k in _FP_KEYS:
            assert scales[k] == pytest.approx(1.0, rel=1e-6)

    def test_falls_back_to_prior_when_pool_std_too_small(self) -> None:
        """Pool 內某一維度幾乎沒變化（std < 1e-4）時不該被噪聲放大。"""
        rows = [
            {k: (1.0 if k != "volatility" else 0.01) for k in _FP_KEYS}
            for _ in range(5)
        ]  # 全部相同 → std=0
        pool = _make_pool(rows)
        scales = _compute_pool_scales(pool, _FP_KEYS, min_entries=3)
        for k in _FP_KEYS:
            assert scales[k] == pytest.approx(SCALE_PRIORS[k]), (
                f"{k}: zero-std 維度應回退到 prior，實際 {scales[k]}"
            )


class TestStandardizedDistance:
    def test_identical_fingerprints_zero_distance(self) -> None:
        fp = {"volatility": 0.02, "autocorrelation": 0.1, "avg_cross_correlation": 0.3,
              "trend_strength": 0.0005, "volume_ratio": 1.2}
        d = _standardized_distance(fp, fp, SCALE_PRIORS)
        assert d == pytest.approx(0.0)

    def test_distance_is_scale_invariant_per_dim(self) -> None:
        """單一維度差 1 個 scale 單位的 mean-form distance = 1/sqrt(N)。

        mean-form 語意：d=1 表示「所有 N 維平均每維差 1 std」；
        只有 1 維差 1 std 時 d = sqrt(1/N) = 1/sqrt(N)。
        """
        a = {k: 0.0 for k in _FP_KEYS}
        b = dict(a)
        b["volatility"] = SCALE_PRIORS["volatility"]  # 差 1 個 prior scale
        d = _standardized_distance(a, b, SCALE_PRIORS)
        assert d == pytest.approx(1.0 / math.sqrt(len(_FP_KEYS)), rel=1e-6)

    def test_volume_ratio_not_dominating(self) -> None:
        """檢驗 Phase B-1 根治的 volume_ratio 主導問題（mean-form 下同樣成立）：
        volume_ratio 差 0.5 prior、volatility 差 0.5 prior——兩維各貢獻 0.5；
        Euclidean = sqrt(0.5^2 + 0.5^2) = sqrt(0.5)；除以 sqrt(N) 得 mean-form d。
        """
        a = {k: 0.0 for k in _FP_KEYS}
        b = dict(a)
        b["volume_ratio"] = 0.5 * SCALE_PRIORS["volume_ratio"]      # 半個 scale
        b["volatility"] = 0.5 * SCALE_PRIORS["volatility"]           # 半個 scale
        d = _standardized_distance(a, b, SCALE_PRIORS)
        # mean-form: sqrt(0.5) / sqrt(N)
        assert d == pytest.approx(math.sqrt(0.5) / math.sqrt(len(_FP_KEYS)), rel=1e-6)


class TestDistanceToSimilarity:
    def test_zero_distance_unit_sim(self) -> None:
        assert _distance_to_similarity(0.0, distance_scale=2.0) == pytest.approx(1.0)

    def test_strictly_decreasing(self) -> None:
        prev = 1.0 + 1e-9
        for d in [0.5, 1.0, 2.0, 4.0, 10.0]:
            cur = _distance_to_similarity(d, distance_scale=2.0)
            assert cur < prev, f"d={d}: similarity 必須嚴格遞減（{cur} < {prev}）"
            assert 0.0 < cur <= 1.0
            prev = cur

    def test_known_reference_points(self) -> None:
        """verify formula `exp(-d/scale)` matches CLAUDE.md docstring 標的。"""
        s = 2.0
        assert _distance_to_similarity(2.0, s) == pytest.approx(math.exp(-1.0), rel=1e-6)
        assert _distance_to_similarity(4.0, s) == pytest.approx(math.exp(-2.0), rel=1e-6)


class TestStalenessFactor:
    def test_now_yields_one(self) -> None:
        now = datetime(2024, 6, 30, 12, 0, 0)
        assert _staleness_factor(now, now, tau_days=180.0) == pytest.approx(1.0)

    def test_one_tau_yields_inverse_e(self) -> None:
        now = datetime(2024, 6, 30)
        old = now - timedelta(days=180)
        assert _staleness_factor(old, now, tau_days=180.0) == pytest.approx(
            math.exp(-1.0), rel=1e-6
        )

    def test_very_old_decays_to_near_zero(self) -> None:
        now = datetime(2024, 6, 30)
        ancient = now - timedelta(days=365 * 3)
        s = _staleness_factor(ancient, now, tau_days=180.0)
        assert 0.0 < s < 0.005, f"3 年前的 entry 衰減應 ~0，實際 {s}"

    def test_none_detected_at_returns_one(self) -> None:
        """缺 detected_at 時不要懲罰，由其他 component 決定。"""
        assert _staleness_factor(None, datetime(2024, 6, 30), tau_days=180.0) == 1.0

    def test_future_detected_at_clamped_to_one(self) -> None:
        """偶發資料異常導致 detected_at 在 now 之後（如時鐘漂移）— 不該回傳 > 1。"""
        now = datetime(2024, 6, 30)
        future = now + timedelta(days=30)
        s = _staleness_factor(future, now, tau_days=180.0)
        assert s == pytest.approx(1.0)


class TestPerformanceGate:
    def test_none_summary_returns_one(self) -> None:
        assert _performance_gate(None, min_rank_ic=0.0) == 1.0

    def test_empty_summary_returns_one(self) -> None:
        assert _performance_gate({}, min_rank_ic=0.0) == 1.0

    def test_below_min_returns_zero(self) -> None:
        assert _performance_gate({"rank_ic": -0.01}, min_rank_ic=0.0) == 0.0

    def test_at_threshold_returns_one(self) -> None:
        assert _performance_gate({"rank_ic": 0.0}, min_rank_ic=0.0) == 1.0

    def test_above_threshold_returns_one(self) -> None:
        assert _performance_gate({"rank_ic": 0.05}, min_rank_ic=0.02) == 1.0

    def test_falls_back_to_ic_when_rank_ic_missing(self) -> None:
        assert _performance_gate({"ic": 0.05}, min_rank_ic=0.02) == 1.0
        assert _performance_gate({"ic": 0.01}, min_rank_ic=0.02) == 0.0

    def test_unparseable_value_does_not_kill_score(self) -> None:
        """rank_ic 為奇怪型別（dict、字串）時保守回 1.0，避免誤殺所有 candidate。"""
        assert _performance_gate({"rank_ic": "n/a"}, min_rank_ic=0.0) == 1.0
        assert _performance_gate({"rank_ic": None, "ic": None}, min_rank_ic=0.0) == 1.0


class TestEndToEndScoring:
    """整合三段相乘，模擬實際 ``find_similar_regime`` 的算分。"""

    def test_perfect_match_recent_high_perf_gives_max_score(self) -> None:
        fp = {k: 0.5 for k in _FP_KEYS}
        d = _standardized_distance(fp, fp, SCALE_PRIORS)
        raw = _distance_to_similarity(d, distance_scale=2.0)
        stale = _staleness_factor(datetime(2024, 6, 30), datetime(2024, 6, 30), tau_days=180.0)
        perf = _performance_gate({"rank_ic": 0.05}, min_rank_ic=0.0)
        assert raw * stale * perf == pytest.approx(1.0)

    def test_bad_perf_kills_otherwise_perfect_match(self) -> None:
        fp = {k: 0.5 for k in _FP_KEYS}
        d = _standardized_distance(fp, fp, SCALE_PRIORS)
        raw = _distance_to_similarity(d, distance_scale=2.0)
        stale = 1.0
        perf = _performance_gate({"rank_ic": -0.05}, min_rank_ic=0.0)
        assert raw * stale * perf == 0.0
