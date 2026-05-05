"""Unit tests for Phase B-3 top-k + quality feedback loop。

涵蓋：
* ``find_similar_regimes(top_k=...)`` 依 score 由高到低排序、僅回傳 ≥ threshold
* ``last_evaluated_ic`` 優先於 ``performance_summary.rank_ic`` 作為 perf gate
* ``last_evaluated_ic`` 為 NULL 時 fallback 到 holdout perf summary 的舊行為
* ``find_similar_regime`` 包裝（top_k=1 + 不過 threshold 時回 best_score）

PG 連線以 monkeypatch ``pd.read_sql`` 模擬，不依賴 docker。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.adaptation.recurring_concept import RecurringConceptPool


# 用「現在」往前 10 天作為測試 entry 的 detected_at，避免 staleness 把分數壓到 0
_RECENT_DT = datetime.utcnow() - timedelta(days=10)


_BASE_FP = {
    "volatility": 0.02,
    "autocorrelation": 0.0,
    "avg_cross_correlation": 0.1,
    "trend_strength": 0.0,
    "volume_ratio": 1.0,
    "cs_return_std": 0.015,
    "cs_return_skew": 0.0,
    "cs_return_kurt": 0.0,
    "cs_tail_spread": 0.02,
    "vol_of_vol_5d": 0.005,
    "vol_of_vol_20d": 0.003,
    "cvar_5pct": -0.02,
    "alpha_ic_mean": 0.02,
    "alpha_ic_std": 0.03,
    "alpha_ic_pos_fraction": 0.6,
}


def _row(
    regime_id: str,
    detected_at: datetime,
    fp_override: dict | None = None,
    *,
    holdout_rank_ic: float = 0.05,
    last_evaluated_ic: float | None = None,
) -> dict:
    fp = dict(_BASE_FP)
    if fp_override:
        fp.update(fp_override)
    return {
        "regime_id": regime_id,
        "detected_at": detected_at,
        "fingerprint": json.dumps(fp),
        "associated_model_id": f"model_{regime_id}",
        "associated_alpha_weights": json.dumps({}),
        "performance_summary": json.dumps({"rank_ic": holdout_rank_ic, "ic": 0.1}),
        "times_reused": 0,
        "last_reused_at": None,
        "last_evaluated_ic": last_evaluated_ic,
        "last_evaluated_at": None,
    }


def _patch_pool(rows: list[dict]):
    """Return a fake ``pd.read_sql`` that yields a DataFrame from ``rows``。"""
    def _fake(sql: str, conn=None, params=None):
        return pd.DataFrame(rows)
    return _fake


@pytest.fixture(autouse=True)
def _stub_pg(monkeypatch):
    """避免 RecurringConceptPool 真的去打 PG（每個測試各自 patch 想要的 pool 內容）。"""
    class _FakeConn:
        def cursor(self):
            raise RuntimeError("should not reach cursor in these tests")
        def commit(self):
            pass
        def close(self):
            pass
    monkeypatch.setattr("src.adaptation.recurring_concept.get_pg_connection", lambda: _FakeConn())


class TestFindSimilarRegimesTopK:
    def test_empty_pool_returns_empty_list(self) -> None:
        pool = RecurringConceptPool(similarity_threshold=0.5)
        with patch("pandas.read_sql", _patch_pool([])):
            results = pool.find_similar_regimes(_BASE_FP, top_k=3)
        assert results == []

    def test_all_below_threshold_returns_empty(self) -> None:
        """構造 3 個都遠離 current 的 entries（預期 score < 0.5）。"""
        pool = RecurringConceptPool(similarity_threshold=0.5)
        # 構造距離很大的 fingerprint（每個維度都偏離 5 個 prior scale）
        far = {k: v + 5.0 for k, v in _BASE_FP.items()}
        rows = [
            _row(f"far_{i}", _RECENT_DT, fp_override=far)
            for i in range(3)
        ]
        with patch("pandas.read_sql", _patch_pool(rows)):
            results = pool.find_similar_regimes(_BASE_FP, top_k=3)
        assert results == []

    def test_returns_top_k_ordered_by_score(self) -> None:
        """Pool 有 4 個漸進偏離的 entries → 應回傳 top-k 中前幾個（最近的）。"""
        pool = RecurringConceptPool(similarity_threshold=0.0)  # 全收
        # detected_at 都一樣，差異只在 fingerprint
        base_t = _RECENT_DT
        rows = []
        # entry_0：完全相同 → sim = 1.0
        rows.append(_row("regime_0", base_t, fp_override=None))
        # entry_1：volatility 偏 0.5 prior → sim ~ exp(-0.5/2) ≈ 0.78
        rows.append(_row("regime_1", base_t, fp_override={"volatility": 0.02 + 0.5 * 0.01}))
        # entry_2：volatility 偏 1.5 prior → sim ~ exp(-1.5/2) ≈ 0.47
        rows.append(_row("regime_2", base_t, fp_override={"volatility": 0.02 + 1.5 * 0.01}))
        # entry_3：volatility 偏 5 prior → sim ~ exp(-5/2) ≈ 0.082
        rows.append(_row("regime_3", base_t, fp_override={"volatility": 0.02 + 5.0 * 0.01}))

        with patch("pandas.read_sql", _patch_pool(rows)):
            results = pool.find_similar_regimes(_BASE_FP, top_k=2)

        assert len(results) == 2
        assert results[0][0] == "regime_0"
        assert results[1][0] == "regime_1"
        assert results[0][1] > results[1][1], "score 必須遞減"

    def test_top_k_clamped_to_pool_size(self) -> None:
        pool = RecurringConceptPool(similarity_threshold=0.0)
        rows = [_row("only", _RECENT_DT)]
        with patch("pandas.read_sql", _patch_pool(rows)):
            results = pool.find_similar_regimes(_BASE_FP, top_k=10)
        assert len(results) == 1


class TestLastEvaluatedIcOverride:
    def test_holdout_used_when_last_eval_null(self) -> None:
        """``last_evaluated_ic`` 為 NULL → 用 ``performance_summary.rank_ic`` 做 perf gate。"""
        pool = RecurringConceptPool(similarity_threshold=0.0, min_rank_ic=0.0)
        rows = [_row(
            "regime_holdout_only",
            _RECENT_DT,
            holdout_rank_ic=0.05,
            last_evaluated_ic=None,
        )]
        with patch("pandas.read_sql", _patch_pool(rows)):
            results = pool.find_similar_regimes(_BASE_FP, top_k=1)
        # holdout rank_ic=0.05 ≥ 0 → perf_gate=1 → score > 0 → 過 threshold=0
        assert len(results) == 1

    def test_last_eval_below_min_kills_score(self) -> None:
        """``last_evaluated_ic`` < min_rank_ic → perf_gate=0 → score=0 → 不過 threshold。"""
        # min_rank_ic=0.0：last_eval=-0.05 < 0 直接歸零
        pool = RecurringConceptPool(similarity_threshold=0.0, min_rank_ic=0.0)
        rows = [_row(
            "regime_bad_last",
            _RECENT_DT,
            holdout_rank_ic=0.10,           # holdout 看起來很好
            last_evaluated_ic=-0.05,        # 但最近 shadow eval 很爛
        )]
        with patch("pandas.read_sql", _patch_pool(rows)):
            results = pool.find_similar_regimes(_BASE_FP, top_k=1)
        # threshold=0 但 score=raw_sim*stale*0=0 不嚴格 ≥ threshold（程式碼 ≥ threshold 條件 0≥0 為真），
        # 不過 score=0 進不了 ranked，所以 results 應為空
        assert results == [], (
            "last_evaluated_ic 低於 min 應讓 perf_gate=0、score=0、不該出現在 results"
        )

    def test_last_eval_above_min_keeps_score(self) -> None:
        pool = RecurringConceptPool(similarity_threshold=0.0, min_rank_ic=0.0)
        rows = [_row(
            "regime_good_last",
            _RECENT_DT,
            holdout_rank_ic=-0.10,          # holdout 很爛
            last_evaluated_ic=+0.05,        # 但最近 shadow eval 好
        )]
        with patch("pandas.read_sql", _patch_pool(rows)):
            results = pool.find_similar_regimes(_BASE_FP, top_k=1)
        assert len(results) == 1
        assert results[0][0] == "regime_good_last"

    def test_last_eval_nan_falls_back_to_holdout(self) -> None:
        """``last_evaluated_ic`` 為 NaN（資料髒）→ 不 trust，fallback holdout。"""
        pool = RecurringConceptPool(similarity_threshold=0.0, min_rank_ic=0.0)
        rows = [_row(
            "regime_nan_last",
            _RECENT_DT,
            holdout_rank_ic=0.05,
            last_evaluated_ic=float("nan"),
        )]
        with patch("pandas.read_sql", _patch_pool(rows)):
            results = pool.find_similar_regimes(_BASE_FP, top_k=1)
        assert len(results) == 1


class TestFindSimilarRegimeWrapper:
    """``find_similar_regime`` （單數）必須維持原 API 行為。"""

    def test_pass_threshold_returns_id_and_score(self) -> None:
        pool = RecurringConceptPool(similarity_threshold=0.5)
        rows = [_row("regime_match", _RECENT_DT)]  # 完全匹配 raw_sim=1.0；staleness 取決於 _RECENT_DT
        with patch("pandas.read_sql", _patch_pool(rows)):
            rid, score = pool.find_similar_regime(_BASE_FP)
        assert rid == "regime_match"
        # _RECENT_DT 在 10 天前，staleness=exp(-10/180)=0.946 → 整體 score ~ 0.94+
        assert score > 0.9

    def test_below_threshold_returns_none_with_best_score(self) -> None:
        """高 threshold 下不過門檻 → (None, best_seen_score)，回傳值供 log 與 retrain reason 使用。"""
        pool = RecurringConceptPool(similarity_threshold=0.95)
        rows = [_row("regime_far", _RECENT_DT,
                    fp_override={"volatility": 0.02 + 1.5 * 0.01})]  # sim ~ 0.47
        with patch("pandas.read_sql", _patch_pool(rows)):
            rid, score = pool.find_similar_regime(_BASE_FP)
        assert rid is None
        assert 0.0 < score < 0.95


class TestUpdateLastEvaluatedIc:
    """``update_last_evaluated_ic`` 對 PG 的 SQL 行為。"""

    def test_update_emits_correct_sql(self, monkeypatch) -> None:
        """確認 update_last_evaluated_ic 用正確的 SQL + 參數。"""
        captured: dict = {}

        class _FakeCursor:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def execute(self, sql: str, params=None):
                captured["sql"] = sql
                captured["params"] = params

        class _FakeConn:
            def cursor(self): return _FakeCursor()
            def commit(self): captured["committed"] = True
            def close(self): pass

        monkeypatch.setattr(
            "src.adaptation.recurring_concept.get_pg_connection",
            lambda: _FakeConn(),
        )
        pool = RecurringConceptPool()
        pool.update_last_evaluated_ic("regime_xyz", 0.0742)
        assert "UPDATE regime_pool SET last_evaluated_ic" in captured["sql"]
        assert "last_evaluated_at" in captured["sql"]
        # params 包括 (ic, timestamp, regime_id)
        assert captured["params"][0] == pytest.approx(0.0742)
        assert captured["params"][2] == "regime_xyz"
        assert captured["committed"] is True

    def test_update_skipped_on_nan(self, monkeypatch) -> None:
        """NaN/Inf IC → 直接 return，不該執行 SQL。"""
        called = {"n": 0}

        class _FakeConn:
            def cursor(self):
                called["n"] += 1
                raise AssertionError("should not be called for NaN")
            def commit(self): pass
            def close(self): pass

        monkeypatch.setattr(
            "src.adaptation.recurring_concept.get_pg_connection",
            lambda: _FakeConn(),
        )
        pool = RecurringConceptPool()
        pool.update_last_evaluated_ic("regime_xyz", float("nan"))
        pool.update_last_evaluated_ic("regime_xyz", float("inf"))
        assert called["n"] == 0
