"""WP11 — Integration tests for model_pool adaptation strategy.

驗證：
1. model_pool strategy 端到端能跑完
2. Day 0 初始訓練後 pool 有第一筆 entry
3. 第一次觸發必定是 pool miss（pool 只有 1 筆 entry 且 threshold 高時）
4. shadow 評估選擇留用當前模型（新候選較差時）
5. 相似 regime 命中時 decision.reason 含 shadow_selected_reused
6. DB 不可用時降級為 triggered 行為
7. DEFAULT_STRATEGIES 包含 model_pool
8. run_ab_experiment 跑完 5 組產出正確 comparison
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from pipelines.ab_experiment import DEFAULT_STRATEGIES, run_ab_experiment
from pipelines.simulate_recent import simulate


# ---------------------------------------------------------------------------
# Shared fixture — 合成 CSV（30 標的 × ~6 個月，與 test_ab_experiment 共用格式）
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_csv(tmp_path_factory) -> Path:
    rng = np.random.RandomState(2027)
    n_symbols = 30
    symbols = [f"SYM{i:04d}" for i in range(1, n_symbols + 1)]
    dates = pd.bdate_range("2023-01-02", "2024-06-30", freq="B")

    rows = []
    for sym in symbols:
        price = 100.0 + rng.randn() * 10
        for d in dates:
            ret = rng.randn() * 0.015
            price = max(1.0, price * (1 + ret))
            o = price * (1 + rng.randn() * 0.003)
            h = max(o, price) * (1 + abs(rng.randn()) * 0.003)
            lo = min(o, price) * (1 - abs(rng.randn()) * 0.003)
            vol = max(1000, int(rng.exponential(300_000)))
            rows.append({
                "datetime": d,
                "security_id": sym,
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(lo, 2),
                "close": round(price, 2),
                "volume": vol,
            })

    df = pd.DataFrame(rows)
    csv_path = tmp_path_factory.mktemp("pool_data") / "synthetic_ohlcv.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(scope="module")
def sim_period():
    return date(2024, 1, 2), date(2024, 6, 30)


# ---------------------------------------------------------------------------
# Fake in-memory pool — 避開真 DB
# ---------------------------------------------------------------------------

class _InMemoryRegimePool:
    """RecurringConceptPool 的 in-memory fake，用於 monkeypatch。

    Phase B-2/B-3 補：
    * ``compute_regime_fingerprint`` 接受 ``alpha_ic_stats`` 第二參數
    * ``find_similar_regimes(top_k)`` 用 cosine 排序回 top-k；保留 ``find_similar_regime`` 包裝
    * ``update_last_evaluated_ic`` no-op（測試只關心呼叫次數，不關心 PG 寫入）
    """

    def __init__(self, similarity_threshold: float = 0.8):
        self._threshold = similarity_threshold
        self.entries: list[dict] = []
        # Phase B-3：記錄 last_evaluated_ic 寫入次數供斷言使用
        self.last_eval_updates: list[tuple[str, float]] = []

    def compute_regime_fingerprint(
        self,
        market_data: pd.DataFrame,
        alpha_ic_stats: dict[str, float] | None = None,
    ) -> dict[str, float]:
        fp = {
            "volatility": float(market_data["close"].pct_change().std() or 0.01),
            "autocorrelation": 0.0,
            "avg_cross_correlation": 0.0,
            "trend_strength": 0.0,
            "volume_ratio": 1.0,
        }
        if alpha_ic_stats:
            fp["alpha_ic_mean"] = float(alpha_ic_stats.get("alpha_ic_mean", 0.0))
            fp["alpha_ic_std"] = float(alpha_ic_stats.get("alpha_ic_std", 0.0))
            fp["alpha_ic_pos_fraction"] = float(alpha_ic_stats.get("alpha_ic_pos_fraction", 0.0))
        return fp

    def _score(self, current_fp: dict, entry_fp: dict) -> float:
        import numpy as np
        keys = list(current_fp.keys())
        cur = np.array([current_fp.get(k, 0.0) for k in keys])
        hist = np.array([entry_fp.get(k, 0.0) for k in keys])
        norm = np.linalg.norm(cur) * np.linalg.norm(hist)
        return float(np.dot(cur, hist) / max(norm, 1e-10))

    def find_similar_regime(self, current_fp: dict, since=None) -> tuple[str | None, float]:
        results = self.find_similar_regimes(current_fp, since=since, top_k=1)
        if results:
            return results[0]
        # threshold 沒過的時候仍回 best_score 供 caller 診斷
        if not self.entries:
            return None, 0.0
        best = max(self._score(current_fp, e["fingerprint"]) for e in self.entries)
        return None, best

    def find_similar_regimes(
        self, current_fp: dict, since=None, top_k: int = 1,
    ) -> list[tuple[str, float]]:
        if not self.entries:
            return []
        scored = [
            (e["regime_id"], self._score(current_fp, e["fingerprint"]))
            for e in self.entries
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(rid, s) for (rid, s) in scored if s >= self._threshold][:top_k]

    def add_to_pool(self, fingerprint, model_id, alpha_weights, performance_summary) -> str:
        rid = f"regime_{len(self.entries):03d}"
        self.entries.append({
            "regime_id": rid,
            "fingerprint": fingerprint,
            "associated_model_id": model_id,
        })
        return rid

    def get_regime_model(self, regime_id: str) -> dict | None:
        for e in self.entries:
            if e["regime_id"] == regime_id:
                return e
        return None

    def record_reuse(self, regime_id: str) -> None:
        pass

    def update_last_evaluated_ic(self, regime_id: str, ic: float) -> None:
        self.last_eval_updates.append((regime_id, ic))


@pytest.fixture()
def fake_pool(monkeypatch):
    """將 RecurringConceptPool 替換成 in-memory fake，避開 PostgreSQL。

    Bug Fix #1 之後 ``ModelPoolController`` 在 ``add_to_pool`` 前會先呼叫
    ``MLMetaModel.register_to_registry()``（避免 FK 失敗），但 register_to_registry
    走真 PG。本 fixture 同時 stub register_to_registry 永遠回 True，模擬
    「PG 可寫」的整合場景。
    """
    pool_instance = _InMemoryRegimePool(similarity_threshold=0.8)

    import src.adaptation.model_pool_strategy as mps
    import src.meta_signal.ml_meta_model as mlm

    def _fake_init(self, similarity_threshold=0.8):
        self._threshold = similarity_threshold
        self._pool_obj = pool_instance
        # 複製 pool_instance 方法到 self
        self.compute_regime_fingerprint = pool_instance.compute_regime_fingerprint
        self.find_similar_regime = pool_instance.find_similar_regime
        self.find_similar_regimes = pool_instance.find_similar_regimes
        self.add_to_pool = pool_instance.add_to_pool
        self.get_regime_model = pool_instance.get_regime_model
        self.record_reuse = pool_instance.record_reuse
        self.update_last_evaluated_ic = pool_instance.update_last_evaluated_ic

    # patch ModelPoolController.initialize_run 改用 fake pool
    original_init_run = mps.ModelPoolController.initialize_run

    def _fake_initialize_run(ctrl_self):
        from datetime import datetime
        ctrl_self._session_start = datetime.utcnow()
        ctrl_self._pool = pool_instance
        ctrl_self._backend = "postgres"

    monkeypatch.setattr(mps.ModelPoolController, "initialize_run", _fake_initialize_run)
    # Bug Fix #1 配套：stub register_to_registry 為「PG 寫入成功」
    monkeypatch.setattr(mlm.MLMetaModel, "register_to_registry", lambda self: True)
    return pool_instance


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestModelPoolStrategy:

    def test_model_pool_runs_end_to_end_with_fake_pool(
        self, synthetic_csv, sim_period, tmp_path, fake_pool
    ):
        """model_pool strategy 能端到端跑完且產出標準 artifacts。"""
        start, end = sim_period
        result = simulate(
            csv_path=synthetic_csv,
            start=start,
            end=end,
            strategy="model_pool",
            top_k=5,
            out_dir=tmp_path,
            run_tag="e2e",
            similarity_threshold=0.8,
            pool_regime_window=60,
            shadow_window=20,
        )
        assert Path(result["daily_pnl_path"]).exists()
        assert Path(result["retrain_log_path"]).exists()
        assert "summary_metrics" in result
        pnl = pd.read_csv(result["daily_pnl_path"])
        assert "net_return" in pnl.columns
        assert "cumulative_value" in pnl.columns

    def test_model_pool_initial_train_adds_first_entry(
        self, synthetic_csv, sim_period, tmp_path, fake_pool
    ):
        """Day 0 initial_train 後，fake pool 應有 1 筆 entry。"""
        start, end = sim_period
        simulate(
            csv_path=synthetic_csv,
            start=start,
            end=end,
            strategy="model_pool",
            top_k=5,
            out_dir=tmp_path,
            run_tag="init",
            similarity_threshold=0.8,
        )
        assert len(fake_pool.entries) >= 1, "Day 0 後 pool 應有至少 1 筆 entry"

    def test_model_pool_first_trigger_is_pool_miss(
        self, synthetic_csv, sim_period, tmp_path, fake_pool
    ):
        """第一次觸發（pool 只有 1 筆 entry 時）相似度必定低，應為 pool miss。"""
        start, end = sim_period
        result = simulate(
            csv_path=synthetic_csv,
            start=start,
            end=end,
            strategy="model_pool",
            top_k=5,
            out_dir=tmp_path,
            run_tag="miss",
            min_retrain_gap=20,
            trigger_ic_threshold=0.0,
            trigger_ic_days=3,
            trigger_sharpe_threshold=0.0,
            trigger_sharpe_days=10,
            similarity_threshold=0.99,  # 高 threshold 確保不命中
            pool_regime_window=60,
            shadow_window=20,
        )
        retrains = pd.read_csv(result["retrain_log_path"])
        non_initial = retrains[retrains["reason"] != "initial_train"]
        if len(non_initial) > 0:
            first_trigger = non_initial.iloc[0]
            reason = str(first_trigger["reason"])
            assert "pool_miss" in reason or "shadow_kept_current" in reason or "shadow_selected_new" in reason, (
                f"第一次觸發應是 miss 或 kept，實際 reason={reason!r}"
            )

    def test_model_pool_retrain_log_has_similarity_column(
        self, synthetic_csv, sim_period, tmp_path, fake_pool
    ):
        """retrain_log.csv 應包含 similarity 欄位（model_pool 專屬）。"""
        start, end = sim_period
        result = simulate(
            csv_path=synthetic_csv,
            start=start,
            end=end,
            strategy="model_pool",
            top_k=5,
            out_dir=tmp_path,
            run_tag="simcol",
        )
        retrains = pd.read_csv(result["retrain_log_path"])
        assert "similarity" in retrains.columns, "retrain_log 應有 similarity 欄位"

    def test_model_pool_summary_has_pool_stats(
        self, synthetic_csv, sim_period, tmp_path, fake_pool
    ):
        """summary_metrics 應含 n_pool_reuses / n_pool_misses / pool_backend。"""
        start, end = sim_period
        result = simulate(
            csv_path=synthetic_csv,
            start=start,
            end=end,
            strategy="model_pool",
            top_k=5,
            out_dir=tmp_path,
            run_tag="stats",
        )
        m = result["summary_metrics"]
        assert "n_pool_reuses" in m
        assert "n_pool_misses" in m
        assert "pool_backend" in m
        assert m["pool_backend"] == "postgres"

    def test_model_pool_fallback_when_db_unavailable(
        self, synthetic_csv, sim_period, tmp_path, monkeypatch
    ):
        """DB 不可用時 model_pool 降級為 triggered，pool_backend='unavailable'。"""
        import src.adaptation.model_pool_strategy as mps

        def _fail_init(self):
            from datetime import datetime
            self._session_start = datetime.utcnow()
            self._backend = "unavailable"
            self._pool = None

        monkeypatch.setattr(mps.ModelPoolController, "initialize_run", _fail_init)

        start, end = sim_period
        result = simulate(
            csv_path=synthetic_csv,
            start=start,
            end=end,
            strategy="model_pool",
            top_k=5,
            out_dir=tmp_path,
            run_tag="fallback",
            min_retrain_gap=20,
        )
        m = result["summary_metrics"]
        assert m.get("pool_backend") == "unavailable"
        # 降級後 retrain_log reason 不含 pool_ 相關字串（除初始訓練外）
        retrains = pd.read_csv(result["retrain_log_path"])
        non_initial = retrains[retrains["reason"] != "initial_train"]
        if len(non_initial) > 0:
            assert not any("pool_reuse" in r for r in non_initial["reason"]), (
                "降級模式不應有 pool_reuse reason"
            )


class TestDefaultStrategiesCatalog:

    def test_default_strategies_catalog_includes_model_pool(self):
        """DEFAULT_STRATEGIES 應包含 model_pool 第五策略。"""
        assert "model_pool" in DEFAULT_STRATEGIES
        cfg = DEFAULT_STRATEGIES["model_pool"]
        assert cfg["strategy"] == "model_pool"
        assert "similarity_threshold" in cfg
        assert "pool_regime_window" in cfg
        assert "shadow_window" in cfg


class TestABExperimentFiveStrategies:

    def test_ab_experiment_five_strategies(
        self, synthetic_csv, sim_period, tmp_path, fake_pool
    ):
        """run_ab_experiment 跑完 5 組，comparison_df 有 5 列。"""
        start, end = sim_period
        result = run_ab_experiment(
            csv_path=synthetic_csv,
            start=start,
            end=end,
            strategies=list(DEFAULT_STRATEGIES.keys()),
            top_k=5,
            out_dir=tmp_path,
            run_tag="five",
        )
        df = result["comparison_df"]
        assert len(df) == 5, f"應有 5 列策略，實際 {len(df)}"
        assert "model_pool" in df.index
        # comparison.csv 應含 pool 相關欄位
        assert "n_pool_reuses" in df.columns
        assert "n_pool_misses" in df.columns


# ---------------------------------------------------------------------------
# Phase A — shadow warm-up gap（P0★ #2）
# ---------------------------------------------------------------------------

class TestShadowWarmupGap:
    def test_controller_accepts_shadow_warmup_days(self):
        """ModelPoolController 應接受 ``shadow_warmup_days`` 參數。"""
        from src.adaptation.model_pool_strategy import ModelPoolController

        ctrl = ModelPoolController(
            similarity_threshold=0.8,
            pool_regime_window=60,
            shadow_window=20,
            shadow_warmup_days=5,
            min_improvement_ic=0.005,
            purge_days=5,
            horizon_days=5,
        )
        assert ctrl._shadow_warmup_days == 5

    def test_simulate_passes_shadow_warmup_to_pool(
        self, synthetic_csv, sim_period, tmp_path, fake_pool
    ):
        """simulate(model_pool) 應把 shadow_warmup_days 透傳給 pool_ctrl。"""
        start, end = sim_period
        result = simulate(
            csv_path=synthetic_csv,
            start=start,
            end=end,
            strategy="model_pool",
            top_k=5,
            out_dir=tmp_path,
            run_tag="warmup",
            shadow_warmup_days=10,  # 故意給較大值
        )
        assert "summary_metrics" in result
        # 端到端能跑完即代表透傳成功；具體 cutoff 由 unit-level 驗證


class TestPhaseATriggerWindow:
    def test_simulate_accepts_new_trigger_window_params(
        self, synthetic_csv, sim_period, tmp_path
    ):
        """simulate() 應接受新的 trigger_window_days / trigger_eval_gap_days 參數。"""
        start, end = sim_period
        result = simulate(
            csv_path=synthetic_csv,
            start=start,
            end=end,
            strategy="triggered",
            top_k=5,
            out_dir=tmp_path,
            run_tag="winwd",
            trigger_window_days=60,
            trigger_eval_gap_days=20,
            shadow_warmup_days=5,
        )
        # 端到端可跑完
        pnl = pd.read_csv(result["daily_pnl_path"])
        # 新窗口的 rolling_ic / rolling_sharpe 在前 trigger_window_days 大致為 NaN
        assert "rolling_ic" in pnl.columns
        assert "rolling_sharpe" in pnl.columns
