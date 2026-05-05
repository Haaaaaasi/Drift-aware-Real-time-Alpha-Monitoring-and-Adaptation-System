"""Unit tests for ModelPoolController FK ordering（regime_pool ↔ model_registry）。

針對「``regime_pool.associated_model_id`` FK 約束導致 ``add_to_pool`` 永遠失敗」
這個 silent bug 加保護。修復後的不變式：

* 任何 ``add_to_pool`` 呼叫之前，``MLMetaModel.register_to_registry()`` 必須先成功
* 若 model 不在 ``_models_by_id`` 或 register 失敗，``add_to_pool`` **不會被呼叫**
* 失敗一律走 ``logger.error``（不再 silent warning）
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.adaptation.model_pool_strategy import ModelPoolController


def _stub_pool(returns_fp: dict | None = None) -> MagicMock:
    """產生 ``RecurringConceptPool`` 介面 stub。"""
    p = MagicMock()
    p.compute_regime_fingerprint.return_value = returns_fp or {
        "volatility": 0.01,
        "autocorrelation": 0.0,
        "avg_cross_correlation": 0.1,
        "trend_strength": 0.0,
        "volume_ratio": 1.0,
    }
    p.add_to_pool.return_value = "regime_stub_001"
    return p


def _stub_model(register_returns: bool = True) -> MagicMock:
    """產生 ``MLMetaModel`` stub，``register_to_registry`` 回傳指定布林。"""
    m = MagicMock()
    m.register_to_registry.return_value = register_returns
    return m


def _bars(t: pd.Timestamp, n_days: int = 80) -> pd.DataFrame:
    """產一段橫跨 t 前 n_days 的合成 bars，給 fingerprint 計算使用。"""
    dates = pd.bdate_range(end=t, periods=n_days, freq="B")
    return pd.DataFrame([
        {"security_id": "A", "tradetime": d, "close": 100.0 + i * 0.1, "vol": 1e6}
        for i, d in enumerate(dates)
    ])


def _train_info(model_id: str = "ml_xgb_test") -> dict:
    return {
        "model_id": model_id,
        "holdout_metrics": {"ic": 0.05, "rank_ic": 0.02},
        "feature_importance": {"wq001": 0.3, "wq014": 0.2},
        "n_train": 1000,
        "n_features": 2,
    }


class TestRegisterInitialOrdering:
    """``register_initial`` 必須先 ``register_to_registry`` 再 ``add_to_pool``。"""

    def test_register_initial_calls_registry_before_pool(self) -> None:
        ctrl = ModelPoolController()
        ctrl._session_start = datetime.utcnow()
        ctrl._backend = "postgres"
        ctrl._pool = _stub_pool()

        model = _stub_model(register_returns=True)
        info = _train_info("ml_xgb_init01")
        bars = _bars(pd.Timestamp("2024-06-30"))

        returned = ctrl.register_initial(model, bars, info)

        assert returned == "ml_xgb_init01"
        # register_to_registry 必須被呼叫且 add_to_pool 也要被呼叫
        model.register_to_registry.assert_called_once()
        ctrl._pool.add_to_pool.assert_called_once()
        # 而且 model 必須進 _models_by_id 給後續查詢
        assert "ml_xgb_init01" in ctrl._models_by_id

    def test_register_initial_skips_pool_when_registry_write_fails(self) -> None:
        """若 register_to_registry 回 False（如 PG 連線中斷），不應再呼叫 add_to_pool
        （否則會被 FK 擋下、留下 dirty state）。
        """
        ctrl = ModelPoolController()
        ctrl._session_start = datetime.utcnow()
        ctrl._backend = "postgres"
        ctrl._pool = _stub_pool()

        model = _stub_model(register_returns=False)  # registry 寫入失敗
        info = _train_info("ml_xgb_init02")
        bars = _bars(pd.Timestamp("2024-06-30"))

        ctrl.register_initial(model, bars, info)

        model.register_to_registry.assert_called_once()
        ctrl._pool.add_to_pool.assert_not_called()

    def test_register_initial_noop_when_backend_unavailable(self) -> None:
        ctrl = ModelPoolController()
        ctrl._session_start = datetime.utcnow()
        ctrl._backend = "unavailable"
        ctrl._pool = None

        model = _stub_model(register_returns=True)
        info = _train_info("ml_xgb_init03")
        bars = _bars(pd.Timestamp("2024-06-30"))

        ctrl.register_initial(model, bars, info)

        # 後端不可用時連 register_to_registry 都不該呼叫（避免 unnecessary DB writes）
        model.register_to_registry.assert_not_called()


class TestTryAddToPoolOrdering:
    """``_try_add_to_pool`` 的 FK / lookup 守則。"""

    def test_try_add_to_pool_calls_registry_before_pool(self) -> None:
        ctrl = ModelPoolController()
        ctrl._session_start = datetime.utcnow()
        ctrl._backend = "postgres"
        ctrl._pool = _stub_pool()

        model = _stub_model(register_returns=True)
        ctrl._models_by_id["ml_xgb_add01"] = model
        bars = _bars(pd.Timestamp("2024-06-30"))

        ctrl._try_add_to_pool(bars, pd.Timestamp("2024-06-30"), "ml_xgb_add01", _train_info("ml_xgb_add01"))

        model.register_to_registry.assert_called_once()
        ctrl._pool.add_to_pool.assert_called_once()

    def test_try_add_to_pool_skips_when_model_not_in_dict(self) -> None:
        """若 model_id 不在 ``_models_by_id``，_try_add_to_pool 應 ERROR log 並 return，
        不該呼叫 add_to_pool（沒有模型物件可註冊 → 一定會踩 FK）。
        """
        ctrl = ModelPoolController()
        ctrl._session_start = datetime.utcnow()
        ctrl._backend = "postgres"
        ctrl._pool = _stub_pool()
        # _models_by_id 故意保持空

        bars = _bars(pd.Timestamp("2024-06-30"))
        ctrl._try_add_to_pool(bars, pd.Timestamp("2024-06-30"), "ml_xgb_missing", _train_info("ml_xgb_missing"))

        ctrl._pool.add_to_pool.assert_not_called()

    def test_try_add_to_pool_skips_when_registry_write_fails(self) -> None:
        ctrl = ModelPoolController()
        ctrl._session_start = datetime.utcnow()
        ctrl._backend = "postgres"
        ctrl._pool = _stub_pool()

        model = _stub_model(register_returns=False)
        ctrl._models_by_id["ml_xgb_failed"] = model
        bars = _bars(pd.Timestamp("2024-06-30"))

        ctrl._try_add_to_pool(bars, pd.Timestamp("2024-06-30"), "ml_xgb_failed", _train_info("ml_xgb_failed"))

        model.register_to_registry.assert_called_once()
        ctrl._pool.add_to_pool.assert_not_called()

    def test_try_add_to_pool_noop_when_backend_unavailable(self) -> None:
        ctrl = ModelPoolController()
        ctrl._session_start = datetime.utcnow()
        ctrl._backend = "unavailable"
        ctrl._pool = None

        model = _stub_model(register_returns=True)
        ctrl._models_by_id["ml_xgb_x"] = model
        bars = _bars(pd.Timestamp("2024-06-30"))

        ctrl._try_add_to_pool(bars, pd.Timestamp("2024-06-30"), "ml_xgb_x", _train_info("ml_xgb_x"))

        # 後端不可用時整段 short-circuit
        model.register_to_registry.assert_not_called()
