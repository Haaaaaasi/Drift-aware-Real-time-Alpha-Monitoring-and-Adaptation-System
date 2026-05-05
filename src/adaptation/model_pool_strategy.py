"""Layer 10 — ModelPoolController：協調 RecurringConceptPool + ShadowEvaluator 的 walk-forward 策略控制器。

在 simulate_recent.py 的 model_pool 策略分支中使用：
1. 每次觸發時，一律訓練新候選模型。
2. 同時計算當前 regime fingerprint 並搜尋 pool（只考慮本次 run 內的 entries）。
3. 若 pool 有相似 regime，把 pool 候選也加入 shadow evaluation。
4. ShadowEvaluator 在最近成熟窗口上比較 current / retrained / reused，選出最佳者。
5. 若 PostgreSQL 不可用，降級為 triggered（只重訓，不搜 pool）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from src.adaptation.shadow_evaluator import ShadowEvaluator
from src.common.logging import get_logger
from src.meta_signal.ml_meta_model import MLMetaModel

logger = get_logger(__name__)


@dataclass
class PoolDecision:
    """decide_on_trigger() 的回傳結構。"""

    best_model: MLMetaModel
    best_model_id: str
    reason: str
    similarity: float
    train_info: dict[str, Any]
    candidates_evaluated: list[str] = field(default_factory=list)


class ModelPoolController:
    """在 walk-forward simulation 中協調 RecurringConceptPool + ShadowEvaluator。

    設計原則：
    - MLMetaModel 無 disk 序列化，model instance 用 process-local dict 維護。
    - RecurringConceptPool（PostgreSQL）只儲存 fingerprint + model_id metadata。
    - `since` 參數確保只搜尋本次 simulation run 內的 pool entries。
    - 若 DB 連線失敗，`_backend` 設為 'unavailable'，退化為 triggered 行為。
    """

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        pool_regime_window: int = 60,
        shadow_window: int = 20,
        shadow_warmup_days: int = 5,
        min_improvement_ic: float = 0.005,
        purge_days: int = 5,
        horizon_days: int = 5,
        top_k_candidates: int = 3,
    ) -> None:
        # ``similarity_threshold`` 預設 0.5 對應 Phase B-1 之後的
        # ``score = exp(-d_zscored / 2)``（約 d ≤ 1.4 std）。舊 raw cosine 默認 0.8
        # 在新公式下相當於 d ≤ 0.45（< 0.5 std），過嚴。
        # ``top_k_candidates``：Phase B-3，shadow 階段最多納入幾個 reused 候選；
        # 1 = 與 Phase B-1 相容；2-3 = ensemble 比較讓 evaluator 自選。
        self._threshold = similarity_threshold
        self._regime_window = pool_regime_window
        self._shadow_window = shadow_window
        self._shadow_warmup_days = shadow_warmup_days
        self._min_improvement = min_improvement_ic
        self._purge_days = purge_days
        self._horizon_days = horizon_days
        self._top_k = max(1, int(top_k_candidates))

        self._models_by_id: dict[str, MLMetaModel] = {}
        self._session_start: datetime | None = None
        self._backend: str = "unavailable"
        self._pool = None
        self._shadow = ShadowEvaluator(
            min_improvement_ic=min_improvement_ic,
            min_evaluation_days=5,
        )

        # 統計
        self.n_pool_reuses: int = 0
        self.n_pool_misses: int = 0

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def initialize_run(self) -> None:
        """建立 DB 連線並記錄 session start，供 find_similar_regime 過濾本次 entries。"""
        self._session_start = datetime.utcnow()
        try:
            from src.adaptation.recurring_concept import RecurringConceptPool
            self._pool = RecurringConceptPool(similarity_threshold=self._threshold)
            self._backend = "postgres"
            logger.info("model_pool_backend_ready", since=self._session_start.isoformat())
        except Exception as exc:
            logger.warning("model_pool_backend_unavailable", error=str(exc))
            self._backend = "unavailable"
            self._pool = None

    # ------------------------------------------------------------------
    # 初始訓練（day 0）
    # ------------------------------------------------------------------

    def register_initial(
        self,
        model: MLMetaModel,
        bars_window: pd.DataFrame,
        train_info: dict[str, Any],
        alpha_ic_stats: dict[str, float] | None = None,
    ) -> str:
        """Day 0 模型：直接加入 local dict + pool（無 shadow），回傳 model_id。

        會先把模型寫進 ``model_registry`` 滿足 ``regime_pool.associated_model_id`` 的 FK，
        再寫入 ``regime_pool``。任何環節失敗都升級為 ERROR log，避免 silent failure。

        ``alpha_ic_stats`` 為 Phase B-2 alpha-side fingerprint 維度（caller 預先以
        最近成熟標籤算好），無提供時 alpha-side 三維 fallback 0。
        """
        model_id = train_info["model_id"]
        self._models_by_id[model_id] = model

        if self._backend == "postgres" and self._pool is not None:
            try:
                # FK 前置：必須先在 model_registry 留紀錄，否則 add_to_pool 會被 FK 擋下
                if not model.register_to_registry():
                    logger.error("model_pool_initial_registry_write_failed", model_id=model_id)
                    return model_id
                fp = self._pool.compute_regime_fingerprint(bars_window, alpha_ic_stats=alpha_ic_stats)
                alpha_weights = train_info.get("feature_importance", {})
                perf = train_info.get("holdout_metrics", {})
                self._pool.add_to_pool(fp, model_id, alpha_weights, perf)
                logger.info("model_pool_initial_registered", model_id=model_id)
            except Exception as exc:
                logger.error("model_pool_initial_register_failed", model_id=model_id, error=str(exc))

        return model_id

    # ------------------------------------------------------------------
    # 觸發時的 3-way shadow 決策
    # ------------------------------------------------------------------

    def decide_on_trigger(
        self,
        *,
        t: pd.Timestamp,
        current_model: MLMetaModel | None,
        current_model_id: str | None,
        bars: pd.DataFrame,
        alpha_panel: pd.DataFrame,
        fwd_returns: pd.Series,
        train_panel: pd.DataFrame,
        train_labels: pd.Series,
        eff_alphas: list[str] | None,
        alpha_ic_stats: dict[str, float] | None = None,
    ) -> PoolDecision:
        """執行 shadow 3-way compare，回傳最佳 PoolDecision。

        無論 pool 狀態如何，都會訓練新候選。差異只在是否加入 pool 候選。
        若 DB 不可用，直接採用 triggered 邏輯（新候選 vs current）。
        """
        # Step 1: 計算 shadow 評估窗口（成熟標籤）
        shadow_cutoff_end = t - pd.Timedelta(days=self._purge_days + self._horizon_days)
        shadow_cutoff_start = shadow_cutoff_end - pd.Timedelta(days=self._shadow_window)
        # Shadow 候選的訓練 cutoff 額外往前推 shadow_warmup_days，
        # 讓新候選不用 shadow window 之內的資料訓練（避免 IS leakage）。
        shadow_train_cutoff = shadow_cutoff_start - pd.Timedelta(days=self._shadow_warmup_days)
        shadow_train_panel = train_panel[train_panel["tradetime"] <= shadow_train_cutoff]
        shadow_train_labels = train_labels[
            train_labels.index.get_level_values("tradetime") <= shadow_train_cutoff
        ]

        # Step 2: 訓練 shadow 候選（用 stricter cutoff，僅供 shadow scoring）
        # 若資料不足則退化為「直接訓練、無 warm-up gap」，避免 simulate 卡住
        if len(shadow_train_labels) >= 100:
            shadow_model = MLMetaModel(feature_columns=eff_alphas)
            shadow_train_info = shadow_model.train(shadow_train_panel, shadow_train_labels)
            shadow_new_id = shadow_train_info["model_id"]
        else:
            shadow_model = MLMetaModel(feature_columns=eff_alphas)
            shadow_train_info = shadow_model.train(train_panel, train_labels)
            shadow_new_id = shadow_train_info["model_id"]
            logger.warning(
                "shadow_warmup_data_insufficient",
                t=str(t.date()),
                n_train=len(shadow_train_labels),
                fallback="full_train_panel",
            )

        # Step 3: 建立 shadow 候選集合（current vs shadow_new vs reused）
        shadow_panel = alpha_panel[
            (alpha_panel["tradetime"] > shadow_cutoff_start)
            & (alpha_panel["tradetime"] <= shadow_cutoff_end)
        ]
        shadow_fwd = fwd_returns[
            fwd_returns.index.get_level_values("tradetime").map(
                lambda ts: shadow_cutoff_start < ts <= shadow_cutoff_end
            )
        ]

        candidates: dict[str, pd.DataFrame] = {}
        if current_model is not None and current_model_id is not None and not shadow_panel.empty:
            try:
                cur_signals = current_model.predict(shadow_panel)
                candidates[current_model_id] = cur_signals.rename(
                    columns={"tradetime": "tradetime"}
                )
            except Exception:
                pass

        if not shadow_panel.empty:
            try:
                new_signals = shadow_model.predict(shadow_panel)
                candidates[shadow_new_id] = new_signals
            except Exception:
                pass

        # Step 4: 若 pool 可用，搜尋相似 regime 並加入 top-k 候選（Phase B-3）
        # ``regime_by_model`` 記錄 model_id → regime_id，供 shadow eval 後寫回 last_evaluated_ic
        regime_by_model: dict[str, str] = {}
        similarity: float = 0.0  # top-1 score（給日誌與 retrain_log 使用）

        if self._backend == "postgres" and self._pool is not None:
            try:
                bars_window_start = t - pd.Timedelta(days=self._regime_window + self._purge_days)
                bars_win = bars[
                    (bars["tradetime"] > bars_window_start)
                    & (bars["tradetime"] <= t - pd.Timedelta(days=self._purge_days))
                ]
                top_candidates: list[tuple[str, float]] = []
                if not bars_win.empty:
                    current_fp = self._pool.compute_regime_fingerprint(
                        bars_win, alpha_ic_stats=alpha_ic_stats
                    )
                    # Phase B-3 診斷：永遠拿到 best_seen_score（即便低於 threshold），
                    # 供 retrain_log / log 顯示真實的 similarity 分布
                    raw_result = self._pool.find_similar_regimes(
                        current_fp, since=self._session_start,
                        top_k=self._top_k, return_best_seen=True,
                    )
                    if isinstance(raw_result, tuple) and len(raw_result) == 2 and isinstance(raw_result[0], list):
                        top_candidates, best_seen = raw_result
                    else:
                        # fallback 給未升級的 fake pool（integration test）
                        top_candidates = raw_result if isinstance(raw_result, list) else []
                        best_seen = top_candidates[0][1] if top_candidates else 0.0
                    similarity = best_seen if best_seen > 0 else (
                        top_candidates[0][1] if top_candidates else 0.0
                    )

                # 把每個過 threshold 的 reused 候選都丟進 candidates dict
                for regime_id, _score in top_candidates:
                    regime_row = self._pool.get_regime_model(regime_id)
                    if regime_row is None:
                        continue
                    candidate_mid = regime_row.get("associated_model_id")
                    if not candidate_mid or candidate_mid in candidates:
                        # 已加過（不重複）或缺 model_id
                        continue
                    candidate_model = self._models_by_id.get(candidate_mid)
                    if candidate_model is None or shadow_panel.empty:
                        continue
                    try:
                        candidates[candidate_mid] = candidate_model.predict(shadow_panel)
                        regime_by_model[candidate_mid] = regime_id
                    except Exception:
                        # 單一候選 predict 失敗不影響其他 top-k
                        continue
            except Exception as exc:
                logger.warning("model_pool_search_failed", error=str(exc))
                regime_by_model = {}

        # Step 5: shadow evaluation
        eval_results: dict[str, dict] = {}
        if candidates and not shadow_fwd.empty:
            eval_results = self._shadow.evaluate_candidates(candidates, shadow_fwd)

        # Phase B-3：對所有被評估的 reused 候選，寫回 last_evaluated_ic
        # 不論最終誰被選中，都更新（quality feedback 是持續累積的訊號）
        if eval_results and self._pool is not None:
            for cand_mid, regime_id in regime_by_model.items():
                metrics = eval_results.get(cand_mid)
                if not metrics:
                    continue
                ic_val = metrics.get("ic")
                if ic_val is None:
                    continue
                try:
                    self._pool.update_last_evaluated_ic(regime_id, float(ic_val))
                except Exception as exc:
                    logger.warning(
                        "model_pool_update_last_eval_failed",
                        regime_id=regime_id,
                        model_id=cand_mid,
                        error=str(exc),
                    )

        best_id = self._shadow.select_best(eval_results, current_model_id=current_model_id)

        # Step 6: 依 best_id 決定結果並更新 pool
        pool_hit = bool(regime_by_model)
        live_train_info = shadow_train_info

        if best_id is None or best_id == current_model_id:
            best_model = current_model if current_model is not None else shadow_model
            best_model_id = current_model_id if current_model is not None else shadow_new_id
            reason = "shadow_kept_current"
            # shadow_model 永遠存進 _models_by_id（避免後續 _try_add_to_pool 找不到）
            self._models_by_id.setdefault(shadow_new_id, shadow_model)
            if not pool_hit:
                self.n_pool_misses += 1
                self._try_add_to_pool(bars, t, shadow_new_id, shadow_train_info, alpha_ic_stats=alpha_ic_stats)
        elif best_id in regime_by_model:
            # Shadow 在多個 reused 候選中選中其一
            best_model = self._models_by_id[best_id]
            best_model_id = best_id
            reason = f"shadow_selected_reused_sim_{similarity:.3f}"
            self.n_pool_reuses += 1
            try:
                self._pool.record_reuse(regime_by_model[best_id])
            except Exception:
                pass
        else:
            # best_id == shadow_new_id：shadow 階段認可新模型，重訓 live 版（用完整 train_panel）
            live_model = MLMetaModel(feature_columns=eff_alphas)
            live_train_info = live_model.train(train_panel, train_labels)
            live_new_id = live_train_info["model_id"]
            self._models_by_id[live_new_id] = live_model
            best_model = live_model
            best_model_id = live_new_id
            hit_label = "hit" if pool_hit else "miss"
            reason = f"shadow_selected_new_pool_{hit_label}_sim_{similarity:.3f}"
            if not pool_hit:
                self.n_pool_misses += 1
                self._try_add_to_pool(bars, t, live_new_id, live_train_info, alpha_ic_stats=alpha_ic_stats)

        logger.info(
            "model_pool_decision",
            t=str(t.date()),
            best_id=best_model_id,
            reason=reason,
            pool_hit=pool_hit,
            similarity=round(similarity, 4),
            n_reused_candidates=len(regime_by_model),
            candidates=list(candidates.keys()),
        )

        return PoolDecision(
            best_model=best_model,
            best_model_id=best_model_id,
            reason=reason,
            similarity=similarity,
            train_info=live_train_info,
            candidates_evaluated=list(eval_results.keys()),
        )

    # ------------------------------------------------------------------
    # 內部 helpers
    # ------------------------------------------------------------------

    def _try_add_to_pool(
        self,
        bars: pd.DataFrame,
        t: pd.Timestamp,
        model_id: str,
        train_info: dict[str, Any],
        alpha_ic_stats: dict[str, float] | None = None,
    ) -> None:
        """嘗試將新模型對應的 regime 加入 pool。

        FK 前置：``regime_pool.associated_model_id`` 對 ``model_registry(model_id)`` 有 FK，
        所以必須先呼叫 ``MLMetaModel.register_to_registry()`` 把模型寫進 ``model_registry``，
        再寫 ``regime_pool``。失敗一律升級為 ERROR log，避免 silent failure。
        """
        if self._backend != "postgres" or self._pool is None:
            return
        try:
            model = self._models_by_id.get(model_id)
            if model is None:
                logger.error(
                    "model_pool_add_missing_instance",
                    model_id=model_id,
                    msg="model not in _models_by_id; cannot register to model_registry",
                )
                return
            if not model.register_to_registry():
                logger.error("model_pool_add_registry_write_failed", model_id=model_id)
                return
            bars_window_start = t - pd.Timedelta(days=self._regime_window + self._purge_days)
            bars_win = bars[
                (bars["tradetime"] > bars_window_start)
                & (bars["tradetime"] <= t - pd.Timedelta(days=self._purge_days))
            ]
            if bars_win.empty:
                return
            fp = self._pool.compute_regime_fingerprint(bars_win, alpha_ic_stats=alpha_ic_stats)
            self._pool.add_to_pool(
                fp,
                model_id,
                train_info.get("feature_importance", {}),
                train_info.get("holdout_metrics", {}),
            )
        except Exception as exc:
            logger.error("model_pool_add_failed", model_id=model_id, error=str(exc))
