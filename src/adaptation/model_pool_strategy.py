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

import numpy as np
import pandas as pd

from src.adaptation.recurring_concept import (
    _compute_pool_scales,
    _distance_to_similarity,
    _performance_gate,
    _staleness_factor,
    _standardized_distance,
)
from src.adaptation.shadow_evaluator import ShadowEvaluator
from src.common.logging import get_logger
from src.meta_signal.ml_meta_model import MLMetaModel

logger = get_logger(__name__)


def _finite_float(value: Any) -> float | None:
    """把 shadow/proxy metric 安全轉成有限浮點數。"""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if np.isfinite(score) else None


def _rank_desc(scores: dict[str, Any]) -> dict[str, int]:
    """由大到小排序，略過缺值；用於 failure audit 的候選名次。"""
    valid_scores = [
        (candidate_id, score)
        for candidate_id, value in scores.items()
        if (score := _finite_float(value)) is not None
    ]
    valid_scores.sort(key=lambda item: item[1], reverse=True)
    return {candidate_id: rank for rank, (candidate_id, _) in enumerate(valid_scores, start=1)}


@dataclass
class PoolDecision:
    """decide_on_trigger() 的回傳結構。"""

    best_model: MLMetaModel
    best_model_id: str
    reason: str
    similarity: float
    train_info: dict[str, Any]
    candidates_evaluated: list[str] = field(default_factory=list)
    candidate_models: dict[str, MLMetaModel] = field(default_factory=dict, repr=False)
    diagnostic_records: list[dict[str, Any]] = field(default_factory=list)


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
        selection_metric: str = "ic",
        shadow_proxy_top_k: int | None = None,
        reuse_min_score: float | None = None,
        reuse_margin: float = 0.0,
        commission_rate: float = 0.0,
        tax_rate: float = 0.0,
        slippage_bps: float = 0.0,
        round_trip_cost_pct: float | None = None,
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
        self._selection_metric = selection_metric
        self._pool_min_rank_ic = (
            0.0 if selection_metric in {"ic", "rank_ic"} else float("-inf")
        )
        self._shadow_proxy_top_k = shadow_proxy_top_k
        self._reuse_min_score = reuse_min_score
        self._reuse_margin = float(reuse_margin)
        self._commission_rate = commission_rate
        self._tax_rate = tax_rate
        self._slippage_bps = slippage_bps
        self._round_trip_cost_pct = round_trip_cost_pct

        self._models_by_id: dict[str, MLMetaModel] = {}
        self._session_start: datetime | None = None
        self._backend: str = "unavailable"
        self._pool = None
        self._local_entries: list[dict[str, Any]] = []
        self._local_entry_seq: int = 0
        self._shadow = ShadowEvaluator(
            min_improvement_ic=min_improvement_ic,
            min_evaluation_days=5,
        )

        # 統計
        self.n_pool_reuses: int = 0
        self.n_pool_misses: int = 0
        self.decision_records: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def initialize_run(self) -> None:
        """建立 DB 連線並記錄 session start，供 find_similar_regime 過濾本次 entries。"""
        self._session_start = datetime.utcnow()
        try:
            from src.adaptation.recurring_concept import RecurringConceptPool
            self._pool = RecurringConceptPool(
                similarity_threshold=self._threshold,
                min_rank_ic=self._pool_min_rank_ic,
            )
            self._backend = "postgres"
            logger.info("model_pool_backend_ready", since=self._session_start.isoformat())
        except Exception as exc:
            logger.warning("model_pool_backend_unavailable", error=str(exc))
            self._backend = "local_fallback"
            self._pool = None

    def _add_local_entry(
        self,
        *,
        fingerprint: dict[str, float],
        model_id: str,
        train_info: dict[str, Any],
        detected_at: pd.Timestamp | datetime | None,
    ) -> str:
        """把模型加入 process-local pool，供 PostgreSQL 不可用時的同 run reuse。"""
        if model_id not in self._models_by_id:
            return ""
        for entry in self._local_entries:
            if entry.get("associated_model_id") == model_id:
                return str(entry["regime_id"])
        self._local_entry_seq += 1
        regime_id = f"local_regime_{self._local_entry_seq:04d}"
        ts = pd.Timestamp(detected_at or datetime.utcnow()).to_pydatetime()
        self._local_entries.append(
            {
                "regime_id": regime_id,
                "fingerprint": fingerprint,
                "associated_model_id": model_id,
                "detected_at": ts,
                "performance_summary": train_info.get("holdout_metrics", {}),
                "reuse_count": 0,
                "last_evaluated_ic": None,
            }
        )
        if self._backend in {"unavailable", "postgres"}:
            self._backend = "local_fallback" if self._backend == "unavailable" else "postgres_with_local_fallback"
        logger.info("model_pool_local_entry_added", model_id=model_id, regime_id=regime_id)
        return regime_id

    def _find_local_candidates(
        self,
        current_fp: dict[str, float],
        *,
        top_k: int,
        now: pd.Timestamp,
    ) -> tuple[list[tuple[str, float]], float]:
        """搜尋 process-local pool，回傳通過 threshold 的 top-k 與 best-seen score。"""
        if not self._local_entries:
            return [], 0.0
        keys = list(current_fp.keys())
        pool_df = pd.DataFrame(
            [
                {"fingerprint": entry["fingerprint"]}
                for entry in self._local_entries
            ]
        )
        scales = _compute_pool_scales(pool_df, keys)
        scored: list[tuple[str, float]] = []
        for entry in self._local_entries:
            d = _standardized_distance(current_fp, entry["fingerprint"], scales)
            score = (
                _distance_to_similarity(d)
                * _staleness_factor(entry["detected_at"], pd.Timestamp(now).to_pydatetime())
                * _performance_gate(
                    entry.get("performance_summary"),
                    min_rank_ic=self._pool_min_rank_ic,
                )
            )
            scored.append((entry["regime_id"], float(score)))
        scored.sort(key=lambda item: item[1], reverse=True)
        best_seen = scored[0][1] if scored else 0.0
        passed = [(rid, score) for rid, score in scored if score >= self._threshold][: self._top_k]
        return passed[:top_k], best_seen

    def _get_local_regime_model(self, regime_id: str) -> dict[str, Any] | None:
        for entry in self._local_entries:
            if entry["regime_id"] == regime_id:
                return entry
        return None

    def _record_local_reuse(self, regime_id: str) -> None:
        entry = self._get_local_regime_model(regime_id)
        if entry is not None:
            entry["reuse_count"] = int(entry.get("reuse_count", 0)) + 1

    def _update_local_last_evaluated_ic(self, regime_id: str, ic: float) -> None:
        entry = self._get_local_regime_model(regime_id)
        if entry is not None:
            entry["last_evaluated_ic"] = float(ic)

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

        fp: dict[str, float] | None = None
        if self._pool is not None:
            try:
                fp = self._pool.compute_regime_fingerprint(
                    bars_window, alpha_ic_stats=alpha_ic_stats
                )
            except Exception as exc:
                logger.warning(
                    "model_pool_initial_fingerprint_failed",
                    model_id=model_id,
                    error=str(exc),
                )
        detected_at = (
            bars_window["tradetime"].max()
            if "tradetime" in bars_window.columns and not bars_window.empty
            else datetime.utcnow()
        )
        if self._backend != "postgres" or self._pool is None:
            if fp is not None:
                self._add_local_entry(
                    fingerprint=fp,
                    model_id=model_id,
                    train_info=train_info,
                    detected_at=detected_at,
                )
            return model_id

        if self._backend == "postgres" and self._pool is not None:
            try:
                # FK 前置：必須先在 model_registry 留紀錄，否則 add_to_pool 會被 FK 擋下
                if not model.register_to_registry():
                    logger.error("model_pool_initial_registry_write_failed", model_id=model_id)
                    if fp is not None:
                        self._add_local_entry(
                            fingerprint=fp,
                            model_id=model_id,
                            train_info=train_info,
                            detected_at=detected_at,
                        )
                    return model_id
                fp = self._pool.compute_regime_fingerprint(bars_window, alpha_ic_stats=alpha_ic_stats)
                alpha_weights = train_info.get("feature_importance", {})
                perf = train_info.get("holdout_metrics", {})
                self._pool.add_to_pool(fp, model_id, alpha_weights, perf)
                logger.info("model_pool_initial_registered", model_id=model_id)
            except Exception as exc:
                logger.error("model_pool_initial_register_failed", model_id=model_id, error=str(exc))
                if fp is not None:
                    self._add_local_entry(
                        fingerprint=fp,
                        model_id=model_id,
                        train_info=train_info,
                        detected_at=detected_at,
                    )

        return model_id

    @staticmethod
    def _compute_shadow_cutoffs(
        *,
        bars: pd.DataFrame,
        t: pd.Timestamp,
        shadow_window: int,
        maturity_gap: int,
        warmup_days: int,
    ) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
        """用實際交易日序列計算 shadow window 與 shadow 訓練 cutoff。"""
        if bars.empty or "tradetime" not in bars.columns:
            end = pd.Timestamp(t) - pd.Timedelta(days=maturity_gap)
            start = end - pd.Timedelta(days=shadow_window)
            train = start - pd.Timedelta(days=warmup_days)
            return start.normalize(), end.normalize(), train.normalize()

        trading_days = pd.DatetimeIndex(
            sorted(set(pd.to_datetime(bars["tradetime"].dropna()).dt.normalize()))
        )
        if trading_days.empty:
            end = pd.Timestamp(t) - pd.Timedelta(days=maturity_gap)
            start = end - pd.Timedelta(days=shadow_window)
            train = start - pd.Timedelta(days=warmup_days)
            return start.normalize(), end.normalize(), train.normalize()

        t_norm = pd.Timestamp(t).normalize()
        pos = trading_days.searchsorted(t_norm, side="left")
        if pos >= len(trading_days) or trading_days[pos] != t_norm:
            pos = min(pos, len(trading_days) - 1)
            if trading_days[pos] > t_norm and pos > 0:
                pos -= 1

        end_pos = max(0, pos - max(0, int(maturity_gap)))
        start_pos = max(0, end_pos - max(0, int(shadow_window)))
        train_pos = max(0, start_pos - max(0, int(warmup_days)))
        return trading_days[start_pos], trading_days[end_pos], trading_days[train_pos]

    @staticmethod
    def _slice_shadow_forward_returns(
        *,
        fwd_returns: pd.Series,
        label_available_at: pd.Series | None,
        shadow_cutoff_start: pd.Timestamp,
        shadow_cutoff_end: pd.Timestamp,
        as_of: pd.Timestamp,
    ) -> pd.Series:
        """取出 shadow 評分用 forward returns，並要求 label 在 as_of 已成熟。"""
        if fwd_returns.empty:
            return fwd_returns

        signal_dates = pd.to_datetime(
            fwd_returns.index.get_level_values("tradetime")
        ).normalize()
        mask = pd.Series(
            (signal_dates > pd.Timestamp(shadow_cutoff_start).normalize())
            & (signal_dates <= pd.Timestamp(shadow_cutoff_end).normalize()),
            index=fwd_returns.index,
        )

        if label_available_at is not None and not label_available_at.empty:
            avail = pd.to_datetime(
                label_available_at.reindex(fwd_returns.index),
                errors="coerce",
            )
            mask &= avail <= pd.Timestamp(as_of)

        return fwd_returns[mask]

    def _apply_reuse_guard(
        self,
        *,
        best_id: str | None,
        eval_results: dict[str, dict[str, float]],
        regime_by_model: dict[str, str],
        current_model_id: str | None,
    ) -> tuple[str | None, dict[str, Any]]:
        """對 reused candidate 加保守 guard，避免 similarity hit 被微弱 shadow edge 放大。

        Guard 只在 shadow 原始最佳候選是 reused 時生效。若 reused 未達最低分數
        或沒有比 best non-reused candidate 多出指定 margin，改從 current/new 裡重選。
        """
        raw_best_id = best_id
        raw_best_score = (
            _finite_float(eval_results.get(best_id, {}).get(self._selection_metric))
            if best_id
            else None
        )
        raw_best_role = "reused" if best_id in regime_by_model else None

        non_reused_results = {
            mid: metrics
            for mid, metrics in eval_results.items()
            if mid not in regime_by_model
        }
        best_non_reused_id: str | None = None
        best_non_reused_score: float | None = None
        if non_reused_results:
            best_non_reused_id, best_non_reused_metrics = max(
                non_reused_results.items(),
                key=lambda item: (
                    _finite_float(item[1].get(self._selection_metric)) or float("-inf")
                ),
            )
            best_non_reused_score = _finite_float(
                best_non_reused_metrics.get(self._selection_metric)
            )

        info: dict[str, Any] = {
            "raw_best_candidate_model_id": raw_best_id,
            "raw_best_role": raw_best_role,
            "raw_best_score": raw_best_score,
            "best_non_reused_model_id": best_non_reused_id,
            "best_non_reused_score": best_non_reused_score,
            "reuse_guard_min_score": self._reuse_min_score,
            "reuse_guard_margin": self._reuse_margin,
            "reuse_guard_passed": None,
            "reuse_guard_reason": None,
        }

        if best_id is None or best_id not in regime_by_model:
            return best_id, info

        score = raw_best_score
        if score is None:
            info["reuse_guard_passed"] = False
            info["reuse_guard_reason"] = "missing_reuse_score"
        elif self._reuse_min_score is not None and score < float(self._reuse_min_score):
            info["reuse_guard_passed"] = False
            info["reuse_guard_reason"] = "below_min_score"
        elif (
            best_non_reused_score is not None
            and score - best_non_reused_score < self._reuse_margin
        ):
            info["reuse_guard_passed"] = False
            info["reuse_guard_reason"] = "below_non_reused_margin"
        else:
            info["reuse_guard_passed"] = True
            info["reuse_guard_reason"] = "passed"
            return best_id, info

        fallback_id = self._shadow.select_best(
            non_reused_results,
            current_model_id=current_model_id,
            metric=self._selection_metric,
        )
        logger.info(
            "model_pool_reuse_guard_rejected",
            raw_best_id=best_id,
            raw_best_score=raw_best_score,
            best_non_reused_id=best_non_reused_id,
            best_non_reused_score=best_non_reused_score,
            fallback_id=fallback_id,
            reason=info["reuse_guard_reason"],
        )
        return fallback_id, info

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
        label_available_at: pd.Series | None = None,
        alpha_ic_stats: dict[str, float] | None = None,
    ) -> PoolDecision:
        """執行 shadow 3-way compare，回傳最佳 PoolDecision。

        無論 pool 狀態如何，都會訓練新候選。差異只在是否加入 pool 候選。
        若 DB 不可用，直接採用 triggered 邏輯（新候選 vs current）。
        """
        # Step 1: 用實際交易日計算 shadow 評估窗口（成熟標籤）
        shadow_cutoff_start, shadow_cutoff_end, shadow_train_cutoff = (
            self._compute_shadow_cutoffs(
                bars=bars,
                t=t,
                shadow_window=self._shadow_window,
                maturity_gap=self._purge_days + self._horizon_days,
                warmup_days=self._shadow_warmup_days,
            )
        )
        # Shadow 候選的訓練 cutoff 額外往前推 shadow_warmup_days，
        # 讓新候選不用 shadow window 之內的資料訓練（避免 IS leakage）。
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
        shadow_fwd = self._slice_shadow_forward_returns(
            fwd_returns=fwd_returns,
            label_available_at=label_available_at,
            shadow_cutoff_start=shadow_cutoff_start,
            shadow_cutoff_end=shadow_cutoff_end,
            as_of=t,
        )

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
        similarity_by_model: dict[str, float] = {}
        bars_window_start = t - pd.Timedelta(days=self._regime_window + self._purge_days)
        bars_win = bars[
            (bars["tradetime"] > bars_window_start)
            & (bars["tradetime"] <= t - pd.Timedelta(days=self._purge_days))
        ]
        current_fp: dict[str, float] | None = None
        if self._pool is not None and not bars_win.empty:
            try:
                current_fp = self._pool.compute_regime_fingerprint(
                    bars_win, alpha_ic_stats=alpha_ic_stats
                )
            except Exception as exc:
                logger.warning("model_pool_fingerprint_failed", error=str(exc))
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
                        similarity_by_model[candidate_mid] = float(_score)
                    except Exception:
                        # 單一候選 predict 失敗不影響其他 top-k
                        continue
            except Exception as exc:
                logger.warning("model_pool_search_failed", error=str(exc))
                regime_by_model = {}

        if self._local_entries and current_fp is not None:
            top_candidates, best_seen = self._find_local_candidates(
                current_fp,
                top_k=self._top_k,
                now=t,
            )
            similarity = max(similarity, best_seen)
            for regime_id, _score in top_candidates:
                regime_row = self._get_local_regime_model(regime_id)
                if regime_row is None:
                    continue
                candidate_mid = regime_row.get("associated_model_id")
                if not candidate_mid or candidate_mid in candidates:
                    continue
                candidate_model = self._models_by_id.get(candidate_mid)
                if candidate_model is None or shadow_panel.empty:
                    continue
                try:
                    candidates[candidate_mid] = candidate_model.predict(shadow_panel)
                    regime_by_model[candidate_mid] = regime_id
                    similarity_by_model[candidate_mid] = float(_score)
                except Exception:
                    continue

        # Step 5: shadow evaluation
        eval_results: dict[str, dict] = {}
        if candidates and not shadow_fwd.empty:
            eval_results = self._shadow.evaluate_candidates(
                candidates,
                shadow_fwd,
                proxy_top_k=self._shadow_proxy_top_k,
                commission_rate=self._commission_rate,
                tax_rate=self._tax_rate,
                slippage_bps=self._slippage_bps,
                round_trip_cost_pct=self._round_trip_cost_pct,
            )

        # Phase B-3：對所有被評估的 reused 候選，寫回 last_evaluated_ic
        # 不論最終誰被選中，都更新（quality feedback 是持續累積的訊號）
        if eval_results:
            for cand_mid, regime_id in regime_by_model.items():
                metrics = eval_results.get(cand_mid)
                if not metrics:
                    continue
                ic_val = metrics.get("ic")
                if ic_val is None:
                    continue
                try:
                    if regime_id.startswith("local_regime_"):
                        self._update_local_last_evaluated_ic(regime_id, float(ic_val))
                    elif self._pool is not None:
                        self._pool.update_last_evaluated_ic(regime_id, float(ic_val))
                except Exception as exc:
                    logger.warning(
                        "model_pool_update_last_eval_failed",
                        regime_id=regime_id,
                        model_id=cand_mid,
                        error=str(exc),
                    )

        raw_best_id = self._shadow.select_best(
            eval_results,
            current_model_id=current_model_id,
            metric=self._selection_metric,
        )
        best_id, reuse_guard_info = self._apply_reuse_guard(
            best_id=raw_best_id,
            eval_results=eval_results,
            regime_by_model=regime_by_model,
            current_model_id=current_model_id,
        )

        # Step 6: 依 best_id 決定結果並更新 pool
        pool_hit = bool(regime_by_model)
        live_train_info = shadow_train_info

        selected_candidate_id = best_id if best_id is not None else current_model_id
        live_model_id: str | None = None

        if best_id is None or best_id == current_model_id:
            best_model = current_model if current_model is not None else shadow_model
            best_model_id = current_model_id if current_model is not None else shadow_new_id
            live_model_id = best_model_id
            reason = "shadow_kept_current"
            if reuse_guard_info.get("reuse_guard_passed") is False:
                reason = f"reuse_guard_rejected_keep_current_{reuse_guard_info.get('reuse_guard_reason')}"
            # shadow_model 永遠存進 _models_by_id（避免後續 _try_add_to_pool 找不到）
            self._models_by_id.setdefault(shadow_new_id, shadow_model)
            if not pool_hit:
                self.n_pool_misses += 1
                self._try_add_to_pool(bars, t, shadow_new_id, shadow_train_info, alpha_ic_stats=alpha_ic_stats)
        elif best_id in regime_by_model:
            # Shadow 在多個 reused 候選中選中其一
            best_model = self._models_by_id[best_id]
            best_model_id = best_id
            live_model_id = best_model_id
            reason = f"shadow_selected_reused_sim_{similarity:.3f}"
            self.n_pool_reuses += 1
            try:
                regime_id = regime_by_model[best_id]
                if regime_id.startswith("local_regime_"):
                    self._record_local_reuse(regime_id)
                elif self._pool is not None:
                    self._pool.record_reuse(regime_id)
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
            live_model_id = live_new_id
            hit_label = "hit" if pool_hit else "miss"
            if reuse_guard_info.get("reuse_guard_passed") is False:
                reason = (
                    f"reuse_guard_rejected_selected_new_{hit_label}_"
                    f"{reuse_guard_info.get('reuse_guard_reason')}_sim_{similarity:.3f}"
                )
            else:
                reason = f"shadow_selected_new_pool_{hit_label}_sim_{similarity:.3f}"
            if not pool_hit:
                self.n_pool_misses += 1
                self._try_add_to_pool(bars, t, live_new_id, live_train_info, alpha_ic_stats=alpha_ic_stats)

        candidate_models: dict[str, MLMetaModel] = {}
        if current_model is not None and current_model_id is not None and current_model_id in candidates:
            candidate_models[current_model_id] = current_model
        if shadow_new_id in candidates:
            candidate_models[shadow_new_id] = shadow_model
        for candidate_mid in regime_by_model:
            candidate_model = self._models_by_id.get(candidate_mid)
            if candidate_model is not None and candidate_mid in candidates:
                candidate_models[candidate_mid] = candidate_model

        def _candidate_role(candidate_id: str | None) -> str | None:
            if candidate_id is None:
                return None
            if candidate_id == current_model_id:
                return "current"
            if candidate_id == shadow_new_id:
                return "new"
            if candidate_id in regime_by_model:
                return "reused"
            return None

        selected_role = _candidate_role(selected_candidate_id)
        selected_similarity = similarity_by_model.get(selected_candidate_id) if selected_candidate_id else None
        selection_scores = {
            candidate_id: metrics.get(self._selection_metric)
            for candidate_id, metrics in eval_results.items()
        }
        selection_rank = _rank_desc(selection_scores)
        topk_net_rank = _rank_desc({
            candidate_id: metrics.get("topk_net_return")
            for candidate_id, metrics in eval_results.items()
        })
        diagnostic_records: list[dict[str, Any]] = []
        for candidate_id in candidates:
            metrics = eval_results.get(candidate_id, {})
            role = _candidate_role(candidate_id)
            if role is None:
                continue
            is_selected = candidate_id == selected_candidate_id
            candidate_score = _finite_float(metrics.get(self._selection_metric))
            best_non_reused_score = reuse_guard_info.get("best_non_reused_score")
            reuse_margin_vs_non_reused = (
                candidate_score - best_non_reused_score
                if role == "reused"
                and candidate_score is not None
                and best_non_reused_score is not None
                else None
            )
            diagnostic_records.append({
                "date": t.strftime("%Y-%m-%d"),
                "day_idx": None,
                "current_model_id": current_model_id,
                "shadow_new_model_id": shadow_new_id,
                "live_model_id": live_model_id,
                "selected_candidate_model_id": selected_candidate_id,
                "applied_model_id": best_model_id,
                "candidate_model_id": candidate_id,
                "candidate_role": role,
                "selected": bool(is_selected),
                "selected_role": selected_role,
                "decision_reason": reason,
                "pool_hit": bool(pool_hit),
                "candidate_similarity": similarity_by_model.get(candidate_id),
                "selected_similarity": selected_similarity,
                "best_seen_similarity": similarity if similarity else None,
                "n_reused_candidates": len(regime_by_model),
                "selection_metric": self._selection_metric,
                "selection_score": candidate_score,
                "shadow_rank_by_selection_metric": selection_rank.get(candidate_id),
                "shadow_rank_by_topk_net_return": topk_net_rank.get(candidate_id),
                "raw_best_candidate_model_id": reuse_guard_info.get("raw_best_candidate_model_id"),
                "raw_best_role": reuse_guard_info.get("raw_best_role"),
                "raw_best_score": reuse_guard_info.get("raw_best_score"),
                "best_non_reused_model_id": reuse_guard_info.get("best_non_reused_model_id"),
                "best_non_reused_score": best_non_reused_score,
                "reuse_score_margin_vs_best_non_reused": reuse_margin_vs_non_reused,
                "reuse_guard_min_score": reuse_guard_info.get("reuse_guard_min_score"),
                "reuse_guard_margin": reuse_guard_info.get("reuse_guard_margin"),
                "reuse_guard_passed": reuse_guard_info.get("reuse_guard_passed"),
                "reuse_guard_reason": reuse_guard_info.get("reuse_guard_reason"),
                "shadow_ic": metrics.get("ic"),
                "shadow_hit_rate": metrics.get("hit_rate"),
                "shadow_sharpe": metrics.get("sharpe"),
                "shadow_n_samples": metrics.get("n_samples"),
                "shadow_topk_gross_return": metrics.get("topk_gross_return"),
                "shadow_topk_net_return": metrics.get("topk_net_return"),
                "shadow_topk_turnover": metrics.get("topk_turnover"),
                "shadow_topk_n_days": metrics.get("topk_n_days"),
            })
        self.decision_records.extend(diagnostic_records)

        logger.info(
            "model_pool_decision",
            t=str(t.date()),
            best_id=best_model_id,
            reason=reason,
            pool_hit=pool_hit,
            similarity=round(similarity, 4),
            n_reused_candidates=len(regime_by_model),
            selection_metric=self._selection_metric,
            candidates=list(candidates.keys()),
        )

        return PoolDecision(
            best_model=best_model,
            best_model_id=best_model_id,
            reason=reason,
            similarity=similarity,
            train_info=live_train_info,
            candidates_evaluated=list(eval_results.keys()),
            candidate_models=candidate_models,
            diagnostic_records=diagnostic_records,
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
        model = self._models_by_id.get(model_id)
        if model is None:
            logger.error(
                "model_pool_add_missing_instance",
                model_id=model_id,
                msg="model not in _models_by_id; cannot register to model_registry",
            )
            return
        bars_window_start = t - pd.Timedelta(days=self._regime_window + self._purge_days)
        bars_win = bars[
            (bars["tradetime"] > bars_window_start)
            & (bars["tradetime"] <= t - pd.Timedelta(days=self._purge_days))
        ]
        if bars_win.empty:
            return

        fp: dict[str, float] | None = None
        if self._pool is not None:
            try:
                fp = self._pool.compute_regime_fingerprint(
                    bars_win, alpha_ic_stats=alpha_ic_stats
                )
            except Exception as exc:
                logger.warning("model_pool_add_fingerprint_failed", model_id=model_id, error=str(exc))

        if self._backend != "postgres" or self._pool is None:
            if fp is not None:
                self._add_local_entry(
                    fingerprint=fp,
                    model_id=model_id,
                    train_info=train_info,
                    detected_at=t,
                )
            return
        try:
            if not model.register_to_registry():
                logger.error("model_pool_add_registry_write_failed", model_id=model_id)
                if fp is not None:
                    self._add_local_entry(
                        fingerprint=fp,
                        model_id=model_id,
                        train_info=train_info,
                        detected_at=t,
                    )
                return
            if fp is None:
                return
            self._pool.add_to_pool(
                fp,
                model_id,
                train_info.get("feature_importance", {}),
                train_info.get("holdout_metrics", {}),
            )
        except Exception as exc:
            logger.error("model_pool_add_failed", model_id=model_id, error=str(exc))
            if fp is not None:
                self._add_local_entry(
                    fingerprint=fp,
                    model_id=model_id,
                    train_info=train_info,
                    detected_at=t,
                )
