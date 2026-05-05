---
name: Model Pool Phase B 重構（B-1 / B-2 / B-3）
description: WP11 model pool 全面重構：cosine→z-scored Euclidean 評分、15 維 fingerprint、top-k shadow ensemble + last_evaluated_ic feedback；尚未 commit
type: project
originSessionId: 0ace12ca-d5f6-411f-b2f0-98abaf818a4c
---

WP11 model_pool 在 Phase A baseline 出現 0% reuse，2026-04-29 起逐步診斷出多條問題鏈，分三個 phase 處理。

## Phase A — Bug fixes（2026-04-30）

1. **FK 違反 silent failure**：`regime_pool.associated_model_id` REFERENCES `model_registry(model_id)`，但 `simulate_recent` 只把 model 存進 process-local dict、沒寫進 model_registry → `add_to_pool` 永遠被 FK 擋下。修法：`ModelPoolController.register_initial` / `_try_add_to_pool` 在 `add_to_pool` 前先呼叫 `model.register_to_registry()`，回 False 直接 ERROR log + return。
2. **Fingerprint 偷渡 NaN**：單一標的或恆定報酬時 `avg_cross_correlation` 為 NaN；PostgreSQL JSONB 拒絕 `NaN` 字面值，且舊 cosine 遇 NaN 退回 0 → silent miss。修法：`compute_regime_fingerprint` 走 `_safe()` helper，所有欄位 NaN/Inf → 0.0。
3. **broad except 掩蓋錯誤**：原本走 `logger.warning`，全升級為 `logger.error`。
4. **`_models_by_id` 不一致**：`shadow_kept_current` 分支用 `setdefault` 兜底。

## Phase B-1 — Similarity 重構（2026-05-01）

原 raw cosine 在 5 維非標準化 fingerprint 上幾乎永遠 > 0.98（`volume_ratio` 量綱主導），pool 形同失能。新公式：

    score = exp(-d_zscored / distance_scale) * staleness * perf_gate

* `_compute_pool_scales`：pool size ≥ 3 用 std；< 3 或 std < 1e-4 用 `SCALE_PRIORS` cold-start 兜底
* `_standardized_distance`：每維度先除以 scale 再 Euclidean
* `_distance_to_similarity`：`exp(-d/2.0)`，d=0→1、d=2→0.37、d=4→0.135
* `_staleness_factor`：`exp(-age_days / tau_days)`，預設 `tau_days=180`
* `_performance_gate`：holdout rank_ic 低於下限歸 0
* 預設 threshold 0.8 → 0.5（4 處入口統一）

**Diagnostic 確認**：2024 H1 6-month sim → similarity=0.453（下手算驗證），pool 從 cosine 永遠 0.98+ 變成有真實區別。

## Phase B-2 — Fingerprint 擴 15 維（2026-05-01）

5 → 15 維，三組新增：
* **Cross-sectional**（4 維）：`cs_return_std` / `_skew` / `_kurt` / `_tail_spread`（每日 top10% - bottom10% spread）
* **Temporal**（3 維）：`vol_of_vol_5d` / `_20d`（rolling vol 的 std）+ `cvar_5pct`（底端 5% 平均報酬）
* **Alpha-side**（3 維，caller 注入）：`alpha_ic_mean` / `_std` / `_pos_fraction`，由 `compute_alpha_ic_stats(alpha_panel, fwd_returns, label_avail, t)` 在 simulate_recent 端計算。最近 60 日成熟標籤窗口，用 spearman rank corr 算每日每 alpha IC，然後跨日平均得 per-alpha IC，再對 alpha 取 mean/std/pos_fraction。

`SCALE_PRIORS` 同步擴到 15 維。簽名變更：
* `RecurringConceptPool.compute_regime_fingerprint(market_data, alpha_ic_stats=None)`
* `ModelPoolController.register_initial(..., alpha_ic_stats=None)`
* `ModelPoolController.decide_on_trigger(..., alpha_ic_stats=None)`
* `ModelPoolController._try_add_to_pool(..., alpha_ic_stats=None)`

## Phase B-3 — Top-k ensemble + quality feedback（2026-05-01）

1. **Schema migration** `migrations/002_phase_b3_regime_pool.sql`：`regime_pool` 加 `last_evaluated_ic NUMERIC` + `last_evaluated_at TIMESTAMPTZ`（已對 dev PG 跑過）。
2. **`find_similar_regimes(top_k=...)`**：依 score 排序回 top-k 候選；保留 `find_similar_regime` 為 top-1 wrapper（維持向後相容）。多了「`score > 0` 才能進結果」的 guard，避免 perf_gate 殺成 0 的 entry 在 threshold=0 時偷渡進來。
3. **`_performance_gate` 升級**：`last_evaluated_ic` 優先於 `performance_summary.rank_ic`（NULL/NaN 時 fallback holdout）→ 形成 quality feedback loop。
4. **Top-k shadow ensemble**：`decide_on_trigger` 用 `find_similar_regimes(top_k=self._top_k)` 把多個 reused 候選都丟進 ShadowEvaluator，由 evaluator 自選；`regime_by_model` map 用來在 best_id 是 reused 時找回對應 regime_id（多候選下不能只記錄 single similar_regime_id）。
5. **`update_last_evaluated_ic`**：每次 shadow eval 後對所有被評估的 reused 候選都寫回 IC（不只是被選中的那個）。
6. `ModelPoolController` 新增 `top_k_candidates: int = 3` 參數；`pipelines.simulate_recent.simulate(..., pool_top_k=3)` + `ab_experiment.DEFAULT_STRATEGIES['model_pool']` 串到底。

## 測試
| 檔案 | 測試數 | 範圍 |
|---|---|---|
| `tests/unit/test_recurring_concept_fingerprint.py` | 6 | NaN/Inf + JSONB 嚴格性 |
| `tests/unit/test_model_pool_fk_ordering.py` | 7 | register_to_registry → add_to_pool 順序 |
| `tests/unit/test_recurring_concept_similarity.py` | 23 | Phase B-1 5 個 helper 數學 |
| `tests/unit/test_recurring_concept_phase_b2.py` | 13 | 15 維存在 / NaN safe / `compute_alpha_ic_stats` |
| `tests/unit/test_recurring_concept_phase_b3.py` | 12 | top-k 排序 / `last_evaluated_ic` 優先邏輯 / SQL |
| `tests/integration/test_model_pool_strategy.py` | 11 | 端到端 + fixture 加上 `find_similar_regimes` + `update_last_evaluated_ic` |

整合測試 fake pool（`_InMemoryRegimePool`）已擴充支援新介面。

## Long-period diagnostic sim（2026-05-01 已跑）

2022-06-01 → 2024-12-31，model_pool 策略，IS-selected 52 alphas，top_k=10，沒套 trigger 過嚴限制。

| 指標 | 值 |
|---|---|
| n_days | 631 |
| n_retrains | 16 |
| **n_pool_misses** | **15** |
| **n_pool_reuses** | **0** |
| Pool 最終大小 | 16 entries（vs. Phase B-1 6 個月只有 2） |
| Sharpe | 3.636 |
| cum_ret | 15,515% |
| max_drawdown | -57.6% |
| best_seen score（log） | 0.40–0.45（threshold 0.5） |

**Pool 確實長出來了，但 0 reuse**。這暴露 Phase B-1 ↔ B-2 介面的隱形問題：

擴 5 → 15 維後，z-scored Euclidean 距離自然 ~sqrt(3) 倍放大（per-dim 1 std → d=sqrt(N)）。同樣「平均每維 0.5 std 差」的兩個 regime：
* 5 維 → d≈1.12 → sim=exp(-1.12/2)=0.57 → 過 threshold
* 15 維 → d≈1.94 → sim=exp(-1.94/2)=0.38 → 不過

threshold 0.5 在 15 維下變得過嚴，**reusable 的 regime 也被擋掉**。後續觸發 best_score 一直在 0.40~0.45（很接近但都沒過）。

### Option C 已實作並驗證（2026-05-01）

**選擇：C — mean-form distance `d = sqrt(Σ((diff/scale)²) / N)`**

實作：`_standardized_distance` 末行加 `/ np.sqrt(n)`，2 個單測期望值更新（`1.0 → 1/sqrt(15)`、`sqrt(0.5) → sqrt(0.5)/sqrt(15)`），65 tests 全綠。

**Long-period sim 驗證（2022-06-01 → 2024-12-31，model_pool，top_k=10）：**

| 指標 | Phase B-1（Option C 前） | Option C 後 |
|---|---|---|
| n_retrains | 16 | 14 |
| n_pool_misses | 15 | 9 |
| **n_pool_reuses** | **0** | **3** ✓ |
| Sharpe | 3.636 | 3.462 |
| cum_ret | 15,515% | 11,290% |
| max_drawdown | -57.6% | -47.9% |

最後一次 reuse：`similarity=0.741`，`reason=shadow_selected_reused_sim_0.741`。Pool 正常運作，reuse 有真實結構相似度支撐。Sharpe 稍降屬正常（reuse 舊模型 vs 新訓練的 trade-off）。

## 同期附加

* **Diagnostic surface 補強**：`find_similar_regimes` 加 `return_best_seen=True` 參數；`decide_on_trigger` 改用 best_seen 做 retrain_log 顯示，避免「沒過 threshold 時 sim=0.000」的誤導。

## 仍懸空

* **沒 commit**：自 Phase A 起所有改動仍在 working tree，使用者明示「累積三週量先不急著推」。等 metric 調整 + 重跑驗證再決定是否 commit。
* **跨 run 持久化（P0★ #6）**：`_models_by_id` 仍是 process-local，要等 `MLMetaModel.save()/load()` + `regime_pool.model_artifact_path` 欄位才能 cross-run reuse。
* **PG 既有遺留**：dev PG 還有 1 筆 2026-04-30 的舊 regime（cosine 時代），長 sim 開始前已 cleanup 2026-05-01 的，2024-04-30 的留著（since-filter 會過濾）。
