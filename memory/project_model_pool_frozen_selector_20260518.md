---
name: Model Pool frozen selector challenge 2026-05-18
description: frozen alpha selector 下測試 model_pool 是否能挑戰 scheduled_20 incumbent；包含 process-local fallback、metric gate 修正與 full OOS 結論。
type: project
---

## 背景

本輪把 model_pool 放到正式 frozen alpha selector 路徑下測試：

```text
frozen selector = incumbent_55 + rolling_topk20_w126_pen10
model_pool trigger mode = scheduled
selection_metric = topk_net_return
execution = next_vwap
baseline = scheduled_20 incumbent
```

## 工程修正

- `simulate_recent` / `ab_experiment` 支援 `--frozen-config` 與 `--frozen-execution`，由 `configs/frozen_alpha_selector_20260517.yaml` 覆寫資料源、selector、portfolio、label、成本與 `train_window_days=500`。
- `model_pool` 新增 `--model-pool-trigger-mode scheduled`，讓它在與 `scheduled_20` 相同 cadence 下挑戰 incumbent，而不是靠不同 trigger 條件混淆歸因。
- `ModelPoolController` 新增 process-local fallback pool。PostgreSQL/registry 不可用時，run 內仍可建立 regime-model entries 與 reused candidates；summary 會顯示 `pool_backend=postgres_with_local_fallback`。
- 修正 pool pre-gate 與 selector metric 不一致問題：`selection_metric=topk_net_return` 時，不再用 `rank_ic >= 0` 預先擋掉 candidate；`selection_metric=ic/rank_ic` 時仍保留 rank IC gate。
- 新增 `tests/unit/test_model_pool_local_fallback.py`。

## 驗證

```text
py_compile passed
tests/unit/test_model_pool_local_fallback.py
tests/unit/test_frozen_alpha_selector.py
tests/unit/test_model_pool_reuse_guard.py
tests/unit/test_model_pool_shadow_window.py
tests/integration/test_model_pool_strategy.py
= 24 passed
```

## 實驗結果

| Period | Variant | Cum Ret % | Sharpe | Max DD % | n_reuses | n_misses |
|---|---|---:|---:|---:|---:|---:|
| 2024-07-01 → 2024-08-30 | scheduled_20 | 0.659 | 0.279 | -14.131 | 0 | 0 |
| 2024-07-01 → 2024-08-30 | model_pool, no reuse min | -0.778 | -0.003 | -13.817 | 1 | 1 |
| 2024-07-01 → 2024-12-31 | scheduled_20 | -8.356 | -0.646 | -14.131 | 0 | 0 |
| 2024-07-01 → 2024-12-31 | model_pool, no reuse min | -5.322 | -0.395 | -13.817 | 1 | 4 |
| 2024-07-01 → 2024-12-31 | model_pool, `reuse_min_score=0` | -4.710 | -0.339 | -13.817 | 0 | 4 |
| 2024-07-01 → 2026-04-30 | scheduled_20 | 62.120 | 1.298 | -30.373 | 0 | 0 |
| 2024-07-01 → 2026-04-30 | model_pool, `reuse_min_score=0` | 14.286 | 0.442 | -37.537 | 3 | 16 |

主要輸出：

- `reports/adaptation_ab/ab_20240701_20240830_top10_frozen_model_pool_local_smoke_gatefix_20260518/`
- `reports/adaptation_ab/ab_20240701_20241231_top10_frozen_model_pool_h2_m0_20260518/`
- `reports/adaptation_ab/ab_20240701_20241231_top10_frozen_model_pool_h2_reuse_min0_20260518/`
- `reports/adaptation_ab/ab_20240701_20260430_top10_frozen_model_pool_full_oos_reuse_min0_20260518/`

## 結論

- model_pool 在 2024H2 局部比 scheduled_20 少虧，但 full frozen OOS 明顯輸給 `scheduled_20`（14.286% vs 62.120%，Sharpe 0.442 vs 1.298，DD 也較深）。
- `reuse_min_score=0` 能避免負 shadow score 的 reused candidate 被選，H2 結果略改善，但 full OOS 仍不足以構成 recurring concept reuse claim。
- 目前不能宣稱找到可用且優於 incumbent 的 model_pool。正式研究主線仍應保留 `incumbent_55 + rolling_topk20_w126_pen10 + scheduled_20`。
- 這次結果使用 process-local fallback，適合在 DB 不可用時做單 run 診斷；不能當作跨 run recurring concept persistence 的證據。

## 2026-05-18 next_open closure check

補跑 `--frozen-execution secondary`（對應 `next_open`）的 full OOS closure check：

```text
reports/adaptation_ab/ab_20240701_20260430_top10_frozen_model_pool_full_oos_nextopen_reuse_min0_20260518/
```

結果：

| Execution | Strategy | Cum Ret % | Sharpe | Max DD % | n_reuses | n_misses |
|---|---|---:|---:|---:|---:|---:|
| next_open | scheduled_20 | 76.252 | 1.385 | -25.615 | 0 | 0 |
| next_open | model_pool, `reuse_min_score=0` | 27.838 | 0.645 | -35.020 | 3 | 16 |

`model_pool` decision log 中實際選中角色為 current 12 次、new 7 次、reused 3 次；reused 發生於 2025-08-25、2025-11-21、2026-01-20。雖然 reused candidate 能被挑到，最終仍明顯輸給 `scheduled_20`。

結論：closure check 完成。`next_vwap` 與 `next_open` 都支持同一個判斷：在 frozen alpha selector（`incumbent_55 + rolling_topk20_w126_pen10`）下，現版 model_pool 不具備挑戰 `scheduled_20` incumbent 的證據。model_pool 主線應收斂為 failure / ablation appendix，正式主線回到 final robustness / holdout 與報告整理。
