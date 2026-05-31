---
name: Model Pool failure audit 與 reuse guard
description: 2026-05-13 對 model_pool selector 的保守修正與失敗審計欄位
type: project
---

## 背景

2026-05-08 current-code formal A/B 顯示 `model_pool(topk_net_return/0.5)` 只略優於 `triggered`，但明顯輸給正式 baseline `scheduled_20`。先前 failure diagnosis 指向 reused candidates 在 shadow selector 中偶爾勝出，但後續持股 proxy / observed post-window PnL 偏弱。

核心漏洞：舊 selector 只要求 raw best candidate 比 current model 好，沒有要求 reused candidate 必須明顯贏過同次 trigger 產生的 `new` candidate。因此舊 regime model 可能因 shadow window noise 被放行。

## 2026-05-13 修正

1. `src/adaptation/model_pool_strategy.py`
   - 新增 `reuse_min_score` 與 `reuse_margin`。
   - 當 raw best 是 reused candidate 時，必須同時通過：
     - selector score 為有限值；
     - 若設定 `reuse_min_score`，score 不低於門檻；
     - 若有 current/new non-reused candidate，reused score 必須比最佳 non-reused score 高出至少 `reuse_margin`。
   - 未通過時改用 non-reused candidate 重新 selection，避免 similarity hit 自動壓過現場新訓練模型。

2. `pipelines/simulate_recent.py` / `pipelines/ab_experiment.py`
   - 新增 CLI / config：
     - `--model-pool-reuse-min-score`
     - `--model-pool-reuse-margin`
   - `model_pool_decisions.csv` 新增 selector audit 欄位：
     - `shadow_rank_by_selection_metric`
     - `shadow_rank_by_topk_net_return`
     - `raw_best_candidate_model_id`
     - `raw_best_role`
     - `raw_best_score`
     - `best_non_reused_model_id`
     - `best_non_reused_score`
     - `reuse_score_margin_vs_best_non_reused`
     - `reuse_guard_min_score`
     - `reuse_guard_margin`
     - `reuse_guard_passed`
     - `reuse_guard_reason`

3. `scripts/diagnose_model_pool_failure.py`
   - failure summary 新增 `Reuse Guard / Selector Audit` 區塊，回報 raw best role 與 guard reason counts。
   - event attribution 也帶入 raw best / guard 欄位，可比較「原本 selector 會選誰」與「guard 後實際套用誰」。

## 測試

已通過：

```powershell
.\.venv\Scripts\python.exe -m py_compile src\adaptation\model_pool_strategy.py pipelines\simulate_recent.py pipelines\ab_experiment.py scripts\diagnose_model_pool_failure.py tests\unit\test_model_pool_reuse_guard.py tests\unit\test_diagnose_model_pool_failure.py tests\integration\test_model_pool_strategy.py
.\.venv\Scripts\python.exe -m pytest tests/unit/test_model_pool_reuse_guard.py tests/unit/test_model_pool_diagnostics.py tests/unit/test_diagnose_model_pool_failure.py tests/integration/test_model_pool_strategy.py -q
```

結果：`18 passed`。警告為既有 pandas `groupby.apply` future warning 與 matplotlib CJK glyph warning。

## 下一步

正式驗證時先跑小矩陣，不要直接宣稱 model_pool 已修好：

- 固定 frozen OOS 設定、next_vwap 優先。
- 以 `topk_net_return` selector 為主。
- 小矩陣建議：
  - `reuse_margin`: `0.0`, `0.0025`, `0.005`, `0.01`
  - `reuse_min_score`: `None`, `0.0`
- 評估順序：
  1. `model_pool` 是否優於 `triggered`；
  2. 是否縮小對 `scheduled_20` 的差距；
  3. 若沒有縮小，問題不在 reuse guard，而更可能是 trigger policy / pool fingerprint / cross-run persistence。

目前仍不可把 model_pool 寫成完成的 recurring concept reuse claim；它仍是待驗證策略，正式 baseline 仍是 `scheduled_20`。
