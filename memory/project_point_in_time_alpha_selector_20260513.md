---
name: Point-in-time AlphaSelector 主線（2026-05-13）
description: 將 alpha selection 從固定 effective_alphas 清單升級為可回放、可審計的 point-in-time 決策層；第一版先以 static_is 重現既有 frozen OOS baseline。
type: project
originSessionId: codex-2026-05-13
---

# Point-in-time AlphaSelector 主線（2026-05-13）

## 背景

使用者確認下一條主線應優先於繼續修 model_pool：如果 alpha selection 仍是固定 55/64 alpha 清單，model_pool 再聰明也只是同一組可能退化的 feature 上換模型，研究價值會被卡住。

正式方向：

- feature store 提供完整 alpha feature。
- selector 做 point-in-time 決策。
- model / portfolio 只吃當下 selector snapshot。
- model_pool 後續必須記錄並尊重模型訓練時的 feature schema。

## 本次落地

新增最小 milestone：

- `docs/alpha_selection_design.md`
  - 將使用者確認的整體規劃整理成專案文件。
  - 明確列出 FeatureStore / AlphaSelector / snapshot / model registry 的責任邊界。
  - 納入本次 alpha 修改的架構圖，作為後續 rolling selector 與 model_pool 串接參考。

- `docs/architecture.md`
  - 主架構圖更新為 `FeatureStore -> Point-in-time AlphaSelector -> Meta Signal`。
  - 補上本次 alpha 架構圖與 Redis 定位。
  - 將 2026-05-13 cache-aligned frozen OOS baseline 更新為目前 reviewer-facing 結論。

- `src/alpha_engine/feature_store.py`
  - 邏輯 feature store wrapper。
  - 底層仍沿用現有 bar-aligned parquet cache，不先搬 partitioned parquet。
  - 提供 `feature_store_version`，供 snapshot / retrain log 追蹤。

- `src/alpha_selection/`
  - `base.py`：`SelectorContext`、`AlphaSelectionSnapshot`、穩定 hash helper。
  - `static_is.py`：`StaticISSelector`，重現目前 `effective_alphas.json` + `--exclude-indclass-cap-alphas` 行為。
  - `snapshot.py`：輸出兩層 artifact。

- `pipelines/simulate_recent.py`
  - 新增 `--selector {static_is, legacy}`。
  - 預設為 `static_is`。
  - `legacy` 保留舊 `_resolve_alpha_ids_for_run` 路徑，供等價比較。
  - 每次 retrain 嘗試會產生 selector snapshot。
  - `retrain_log.csv` 新增：
    - `selector`
    - `selector_snapshot_hash`
    - `feature_store_version`
    - `feature_columns_hash`
    - `n_feature_alphas`
  - `config.json` 新增 selector / feature store / bars / universe / feature columns hash 與 snapshot 路徑。

- `src/meta_signal/ml_meta_model.py`
  - `train()` 回傳 `feature_columns` 與 `feature_columns_hash`。
  - `holdout_metrics` 也寫入 `feature_columns_hash`，因此 model registry 的 JSON 欄位可追蹤 schema。

## Snapshot 輸出

每個 simulation run 會輸出：

- `alpha_selection_snapshots.csv`
  - event-level metadata。
  - 包含 `snapshot_hash`、`feature_store_version`、`bars_snapshot_hash`、`universe_hash`、`selector_config_hash`、`feature_columns_hash`。

- `alpha_scores.csv`
  - per-alpha score / selected / excluded_reason。

- `alpha_weights_by_date.csv`
  - selected alpha 的 date-weight table。

## 驗證

已通過：

- `py_compile`：
  - `src/alpha_engine/feature_store.py`
  - `src/alpha_selection/*.py`
  - `src/meta_signal/ml_meta_model.py`
  - `pipelines/simulate_recent.py`

- `pytest tests/unit/test_alpha_selection_static.py tests/unit/test_ml_meta_model.py::TestTraining::test_train_returns_expected_fields -q`
  - 4 passed

- `pytest tests/integration/test_ab_experiment.py::TestSimulateStrategies::test_strategy_none_trains_once -q`
  - 1 passed

- `pytest tests/unit/test_execution_alpha_universe.py tests/unit/test_alpha_cache.py -q`
  - 16 passed（僅既有 numpy/pandas warning）

## 2026-05-14 overnight 等價實驗

使用者允許跑長程 overnight 實驗後，先跑 `static_is` vs `legacy` frozen OOS 等價驗證。

Smoke 先抓到一個重要 bug：`build_snapshot()` 會把 `alpha_id` 排序，導致 `static_is` 的 `selected_alphas` 變成 alphabetical order，而 legacy 使用 `effective_alphas.json` 原始順序。因為 `MLMetaModel` 以 `to_numpy()` 餵 XGBoost，feature column order 是 schema 的一部分；原本 `hash_alpha_ids()` 又排序後才 hash，無法抓到這個 schema drift。

已修正：

- `src/alpha_selection/base.py`
  - `build_snapshot()` 保留 selector 輸出順序。
  - `hash_alpha_ids()` 改為 order-sensitive。
- `src/alpha_selection/static_is.py`
  - per-alpha scores 保留 effective list order。
- `tests/unit/test_alpha_selection_static.py`
  - 新增 order-preserving 與 order-sensitive hash 測試。

驗證：

- `pytest tests/unit/test_alpha_selection_static.py -q`：4 passed。
- 完整 frozen OOS 4 路徑已完成：
  - `static_is × next_vwap`
  - `legacy × next_vwap`
  - `static_is × next_open`
  - `legacy × next_open`

輸出：

- `reports/adaptation_ab/selector_equivalence_full_20260514/`
- 摘要：`reports/adaptation_ab/selector_equivalence_full_20260514/selector_equivalence_summary.md`

結果：

- `next_vwap`：static_is 與 legacy 都是 cum +22.337%、Sharpe 0.587、max DD -41.907%、23 retrains。
- `next_open`：static_is 與 legacy 都是 cum +36.606%、Sharpe 0.771、max DD -36.374%、23 retrains。
- `holdings.csv` SHA256 完全相同。
- `daily_pnl.csv` 的 `gross_return` / `net_return` / `cumulative_value` 完全相同，其他成本欄位僅 `1e-16` 浮點尾差。
- `retrain_log.csv` common fields 完全一致。
- `static_is` 產生 23 個 snapshot events、1472 score rows、1265 weight rows，feature hash prefix `1b6b75acaeb0...`。

結論：`--selector static_is` 通過 frozen OOS 等價驗收，可以成為正式預設路徑；下一步才做 `rolling_topk` 與 leakage toy test。

## 2026-05-14 rolling_topk 初版

已新增 `RollingTopKSelector`：

- `src/alpha_selection/rolling_topk.py`
- CLI：`simulate_recent --selector rolling_topk`
- 參數：
  - `--selector-alpha-top-k`
  - `--selector-window-days`
  - `--selector-min-coverage`
  - `--selector-min-observations`
- score：`abs(rolling_rank_ic) × coverage`
- mature gate：只使用 `label_available_at <= as_of_date`

新增 leakage toy test：

- `tests/unit/test_alpha_selection_rolling_topk.py`
- toy data 故意讓未來 label 有強訊號。
- 在 label mature 前，`rolling_topk` 不得選到未來 alpha；label mature 後才可被選。

目標測試：

- `pytest tests/unit/test_alpha_selection_rolling_topk.py tests/unit/test_alpha_selection_static.py tests/integration/test_ab_experiment.py::TestSimulateStrategies::test_strategy_none_trains_once -q`
- 結果：7 passed。

TEJ smoke：

- `reports/adaptation_ab/rolling_topk_smoke_20260514/`
- `rolling_topk30 + next_vwap` 短窗 2024-07-01→2024-08-30 可跑通，產生 3 個 selector snapshots。

Frozen OOS：

- `reports/adaptation_ab/rolling_topk_oos_20260514/`
- 摘要：`reports/adaptation_ab/rolling_topk_oos_20260514/rolling_topk_summary.md`

結果：

| Execution | Selector | Cum Ret % | Sharpe | Max DD % |
|---|---|---:|---:|---:|
| next_vwap | static_is baseline | 22.337 | 0.587 | -41.907 |
| next_vwap | rolling_topk30 | 24.941 | 0.629 | -45.149 |
| next_open | static_is baseline | 36.606 | 0.771 | -36.374 |
| next_open | rolling_topk30 | 37.890 | 0.785 | -41.080 |

判讀：rolling_topk30 對 return / Sharpe 有小幅改善，但 drawdown 變深；目前只能視為初步正向訊號，不可直接 claim 穩健勝利。下一步要跑 selector sensitivity：`alpha_top_k ∈ {20,30,40}` × `window_days ∈ {126,252,504}`，並加入 stability penalty / bootstrap。

## 後續順序

1. 跑 rolling_topk sensitivity 小矩陣：`alpha_top_k ∈ {20,30,40}` × `window_days ∈ {126,252,504}`。
2. 加入 selector stability penalty，避免 alpha set 過度跳動。
3. 做 `static_is vs rolling_topk` paired / block bootstrap，特別檢查 drawdown 與 regime 分段。
4. `soft_weighted_all` 先作為 rule aggregator / reporting 權重，不急著改 XGBoost feature scaling。
5. 最後才接 `regime_aware + model_pool`，且 reused model 必須使用自己訓練時的 `feature_columns_hash` / `selector_snapshot_hash`。

## 2026-05-14 rolling_topk stability penalty 與小矩陣

使用者要求開始跑小矩陣並加入 stability penalty。本輪已完成：

- `RollingTopKSelector` 新增 `stability_penalty`。
- CLI 新增 `--selector-stability-penalty`。
- `alpha_scores.csv` 新增 / 確認輸出：
  - `raw_score`
  - `score`
  - `stability`
  - `turnover_penalty`
- stability 規則：
  - 原始分數仍是 `abs(rolling_rank_ic) × coverage`。
  - 若 alpha 不在上一期 selected set，排序分數打折：`score = raw_score × (1 - stability_penalty)`。
  - 第一個 selector event 沒有上一期 selected set，不套 penalty。

新增測試：

- leakage toy test 仍通過。
- stability penalty test：當新 alpha 分數只小幅勝過上一期 alpha 時，`stability_penalty=0.20` 會保留上一期 alpha。

目標測試：

```text
pytest tests/unit/test_alpha_selection_rolling_topk.py tests/unit/test_alpha_selection_static.py tests/integration/test_ab_experiment.py::TestSimulateStrategies::test_strategy_none_trains_once -q
```

結果：8 passed。

### 3x3 小矩陣

設定：

- OOS：2024-07-01 → 2026-04-30
- Data source：TEJ survivorship-correct parquet
- Strategy：`scheduled_20`
- Portfolio：`turnover_aware_topk`，entry20 / exit60 / max_turnover0.25 / min_holding_days10 / tail25
- Execution：`next_vwap`
- Grid：`selector_top_k ∈ {20, 30, 40}` × `window_days ∈ {126, 252, 504}`
- 固定：`stability_penalty=0.10`

輸出：

- `reports/adaptation_ab/rolling_topk_stability_matrix_20260514/`
- 摘要：`reports/adaptation_ab/rolling_topk_stability_matrix_20260514/rolling_topk_stability_matrix_summary.md`

結果排序（Sharpe）：

| selector_top_k | window_days | Cum Ret % | Sharpe | Max DD % |
|---:|---:|---:|---:|---:|
| 20 | 126 | 62.120 | 1.298 | -30.373 |
| 20 | 504 | 46.183 | 0.957 | -38.296 |
| 30 | 126 | 40.934 | 0.923 | -36.332 |
| 20 | 252 | 31.748 | 0.790 | -31.509 |
| 40 | 252 | 28.154 | 0.691 | -44.120 |
| 30 | 252 | 22.973 | 0.614 | -39.560 |
| 30 | 504 | 19.768 | 0.542 | -39.128 |
| 40 | 504 | 14.779 | 0.441 | -38.609 |
| 40 | 126 | 4.187 | 0.219 | -39.852 |

相對 static baseline next_vwap（cum +22.337%、Sharpe 0.587、max DD -41.907%），`rolling_topk20_w126_pen10` 明顯改善。

### Penalty ablation

固定 `selector_top_k=20`、`window_days=126`，補跑 `stability_penalty ∈ {0.00, 0.05, 0.20}`，並用小矩陣中的 `0.10` 作對照。

輸出：

- `reports/adaptation_ab/rolling_topk_penalty_ablation_20260514/`
- 摘要：`reports/adaptation_ab/rolling_topk_penalty_ablation_20260514/penalty_ablation_summary.md`

結果：

| stability_penalty | Cum Ret % | Sharpe | Max DD % | Avg selected Jaccard | Avg new fraction |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 49.685 | 1.074 | -34.356 | 0.560 | 0.291 |
| 0.05 | 36.452 | 0.888 | -33.316 | 0.580 | 0.273 |
| 0.10 | 62.120 | 1.298 | -30.373 | 0.599 | 0.257 |
| 0.20 | 39.684 | 0.942 | -30.962 | 0.632 | 0.232 |

判讀：penalty 效果不單調；`0.10` 是目前最佳，但應視為排序邊界上的 tie-break，不可解讀成「越穩越好」。

### Execution price check

補跑最佳組合 `rolling_topk20_w126_pen10` 的 `next_open`：

- 輸出：`reports/adaptation_ab/rolling_topk_best_execution_check_20260514/`
- 摘要：`reports/adaptation_ab/rolling_topk_best_execution_check_20260514/execution_check_summary.md`

| Execution | Selector | Cum Ret % | Sharpe | Max DD % |
|---|---|---:|---:|---:|
| next_vwap | static_is baseline | 22.337 | 0.587 | -41.907 |
| next_vwap | rolling_topk20_w126_pen10 | 62.120 | 1.298 | -30.373 |
| next_open | static_is baseline | 36.606 | 0.771 | -36.374 |
| next_open | rolling_topk20_w126_pen10 | 76.252 | 1.385 | -25.615 |

目前結論：`rolling_topk20_w126_pen10` 是下一輪候選 incumbent，但還不能直接成為 reviewer-facing claim；必須補 shuffled-signal placebo、liquidity-filtered EW benchmark sensitivity、regime 分段、paired / block bootstrap，再決定是否取代 static_is frozen baseline。

## 2026-05-15 rolling_topk 防守性驗證完成

使用者要求補 shuffled-signal placebo、liquidity-filtered EW benchmark sensitivity、regime 分段、paired/block bootstrap。已新增並跑完：

- `scripts/run_rolling_topk_validation_workflow.py`
- 輸出：`reports/adaptation_ab/rolling_topk_validation_20260514/`
- workflow：`python scripts/run_rolling_topk_validation_workflow.py --n-vwap-seeds 30 --n-open-seeds 10 --n-boot 3000 --block-len 20`

注意：第一次長程跑到 `next_vwap seed=29` 中途被中斷；第二次續跑自動跳過 seed0–28、重跑 seed29，並完成 `next_open` 10 seeds 與最終彙整。

### Placebo

| Execution | Metric | Real | Placebo p95 | Real percentile | Seeds |
|---|---|---:|---:|---:|---:|
| next_vwap | cum ret % | 62.120 | 2.621 | 100.0 | 30 |
| next_vwap | Sharpe | 1.298 | 0.173 | 100.0 | 30 |
| next_open | cum ret % | 76.252 | 7.402 | 100.0 | 10 |
| next_open | Sharpe | 1.385 | 0.322 | 100.0 | 10 |

結論：return / Sharpe 明顯高於 shuffled signal null，支持結果不是 portfolio / pipeline 自帶正報酬。

### Benchmark sensitivity

| Execution | Series | Cum Ret % | Sharpe | Max DD % |
|---|---|---:|---:|---:|
| next_vwap | rolling_topk20_w126_pen10 | 62.120 | 1.298 | -30.373 |
| next_vwap | static_is_scheduled_20 | 22.337 | 0.587 | -41.907 |
| next_vwap | ew_same_cadence_liq100m | 19.585 | 0.563 | -36.069 |
| next_vwap | ew_same_cadence_liq200m | 27.493 | 0.710 | -36.340 |
| next_open | rolling_topk20_w126_pen10 | 76.252 | 1.385 | -25.615 |
| next_open | static_is_scheduled_20 | 36.606 | 0.771 | -36.374 |
| next_open | ew_same_cadence_liq100m | 27.291 | 0.671 | -33.507 |
| next_open | ew_same_cadence_liq200m | 35.762 | 0.795 | -33.904 |

### Regime 分段

rolling_topk20_w126_pen10：

| Execution | Regime | Cum Ret % | Sharpe | Max DD % |
|---|---|---:|---:|---:|
| next_vwap | 2024 H2 | -8.356 | -0.646 | -14.131 |
| next_vwap | 2025 H1 | 9.509 | 0.804 | -27.983 |
| next_vwap | 2025 H2 | 19.941 | 2.661 | -4.297 |
| next_vwap | 2026 YTD | 34.683 | 4.362 | -8.139 |
| next_open | 2024 H2 | -3.598 | -0.127 | -13.280 |
| next_open | 2025 H1 | 11.428 | 0.910 | -25.283 |
| next_open | 2025 H2 | 21.794 | 2.431 | -4.724 |
| next_open | 2026 YTD | 34.719 | 4.142 | -8.292 |

判讀：主要貢獻集中在 2025 H2 與 2026 YTD；2024 H2 仍是負報酬，2025 H1 仍有較深 drawdown。正式文字應避免宣稱所有 regime 都穩健勝出。

### Paired / block bootstrap

| Execution | Comparison | Mean excess bps/day | Paired p | Block bootstrap p |
|---|---|---:|---:|---:|
| next_vwap | vs static_is | 6.215 | 0.022 | 0.009 |
| next_vwap | vs liq100m EW | 6.925 | 0.004 | 0.002 |
| next_vwap | vs liq200m EW | 5.426 | 0.020 | 0.023 |
| next_open | vs static_is | 5.486 | 0.084 | 0.024 |
| next_open | vs liq100m EW | 7.399 | 0.020 | 0.002 |
| next_open | vs liq200m EW | 5.845 | 0.055 | 0.019 |

結論更新：

- `next_vwap` 可作為主要 reviewer-facing result：placebo 通過，且對 static / liquidity-filtered EW 的 paired 與 block bootstrap 都達 5% 單尾顯著。
- `next_open` 作為 supportive evidence：block bootstrap 通過，但 paired t-test 對 static / liq200m 只接近顯著。
- `rolling_topk20_w126_pen10` 可升級為目前 alpha selection 新 incumbent；後續 model_pool 必須挑戰 dynamic selector，而不是只挑戰 static_is。

## 2026-05-16 all_valid alpha audit 與擴充實驗

使用者要求先做 alpha audit，將 101 個 alpha 分成 pure price/volume、需要真實 cap、需要真實 indclass、coverage/NaN/constant 過差與不應進正式研究者，並測試 `rolling_topk_all_valid`：不再先用 IS IC performance 篩成 64/55，只排除資料語意不正確的 alpha。

已產出：

- `reports/alpha_audit/all_valid_alpha_audit_20260516/`
- `reports/adaptation_ab/rolling_topk_all_valid_oos_20260516/`
- `scripts/analyze_all_valid_alpha_selection.py`

Audit 結論：

| 分類 | 數量 |
|---|---:|
| WQ101 total | 101 |
| pure price / volume all_valid | 82 |
| requires indclass or cap | 19 |
| requires indclass | 18 |
| requires cap | 1 |
| TEJ IS effective | 64 |
| TEJ IS effective 且排除 indclass/cap | 55 |

實作注意：

- `simulate_recent --skip-effective-filter --exclude-indclass-cap-alphas` 可得到 all_valid 82-alpha 候選池。
- 初次全量讀取 82 alpha 會 OOM；已在 `simulate_recent` 對 `train_window_days` 啟用 alpha cache load window，正式 500-day training window 不再從 2018 全量讀 alpha panel。
- 相關檢查：`py_compile pipelines\simulate_recent.py scripts\analyze_all_valid_alpha_selection.py`；`pytest tests/unit/test_alpha_selection_rolling_topk.py tests/unit/test_alpha_selection_static.py -q` 通過。

OOS 結果：

| Execution | Series | Cum Ret % | Sharpe | Max DD % |
|---|---|---:|---:|---:|
| next_vwap | all_valid_82 | 6.717 | 0.280 | -39.352 |
| next_vwap | incumbent_55 | 62.120 | 1.298 | -30.373 |
| next_vwap | static_is_55 | 22.337 | 0.587 | -41.907 |
| next_vwap | ew_same_cadence_liq100m | 19.585 | 0.563 | -36.069 |
| next_vwap | ew_same_cadence_liq200m | 27.493 | 0.710 | -36.340 |
| next_open | all_valid_82 | 16.319 | 0.484 | -35.993 |
| next_open | incumbent_55 | 76.252 | 1.385 | -25.615 |
| next_open | static_is_55 | 36.606 | 0.771 | -36.374 |
| next_open | ew_same_cadence_liq100m | 27.291 | 0.671 | -33.507 |
| next_open | ew_same_cadence_liq200m | 35.762 | 0.795 | -33.904 |

Bootstrap：

| Execution | Comparison | Mean excess bps/day | Paired p | Block p |
|---|---|---:|---:|---:|
| next_vwap | all_valid_82 vs incumbent_55 | -9.596 | 0.999 | 0.998 |
| next_vwap | incumbent_55 vs all_valid_82 | 9.596 | 0.001 | 0.002 |
| next_open | all_valid_82 vs incumbent_55 | -9.603 | 0.995 | 0.998 |
| next_open | incumbent_55 vs all_valid_82 | 9.603 | 0.005 | 0.002 |

Selector pool shift：

- 新增候選 27 個；23 次 snapshot 中平均每次選到 5.391 個新增 alpha，平均權重占比 26.96%，最多一次選到 10/20 個。
- 最常被選入的新增候選：`wq094`、`wq072`、`wq074`、`wq061`、`wq065`、`wq075`、`wq086`。

結論：`all_valid_82` 可跑通，但明顯輸給 55-alpha incumbent、static baseline 及 liquidity-filtered EW；不應再投入長 placebo 作為 promoted candidate。新增 27 個 alpha 不能一次全量放進 live selector，應先進 quarantine/admission gate，以 point-in-time coverage、stability、成熟 label rolling score 與 family diversity 通過後再升級。

## 2026-05-16 quarantine / admission gate 實作

已把 all_valid 實驗後的修正策略落地：

- 修改 `src/alpha_selection/rolling_topk.py`：`RollingTopKSelector` 新增 `base_alpha_ids` 與 admission gate 參數。
- 修改 `src/alpha_selection/base.py`：snapshot score table 新增 `alpha_pool`、`admission_status`、`admission_score`、`admission_reason`、`admission_subwindow_pass_count`、`max_abs_corr_to_live`。
- 修改 `pipelines/simulate_recent.py`：新增 CLI `--selector-admission-gate` 與 admission gate 相關參數；啟用時 candidate pool 可用 all_valid 82，但 base live alpha 會從 `effective_alphas.json` 取 55-alpha incumbent，新增 27 alpha 自動視為 quarantine。
- 新增測試：`tests/unit/test_alpha_selection_rolling_topk.py` 補四個 admission gate cases。

Gate 規則：

- live alpha：不需 admission，仍按 rolling score 競爭。
- quarantine alpha：需通過樣本數、coverage、`admission_min_score`、子窗穩定性與 live alpha 最大相關性上限，且每次 event 最多 `admission_max_promoted` 個 alpha 進入 live selector 候選集合。
- 若未通過，`excluded_reason` 會寫成 `admission_low_score`、`admission_unstable_subwindows`、`admission_redundant_family`、`admission_capacity` 等可審計原因。

驗證：

```powershell
python -m py_compile src\alpha_selection\base.py src\alpha_selection\rolling_topk.py pipelines\simulate_recent.py tests\unit\test_alpha_selection_rolling_topk.py
python -m pytest tests\unit\test_alpha_selection_rolling_topk.py tests\unit\test_alpha_selection_static.py -q
```

結果：`11 passed`。

TEJ/cache smoke：

```text
reports/adaptation_ab/admission_gate_smoke_20260516/
```

使用 `wq019` 作為 incumbent base、`wq094` 作為 quarantine alpha。`wq094` 兩次 snapshot 都因 `admission_low_score` 被擋下，model 只使用 `wq019`；config 正確記錄 `selector_admission_gate=true`、`n_admission_base_alphas=1`、`n_quarantine_alphas=1`。

下一步應跑正式 `all_valid_82 + admission_gate` OOS 小矩陣：`admission_max_promoted ∈ {2,4}`、`admission_min_score ∈ {0.02,0.03,0.05}`、`admission_max_abs_corr_to_live ∈ {0.95,0.98}`，主比較仍是 incumbent_55 的 `rolling_topk20_w126_pen10`。

## 2026-05-17 admission gate matrix 完成

已跑完 `next_vwap` 主結果矩陣：

- 輸出：`reports/adaptation_ab/admission_gate_matrix_20260517/`
- summary：`reports/adaptation_ab/admission_gate_matrix_20260517/matrix_summary.csv`
- 矩陣：`admission_max_promoted ∈ {2,4}`、`admission_min_score ∈ {0.02,0.03,0.05}`、`admission_max_abs_corr_to_live ∈ {0.95,0.98}`
- 共 12 組，全部 `done`。

最佳組合：

| Execution | max_promoted | min_score | max_corr | Cum Ret % | Sharpe | Max DD % | Avg admitted |
|---|---:|---:|---:|---:|---:|---:|---:|
| next_vwap | 4 | 0.02 | 0.95 / 0.98 | 14.693 | 0.447 | -42.722 | 3.913 |

對照：

| Series | Cum Ret % | Sharpe | Max DD % |
|---|---:|---:|---:|
| all_valid_82 直接全放 | 6.717 | 0.280 | -39.352 |
| admission gate best | 14.693 | 0.447 | -42.722 |
| incumbent_55 | 62.120 | 1.298 | -30.373 |

判斷：

- Admission gate 比 all_valid 直接全放好，但仍遠輸 incumbent_55。
- `admission_min_score=0.03/0.05` 更嚴反而更差；目前 gate 只靠 admission 前 rolling IC / coverage / subwindow stability 不足以找到會改善 portfolio 的新增 alpha。
- `max_corr=0.95` 與 `0.98` 結果相同，family redundancy 不是這輪主因。
- 暫時不補 `next_open`，因為主結果沒有接近 incumbent，不值得做 supportive evidence。

下一步應做新增 27 alpha 的 failure attribution：追蹤被 admit 的 alpha、進入日期、進入後 portfolio 邊際貢獻，並設計 probation / shadow contribution gate；不要繼續只掃 admission threshold。

## 2026-05-17 admitted alpha failure attribution 完成

已針對最佳 admission gate run 做 period-level attribution：

- 輸出：`reports/adaptation_ab/admission_gate_attribution_20260517/`
- 腳本：`scripts/analyze_admitted_alpha_attribution.py`
- Best admission run：`adgate_p4_s0p02_c0p95_nextvwap`
- Incumbent：`rolling_topk20_w126_pen10_nextvwap`

注意：這不是逐 alpha causal counterfactual；同一 period 可能有多個 admitted alpha 且會與 XGB / portfolio state 交互。本診斷只用來判斷 alpha expansion 是否值得繼續救。

整體：

| 指標 | 值 |
|---|---:|
| selection periods | 23 |
| periods with admission | 23 |
| negative excess rate vs incumbent | 69.6% |
| avg period excess vs incumbent | -7.816 bps/day |

Alpha-level association：

| Alpha | Admitted periods | Avg excess bps/day | Median excess bps/day | Negative window rate |
|---|---:|---:|---:|---:|
| wq051 | 3 | -19.955 | -10.940 | 100.0% |
| wq047 | 2 | -17.396 | -17.396 | 100.0% |
| wq065 | 9 | -15.268 | -12.932 | 77.8% |
| wq061 | 13 | -13.040 | -10.940 | 84.6% |
| wq085 | 1 | -12.932 | -12.932 | 100.0% |
| wq007 | 3 | -9.823 | -1.953 | 66.7% |
| wq072 | 12 | -7.288 | -5.426 | 58.3% |
| wq086 | 4 | -5.698 | -9.436 | 50.0% |
| wq094 | 17 | -5.088 | -2.600 | 64.7% |
| wq075 | 11 | -3.581 | -2.600 | 63.6% |
| wq045 | 4 | -3.338 | -1.900 | 75.0% |
| wq074 | 11 | 0.879 | -1.277 | 54.5% |

判斷：不是少數壞 alpha 拖垮，而是 admitted quarantine 整批沒有穩定正向邊際貢獻。`wq074` 是唯一平均略正者，但中位數與過半 window 仍為負，不足以升級。結論是停止 alpha expansion 主線，正式保留 `incumbent_55 + rolling_topk20_w126_pen10`；admission gate 保留為未來新資料源或真實 indclass/cap 接入後的研究工具。
## 2026-05-17 P0 frozen selector 與 prospective holdout protocol

已完成 P0 收斂，正式把目前 alpha selection 主線鎖定為：

```text
incumbent_55 + rolling_topk20_w126_pen10 + scheduled_20
```

新增檔案：

- `configs/frozen_alpha_selector_20260517.yaml`
- `docs/final_robustness_holdout_protocol.md`
- `reports/adaptation_ab/final_robustness_20260517/final_robustness_summary.md`
- `reports/adaptation_ab/final_robustness_20260517/manifest.json`

關鍵決策：

- `2024-07-01` 到 `2026-04-30` 改定義為 frozen validation，不再稱為 untouched holdout，因為已用於 selector、portfolio hygiene、benchmark sensitivity、all_valid expansion 與 admission gate 決策。
- 真正 prospective holdout 必須從 2026-05-17 freeze 後的新 TEJ 資料開始，依目前資料狀態預期為 `2026-05-01` 或之後。
- `next_vwap` 是正式主結果，`next_open` 只作支持結果。
- Alpha expansion 暫停；all_valid_82 與 admission gate 不進 live selector。
- 後續 model_pool 必須用此 frozen selector 當 alpha input，並以 `scheduled_20 incumbent` 作主要比較對象。

P0 robustness 匯總：

| Execution | Series | Cum Ret % | Sharpe | Max DD % |
|---|---|---:|---:|---:|
| next_vwap | rolling_topk20_w126_pen10 | 62.120 | 1.298 | -30.373 |
| next_vwap | static_is_scheduled_20 | 22.337 | 0.587 | -41.907 |
| next_vwap | ew_same_cadence_liq100m | 19.585 | 0.563 | -36.069 |
| next_vwap | ew_same_cadence_liq200m | 27.493 | 0.710 | -36.340 |
| next_open | rolling_topk20_w126_pen10 | 76.252 | 1.385 | -25.615 |
| next_open | static_is_scheduled_20 | 36.606 | 0.771 | -36.374 |

主要 caveat：

- 2024_H2 為負報酬，績效集中於 2025_H2 與 2026_YTD。
- drawdown 仍高，`next_vwap` max DD 為 -30.373%。
- 若依 holdout 結果調參、改 alpha universe、重啟 admission gate 或把 `next_open` 提升為主結果，該結果必須另開 experiment family，不能沿用 P0 frozen holdout claim。
