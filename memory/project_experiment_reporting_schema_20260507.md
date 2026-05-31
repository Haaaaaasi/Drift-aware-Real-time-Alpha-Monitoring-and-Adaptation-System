# Experiment reporting schema（2026-05-07）

新增獨立 PostgreSQL experiment reporting 層，用於把既有研究輸出轉成可查詢、可展示、可比較的資料，不再把實驗成果塞進 `monitoring_metrics`。

## 已落地範圍

- `migrations/002_experiment_reporting.sql`
  - `experiment_runs`
  - `experiment_strategy_results`
  - `experiment_daily_pnl`
  - `experiment_model_pool_decisions`
- `scripts/ingest_experiment_report.py`
  - 只 ingest existing reports，不改 `simulate_recent.py` / `ab_experiment.py`
  - 支援 `--dry-run`
  - official run 必須明確指定 `--data-source`
  - 寫入採 transaction：run row upsert + child rows delete/insert
- `dashboards/experiment_results.json`
  - 使用 `$run_id` variable
  - 顯示 run summary、strategy comparison、cumulative value、gross/cost/net、turnover、model_pool decisions、selector matrix、cost sensitivity

## 重要語意

- `strategy` 與 `variant_name` 分離；selector matrix 的 `topknet_t05` 這類 label 存在 `variant_name`，`strategy` 固定為 `model_pool`。
- `scenario_name` 與 `round_trip_cost_pct` 用於 cost sweep；baseline 情境為 `scenario_name='baseline'`。
- `variant_name` 預設空字串，`scenario_name` 預設 `baseline`，避免 PostgreSQL unique key 遇到 NULL 語意問題。
- benchmark row 會匯入 strategy results 與 daily pnl，但 `rank_by_net_return` 存 NULL，不參與策略排名。
- `experiment_model_pool_decisions.raw_record` 保留原始 CSV row，避免 diagnostics 欄位後續變動時立刻破 schema。

## 已驗證

- 新單元測試：`tests/unit/test_experiment_report_ingest.py`
- dry-run 正式 A/B：
  - `ab_20220601_20241231_top10_horizon5_reb10_formal_5strategy_tw500`
  - 6 strategy rows、3786 daily pnl rows
- dry-run selector matrix：
  - `model_pool_selector_matrix_20260507`
  - 5 strategy rows、3473 daily pnl rows、328 decision rows
