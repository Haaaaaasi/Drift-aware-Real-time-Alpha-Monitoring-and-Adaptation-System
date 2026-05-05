# Adaptation A/B 實驗摘要：ab_20220601_20241231_top30_horizon5_top30_reb15_sched20_tw500

## 實驗設計
- 期間：2022-06-01 → 2024-12-31
- Top-K：30
- 對照策略：scheduled_20
- 共用條件：同一份資料、同一組 XGBoost 超參數、同一份 effective alpha subset、
  同一 portfolio constructor（equal_weight_topk, long_only）與 risk_manager。

## 窗口與成本（Phase A 修正）
- Trigger window（rolling IC / Sharpe）：`signal_time ∈ [t-60, t-20]`（calendar days）。
- Shadow eval window（model_pool）：`[t-30, t-10]`（shadow_window=20，與 trigger 不重疊）。
- Shadow warm-up gap：`5` 日（新候選訓練 cutoff 額外往前推，避免 IS leakage）。
- 交易成本：commission `0.0926%`/side、tax `0.300%`/sell-side、slippage `5.0` bps/side；round-trip 等效 ≈ `0.5852%`。

## 對照結果

| strategy | n_retrains | cumulative_return_pct | annualized_return_pct | sharpe | max_drawdown_pct | win_rate_pct | avg_turnover | avg_gross_return_bps | avg_total_cost_bps | avg_net_return_bps | final_value | n_pool_reuses | n_pool_misses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scheduled_20 | 32.0 | 27.66 | 10.244 | 0.593 | -21.175 | 53.25 | 0.064 | 7.957 | 3.309 | 4.649 | 12766010.64 | 0.0 | 0.0 |
| ew_buy_hold_universe | 0.0 | 43.612 | 15.552 | 1.054 | -13.907 | 58.95 | 0.002 | 6.196 | 0.023 | 6.173 | 14361204.23 | 0.0 | 0.0 |

## 觀察

- Sharpe 最高：`ew_buy_hold_universe` (1.054)；最低：`scheduled_20` (0.593)。
- 重訓次數範圍：0 → 32。

## 延伸分析
詳細的 regime-stratified 分析、paired t-test、drift 指標疊圖請見 
`notebooks/03_adaptation_evaluation.py`。
