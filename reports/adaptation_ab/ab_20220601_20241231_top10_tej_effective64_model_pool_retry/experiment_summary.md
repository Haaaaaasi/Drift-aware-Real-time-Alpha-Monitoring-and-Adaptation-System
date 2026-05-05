# Adaptation A/B 實驗摘要：ab_20220601_20241231_top10_tej_effective64_model_pool_retry

## 實驗設計
- 期間：2022-06-01 → 2024-12-31
- Top-K：10
- 對照策略：model_pool
- 共用條件：同一份資料、同一組 XGBoost 超參數、同一份 effective alpha subset、
  同一 portfolio constructor（equal_weight_topk, long_only）與 risk_manager。

## 窗口與成本（Phase A 修正）
- Trigger window（rolling IC / Sharpe）：`signal_time ∈ [t-60, t-20]`（calendar days）。
- Shadow eval window（model_pool）：`[t-30, t-10]`（shadow_window=20，與 trigger 不重疊）。
- Shadow warm-up gap：`5` 日（新候選訓練 cutoff 額外往前推，避免 IS leakage）。
- 交易成本：commission `0.0926%`/side、tax `0.300%`/sell-side、slippage `5.0` bps/side；round-trip 等效 ≈ `0.5852%`。

## 對照結果

| strategy | n_retrains | cumulative_return_pct | annualized_return_pct | sharpe | max_drawdown_pct | win_rate_pct | avg_turnover | final_value | n_pool_reuses | n_pool_misses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model_pool | 28.0 | -63.659 | -33.252 | -1.58 | -66.305 | 43.74 | 0.88 | 3634062.18 | 0.0 | 26.0 |

## 觀察

- Sharpe 最高：`model_pool` (-1.580)；最低：`model_pool` (-1.580)。
- 重訓次數範圍：28 → 28。

## 延伸分析
詳細的 regime-stratified 分析、paired t-test、drift 指標疊圖請見 
`notebooks/03_adaptation_evaluation.py`。
