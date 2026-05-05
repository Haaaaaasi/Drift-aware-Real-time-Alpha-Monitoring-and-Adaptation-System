name: WP9 訊號強度診斷
description: 用 TEJ effective alphas 拆解 gross edge、成本、meta model 與持股週期責任
date: 2026-05-05

# WP9 訊號強度診斷

## 背景

TEJ + 新 `effective_alphas.json` 的正式 WP9 baseline 扣成本後全策略為負，先前一度懷疑「alpha/model signal edge 太弱」。為避免直接開 Phase 1 大矩陣，新增並執行 `scripts/wp9_signal_diagnostics.py`，把以下因素拆開：

- zero-cost 原 daily top-k gross edge
- zero-cost turnover-aware portfolio gross edge
- raw/simple signed IC ensemble decile test
- XGBoost vs simple signed IC ensemble
- 5 日訓練目標 vs 1 日/5 日持有週期對齊

腳本修正過 temporal gate：XGBoost 訓練標籤必須滿足 `label_available_at <= start`，避免回測起點附近 5 日 label leakage。

## 主要輸出

- 短窗：`reports/wp9_signal_diagnostics/diag_20220601_20220831_short_smoke_v2/`
- 完整 OOS：`reports/wp9_signal_diagnostics/diag_20220601_20241231_full_oos_v2/`

完整 OOS（2022-06-01 → 2024-12-31）重點：

| Experiment | Cum Ret | Avg Gross bps/day | Sharpe | Avg Turnover | Avg Holdings |
|---|---:|---:|---:|---:|---:|
| simple_daily_topk_zero_cost | 18.48% | 3.85 | 0.40 | 0.783 | 10.0 |
| simple_turnover_aware_zero_cost | 61.90% | 8.42 | 1.08 | 0.052 | 150.4 |
| xgb_daily_topk_zero_cost | 351.16% | 24.93 | 2.77 | 0.875 | 9.1 |
| xgb_turnover_aware_zero_cost | 62.05% | 8.24 | 1.22 | 0.070 | 69.9 |
| simple_rebalance5_hold5_zero_cost | 40.87% | 6.53 | 0.70 | 0.188 | 10.0 |
| xgb_rebalance5_hold5_zero_cost | 109.81% | 12.65 | 1.50 | 0.200 | 9.5 |

IC by horizon：

- Simple signed IC ensemble：rank IC 約 0.060 / 0.050 / 0.037（1/5/10 日）
- XGBoost：rank IC 約 0.013 / 0.012 / 0.016（1/5/10 日），但 top-k forward mean bps 明顯較高（5 日約 64 bps）

XGBoost 5 日 decile test 大致單調：decile 1 約 27.0 bps、decile 10 約 39.3 bps。Simple ensemble 的 decile 不單調，雖有 rank IC，但直接 top-k 使用較不穩。

## 研究判讀

目前不應把「不賺錢」主要歸因為 alpha 完全太弱。更準確的結論是：

1. **gross edge 存在**：XGBoost daily top-k zero-cost 在完整 OOS 有明顯正 gross edge。
2. **成本是主要破壞因子之一**：daily top-k turnover 約 0.875，若套台股真實成本，約 25 bps/day 的 gross edge 仍容易被 45-50 bps/day 成本吃掉。
3. **5 日 horizon 對齊很重要**：XGBoost 5 日持有 zero-cost 仍有正報酬（avg gross 約 12.65 bps/day、turnover 約 0.20），比 daily top-k 更接近成本可行區。
4. **turnover-aware_topk 目前不是乾淨 top-10**：`max_turnover` 會留下大量 residual holdings（完整 OOS avg holdings 約 70～150），所以它降 turnover 的代價是多頭長尾倉位與訊號稀釋。正式 Phase 1 前要重設 residual sleeve / cash / liquidation 規則。
5. **XGBoost 優於 naive simple ensemble 的 top-k extraction**：simple ensemble 有 rank IC，但 top decile 不夠單調；XGBoost 對 top-k tail 的排序更有用。

## 下一步

- 不要再用「alpha 太弱」作為主要結論；改成「TEJ WQ101 + XGBoost 有 gross signal，但 naive daily long-only top-k 需要過高 turnover，成本後失效」。
- Phase 1 portfolio grid 應優先測：
  - `rebalance_every=5`
  - 無 turnover cap 或較寬 cap 的 5 日持有 top-k
  - 明確 cash / residual liquidation 的 turnover-aware 版本
- 在正式實驗報告中固定輸出 gross return、cost drag、net return、turnover、avg holdings，避免只看 Sharpe。
