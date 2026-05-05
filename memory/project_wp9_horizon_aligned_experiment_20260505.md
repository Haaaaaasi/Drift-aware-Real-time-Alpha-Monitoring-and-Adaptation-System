name: WP9 5 日 horizon 對齊實驗
description: scheduled_20 + 10 日換倉在 TEJ 正式成本下首次略勝 buy-and-hold
date: 2026-05-05

# WP9 5 日 Horizon 對齊實驗

## 背景

WP9 signal diagnostics 顯示 XGBoost daily top-k 有 gross edge，但 turnover 約 0.87，成本後失效。本輪改測「5 日 forward-return 目標」與「5/10 日持有週期」對齊，並用 TEJ survivorship-correct universe、TEJ effective 64 alphas、真實台股成本評估。

## 已執行輸出

- 設計與結果摘要：`reports/adaptation_ab/wp9_horizon_aligned_experiment_plan_20260505.md`
- `horizon5_reb5_none_baseline`：`reports/adaptation_ab/ab_20220601_20241231_top10_horizon5_reb5_none_baseline/`
- `horizon5_reb5_none_tw500`：`reports/adaptation_ab/ab_20220601_20241231_top10_horizon5_reb5_none_tw500/`
- `horizon5_reb5_adapt_tw500`：`reports/adaptation_ab/ab_20220601_20241231_top10_horizon5_reb5_adapt_tw500/`
- `horizon5_reb10_sched20_tw500`：`reports/adaptation_ab/ab_20220601_20241231_top10_horizon5_reb10_sched20_tw500/`

## 主要結果

1. `none + rebalance_every=5` 失敗：
   - expanding train：cum -14.884%，gross 9.135 bps/day，cost 10.790 bps/day
   - rolling 500 train：cum -31.629%，gross 5.290 bps/day，cost 10.398 bps/day

2. `scheduled_20 + rebalance_every=5 + rolling500` 轉正但不夠：
   - cum +9.747%
   - gross 13.673 bps/day
   - cost 10.725 bps/day
   - net 2.948 bps/day
   - buy-and-hold benchmark +43.612%

3. `scheduled_20 + rebalance_every=10 + rolling500` 是目前第一個可防守候選：
   - cum +47.679%，略勝 buy-and-hold +43.612%
   - gross 13.209 bps/day
   - cost 5.633 bps/day
   - net 7.576 bps/day
   - turnover 0.099
   - Sharpe 0.721，仍低於 benchmark 1.054
   - max drawdown -22.972%，仍差於 benchmark -13.907%

4. 從已完成 daily PnL 重算 cost sensitivity：
   - 0.0% round-trip：+110.816%
   - 0.2% round-trip：+86.059%
   - 0.4% round-trip：+64.169%
   - 0.6% round-trip：+44.818%
   - 0.6% 下仍略高於 benchmark +43.612%，但 margin 很薄。

## 研究判讀

目前可以說：

> TEJ WQ101 + XGBoost 存在 gross signal；daily top-k 因 turnover 過高失效。當持股週期延長到 10 日換倉，並用 scheduled_20 維持模型新鮮度後，策略首次在真實成本下略勝 buy-and-hold，但風險調整後仍不如 benchmark。

不應說：

- alpha 完全太弱。
- turnover-aware residual 長尾版本已解決問題。
- adaptation 已明顯戰勝 benchmark。現在只是 cumulative return 略勝，Sharpe/MaxDD 仍落後。

## 下一步

- 優先測 `scheduled_20 + rebalance_every=10 + top_k=20/30`，看能否降低 max drawdown。
- 測 `rebalance_every=15/20`，看是否能維持 net edge 並進一步降成本。
- 若候選仍勝 benchmark，再跑正式五策略與正式 cost sweep。
## 2026-05-05 小矩陣補跑

輸出摘要：`reports/adaptation_ab/wp9_horizon_aligned_matrix_20260505.md`

本輪固定 TEJ、2022-06-01 至 2024-12-31、rolling 500 calendar days、正式 baseline cost、`equal_weight_topk`，補跑：

- `top_k=20 / rebalance_every=15 / scheduled_20`：cum +41.423%，Sharpe 0.747，max DD -22.106%，turnover 0.0649。
- `top_k=20 / rebalance_every=20 / scheduled_20`：cum +34.456%，Sharpe 0.632，max DD -22.739%，turnover 0.0510。
- `top_k=30 / rebalance_every=15 / scheduled_20`：cum +27.660%，Sharpe 0.593，max DD -21.175%，turnover 0.0642。
- `top_k=30 / rebalance_every=20 / scheduled_20`：cum +32.671%，Sharpe 0.645，max DD -21.090%，turnover 0.0504。
- `top_k=10 / rebalance_every=10 / model_pool`：cum -1.248%，Sharpe 0.109，max DD -34.157%，turnover 0.0968，`n_pool_reuses=0`、`n_pool_misses=23`。

判讀：

1. `scheduled_20 + top_k=10 + rebalance_every=10` 仍是唯一累積報酬勝過 `ew_buy_hold_universe` 的正式 TEJ 成本候選。
2. top20/top30 沒有改善，反而稀釋 gross edge；目前訊號主要集中在前 10 名。
3. rebalance 15/20 雖降低 turnover，但 gross return 下降更多，因此 net edge 未提升。
4. `model_pool` 本輪不能解讀為 recurring concept reuse 失敗，因 PostgreSQL registry 連線失敗，pool reuse 實際為 0；需先補 offline filesystem 或 in-memory pool backend 才能正式評估 WP11。
