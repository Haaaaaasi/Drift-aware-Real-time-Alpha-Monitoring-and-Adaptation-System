# WP9 5 日 Horizon 對齊小矩陣實驗

日期：2026-05-05

## 目的

延續 `horizon5_reb10_sched20_tw500` 的初步勝出結果，檢查這個改善是否來自穩健的「5 日預測目標與持股週期對齊」，或只是 `top_k=10 / rebalance_every=10` 的偶然組合。

本輪固定條件：

- 資料來源：TEJ survivorship-correct universe
- alpha 清單：`reports/alpha_ic_analysis/effective_alphas.json`
- 期間：2022-06-01 至 2024-12-31
- 模型：XGBoost regression
- 訓練窗：rolling 500 calendar days
- Portfolio：`equal_weight_topk`
- 成本：正式 baseline cost
- Benchmark：`ew_buy_hold_universe`

## 實驗矩陣

| Run tag | Strategy | Top K | Rebalance every | Cum Ret | Sharpe | Max DD | Avg turnover | Gross bps/day | Cost bps/day | Net bps/day |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `horizon5_reb10_sched20_tw500` | scheduled_20 | 10 | 10 | **47.679%** | 0.721 | -22.972% | 0.0990 | 13.209 | 5.633 | **7.576** |
| `horizon5_top20_reb15_sched20_tw500` | scheduled_20 | 20 | 15 | 41.423% | **0.747** | -22.106% | 0.0649 | 9.780 | 3.347 | 6.432 |
| `horizon5_top20_reb20_sched20_tw500` | scheduled_20 | 20 | 20 | 34.456% | 0.632 | -22.739% | 0.0510 | 8.562 | 2.823 | 5.738 |
| `horizon5_top30_reb15_sched20_tw500` | scheduled_20 | 30 | 15 | 27.660% | 0.593 | -21.175% | 0.0642 | 7.957 | 3.309 | 4.649 |
| `horizon5_top30_reb20_sched20_tw500` | scheduled_20 | 30 | 20 | 32.671% | 0.645 | **-21.090%** | 0.0504 | 8.140 | 2.786 | 5.354 |
| `horizon5_top10_reb10_model_pool_tw500` | model_pool | 10 | 10 | -1.248% | 0.109 | -34.157% | 0.0968 | 6.467 | 5.355 | 1.112 |
| `ew_buy_hold_universe` | benchmark | universe | buy-hold | 43.612% | **1.054** | **-13.907%** | 0.0016 | 6.196 | 0.023 | 6.173 |

## 主要結論

1. `top_k=10 / rebalance_every=10 / scheduled_20` 仍是目前唯一在正式 TEJ 成本下累積報酬勝過 buy-and-hold 的候選。
2. 擴大到 `top_k=20/30` 沒有改善總績效，反而稀釋 gross edge。這表示目前 XGBoost 排名前 10 的集中訊號比更分散的 top-k sleeve 更有效。
3. `rebalance_every=15/20` 雖能降低 turnover 與 cost drag，但 gross return 同步下降，net edge 沒有因更低交易頻率而增加。
4. `model_pool` 本輪不能視為 recurring concept reuse 的正式失敗。實驗期間 PostgreSQL registry 連線失敗，`n_pool_reuses=0`、`n_pool_misses=23`，實際上接近「每次 shadow 選新模型」而不是 reusable concept pool。

## 研究判讀

目前 evidence 支持較窄的說法：

> TEJ WQ101 + XGBoost 在 OOS 上有可被 portfolio layer 保留的 gross signal，但 edge 主要集中於 top-10；將持股週期拉到 10 個交易日可以讓成本下降到足以略勝 buy-and-hold 的累積報酬，但風險調整後仍不如 benchmark。

不應過度主張：

- 不能說 `model_pool` 失敗，因為 offline pool backend 沒有真正可重用。
- 不能說低 turnover 單調有效，因為 15/20 日再平衡的 gross edge 明顯下降。
- 不能說 top-k 分散有效，因為 top20/top30 皆弱於 top10。

## 下一步

1. 先固定 `top_k=10 / rebalance_every=10`，比較 `scheduled_20`、`scheduled_60`、`triggered`、`none`，確認勝出來自 scheduled_20 還是 horizon 對齊本身。
2. 修正 offline `model_pool` backend：沒有 PostgreSQL 時應使用明確的 filesystem 或 in-memory pool，並在 summary 裡標示 backend，否則 WP11 的 recurring reuse 無法被正確評估。
3. 對最佳候選做正式 cost sweep 與 paired test；目前 0.6% round-trip 的 margin 只有約 1.2 個百分點，不足以作為強結論。
