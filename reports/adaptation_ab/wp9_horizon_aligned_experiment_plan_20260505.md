# WP9 5 日 Horizon 對齊實驗設計與初步結果

日期：2026-05-05

## 目的

前一輪 TEJ 訊號診斷顯示，XGBoost meta signal 在 zero-cost 下仍有 gross edge，但 daily top-k turnover 過高，真實台股成本會吃掉 edge。因此本輪實驗不再從 `turnover_aware_topk` 長尾 residual 版本開始，而改測「5 日 forward-return 目標」與「5/10 日持有週期」是否更一致。

核心問題：

1. 5 日持有是否能把 turnover 降到成本可承受區間？
2. scheduled adaptation 是否能恢復 gross edge？
3. 真實成本與 0.6% round-trip 壓力下，是否能勝過 TEJ 等權 buy-and-hold benchmark？

## 實驗矩陣

### 已執行

| Run tag | Strategy | Train window | Rebalance every | Top K | Portfolio | Benchmark |
|---|---|---:|---:|---:|---|---|
| `horizon5_reb5_none_baseline` | none | expanding | 5 | 10 | equal weight top-k | ew buy-hold |
| `horizon5_reb5_none_tw500` | none | 500 calendar days | 5 | 10 | equal weight top-k | ew buy-hold |
| `horizon5_reb5_adapt_tw500` | scheduled_20 / scheduled_60 / triggered | 500 calendar days | 5 | 10 | equal weight top-k | ew buy-hold |
| `horizon5_reb10_sched20_tw500` | scheduled_20 | 500 calendar days | 10 | 10 | equal weight top-k | ew buy-hold |

### 下一步候選

| Priority | Candidate | Purpose |
|---|---|---|
| P0 | `scheduled_20`, rebalance_every=10, cost sweep formal rerun | 確認最佳候選在正式 cost sweep 報告中穩健 |
| P1 | `scheduled_20`, rebalance_every=10, top_k=20/30 | 測試分散化是否降低 drawdown 並保留 net edge |
| P1 | `scheduled_20`, rebalance_every=15/20 | 測試成本更低時是否犧牲過多 gross |
| P2 | `model_pool`, rebalance_every=10 | 檢查 recurring concept reuse 是否在成本可行設定下勝出 |
| P2 | residual-aware turnover-aware redesign | 重新設計 residual sleeve / cash / liquidation，再與 rebalance_every=10 比較 |

## 初步結果

### 1. 5 日持有但不 adaptation：失敗

`horizon5_reb5_none_baseline`：

- Gross：9.135 bps/day
- Cost：10.790 bps/day
- Net：-1.654 bps/day
- Cumulative return：-14.884%
- Benchmark：+43.612%

`horizon5_reb5_none_tw500`：

- Gross：5.290 bps/day
- Cost：10.398 bps/day
- Net：-5.108 bps/day
- Cumulative return：-31.629%

解讀：單次訓練後凍結模型不足以穩定吃到 5 日 edge。

### 2. 5 日持有 + scheduled_20：轉正但不夠

`horizon5_reb5_adapt_tw500`：

| Strategy | Cum Ret | Gross bps/day | Cost bps/day | Net bps/day | Turnover |
|---|---:|---:|---:|---:|---:|
| scheduled_20 | 9.747% | 13.673 | 10.725 | 2.948 | 0.193 |
| scheduled_60 | -26.636% | 7.296 | 10.864 | -3.568 | 0.191 |
| triggered | -26.917% | 6.753 | 10.567 | -3.814 | 0.190 |
| ew_buy_hold_universe | 43.612% | 6.196 | 0.023 | 6.173 | 0.002 |

解讀：scheduled_20 恢復 gross edge，但 5 日換倉的成本仍偏高。

### 3. 10 日換倉 + scheduled_20：第一個可防守候選

`horizon5_reb10_sched20_tw500`：

- Gross：13.209 bps/day
- Cost：5.633 bps/day
- Net：7.576 bps/day
- Turnover：0.099
- Cumulative return：47.679%
- Benchmark cumulative return：43.612%
- Sharpe：0.721（benchmark 1.054）
- Max drawdown：-22.972%（benchmark -13.907%）

解讀：這是目前第一個在正式 TEJ、真實成本、survivorship-correct universe 下，net cumulative return 勝過 buy-and-hold 的策略候選。但它的 risk-adjusted quality 仍不如 benchmark。

### 4. 由已完成 daily PnL 重算的成本敏感度

因成本不會回饋進選股與持倉，本次用已完成 `daily_pnl.csv` 直接重算 round-trip cost sensitivity。

| Round-trip cost | Cum Ret | Gross bps/day | Cost bps/day | Net bps/day | Sharpe |
|---:|---:|---:|---:|---:|---:|
| 0.0% | 110.816% | 13.209 | 0.000 | 13.209 | 1.264 |
| 0.2% | 86.059% | 13.209 | 1.981 | 11.228 | 1.074 |
| 0.4% | 64.169% | 13.209 | 3.962 | 9.247 | 0.883 |
| 0.6% | 44.818% | 13.209 | 5.943 | 7.266 | 0.692 |

0.6% round-trip 下仍略高於 benchmark 的 43.612%，但 margin 很薄。

## 暫定研究結論

目前最可防守的說法不是「alpha 太弱」，而是：

> TEJ WQ101 + XGBoost 存在 gross signal；daily top-k 因 turnover 過高失效。當 portfolio horizon 對齊至 10 日換倉、scheduled_20 adaptation 維持模型新鮮度後，策略首次在真實成本下略勝 buy-and-hold，但風險調整後仍需改善。

## 後續決策

下一步不應盲目重跑完整五策略，而應先優化 `horizon5_reb10_sched20_tw500` 的風險：

1. 測 top_k=20/30 是否降低 drawdown。
2. 測 rebalance_every=15/20 是否維持 net edge。
3. 若以上仍勝 benchmark，再跑正式五策略與正式 cost sweep。
