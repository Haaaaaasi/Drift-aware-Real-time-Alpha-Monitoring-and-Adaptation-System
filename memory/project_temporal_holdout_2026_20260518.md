---
name: 2026 temporal holdout replay 2026-05-18
description: Phase A 用 2024-07-01 至 2025-12-31 重新做 rolling_topk 選參，再用 2026-01-01 至 2026-04-30 做 temporal holdout replay。
type: project
---

## 背景

使用者指出目前高績效代表性不足，因為 selector / portfolio 參數是在 `2024-07-01` 到 `2026-04-30` 附近挑出來的。為降低疑慮，本輪執行 temporal holdout replay：

- Phase A calibration：`2024-07-01` 至 `2025-12-31`
- Phase B replay holdout：`2026-01-01` 至 `2026-04-30`
- Phase A 只看 `next_vwap`
- 選參規則預先固定為：Sharpe 由高到低，其次 max drawdown 較淺者，其次 cumulative return 較高者
- 這不是 untouched holdout，因為研究過程已經看過 2026 YTD；正式稱為 temporal holdout replay

新增 workflow：

- `scripts/run_temporal_holdout_2026_workflow.py`
- 輸出：`reports/adaptation_ab/temporal_holdout_2026_20260518/`

## Phase A 結果

3x3 matrix：`selector_alpha_top_k ∈ {20,30,40}` × `selector_window_days ∈ {126,252,504}`，`stability_penalty=0.10`。

| Rank | top_k | window | Cum Ret % | Sharpe | Max DD % |
|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 126 | 20.371 | 0.666 | -30.373 |
| 2 | 20 | 252 | 8.759 | 0.379 | -31.509 |
| 3 | 30 | 126 | 6.943 | 0.314 | -36.332 |
| 4 | 20 | 504 | 3.198 | 0.214 | -38.296 |
| 5 | 40 | 504 | -5.406 | -0.042 | -38.609 |
| 6 | 40 | 252 | -7.266 | -0.095 | -44.120 |
| 7 | 30 | 504 | -7.616 | -0.112 | -39.128 |
| 8 | 30 | 252 | -9.746 | -0.196 | -39.560 |
| 9 | 40 | 126 | -12.804 | -0.302 | -39.852 |

選出的 config 仍是 `rolling_topk20_w126_pen10`，與目前 P0 incumbent 相同。這是強正面訊號：不是靠 2026 YTD 才挑出該 config。

## Phase B 結果

`2026-01-01` 至 `2026-04-30`，使用 Phase A 選出的 `rolling_topk20_w126_pen10`，不看 2026 調參。

| Execution | Series | Cum Ret % | Sharpe | Max DD % |
|---|---|---:|---:|---:|
| next_vwap | selected_rtop20_w126_pen10 | 38.885 | 3.706 | -12.674 |
| next_vwap | static_is_scheduled_20 | 30.717 | 2.781 | -10.201 |
| next_vwap | ew_same_cadence_liq100m | 22.330 | 2.855 | -9.501 |
| next_vwap | ew_same_cadence_liq200m | 26.273 | 3.094 | -10.062 |
| next_open | selected_rtop20_w126_pen10 | 39.629 | 3.473 | -12.421 |
| next_open | static_is_scheduled_20 | 29.898 | 2.759 | -10.661 |
| next_open | ew_same_cadence_liq100m | 23.727 | 2.794 | -10.598 |
| next_open | ew_same_cadence_liq200m | 27.791 | 2.998 | -11.225 |

## 統計解讀

Phase B 只有 75 trading days，因此統計力有限。

- `next_vwap` vs liq100m EW：mean excess +17.675 bps/day，paired p=0.035，block p=0.043，5% block CI 為 +0.918 bps/day，通過 5% 單尾。
- `next_vwap` vs liq200m EW：mean excess +13.262 bps/day，paired p=0.073，block p=0.090，未過 5%。
- `next_vwap` vs static_is：mean excess +7.658 bps/day，paired p=0.256，block p=0.334，未過 5%。
- `next_open` 方向一致，但對 static / liq200m 也未過 5%；對 liq100m block p 約 0.050，邊界。

## 結論

本輪結果支持「目前高績效不是單純把 2026 放進選參才得到」，因為截至 `2025-12-31` 的 Phase A 自動選參仍選出 `rtop20_w126_pen10`。

但仍不能把 2026 replay 稱為 untouched holdout，也不能過度宣稱統計顯著勝過所有 benchmark。比較合適的正式說法是：

> As-of 2025-12-31 的 temporal replay 選出與目前相同的 selector config，且 2026 Jan-Apr 方向上優於 static 與 liquidity-filtered EW；主結果對 liq100m EW 通過 5% paired / block bootstrap，但對 static 與 liq200m 的統計顯著性不足。真正最終驗證仍需 2026-05 之後 prospective holdout。

## Artifact

- `reports/adaptation_ab/temporal_holdout_2026_20260518/temporal_holdout_summary.md`
- `reports/adaptation_ab/temporal_holdout_2026_20260518/phase_a_matrix_summary.csv`
- `reports/adaptation_ab/temporal_holdout_2026_20260518/selected_config.json`
- `reports/adaptation_ab/temporal_holdout_2026_20260518/phase_b_summary.csv`
- `reports/adaptation_ab/temporal_holdout_2026_20260518/phase_b_bootstrap.csv`
- `scripts/run_temporal_holdout_2026_workflow.py`
