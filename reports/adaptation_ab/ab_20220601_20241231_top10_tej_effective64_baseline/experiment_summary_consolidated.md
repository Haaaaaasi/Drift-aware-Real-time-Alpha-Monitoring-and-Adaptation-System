# WP9 TEJ baseline consolidated（2026-05-04）

## 設定

- 期間：2022-06-01 至 2024-12-31
- 資料源：`data/tw_stocks_tej.parquet`
- Alpha universe：`reports/alpha_ic_analysis/effective_alphas.json`，TEJ IS-only 64 alphas
- Alpha source：`python`
- Top-K：10
- 成本：commission 0.0926%/side，tax 0.300%/sell-side，slippage 5 bps/side
- Trigger window：`[t-60, t-20]` calendar days
- Shadow warmup：5 days

## 執行說明

原始五策略 baseline 指令：

```powershell
.\.venv\Scripts\python.exe -m pipelines.ab_experiment --data-source tej --start 2022-06-01 --end 2024-12-31 --top-k 10 --run-tag tej_effective64_baseline
```

該 run 在工具 timeout 後仍完成 `none`、`scheduled_20`、`scheduled_60`、`triggered` 四個策略，但 `model_pool` 子目錄未寫出 summary。為避免重跑已完成策略，後續單獨補跑：

```powershell
.\.venv\Scripts\python.exe -m pipelines.ab_experiment --data-source tej --start 2022-06-01 --end 2024-12-31 --top-k 10 --strategies model_pool --run-tag tej_effective64_model_pool_retry
```

`model_pool` 補跑成功；PostgreSQL 5433 未啟動只造成 model registry warning，流程使用本地 fallback 繼續，不影響本次 simulation 輸出。

## Baseline 結果

| strategy | Sharpe | Cum Ret % | Ann Ret % | Max DD % | n_retrains |
| --- | ---: | ---: | ---: | ---: | ---: |
| model_pool | -1.580 | -63.659 | -33.252 | -66.305 | 28 |
| scheduled_60 | -2.099 | -72.095 | -39.934 | -73.744 | 11 |
| triggered | -2.285 | -74.561 | -42.114 | -75.741 | 29 |
| none | -2.458 | -74.384 | -41.953 | -74.216 | 1 |
| scheduled_20 | -2.820 | -80.859 | -48.330 | -80.967 | 32 |

## 初步解讀

TEJ + 新 effective alpha 清單下，舊 yfinance Phase A 的高報酬結論完全消失；五個策略全為負 Sharpe 與負累積報酬。baseline 排序暫時變成 `model_pool > scheduled_60 > triggered > none > scheduled_20`，但仍需跑 cost sweep 與 paired test 後才能寫成正式 WP9 結論。

## 待補

- 重新跑 0 / 0.2 / 0.4 / 0.6 round-trip cost sweep。
- 用 `notebooks/03_adaptation_evaluation.py` 對 TEJ run 做 paired test 與成本敏感度圖。
