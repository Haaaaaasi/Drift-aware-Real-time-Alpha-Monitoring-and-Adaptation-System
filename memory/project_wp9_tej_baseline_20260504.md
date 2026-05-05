---
name: WP9 TEJ baseline 重跑啟動
description: 2026-05-04 用 TEJ + 新 effective_alphas.json 重跑 WP9 baseline；五策略皆為負績效，舊 yfinance 高報酬結論不成立
type: project
originSessionId: codex-2026-05-04
---

2026-05-04 已開始重跑 WP9，設定為 TEJ survivorship-correct data + `reports/alpha_ic_analysis/effective_alphas.json`（TEJ IS-only 64 alphas）。

## 指令

```powershell
.\.venv\Scripts\python.exe -m pipelines.ab_experiment --data-source tej --start 2022-06-01 --end 2024-12-31 --top-k 10 --run-tag tej_effective64_baseline
```

第一次整包 run 因工具 timeout 後無法正常完成 `model_pool` 彙整；四個策略已完成。後續單獨補跑：

```powershell
.\.venv\Scripts\python.exe -m pipelines.ab_experiment --data-source tej --start 2022-06-01 --end 2024-12-31 --top-k 10 --strategies model_pool --run-tag tej_effective64_model_pool_retry
```

## Baseline consolidated result

位置：`reports/adaptation_ab/ab_20220601_20241231_top10_tej_effective64_baseline/comparison_consolidated.csv`

| strategy | Sharpe | Cum Ret % | n_retrains |
| --- | ---: | ---: | ---: |
| model_pool | -1.580 | -63.659 | 28 |
| scheduled_60 | -2.099 | -72.095 | 11 |
| triggered | -2.285 | -74.561 | 29 |
| none | -2.458 | -74.384 | 1 |
| scheduled_20 | -2.820 | -80.859 | 32 |

## 研究意義

- 舊 yfinance Phase A 的超高累積報酬已確認不是正式研究結論；TEJ baseline 下五策略全負。
- TEJ baseline 的暫定排序是 `model_pool > scheduled_60 > triggered > none > scheduled_20`。
- 這只是 baseline cost setting；正式 WP9 結論仍需補 TEJ cost sweep（0 / 0.2 / 0.4 / 0.6 round-trip cost）與 `notebooks/03_adaptation_evaluation.py`。

## 注意

PostgreSQL 5433 未啟動時，model_pool 會記錄 registry warning 並使用本地 fallback；本次 simulation 已成功輸出，但 `n_pool_reuses=0`、`n_pool_misses=26`，代表 recurring concept reuse 尚未在此 baseline 發揮。
