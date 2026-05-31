---
name: Final robustness bundle 2026-05-18
description: 將 incumbent_55 + rolling_topk20_w126_pen10 + scheduled_20 收束為 P0 正式主線，並把 alpha expansion 與 model_pool closure 納入負面證據。
type: project
---

## 結論

2026-05-18 建立 `reports/adaptation_ab/final_robustness_20260518/`，正式把 P0 主線鎖定為：

```text
incumbent_55 + rolling_topk20_w126_pen10 + scheduled_20
```

`next_vwap` 是主結果，`next_open` 是支持結果。`2024-07-01` 到 `2026-04-30` 只能稱為 frozen validation，不可稱為 untouched holdout；真正 forward holdout 必須等 2026-05 之後新增 TEJ 資料。

## 主要數字

| Execution | rolling_topk20 Cum Ret % | Sharpe | Max DD % | static_is Cum Ret % | model_pool Cum Ret % |
|---|---:|---:|---:|---:|---:|
| next_vwap | 62.120 | 1.298 | -30.373 | 22.337 | 14.286 |
| next_open | 76.252 | 1.385 | -25.615 | 36.606 | 27.838 |

`next_vwap` 對 static selector、liq100m EW、liq200m EW 的 paired 與 block bootstrap 都通過 5% 單尾檢定。`next_open` 方向一致，但 static / liq200m 的 paired p 較弱，正式 claim 要保守。

## 負面證據

- all_valid_82 明顯輸 incumbent_55：next_vwap +6.717% / Sharpe 0.280；next_open +16.319% / Sharpe 0.484。
- admission gate best next_vwap +14.693% / Sharpe 0.447，仍遠輸 incumbent_55。
- admitted alpha failure attribution：23 個 admission periods 全部有 quarantine alpha 進入，negative excess rate 69.6%，平均 excess -7.816 bps/day。
- model_pool 在 frozen selector 下能 reuse，但輸給 scheduled_20：next_vwap +14.286% vs +62.120%；next_open +27.838% vs +76.252%。因此 model_pool 收斂為 failure / ablation appendix。

## Artifact

- `reports/adaptation_ab/final_robustness_20260518/final_robustness_summary.md`
- `reports/adaptation_ab/final_robustness_20260518/manifest.json`
- `reports/adaptation_ab/final_robustness_20260518/grafana_tables.json`
- `docs/final_robustness_holdout_protocol.md`
- `dashboards/final_robustness.json`
- `migrations/003_final_robustness_reporting.sql`
- `scripts/ingest_final_robustness_bundle.py`

## Grafana integration 2026-05-18

新增 final robustness 專用 Grafana reporting schema 與 dashboard：

- `migrations/003_final_robustness_reporting.sql`
- `scripts/ingest_final_robustness_bundle.py`
- `dashboards/final_robustness.json`
- `reports/adaptation_ab/final_robustness_20260518/grafana_tables.json`
- `tests/unit/test_final_robustness_ingest.py`

已在本機正在執行的 `darams-postgres` 套用 v003 migration，並成功匯入 `final_robustness_20260518`。DB 驗證行數：strategy 13、checks 10、regime 8、decisions 5、artifacts 10。Grafana API 已看到 dashboard `darams-final-robustness`，URL 為 `http://127.0.0.1:3000/d/darams-final-robustness/darams-final-robustness`。

驗證：

```text
py_compile passed
ingest_final_robustness_bundle.py --dry-run passed
tests/unit/test_final_robustness_ingest.py = 4 passed
```
