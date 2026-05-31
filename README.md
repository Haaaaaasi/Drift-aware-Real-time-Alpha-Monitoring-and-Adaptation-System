# Drift-aware Real-time Alpha Monitoring and Adaptation System

DARAMS 是一個研究 alpha 在非平穩市場中如何退化、如何被監控、以及 adaptation 是否能改善績效的量化研究系統。正式研究主線使用 TEJ survivorship-correct 台股資料、Python pandas 版 WorldQuant 101 Alpha、XGBoost meta model、成本感知 portfolio / execution，以及 Data / Alpha / Model / Strategy 四層 monitoring。

## 目前主線

| 項目 | 狀態 |
|---|---|
| 正式資料 | `data/tw_stocks_tej.parquet`，2018-01 至 2026-04，1105 檔，含 51 檔期間下市股 |
| Alpha engine | `src/alpha_engine/wq101_python.py` 為預設；DolphinDB 保留為 real mode / streaming 備援 |
| Effective alphas | `reports/alpha_ic_analysis/effective_alphas.json`，TEJ IS-only 64 / 101 |
| 短期防守 universe | `--exclude-indclass-cap-alphas` 後剩 55 個純量價 alpha |
| Execution | `--execution-price next_open` / `next_vwap` 為正式 rerun；`close` 僅保留 legacy proxy |

## 2026-05-11 rerun 結論

報告：`reports/adaptation_ab/no_indcap_execution_rerun_summary_20260511.md`

正式 OOS（2024-07-01 → 2026-04-30）：

| Execution | none Cum % | scheduled_20 Cum % | benchmark Cum % |
|---|---:|---:|---:|
| next_open | -18.945 | 1.377 | 10.618 |
| next_vwap | -21.758 | 0.761 | 4.178 |

Legacy comparability（2022-06-01 → 2024-12-31）：

| Execution | none Cum % | scheduled_20 Cum % | benchmark Cum % |
|---|---:|---:|---:|
| next_open | -1.439 | -9.414 | 47.882 |
| next_vwap | -0.644 | -4.983 | 38.505 |

結論：`scheduled_20` 在正式 OOS 中優於 `none`，但輸給等權 buy-and-hold benchmark；在 legacy comparability window 中甚至輸給 `none`。因此不能再把 `scheduled_20` 包裝為主策略勝利。較可防守的敘事是：adaptation 可降低 no-adapt 退化，但簡單 scheduled retrain 尚不足以勝過市場基準。

## 近期優先事項

1. 檢查 55-alpha universe 的 alpha-level OOS IC 與 signal rank 分布。
2. 降低 turnover：測 `turnover_aware_topk`、較寬 entry/exit buffer、較長 rebalance interval。
3. 補真實 `indclass` / `cap` 後重跑 structural-alpha 對照。
4. 報告主軸改為 benchmark-aware，不只和 `none` 比。
5. `model_pool` 目前不宣稱完整 recurring concept reuse；等 base execution 與 portfolio mapping 穩定後再處理。

## 常用 rerun

```powershell
python -m pipelines.ab_experiment `
  --data-source tej `
  --start 2024-07-01 `
  --end 2026-04-30 `
  --strategies none scheduled_20 `
  --top-k 10 `
  --rebalance-every 10 `
  --train-window-days 500 `
  --benchmark ew_buy_hold_universe `
  --exclude-indclass-cap-alphas `
  --execution-price next_open `
  --run-tag oos_no_indcap_nextopen
```

## 文件入口

- [架構與方法學狀態](docs/architecture.md)
- [資料庫 schema](docs/database_schema.md)
- [名詞表](docs/glossary.md)
- [Grafana 設定](docs/grafana_setup.md)

## 測試

```powershell
pytest -q
```

Codex 沙盒內若 `.venv\Scripts\python.exe` 因 Windows ACL 無法啟動，請依 memory 記錄改用非沙盒權限執行專案虛擬環境，不要改用本機 Anaconda。
