# yfinance guardrails（2026-05-04）

## 背景

2026-05-03 已確認 `data/tw_stocks_ohlcv.csv` 的 yfinance 資料存在 stock 8476 split-adjustment 污染，會把舊 WP9 / Phase A / Phase B 回測的高累積報酬放大成假訊號。2026-05-04 將 yfinance 從「可選資料源」降級為「必須明確解鎖的 demo / 反例資料源」。

## 已完成的防線

1. 所有正式 pipeline 預設資料源統一為 `tej`。
2. 新增 `src/config/data_sources.py`，已知 yfinance 路徑 `data/tw_stocks_ohlcv.csv` 預設會被 `assert_yfinance_allowed()` 擋下；只有傳入 `allow_yfinance=True` 或 CLI 加 `--allow-yfinance` 才能使用。
3. `predict_next_day`、`simulate_recent`、`ab_experiment`、`daily_batch_pipeline`、`adaptation_pipeline`、`replay_pipeline`、`scripts/generate_report.py` 都已接上 `--allow-yfinance`。
4. alpha cache 依資料源分流：
   - TEJ：`data/alpha_cache/wq101_alphas.parquet`
   - yfinance demo：`data/alpha_cache/wq101_alphas_csv.parquet`
5. `src/alpha_engine/alpha_cache.py` 新增 sidecar manifest：
   - `<cache>.manifest.json`
   - 記錄 `data_source`、`alpha_engine`、rows、股票數、alpha 數、日期範圍
   - 讀取時可用 `expected_data_source` 強制驗證，避免手動替換檔案後污染
6. 現有 TEJ cache 已驗證並補寫 manifest：
   - rows：200,032,808
   - securities：1105
   - alphas：101
   - period：2018-01-02 → 2026-04-30
7. 新增 `src/config/alpha_selection.py`，正式 effective alpha 清單唯一入口為 `reports/alpha_ic_analysis/effective_alphas.json`；不再默默 fallback 到舊 yfinance / DolphinDB 時代的 `V3_EFFECTIVE_ALPHA_IDS`。

## 後續代理人注意

- 不要直接使用 `data/tw_stocks_ohlcv.csv` 跑正式研究、報告、論文圖或 reviewer-facing 結論。
- 若真的要重現舊 artifact 或做資料品質反例，命令必須明確加 `--allow-yfinance`。
- 若 alpha cache 缺 manifest 或 manifest 的 `data_source` 不符，應視為 cache 污染風險，不要直接信任。
- `V3_EFFECTIVE_ALPHA_IDS` 只保留歷史相容；正式流程請讀 `reports/alpha_ic_analysis/effective_alphas.json`。
