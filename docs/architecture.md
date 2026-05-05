# DARAMS 系統架構說明文件

> Drift-aware Real-time Alpha Monitoring and Adaptation System  
> 更新日期：2026-04-25

---

## 目錄

1. [系統定位](#1-系統定位)
2. [整體資料流](#2-整體資料流)
3. [十層模組職責](#3-十層模組職責)
4. [支援模組職責](#4-支援模組職責)
5. [執行模式](#5-執行模式)
6. [測試方法](#6-測試方法)
7. [目前能否上線](#7-目前能否上線)
8. [已完成里程碑](#8-已完成里程碑)
9. [下一步行動計畫](#9-下一步行動計畫)
10. [已知限制與風險](#10-已知限制與風險)

---

## 1. 系統定位

DARAMS 是一個**模組化量化研究系統**，研究核心不是「找最強策略」，而是：

- **監控**（Monitoring）：偵測 alpha 因子 / 模型 / 策略在不同市場 regime 下的退化
- **自適應**（Adaptation）：在退化發生後，用三種策略自動或半自動地恢復系統效能

使用 WorldQuant 101 Alpha（WQ101）作為現成的 alpha feature universe，以 DolphinDB 計算因子，以 Python 完成後端所有研究邏輯。

### 三大研究問題

1. WorldQuant 101 Alpha 因子在不同市場 regime 下能否穩定產生訊號？
2. 如何系統化監控 alpha / model / strategy 的退化現象？
3. Adaptation（scheduled / performance-triggered / recurring concept reuse）能否量化改善績效？

---

## 2. 整體資料流

```
╔══════════════════════════════════════════════════════════════════════════╗
║                         【資料層】 Layer 1-2                             ║
╠══════════════════════════════════════════════════════════════════════════╣
║  data/tw_stocks_ohlcv.csv                                                ║
║  1083 檔台股 × 2022-2026                                                 ║
║  欄位：security_id, tradetime, open, high, low, close, vol               ║
╚══════════════════════════════════╦═══════════════════════════════════════╝
                                   ║ load_csv_data()
                                   ▼
╔══════════════════════════════════════════════════════════════════════════╗
║                      【Alpha 計算層】 Layer 3                            ║
╠══════════════════════════════════════════════════════════════════════════╣
║  wq101_python.py（主路徑）/ DolphinDB alpha_batch.dos（real mode）       ║
║  ├─ 計算 101 個 WQ101 alpha（WorldQuant 2016 論文公式）                  ║
║  ├─ 輸出：long format (security_id, tradetime, alpha_id, alpha_value)    ║
║  └─ 快取：data/alpha_cache/wq101_alphas.parquet（冷啟動 5-10 分）        ║
╚══════════════════════════════════╦═══════════════════════════════════════╝
                                   ║ 101 alphas × 1083 stocks × ~1000 days
                                   ▼
╔══════════════════════════════════════════════════════════════════════════╗
║               【Alpha 篩選】 WP2（一次性離線步驟）                       ║
╠══════════════════════════════════════════════════════════════════════════╣
║  01_alpha_ic_analysis.py --train-end 2024-06-30                          ║
║  ├─ IS 期（2022-01 → 2024-06）計算 pooled Rank IC / Coverage            ║
║  ├─ 篩選：|rank_ic| >= 0.01 AND coverage >= 0.80                        ║
║  ├─ 101 alphas → 52 個通過（6 個 IS→OOS sign-flip 列入警示）            ║
║  └─ 輸出：reports/alpha_ic_analysis/effective_alphas.json               ║
╚══════════════════════════════════╦═══════════════════════════════════════╝
                                   ║ 52 個 effective alpha IDs
                         ┌─────────┴─────────┐
                         ▼                   ▼
╔═══════════════════════════════════╗   ╔══════════════════════════════════════════╗
║       【標籤生成】Layer 8         ║   ║      【XGB Meta Model 訓練】Layer 4      ║
╠═══════════════════════════════════╣   ╠══════════════════════════════════════════╣
║  LabelGenerator                   ║   ║  訓練資料：                              ║
║  forward_return(t) =              ║   ║  X：features，tradetime ≤ T − purge_gap ║
║    close[t+h]/close[t] − 1        ║   ║  y：labels，label_available_at ≤ T      ║
║                                   ║   ║  （只用已成熟的標籤，不依曆日近似）      ║
║  label_available_at               ║   ║                                          ║
║  = signal_time 後第 (h + buffer)  ║   ║  XGBRegressor（回歸）：                 ║
║    個實際 trading bar 的 ts       ║   ║  n_estimators=200, max_depth=4          ║
║    （由 market_trading_days 查找  ║   ║  subsample=0.8, colsample_bytree=0.8    ║
║     ，非曆日加法；bar 不足回 None）║   ║                                          ║
║                                   ║   ║  Purged Expanding-Window CV（3 folds）： ║
║  訓練資格：label_available_at ≤ T ║   ║  fold 間插 purge_gap 天防標籤洩漏       ║
║  purge_days 僅用於 feature-side   ║   ║  評估：holdout IC / rank_IC / dir_acc    ║
║  purge 或 CV fold 間距            ║   ║                                          ║
╚══════════════╦════════════════════╝   ╚═══════════════════════╦══════════════════╝
               ╚═══════════════════════╩═══════════════════════╝
                                   ║ trained XGBRegressor
                                   ▼
╔══════════════════════════════════════════════════════════════════════════╗
║                    【T 日截面預測】（每日執行）                           ║
╠══════════════════════════════════════════════════════════════════════════╣
║  輸入：T 日 1083 檔股票的 52 個 alpha 值（1083 × 52 矩陣）               ║
║  XGB.predict()：200 棵決策樹加總                                         ║
║  輸出 per stock：signal_score（預測 5 日報酬，僅相對排名有意義）          ║
║                  signal_direction（+1 / -1）, confidence（|score|）      ║
╚══════════════════════════════════╦═══════════════════════════════════════╝
                                   ║ 1083 stocks × signal_score
                                   ▼
╔══════════════════════════════════════════════════════════════════════════╗
║                    【組合構建】 Layer 5                                   ║
╠══════════════════════════════════════════════════════════════════════════╣
║  PortfolioConstructor(method="equal_weight_topk", top_k=10, long_only)  ║
║  ├─ 過濾 signal_direction < 0（long only）                               ║
║  ├─ 按 signal_score 降序排列                                             ║
║  └─ 取前 10 名，各給 10% 等權                                            ║
╚══════════════════════════════════╦═══════════════════════════════════════╝
                                   ║ 10 stocks × 10% weight
                                   ▼
╔══════════════════════════════════════════════════════════════════════════╗
║                      【風控層】 Layer 6                                  ║
╠══════════════════════════════════════════════════════════════════════════╣
║  RiskManager：                                                           ║
║  ├─ 單一持股上限 10%（等權已符合）                                        ║
║  ├─ 總曝險上限 100%                                                      ║
║  └─ Drawdown 熔斷：累積虧損 > 20% 停止交易                               ║
╚══════════════════════════════════╦═══════════════════════════════════════╝
                                   ║ risk-adjusted targets
                                   ▼
╔══════════════════════════════════════════════════════════════════════════╗
║                   【模擬執行】 Layer 7                                   ║
╠══════════════════════════════════════════════════════════════════════════╣
║  PaperTradingEngine（初始資金 1000 萬）                                  ║
║  T 日收盤建倉 → T+1 日收盤平倉/重平衡                                   ║
║  成本模型：                                                               ║
║  ├─ Commission（買賣雙向）：0.0926%                                      ║
║  ├─ Tax（賣出單向）：0.3%（台股證交稅）                                   ║
║  └─ Slippage：turnover × 5 bps                                          ║
║  每日 PnL：gross - commission - tax - slippage = net_return              ║
╚══════════════════════════════════╦═══════════════════════════════════════╝
                                   ║ fills, positions, daily_pnl
                                   ▼
╔══════════════════════════════════════════════════════════════════════════╗
║                    【監控層】 Layer 9（四大 Monitor）                    ║
╠══════════════════════════════════════════════════════════════════════════╣
║  ┌──────────────────────┐  ┌──────────────────────────────────────────┐ ║
║  │    DataMonitor        │  │              AlphaMonitor                │ ║
║  │  缺值率 / 價格異常    │  │  rolling IC / rank_IC（近 ic_window×50） │ ║
║  │  volume_zscore        │  │  alpha turnover（相鄰兩天排名變動率）     │ ║
║  │  PSI 分布偏移         │  │  PSI 分布偏移（vs 訓練期基準）           │ ║
║  └──────────────────────┘  │  alpha 相關性 drift（vs baseline 矩陣）  │ ║
║  ┌──────────────────────┐  └──────────────────────────────────────────┘ ║
║  │    ModelMonitor       │  ┌──────────────────────────────────────────┐ ║
║  │  預測 IC 追蹤         │  │            StrategyMonitor               │ ║
║  │  calibration error    │  │  rolling Sharpe（60 日）                 │ ║
║  │  score 分布 KS-test   │  │  Max Drawdown                            │ ║
║  └──────────────────────┘  │  實際 vs 預期報酬 / 組合換手率            │ ║
║                             └──────────────────────────────────────────┘ ║
║  Alert 門檻：WARNING / CRITICAL                                          ║
║  所有 metrics → monitoring_metrics（PostgreSQL）→ Grafana Dashboard      ║
╚══════════════════════════════════╦═══════════════════════════════════════╝
                                   ║ alerts + metrics
                                   ▼
╔══════════════════════════════════════════════════════════════════════════╗
║                   【適應層】 Layer 10（三種 Policy）                     ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Policy 1 Scheduled：每 N 個交易日強制重訓（N=20 最佳，Sharpe=3.757）   ║
║                                                                          ║
║  Policy 2 Triggered：rolling IC < 0 連 5 天 OR Sharpe < 0 連 10 天     ║
║           OR CRITICAL alerts ≥ 3 → 重訓（有冷卻期）                     ║
║                                                                          ║
║  Policy 3 Model Pool：                                                   ║
║  觸發 → 計算 regime fingerprint（5 維）→ 搜尋歷史 pool（cosine ≥ 0.8） ║
║       → Shadow 3-way compare on [t-40, t-20] 成熟標籤窗口              ║
║          A=current  B=新重訓  C=pool 歷史模型                           ║
║       → 選最高 IC 者，若選 B 再 dual-train（完整資料）                  ║
║                                                                          ║
║  重訓記錄 → model_registry → 下一個 T 日用新模型預測（回到預測步驟）    ║
╚══════════════════════════════════╦═══════════════════════════════════════╝
                                   ║
                                   ▼
╔══════════════════════════════════════════════════════════════════════════╗
║                         【最終輸出】                                     ║
╠══════════════════════════════════════════════════════════════════════════╣
║  predict_<date>.csv：rank / security_id / target_weight / target_shares  ║
║                       signal_score / confidence / signal_direction       ║
║                                                                          ║
║  daily_pnl.csv：gross / commission / tax / slippage / net_return         ║
║                 cumulative_value / rolling_ic / rolling_sharpe           ║
║                                                                          ║
║  Phase A 五策略績效（2022-06 → 2024-12，IS-selected 52 alphas，含成本）  ║
║  ┌─────────────────┬────────┬──────────┬──────────┐                     ║
║  │ Strategy        │ Sharpe │ n_retrain│ Cum Ret% │                     ║
║  ├─────────────────┼────────┼──────────┼──────────┤                     ║
║  │ scheduled_20    │  3.757 │    32    │  15754   │                     ║
║  │ triggered       │  3.630 │    12    │  15269   │                     ║
║  │ model_pool      │  3.518 │    11    │  12925   │                     ║
║  │ scheduled_60    │  3.514 │    11    │  10938   │                     ║
║  │ none（基準）    │  3.158 │     1    │   7625   │                     ║
║  └─────────────────┴────────┴──────────┴──────────┘                     ║
║  → 4 段成本敏感度（0/0.2/0.4/0.6%）下排序 100% 一致（rank_std=0）      ║
╚══════════════════════════════════════════════════════════════════════════╝

回饋迴圈：T+5 日 label 成熟 → 四大 Monitor 重新評估 → 觸發 Adaptation → 重訓 → 循環
```

---

## 3. 十層模組職責

### Layer 1 — Data Ingestion　`src/ingestion/`

| 檔案 | 職責 |
|------|------|
| `historical_loader.py` | 讀取 CSV（或 DataFrame）並寫入 PostgreSQL `raw_market_events` |
| `shioaji_stream.py` | Shioaji 即時串流接入（目前為 mock，帳號才能啟用真實連線） |
| `replay.py` | 以歷史資料模擬串流（用於離線測試） |

**輸出格式**：`raw_market_events` table（security_id, event_ts, open/high/low/close/volume/vwap, …）

---

### Layer 2 — Standardization　`src/standardization/`

| 檔案 | 職責 |
|------|------|
| `schema_mapper.py` | 將 raw_market_events 轉換為 DolphinDB panel matrix 所需欄位格式 |
| `calendar.py` | 台股交易日曆（剔除非交易日） |
| `quality_check.py` | 欄位完整性檢查，設定 `missing_flags` bitmask |

**輸出格式**：`standardized_bars` DolphinDB 分區表，欄位：`(security_id SYMBOL, tradetime TIMESTAMP, bar_type, open, high, low, close, vol, vwap, cap, indclass INT, is_tradable, missing_flags)`

> **注意**：`security_id` 必須為 STRING / SYMBOL 型別（不可為數字）；`vwap` 若缺失以 `(high+low+close)/3` 補算；`cap` 以 `close × 1,000,000` 近似。

---

### Layer 3 — Alpha Engine　`src/alpha_engine/` + `dolphindb/`

| 檔案 | 職責 |
|------|------|
| `dolphindb_client.py` | DolphinDB session 管理、模組載入、查詢封裝 |
| `batch_compute.py` | 觸發 DolphinDB `computeBatchAlphas()`，支援完整 101 個 WQ101 alpha |
| `stream_compute.py` | 串流 alpha 計算（streamEngineParser，已完整實作） |
| `alpha_registry.py` | alpha 元資料管理（版本、狀態） |
| `dolphindb/modules/alpha_batch.dos` | DolphinDB 端 WQ101 批次計算腳本（含 `preparePanels` + dispatch + `matrixToLongTable`） |

**輸出格式**：long-format `(security_id, tradetime, alpha_id, alpha_value)`。alpha_id 命名為 `wq001` … `wq101`。

**當前資料規模**：101 alphas × 556 檔台股 × 2022-2026 = **53,771,183 rows**（`dfs://darams_alpha/alpha_features`）

**wq101alpha 真實 API（已驗證）**：
- `prepare101` 模組**不存在**於 DolphinDB v2.00.18；以自訂 `preparePanels()` 取代
- 每個 WQAlpha 函式接收**各自需要的獨立 panel matrix**（非字典），例如：
  - `WQAlpha1(close)` — 單一 matrix
  - `WQAlpha2(vol, close, open)` — 三個 matrix
- 返回 `DOUBLE MATRIX`（rows = 日期數，cols = 股票數，column-major）
- 轉回 long format 使用 `flatten(mat)` + `stretch(securityIds, total)` + `take(tradeDates, total)`
- wq061、wq075、wq079、wq095 回傳 BOOL matrix，寫入前需 `double(mat)` 轉型

**DolphinDB 分區設定注意事項**：
- `standardized_bars`：VALUE(日期) × HASH(security_id, 50)，TSDB sortColumns = `security_id, tradetime`
- `alpha_features`：VALUE(日期) × HASH(security_id, 50)，TSDB sortColumns = `security_id, alpha_id, tradetime`（最後欄必須是 temporal/int 型別）
- 分區數龐大，需在 `dolphindb.cfg` 設定 `maxPartitionNumPerQuery=500000`

> **重要**：Python 端的 `compute_python_alphas()`（在 `daily_batch_pipeline.py`）是 WQ101 的**近似替代**，僅用於 `--csv` 執行模式。正式研究必須使用 DolphinDB WQ101。

---

### Layer 4 — Meta Signal　`src/meta_signal/`

| 檔案                    | 職責                                                                                      |
| --------------------- | --------------------------------------------------------------------------------------- |
| `rule_based.py`       | IC-weighted 加權合成（每個 alpha 的歷史 IC 作為權重）                                                  |
| `ml_meta_model.py`    | XGBoost meta model（regression，purged expanding-window CV，gain-based feature importance） |
| `regime_ensemble.py`  | 根據市場 regime 切換不同 alpha 組合（MVP v3）                                                       |
| `signal_generator.py` | Facade，統一呼叫入口，路由至對應方法；支援 `--signal-method rule_based / ml_meta`                         |

**輸出格式**：`meta_signals` table（security_id, signal_time, signal_score, signal_direction, confidence, method, model_version_id）

**XGBoost 回測結果**（10 檔台股，2022-01-01 → 2024-12-31）：Sharpe 0.77（vs rule-based 0.53）

> **設計原則**：wq101alpha 輸出只是 feature，必須經過此層聚合才能產生 trading signal。禁止直接用單一 alpha 下單。

---

### Layer 5 — Portfolio Construction　`src/portfolio/`

| 檔案 | 職責 |
|------|------|
| `constructor.py` | 將 signal_score 轉換為 target weights（equal-weight top-k long-only） |

目前支援：`equal_weight_topk`（選 signal_score 最高的 top-k 股票，等權分配）。

---

### Layer 6 — Risk Management　`src/risk/`

| 檔案 | 職責 |
|------|------|
| `risk_manager.py` | 強制執行風控約束：position cap（單股 ≤10%）、gross exposure cap（≤100%）、turnover limit、最大回撤熔斷 |

output：調整後的 `adjusted_weights`，傳給 Execution 層。

---

### Layer 7 — Execution　`src/execution/`

| 檔案 | 職責 |
|------|------|
| `paper_engine.py` | Paper trading 引擎（模擬成交，含 slippage bps） |
| `order_manager.py` | 訂單生命週期管理（CREATED → SUBMITTED → FILLED / CANCELLED） |
| `reconciliation.py` | 持倉對帳（paper position vs target weights 比對） |

**輸出**：`orders` / `fills` / `positions` 三張 PostgreSQL table。

---

### Layer 8 — Labeling　`src/labeling/`

| 檔案 | 職責 |
|------|------|
| `label_generator.py` | 計算 forward return（horizon = 1 / 5 / 10 / 20 days）並設定 `label_available_at` |
| `evaluator.py` | 三層評估：per-alpha IC/RankIC、model hit-rate、strategy Sharpe/DD |

**核心不變式**：

```
label_available_at = signal_time + horizon + buffer_bars (≥1)
```

任何在 `label_available_at` 之前使用該 label 作為訓練特徵的行為都是 **look-ahead bias**，程式碼禁止此操作。

---

### Layer 9 — Monitoring　`src/monitoring/`

四個 Monitor 相互獨立，各自計算 metrics、各自觸發 alert：

| 監控器                   | 職責                                                     | 主要指標                                                |
| --------------------- | ------------------------------------------------------ | --------------------------------------------------- |
| `data_monitor.py`     | 資料品質：缺值率、volume 異常、價格跳空、PSI 分布偏移                       | missing_rate, volume_zscore, feature_dist_shift_psi |
| `alpha_monitor.py`    | Alpha 退化：rolling IC、IC decay、distribution shift        | rolling_IC, KS-stat, PSI（per-alpha）                 |
| `model_monitor.py`    | 模型退化：calibration drift、prediction distribution shift   | KS-test on scores, ECE                              |
| `strategy_monitor.py` | 策略退化：rolling Sharpe、max drawdown                       | rolling_sharpe, max_dd                              |
| `alert_manager.py`    | 統一 alert 路由（PostgreSQL `alerts` table + Redis pub/sub） | severity: WARNING / CRITICAL                        |

**PSI 門檻**：< 0.10 無漂移 / 0.10–0.25 WARNING / > 0.25 CRITICAL

**輸出格式**：所有 metrics 寫入 `monitoring_metrics` table，schema 為：
```
(metric_time, monitor_type, metric_name, metric_value, dimension, window_size)
```

每次 pipeline 執行後自動呼叫 `alert_mgr.persist_metrics()` / `alert_mgr.fire_alerts()`，52 個 metrics / 26 個 alerts 寫入 PostgreSQL。

---

### Layer 10 — Adaptation　`src/adaptation/`

| 檔案 | Policy | 觸發條件 |
|------|--------|----------|
| `scheduler.py` | Policy 1: Scheduled retrain | 固定週期（每月 / 每季） |
| `performance_trigger.py` | Policy 2: Performance-triggered | Monitoring alert（rolling Sharpe < 門檻 / CRITICAL PSI 累積） |
| `recurring_concept.py` | Policy 3: Recurring concept pool | 當前 regime 與歷史 regime fingerprint 相似時，重用舊模型 |
| `model_pool_strategy.py` | Policy 3 協調器 | 封裝 `ModelPoolController`：shadow 3-way compare 主邏輯、pool 搜尋、DB 降級 |
| `shadow_evaluator.py` | Shadow deployment | 並行評估 current / reused / retrained 三個候選模型 |
| `model_registry.py` | Model versioning | 記錄 training_window / features_used / holdout_metrics / regime_fingerprint |

**設計原則**：Adaptation 在 Monitoring **之後**觸發，絕不在同一層。

### Policy 3 詳細設計（WP11 新增）

`model_pool` 策略的決策流程（每次觸發時）：

```
觸發後（觸發條件與 triggered 策略相同）：
  Step 1: 一律訓練新候選（new_model）
  Step 2: 計算當前 regime fingerprint（5 維：volatility / autocorr /
          avg_cross_corr / trend_strength / volume_ratio）
  Step 3: 搜尋 regime_pool（僅限本次 run 的 entries）
          → cosine similarity；threshold=0.8 時視為命中
  Step 4: 建立候選集合
          candidates = {current_model_id: ..., new_id: ..., reused_id: ...（命中時）}
  Step 5: ShadowEvaluator 在成熟窗口（t - purge - horizon 往前 20 日）
          評估各候選的 IC / Sharpe → select_best（需超越 current IC ≥ 0.005）
  Step 6: 依最佳者更新 active model + 更新 pool
```

**Pool 儲存**：PostgreSQL `regime_pool` 表存 fingerprint + model_id；
`ModelPoolController._models_by_id` dict 存 MLMetaModel instance（因無 disk 序列化）。

**DB 降級**：連線失敗時 `_backend='unavailable'`，行為等同 `triggered`。

**關鍵隔離**：`find_similar_regime(since=session_start)` 過濾 SQL，避免跨 run 污染。

**A/B 實驗框架**（共 5 策略）：

| 策略 | 觸發條件 | Pool 使用 | 重訓次數（Phase A 實測）|
|------|----------|-----------|------------------------|
| none | — | — | 1（僅初始） |
| scheduled_20 | 固定每 20 日 | — | 32 |
| scheduled_60 | 固定每 60 日 | — | 11 |
| triggered | IC/Sharpe 退化 + 冷卻 20 日 | — | 12 |
| model_pool | 同 triggered | ✅ fingerprint 搜尋 + shadow 3-way compare | 11 |

**Phase A 重跑結果**（2022-06-01 → 2024-12-31，IS-selected 52 alphas，含完整成本）：

| 策略 | Sharpe | n_retrain | Cum Ret % |
|------|--------|-----------|-----------|
| **scheduled_20** | **3.757** | 32 | 15754 |
| triggered | 3.630 | 12 | 15269 |
| model_pool | 3.518 | 11 | 12925 |
| scheduled_60 | 3.514 | 11 | 10938 |
| none（基準） | 3.158 | 1 | 7625 |

> **方法學修正（P0★ Phase A，2026-04-27 完成）**：舊版 WP9 結論「no-adapt 勝出」已被識別為兩項 selection bias 所致：（a）trigger 窗口與 shadow 評估窗口重疊，偏好擬合最近 noise 的新模型；（b）未扣台股實際交易成本（~46 bps/day）。修正後——拆開觸發窗口 `[t-60, t-20]`（Phase A #2）+ 加入完整成本模型（Phase A #4）——所有 adaptation 策略均勝過 no-adapt。
>
> **成本敏感度驗證**：4 段 round-trip cost（0 / 0.2 / 0.4 / 0.6%）下 5 策略排序 rank_std=0（100% 一致）。Paired t-test vs none：4 個 adaptation 策略 mean_daily_excess_ret 均正向，direction 對但 power 不足（p_one_sided > 5%，最小 triggered p=0.096）。

---

## 4. 支援模組職責

| 模組 | 目錄 | 職責 |
|------|------|------|
| Config | `src/config/settings.py` | Pydantic Settings，從 `.env` 讀取所有連線參數 |
| Constants | `src/config/constants.py` | Enum 定義（BarType, OrderSide, MonitorType …）、`MVP_V1_ALPHA_IDS` |
| DB | `src/common/db.py` | `get_pg_connection()`, `get_dolphindb()`, `get_redis()` 統一連線工廠；連線檢查使用 `session.isClosed()` |
| Logging | `src/common/logging.py` | structlog 封裝，`get_logger()` / `setup_logging()` |
| Metrics | `src/common/metrics.py` | 通用計算函式：IC, Sharpe, max_drawdown, KS-test, winsorize, PSI, calibration_error |
| Time Utils | `src/common/time_utils.py` | `compute_label_available_at()` 等時間計算 |
| API | `src/api/` | FastAPI app + 4 route groups（monitoring / signals / adaptation / backtest） |
| Pipelines | `pipelines/` | daily_batch / monitoring / adaptation / label_update / replay / ab_experiment / simulate_recent / predict_next_day |
| Scripts | `scripts/` | download_tw_stocks / generate_report / validate_infrastructure / backfill_alpha / ingest_csv_to_dolphindb / demo_monitoring_2026 |

---

## 5. 執行模式

### 模式 A — Synthetic（無任何外部依賴）

```bash
python -m pipelines.daily_batch_pipeline --synthetic
python scripts/generate_report.py
```

- 資料：隨機生成的 OHLCV（`generate_synthetic_data`）
- Alpha：隨機數（`generate_synthetic_alphas`）
- 用途：CI 測試、pipeline 邏輯驗證、環境確認
- 限制：報酬率完全不具參考意義

### 模式 B — CSV + Python Alpha（需網路，無 Docker）

```bash
# 步驟 1：下載台股資料（一次性）
python scripts/download_tw_stocks.py --tickers 2330 2317 2454 ...

# 步驟 2：執行 pipeline（rule-based 或 XGBoost meta）
python -m pipelines.daily_batch_pipeline --csv data/tw_stocks_ohlcv.csv
python -m pipelines.daily_batch_pipeline --csv data/tw_stocks_ohlcv.csv --signal-method ml_meta

# 步驟 3：產生報告
python scripts/generate_report.py --csv data/tw_stocks_ohlcv.csv
```

- 資料：真實台股 OHLCV（yfinance），已下載 1083 檔 × 2022-2026
- Alpha：Python 端 WQ101 近似版本（reversal / momentum / volume-price 等 15 個）
- 用途：驗證 Pipeline 邏輯流程、初步 IC 分析、展示用 backtest report
- 限制：Alpha 是近似值，研究結論不可直接引用

**已驗證結果**（10 檔台股，2022-01-01 → 2024-12-31）：

| 指標 | rule-based | XGBoost meta |
|------|-----------|-------------|
| 累積報酬 | +184% | +79.3% |
| 年化報酬 | +43.8% | +22.6% |
| Sharpe Ratio | 0.94 | **0.77** |
| Max Drawdown | -21.8% | -21.9% |
| Win Rate | 51.9% | 52.3% |

> rule-based 近似版數字可能因 Python alpha 精度問題高估效果，研究不可直接引用。

### 模式 C — Full Real Pipeline（需 Docker）✅ 已驗證

```bash
# 步驟 1：啟動服務（PostgreSQL 使用 port 5433，避免與本機 PG 衝突）
docker compose up -d
python scripts/validate_infrastructure.py --run-migrations

# 步驟 2：DolphinDB 分區表初始化（執行一次）
python -c "
import dolphindb as ddb, pathlib
s = ddb.session(); s.connect('127.0.0.1', 8848, 'admin', '123456')
s.run(pathlib.Path('dolphindb/scripts/setup_database.dos').read_text(encoding='utf-8'))
s.close()
"

# 步驟 3：將 CSV 資料灌入 DolphinDB standardized_bars
python scripts/ingest_csv_to_dolphindb.py --csv data/tw_stocks_ohlcv.csv

# 步驟 4：執行完整 pipeline
python -m pipelines.daily_batch_pipeline --start 2022-01-01 --end 2024-12-31
```

- Alpha：真實 DolphinDB WQ101（`alpha_batch.dos`），~1.2s 計算 101,003 rows（15 alphas × 10 stocks）
- 所有 metrics 寫入 PostgreSQL / Redis；Grafana dashboard 自動 provisioning（`http://localhost:3000`）

**已驗證結果**（10 檔台股，2022-01-01 → 2024-12-31，DolphinDB 真實 WQ101）：

| 指標 | 數值 |
|------|------|
| 累積報酬 | +58.6% |
| 年化報酬 | +17.4% |
| Sharpe Ratio | 0.53 |
| Max Drawdown | -22.1% |
| Win Rate | 53.0% |
| Profit Factor | 1.42 |

### 模式 D — Adaptation A/B 實驗（五策略）

```bash
# 單一策略模擬
python -m pipelines.simulate_recent --csv data/tw_stocks_ohlcv.csv \
    --start 2022-06-01 --end 2024-12-31 --strategy none
python -m pipelines.simulate_recent --csv data/tw_stocks_ohlcv.csv \
    --start 2022-06-01 --end 2024-12-31 --strategy scheduled --retrain-interval 60
python -m pipelines.simulate_recent --csv data/tw_stocks_ohlcv.csv \
    --start 2022-06-01 --end 2024-12-31 --strategy model_pool \
    --similarity-threshold 0.8 --pool-regime-window 60 --shadow-window 20

# 五策略完整對比（自動依序執行）
python -m pipelines.ab_experiment --csv data/tw_stocks_ohlcv.csv \
    --start 2022-06-01 --end 2024-12-31

# 僅跑 model_pool 策略（快速煙霧測試）
python -m pipelines.ab_experiment --csv data/tw_stocks_ohlcv.csv \
    --start 2023-01-01 --end 2024-06-30 --strategies triggered model_pool
```

輸出：`reports/adaptation_ab/<run_id>/`（comparison.csv / comparison.png / experiment_summary.md / config.json）

**model_pool 注意事項**：
- 無 Docker / PostgreSQL 時自動降級為 triggered 行為，`summary.pool_backend="unavailable"`
- Pool 搜尋僅限本次 run 產生的 entries（`since=session_start` SQL 過濾）
- Panel C 圖中 `model_pool` 的重訓標記分色：綠圓=pool reuse / 紅三角=new model / 灰叉=kept current

---

## 6. 測試方法

### 6.1 單元測試

```bash
# 執行全部 83 個 tests
pytest -q

# 分模組執行
pytest tests/unit/test_standardization.py -v
pytest tests/unit/test_risk.py -v
pytest tests/unit/test_meta_signal.py -v
pytest tests/unit/test_monitoring.py -v
pytest tests/unit/test_drift_metrics.py -v
pytest tests/unit/test_ml_meta_model.py -v
```

**覆蓋範圍**：

| Test 檔案 | 測試對象 | 主要斷言 |
|-----------|----------|----------|
| `test_standardization.py` | `SchemaMapper`, `QualityChecker` | 欄位映射正確、missing_flags 設定 |
| `test_risk.py` | `RiskManager` | position cap、exposure 上限、turnover limit |
| `test_meta_signal.py` | `RuleBasedSignalGenerator` | IC 計算、signal 方向一致性 |
| `test_monitoring.py` | 四大 Monitor | metrics 格式、alert 觸發條件 |
| `test_drift_metrics.py` | `population_stability_index`, `calibration_error`, KS-test | PSI 閾值、ECE 計算、邊界條件（12 tests） |
| `test_ml_meta_model.py` | `MLMetaModel`（XGBoost） | 訓練 schema、holdout IC、feature importance、long format pivot（9 tests） |

### 6.2 整合測試

```bash
pytest tests/integration/ -v
```

| Test 檔案 | 測試對象 | 測試數 |
|-----------|----------|--------|
| `test_pipeline_batch.py` | 完整 8 步 pipeline（synthetic data） | — |
| `test_adaptation_loop.py` | drift → monitor → trigger → retrain → shadow → select 完整鏈路 | 11 |
| `test_replay_pipeline.py` | 批次 vs 串流一致性（match rate ≥ 0.95） | 14 |
| `test_ab_experiment.py` | 五種 adaptation 策略 artifact 產生（含 model_pool） | 10 |
| `test_model_pool_strategy.py` | model_pool 端到端、pool stats、DB 降級、五策略 A/B（fake pool） | 8 |

### 6.3 端對端 smoke test（模式 B）

```bash
python -m pipelines.daily_batch_pipeline \
    --csv data/tw_stocks_ohlcv.csv \
    --start 2022-01-01 --end 2022-06-30

python scripts/generate_report.py \
    --csv data/tw_stocks_ohlcv.csv \
    --start 2022-01-01 --end 2022-06-30
```

預期輸出：`reports/<timestamp>_backtest_report.png` 含 4 面板圖，終端印出各項指標。

### 6.4 基礎設施連線測試（模式 C 前置）

```bash
python scripts/validate_infrastructure.py --skip-dolphindb   # 先檢查 PG + Redis
python scripts/validate_infrastructure.py                     # 含 DolphinDB
```

---

## 7. 目前能否上線

### 結論：**不可以用於真實交易，可以用於研究展示**

| 面向 | 狀態 | 說明 |
|------|------|------|
| 程式碼完整性 | ✅ | 10 層全部有完整實作 |
| 邏輯正確性（合成） | ✅ | 83 tests 通過、pipeline 流程正確 |
| 邏輯正確性（真實資料） | ✅ | CSV 模式 + DolphinDB 真實 WQ101 均已驗證 |
| DolphinDB WQ101 驗證 | ✅ | 完整 101 個因子 × 556 檔 × 53.8M rows |
| PostgreSQL 全鏈路 | ✅ | docker compose up -d，port 5433，migration 自動套用 |
| Grafana dashboard | ✅ | 4 面板自動 provisioning（datasource + dashboard JSON） |
| XGBoost meta model | ✅ | purged CV 訓練，Sharpe 0.77 vs rule-based 0.53 |
| Adaptation A/B 實驗（四策略） | ✅ | 四策略對比完成，含 paired t-test 統計檢定（WP9） |
| Model Pool 第五策略（WP11） | ✅ | 程式碼完整實作，fake pool 整合測試通過，DB 降級路徑驗證 |
| Replay streaming 驗證 | ✅ | 15/15 alphas 100% match rate（batch vs 串流） |
| 即時串流（Shioaji） | ❌ | 現為 mock，需要帳號 |
| 風控完整性 | ⚠️ | position cap / exposure 有實作，但缺乏 real-time P&L 監控 |
| Look-ahead bias | ✅ | `label_available_at` 機制已實作並測試 |
| 回測過擬合風險 | ⚠️ | 目前無 walk-forward validation |
| 上線審查 | ❌ | 無 trading license，僅為研究用途 |

### 可以做的事（現在）

- ✅ 展示完整的系統架構與研究思路（研究所申請 / GitHub）
- ✅ 以合成或 CSV 資料展示 Monitoring + Adaptation 設計
- ✅ IC 分析、per-alpha 篩選研究（v3：45 個有效 alpha，556 檔台股）
- ✅ Drift detection 實驗：KS-test / PSI / ECE 跨 regime 比較
- ✅ Adaptation A/B 實驗與反直覺研究結論（論文 RQ3 輸出）
- ✅ 作為研究論文的系統實作基礎

### 不可以做的事（現在）

- ❌ 真實台股下單（非 paper trading）
- ❌ 引用 CSV 模式的 backtest 數字作為 WQ101 alpha 效果的研究結論（Python 近似版高估效果）
- ❌ 以 10 檔、3 年資料的結果作為最終研究結論（樣本數不足，尚未做 walk-forward validation）

---

## 8. 已完成里程碑

### MVP v1（2026-04-06 前）✅

- Docker 基礎設施（DolphinDB v2.00.18 + PostgreSQL 5433 + Redis + Grafana）全部 Up
- DolphinDB 分區表建立、wq101alpha API 確認、alpha_batch.dos 重寫
- 7260 rows 台股資料寫入 DolphinDB，15 個真實 WQ101 因子端到端驗證
- Full real pipeline 跑通：+58.6% 累積報酬，Sharpe 0.53

### MVP v2（2026-04-09 至 2026-04-15）✅

| 工作包 | 內容 |
|--------|------|
| WP1 | PSI / calibration_error drift metrics 補齊 + monitor 整合 + 12 unit tests |
| WP2 | Per-alpha IC 分析 notebook；v2 有效 alpha：10/15（基準 10 檔） |
| WP3 | XGBoost MLMetaModel（regression + purged CV）+ 9 unit tests |
| WP4 | Drift detection 實驗：三 regime 比較，PSI 2023 年中突破 0.25 critical threshold |
| WP5 | Performance-triggered adaptation 整合測試（11 tests，shadow IC 0.543 vs 0.016） |
| WP6 | Grafana 4 面板 dashboard（自動 provisioning） |
| WP7 | Replay streaming 串接（14 tests，15/15 alphas 100% match rate） |

### MVP v3（2026-04-20 至 2026-04-25）✅

| 工作包 | 內容 |
|--------|------|
| WP9 | Adaptation A/B 實驗（論文 RQ3 核心）：四策略對比 + paired t-test + regime-stratified 分析 |
| WP10 | Alpha Universe 擴充：101 alphas × 556 stocks × 2022-2026 = 53,771,183 rows；v3 有效 alpha：45 個 |
| WP11 | Model Pool 第五策略：`ModelPoolController` + `PoolDecision` 新建，shadow 3-way compare，8 新測試，fake pool 整合測試 |

**測試數量演進**：19（v1）→ 31（WP1）→ 40（WP3）→ 51（WP5）→ 65（WP7）→ 75（WP9）→ 83（WP11）→ **136（Phase A）**

---

## 9. 下一步行動計畫

### MVP v3 剩餘（未開始）

| 工作包 | 內容 | 優先級 |
|--------|------|--------|
| WP8 | HMM-based regime identification + regime-aware ensemble | P2 |
| WP11-DB | model_pool 真實 PostgreSQL 驗證（需 docker-compose up，長期歷史資料）；驗證 shadow guard 是否讓 model_pool 在 recurring regime 優於 triggered | P2 |
| WP11-NB | `notebooks/04_model_pool_evaluation.py`：regime-stratified 比較 + paired t-test vs triggered | P2 |
| WP12 | Reproducibility：config versioning + data snapshot | P3 |
| WP13 | 研究報告 notebook（整合 WP4 + WP9 + WP11 findings） | P3 |

### 延到 v3 後（長期）

- Shioaji streaming 真實連線（需帳號）
- MLflow experiment tracking 整合
- Walk-forward validation（解決回測過擬合風險）

---

## 10. 已知限制與風險

1. **DolphinDB 社區版**有連線數與記憶體限制，需先確認 license。
2. **wq101alpha 部分 alpha 在台股的適用性**：v3 篩選後 101 個中 45 個有效（`|rank_ic| ≥ 0.01 AND coverage ≥ 0.80`）。
3. **Recurring concept pool 在短歷史下樣本不足**：WP11 已完成程式碼實作與 fake pool 整合測試；真實 PostgreSQL 驗證需長期歷史資料（需 docker-compose up）。
4. **Shioaji streaming 是 mock 狀態**，需要帳號才能接真實資料。
5. **單人開發風險**：容易在某一層花太多時間，應嚴格遵守 MVP scope。
6. **DolphinDB 分區數限制**：預設 `maxPartitionNumPerQuery=65536`，已調整為 500000（寫入 dolphindb.cfg）；VALUE 日分區 × HASH 50 共 ~200k 分區，大查詢需加 WHERE 時間範圍。
7. **本機 PostgreSQL 衝突**：本機已安裝 PostgreSQL 佔用 5432，Docker 容器改用 5433 port；`.env` 與 `settings.py` 已對應修改。
8. **wq101alpha API 陷阱**：`prepare101` 模組不存在；各 WQAlpha 函式接收獨立 panel matrix（column-major），不是字典；`matrixToLongTable` 使用 `flatten` + `stretch/take` 轉 long format。
9. **TSDB 寫入緩衝 OOM**：寫入大量 alpha 時（101 alphas × 556 stocks），TSDB write cache 跨 alpha 累積導致 OOM；必須在每次 `db.append!()` 後立即呼叫 `flushTSDBCache()` 釋放緩衝。若 DolphinDB 因 OOM 異常重啟，TSDBRedo redo logs（可達 500MB+）會在下次啟動時重播，可能再次觸發 OOM——需先停容器、用 alpine 刪除 `/data/ddb/server/data/local8848/log/TSDBRedo/*.log`，再重啟。
10. **BOOL 型 alpha**：wq061、wq075、wq079、wq095 回傳 BOOL matrix（非 DOUBLE），直接寫入 `alpha_features.alpha_value`（DOUBLE）會報型態錯誤；在 `alpha_batch.dos` 中 `matrixToLongTable()` 前加 `if (typestr(mat) != "FAST DOUBLE MATRIX") { mat = double(mat) }` 轉型。
11. **DolphinDB Python SDK `isClosed()`**：較新版 SDK 不存在 `isConnected` 屬性，連線狀態檢查須使用 `session.isClosed()`（`src/common/db.py` 已修正）。
12. **per-alpha IC 計算效能**：`alpha_id` 不是分區鍵，每次 `where alpha_id = 'wqXXX'` 需掃描全部 ~200k 分區，101 個 alpha 約需 4-6 小時（`compute_ic_summary.dos`）；為避免 Python 端 OOM，IC 計算完整在 DolphinDB 端執行。
13. **Adaptation 結論（Phase A 修正後）**：舊版「no-adapt 勝出」結論已於 2026-04-27 Phase A 翻轉。修正觸發窗口拆分（#2）與完整成本模型（#4）後，所有 adaptation 策略 Sharpe 均高於 no-adapt，4 段成本場景下排序 100% 一致。詳見 `reports/adaptation_ab/` 與 `CLAUDE.md §七`。
