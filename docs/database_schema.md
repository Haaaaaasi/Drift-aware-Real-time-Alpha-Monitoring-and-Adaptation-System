# DARAMS 資料庫表結構與用途說明

> 更新日期：2026-04-16
> 本文件詳細說明 DARAMS 系統中三大儲存層（PostgreSQL、DolphinDB、Redis）的所有資料表、欄位、用途，以及它們在資料流中扮演的角色。

---

## 目錄

1. [儲存層架構總覽](#一儲存層架構總覽)
2. [PostgreSQL — 元資料與交易紀錄](#二postgresql--元資料與交易紀錄)
3. [DolphinDB — 時序資料與串流](#三dolphindb--時序資料與串流)
4. [Redis — 快取與即時狀態](#四redis--快取與即時狀態)
5. [資料流與寫入時機](#五資料流與寫入時機)
6. [查詢範例](#六查詢範例)

---

## 一、儲存層架構總覽

DARAMS 採用**三層異質儲存**設計，各層職責明確分離：

| 儲存層 | 角色 | 資料特性 | 表/集合數 |
|--------|------|----------|----------|
| **DolphinDB** | 時序資料引擎 | 高頻寫入、向量化查詢、wq101alpha 計算 | 2 個分區表 + 2 個串流表 |
| **PostgreSQL** | 元資料與交易紀錄 | ACID 一致性、外鍵約束、JSONB 彈性欄位 | 13 張表 |
| **Redis** | 即時快取 | KV 存取、毫秒延遲、不持久化關鍵狀態 | N/A（key-value） |

### 為何三層分離？

- **DolphinDB**：用 panel matrix 結構處理 standardized OHLCV 與 alpha values，內建 wq101alpha 模組，避免將 GB 級時序資料塞進 PostgreSQL
- **PostgreSQL**：負責 orders / fills / positions / model_registry 等需要強一致性與外鍵關係的元資料
- **Redis**：放當前持倉、最新訊號、alert 去重狀態等存取頻繁但可重建的資料

---

## 二、PostgreSQL — 元資料與交易紀錄

PostgreSQL 中共 **13 張表**，定義於 `migrations/001_init_tables.sql`。容器使用 port **5433**（避開本機 5432 衝突）。

### 1. `raw_market_events` — 原始市場事件記錄表

**用途**：append-only 的原始事件流，所有 ingestion layer 接收的 tick / kbar 都先寫入此表，作為審計與重播來源。

| 欄位 | 型別 | 說明 |
|------|------|------|
| `event_id` | BIGSERIAL PK | 流水號 |
| `security_id` | VARCHAR(20) | 標的代號（例：'2330'） |
| `event_type` | VARCHAR(20) | tick / kbar_1m / kbar_5m / kbar_daily / bidask |
| `event_ts` | TIMESTAMPTZ | 事件在市場發生時間 |
| `ingestion_ts` | TIMESTAMPTZ | 系統接收時間（用於延遲分析） |
| `open/high/low/close/volume/vwap` | DOUBLE | OHLCV |
| `bid_price/ask_price/bid_size/ask_size` | DOUBLE | 報價簿快照 |
| `raw_payload` | JSONB | 原始 payload（例：Shioaji 回傳物件） |

**索引**：
- `ux_raw_event_dedup` (UNIQUE on security_id, event_ts, event_type) — 防重複寫入
- `ix_raw_event_ts` — 時間查詢加速

**寫入者**：`src/ingestion/historical_loader.py`、`src/ingestion/shioaji_stream.py`

---

### 2. `security_master` — 標的主檔

**用途**：股票池 metadata，提供 standardization 層做欄位映射與 industry code lookup。

| 欄位 | 型別 | 說明 |
|------|------|------|
| `security_id` | VARCHAR(20) PK | 標的代號 |
| `name` | VARCHAR(100) | 中文名稱 |
| `exchange` | VARCHAR(10) | TWSE / OTC / NYSE… |
| `industry_code` | INT | 產業分類碼（給 industry-aware alpha 用） |
| `listing_date` | DATE | 上市日 |
| `is_active` | BOOLEAN | 是否仍在交易 |
| `updated_at` | TIMESTAMPTZ | 異動時間 |

**寫入者**：`scripts/seed_security_master.py`（初始化）、定期同步腳本

---

### 3. `meta_signals` — 聚合訊號表

**用途**：儲存 Layer 4（Meta Signal）產生的綜合訊號，下游 Portfolio 由此取得 target signal。

| 欄位 | 型別 | 說明 |
|------|------|------|
| `signal_id` | BIGSERIAL PK | 流水號 |
| `security_id` | VARCHAR(20) | 標的 |
| `signal_time` | TIMESTAMPTZ | 訊號產生時間（**look-ahead 防呆關鍵**） |
| `bar_type` | VARCHAR(10) | daily / 5m… |
| `signal_score` | DOUBLE | 連續訊號值（例：composite IC-weighted score） |
| `signal_direction` | SMALLINT | +1 / 0 / -1 |
| `confidence` | DOUBLE | 訊號信心度（ML 模型可填 prediction probability） |
| `method` | VARCHAR(20) | rule_based / ml_meta / regime_ensemble |
| `model_version_id` | VARCHAR(50) | 對應 `model_registry.model_id` |

**索引**：`ix_meta_signals_lookup` (security_id, signal_time)

**寫入者**：`src/meta_signal/signal_generator.py`

---

### 4. `portfolio_targets` — 目標權重表

**用途**：Portfolio Construction 層輸出的 desired weights，下一步進入 Risk Manager 約束調整。

| 欄位 | 型別 | 說明 |
|------|------|------|
| `target_id` | BIGSERIAL PK | 流水號 |
| `rebalance_time` | TIMESTAMPTZ | 再平衡時點 |
| `security_id` | VARCHAR(20) | 標的 |
| `target_weight` | DOUBLE | 目標權重 [-1, 1] |
| `target_shares` | INT | 目標股數（依當下價格換算） |
| `construction_method` | VARCHAR(30) | equal_weight / risk_parity / mean_variance |
| `pre_risk` | BOOLEAN | TRUE = 風控前；FALSE = 風控後最終值 |

**索引**：`ix_portfolio_rebalance` (rebalance_time)

**寫入者**：`src/portfolio/constructor.py`、`src/risk/risk_manager.py`

---

### 5. `orders` — 訂單表

**用途**：紀錄所有送出訂單（含 paper trading），追蹤 lifecycle 狀態。

| 欄位 | 型別 | 說明 |
|------|------|------|
| `order_id` | VARCHAR(50) PK | UUID 或業務主鍵 |
| `security_id` | VARCHAR(20) | 標的 |
| `order_time` | TIMESTAMPTZ | 下單時間 |
| `side` | VARCHAR(4) | BUY / SELL |
| `order_type` | VARCHAR(10) | MARKET / LIMIT |
| `quantity` | INT | 委託股數 |
| `limit_price` | DOUBLE | 限價（MARKET 為 NULL） |
| `status` | VARCHAR(20) | CREATED / SENT / PARTIAL / FILLED / REJECTED / CANCELLED |
| `expected_price` | DOUBLE | 模型預期成交價（給 slippage 分析） |
| `updated_at` | TIMESTAMPTZ | 最後狀態更新時間 |

**索引**：`ix_orders_lookup` (security_id, order_time)

**寫入者**：`src/execution/paper_engine.py`、`src/execution/order_manager.py`

---

### 6. `fills` — 成交回報表

**用途**：紀錄實際成交明細，與 `orders` 透過 order_id 關聯。為 Strategy Monitor 計算實際 PnL 的來源。

| 欄位 | 型別 | 說明 |
|------|------|------|
| `fill_id` | VARCHAR(50) PK | 成交流水 |
| `order_id` | VARCHAR(50) FK | → `orders.order_id` |
| `security_id` | VARCHAR(20) | 標的 |
| `fill_time` | TIMESTAMPTZ | 成交時間 |
| `fill_price` | DOUBLE | 成交價 |
| `fill_quantity` | INT | 成交股數（partial fill 可能多筆） |
| `commission` | DOUBLE | 手續費 |
| `slippage_bps` | DOUBLE | 滑點（bps）= (fill_price − expected_price) / expected_price × 10000 |

**索引**：`ix_fills_order` (order_id)

**寫入者**：`src/execution/paper_engine.py`、`src/execution/reconciliation.py`

---

### 7. `positions` — 持倉快照表

**用途**：時間點持倉快照（point-in-time），供回測 PnL 重建與 risk monitor 取當前曝險。

| 欄位 | 型別 | 說明 |
|------|------|------|
| `snapshot_time` | TIMESTAMPTZ | 快照時點（PK 之一） |
| `security_id` | VARCHAR(20) | 標的（PK 之二） |
| `quantity` | INT | 持有股數（負值代表空頭） |
| `avg_cost` | DOUBLE | 平均成本 |
| `market_value` | DOUBLE | 市值 |
| `unrealized_pnl` | DOUBLE | 未實現損益 |

**主鍵**：(snapshot_time, security_id) 複合

**寫入者**：每日收盤批次 `pipelines/daily_batch_pipeline.py`

---

### 8. `labels_outcomes` — 延遲標籤表

**用途**：儲存 forward return 標籤，是 ML 訓練的 ground truth。`label_available_at` 是**防 look-ahead 的關鍵欄位**。

| 欄位 | 型別 | 說明 |
|------|------|------|
| `security_id` | VARCHAR(20) | 標的（PK 之一） |
| `signal_time` | TIMESTAMPTZ | 訊號時間（PK 之二） |
| `horizon` | INT | 預測 horizon（bars，例：5 = T+5）（PK 之三） |
| `forward_return` | DOUBLE | 實際前向報酬 |
| `forward_direction` | SMALLINT | +1 / -1 |
| `realized_pnl` | DOUBLE | 實際 PnL（含手續費） |
| `label_available_at` | TIMESTAMPTZ | **此 label 可用於訓練的最早時間** = signal_time + horizon + buffer |

**主鍵**：(security_id, signal_time, horizon)
**索引**：`ix_labels_available` (label_available_at)

**用法（防 look-ahead）**：訓練時必須 `WHERE label_available_at <= train_cutoff_time`

**寫入者**：`src/labeling/label_generator.py`、`pipelines/label_update_pipeline.py`

---

### 9. `monitoring_metrics` — 統一監控指標表

**用途**：四層 monitor（data / alpha / model / strategy）的所有計算結果統一寫入此表，Grafana 由此繪圖。

| 欄位 | 型別 | 說明 |
|------|------|------|
| `metric_id` | BIGSERIAL PK | 流水號 |
| `metric_time` | TIMESTAMPTZ | 指標計算時間 |
| `monitor_type` | VARCHAR(20) | data / alpha / model / strategy |
| `metric_name` | VARCHAR(50) | 例：missing_ratio, alpha_value_psi, calibration_ece, rolling_sharpe |
| `metric_value` | DOUBLE | 指標數值 |
| `dimension` | VARCHAR(50) | 維度標籤（例：security_id='2330' / alpha_id='wq001' / model_id='xgb_v3'） |
| `window_size` | INT | 計算窗口大小（bars） |

**索引**：`ix_monitoring_lookup` (monitor_type, metric_name, metric_time)

**設計原則**：所有 monitor 都用同一張表 + dimension 欄位區分，避免每加一個 metric 就建新表。

**寫入者**：`src/monitoring/alert_manager.py::persist_metrics()`

---

### 10. `alerts` — 告警表

**用途**：當 metric 超過 threshold 時觸發 alert，供 Grafana 顯示與 adaptation 決策參考。

| 欄位 | 型別 | 說明 |
|------|------|------|
| `alert_id` | BIGSERIAL PK | 流水號 |
| `alert_time` | TIMESTAMPTZ | 告警時間（預設 now()） |
| `monitor_type` | VARCHAR(20) | 對應 `monitoring_metrics.monitor_type` |
| `metric_name` | VARCHAR(50) | 觸發告警的指標名 |
| `severity` | VARCHAR(10) | WARNING / CRITICAL |
| `current_value` | DOUBLE | 觸發時的指標值 |
| `threshold` | DOUBLE | 門檻值 |
| `message` | TEXT | 人類可讀說明 |
| `is_acknowledged` | BOOLEAN | 是否已確認（手動確認） |
| `triggered_adaptation` | BOOLEAN | 是否曾觸發 adaptation（避免重複觸發） |

**索引**：`ix_alerts_time`, `ix_alerts_severity`

**寫入者**：`src/monitoring/alert_manager.py::fire_alerts()`
**消費者**：`src/adaptation/performance_trigger.py`

---

### 11. `model_registry` — 模型版本表

**用途**：MLOps 核心 — 每次訓練的模型都註冊一筆，記錄完整訓練上下文，支援 shadow deployment 與 rollback。

| 欄位 | 型別 | 說明 |
|------|------|------|
| `model_id` | VARCHAR(50) PK | 模型 ID（例：'xgb_2026-04-15_a1b2c3'） |
| `model_type` | VARCHAR(30) | xgboost / logistic_regression / ensemble |
| `trained_at` | TIMESTAMPTZ | 訓練完成時間 |
| `training_window_start` | TIMESTAMPTZ | 訓練資料起始 |
| `training_window_end` | TIMESTAMPTZ | 訓練資料終止 |
| `features_used` | JSONB | feature list（例：`["wq001","wq002",...]`） |
| `hyperparams` | JSONB | 超參數 dict |
| `holdout_metrics` | JSONB | holdout IC / RMSE / direction_acc 等 |
| `status` | VARCHAR(20) | shadow / production / archived / failed |
| `regime_fingerprint` | JSONB | 訓練期間的市場 regime 特徵（給 recurring concept 用） |
| `parent_model_id` | VARCHAR(50) | 前一版模型（追蹤血緣） |
| `artifact_path` | VARCHAR(200) | 模型檔案路徑（pickle / MLflow URI） |

**索引**：`ix_model_status` (status)

**寫入者**：`src/meta_signal/ml_meta_model.py::register_to_registry()`、`src/adaptation/model_registry.py`

---

### 12. `alpha_registry` — Alpha 因子目錄

**用途**：所有 alpha 的 metadata 與啟用狀態，供 batch_compute 動態 dispatch 與 alpha_monitor 取 lookback window。

| 欄位 | 型別 | 說明 |
|------|------|------|
| `alpha_id` | VARCHAR(20) PK | 例：'wq001' |
| `alpha_name` | VARCHAR(50) | 例：'WQAlpha1' |
| `category` | VARCHAR(20) | price_volume / industry_aware / cap_aware |
| `requires_industry` | BOOLEAN | 是否需要 industry code 輸入 |
| `requires_cap` | BOOLEAN | 是否需要 market cap 輸入 |
| `lookback_window` | INT | 計算所需歷史長度 |
| `is_active` | BOOLEAN | 是否啟用（false 可暫停某個 alpha） |
| `notes` | TEXT | 公式描述 |

**Seed data**：migration 中內建 15 個 WQ101 alpha（wq001, wq002, …, wq041）

---

### 13. `regime_pool` — 重現概念池表

**用途**：Adaptation Policy 3 — recurring concept reuse。當偵測到的市場 regime 與歷史相似時，重用對應的舊模型。

| 欄位 | 型別 | 說明 |
|------|------|------|
| `regime_id` | VARCHAR(50) PK | regime 識別碼 |
| `detected_at` | TIMESTAMPTZ | 首次偵測到該 regime 的時間 |
| `fingerprint` | JSONB | regime 特徵向量（例：vol / trend / dispersion） |
| `associated_model_id` | VARCHAR(50) FK | → `model_registry.model_id` |
| `associated_alpha_weights` | JSONB | 該 regime 下 alpha 的最佳權重 |
| `performance_summary` | JSONB | 歷史在此 regime 的表現摘要 |
| `times_reused` | INT | 重用次數 |
| `last_reused_at` | TIMESTAMPTZ | 最後一次重用時間 |

**寫入者**：`src/adaptation/recurring_concept.py`

---

## 三、DolphinDB — 時序資料與串流

DolphinDB 是專為時序量化設計的資料庫，DARAMS 用兩個 DFS database 與兩個 streaming table。

定義於 `dolphindb/scripts/setup_database.dos`。

### A. `dfs://darams_market` / `standardized_bars` — 標準化市場資料分區表

**用途**：所有 OHLCV 資料的標準化儲存格式，是 wq101alpha 計算的輸入來源。

**分區**：
- 第一層：`VALUE` partition by date（每日一個分區，2020.01.01 ~ 2030.12.31）
- 第二層：`HASH` partition by SYMBOL（50 個 hash buckets）
- 引擎：`TSDB`，sortColumns = `security_id, tradetime`

| 欄位 | 型別 | 說明 |
|------|------|------|
| `security_id` | SYMBOL | 標的 |
| `tradetime` | TIMESTAMP | 交易時間 |
| `bar_type` | SYMBOL | daily / 5m / 1m |
| `open/high/low/close` | DOUBLE | OHLC |
| `vol` | DOUBLE | 成交量 |
| `vwap` | DOUBLE | 成交量加權均價（缺值時用 (high+low+close)/3 補算） |
| `cap` | DOUBLE | 市值（給 cap-aware alpha） |
| `indclass` | INT | 產業分類碼 |
| `is_tradable` | BOOL | 該日是否可交易（停牌/漲跌停限制） |
| `missing_flags` | INT | 缺值位元旗標（debug 用） |

**寫入者**：`src/standardization/schema_mapper.py` → DolphinDB client `tableInsert`

---

### B. `dfs://darams_alpha` / `alpha_features` — Alpha 因子分區表

**用途**：批次計算的 alpha 結果儲存，**long format**（一筆資料 = 一個 (security, time, alpha) 三元組的值）。

**分區**：與 `standardized_bars` 同樣 VALUE+HASH 雙層，sortColumns = `security_id, alpha_id, tradetime`

| 欄位 | 型別 | 說明 |
|------|------|------|
| `security_id` | SYMBOL | 標的 |
| `tradetime` | TIMESTAMP | 計算時點 |
| `bar_type` | SYMBOL | daily / 5m |
| `alpha_id` | SYMBOL | 例：'wq001'（對應 PG 的 `alpha_registry.alpha_id`） |
| `alpha_value` | DOUBLE | 該 alpha 在該標的該時點的值 |
| `alpha_version_id` | SYMBOL | alpha 版本（公式變更時遞增） |
| `computed_at` | TIMESTAMP | 計算寫入時間（debug 用） |

**重要設計**：long format（非 wide）— 不會因為新增 alpha 而改 schema，且查詢 single alpha 時可用分區裁剪加速。

**寫入者**：`dolphindb/modules/alpha_batch.dos::computeAllAlphas()` → `matrixToLongTable()` flatten

---

### C. `standardized_stream` — 即時市場資料串流表（共享）

**用途**：線上 pipeline 的市場資料入口，Shioaji 串流推進此表，下游 stream engine 訂閱。

| 欄位 | 型別 | 說明 |
|------|------|------|
| `security_id` | SYMBOL | 標的 |
| `tradetime` | TIMESTAMP | 時間 |
| `open/high/low/close/vol/vwap/cap` | DOUBLE | OHLCV + 衍生 |
| `indclass` | INT | 產業 |

**容量**：100,000 列 ring buffer

**寫入者**：`src/ingestion/shioaji_stream.py`、`pipelines/replay_pipeline.py::push_bar_df()`

---

### D. `alpha_output_stream` — Alpha 串流計算輸出表（共享）

**用途**：DolphinDB streamEngineParser 計算出的 alpha 即時推送至此表，Python 端可訂閱回 callback。

| 欄位 | 型別 | 說明 |
|------|------|------|
| `security_id` | SYMBOL | 標的 |
| `tradetime` | TIMESTAMP | 計算時點 |
| `alpha_id` | SYMBOL | alpha 識別碼 |
| `alpha_value` | DOUBLE | alpha 數值 |

**容量**：100,000 列 ring buffer

**訂閱者**：`src/alpha_engine/stream_compute.py::subscribe_output()`

---

## 四、Redis — 快取與即時狀態

Redis 不持久化關鍵業務資料，僅作為**讀寫頻繁但可重建**的快取層。連線設定於 `src/common/db.py::get_redis()`。

| Key Pattern | 型別 | 用途 | TTL |
|-------------|------|------|-----|
| `position:{security_id}` | Hash | 當前持倉快照（quantity, avg_cost, market_value） | 無（隨成交更新） |
| `signal:latest:{security_id}` | String (JSON) | 最新 meta_signal（給 API 即時查詢） | 1 day |
| `alert:state:{metric_name}` | String | Alert 去重狀態（避免短時間重複觸發） | 1 hour |
| `model:active` | String | 當前 production model_id | 無 |
| `regime:current` | String (JSON) | 當前偵測到的 regime fingerprint | 1 hour |

**重建來源**：所有 Redis key 都可從 PostgreSQL 重建（positions / meta_signals / alerts / model_registry）。Redis 掛了不會丟資料，只會增加查詢延遲。

---

## 五、資料流與寫入時機

下表整理「資料流經哪一張表 / 由哪個模組寫入 / 何時觸發」：

| 階段 | 寫入目標 | 模組 | 觸發時機 |
|------|----------|------|----------|
| Ingestion | PG `raw_market_events` | `ingestion/historical_loader.py` | 載入 CSV / 收到 tick |
| Standardization | DolphinDB `standardized_bars` | `standardization/schema_mapper.py` | Ingestion 完成後 |
| Alpha 計算（批次） | DolphinDB `alpha_features` | `alpha_engine/batch_compute.py` | 收盤後 daily batch |
| Alpha 計算（串流） | DolphinDB `alpha_output_stream` | `alpha_engine/stream_compute.py` | 即時，每根 bar |
| Meta Signal | PG `meta_signals` | `meta_signal/signal_generator.py` | Alpha 計算完成後 |
| Portfolio | PG `portfolio_targets` | `portfolio/constructor.py` | 每次 rebalance |
| Risk Manager | PG `portfolio_targets` (pre_risk=FALSE) | `risk/risk_manager.py` | Portfolio 後 |
| Order | PG `orders` | `execution/paper_engine.py` | 風控通過後 |
| Fill | PG `fills` | `execution/paper_engine.py` | 模擬成交 |
| Position | PG `positions` + Redis `position:*` | `pipelines/daily_batch_pipeline.py` | 每日收盤 |
| Label | PG `labels_outcomes` | `labeling/label_generator.py` | T+horizon+buffer |
| Monitoring | PG `monitoring_metrics` | `monitoring/alert_manager.py::persist_metrics()` | 每次 monitor 跑完 |
| Alert | PG `alerts` | `monitoring/alert_manager.py::fire_alerts()` | metric 越界時 |
| Model 訓練 | PG `model_registry` | `meta_signal/ml_meta_model.py::register_to_registry()` | 每次 retrain |
| Regime 入池 | PG `regime_pool` | `adaptation/recurring_concept.py` | 偵測到新 regime |

---

## 六、查詢範例

### 1. 取某標的最近一個月的訊號與報酬比對

```sql
SELECT s.signal_time, s.signal_score, l.forward_return
FROM meta_signals s
LEFT JOIN labels_outcomes l
  ON l.security_id = s.security_id
 AND l.signal_time = s.signal_time
 AND l.horizon = 5
WHERE s.security_id = '2330'
  AND s.signal_time >= now() - INTERVAL '30 days'
ORDER BY s.signal_time;
```

### 2. 列出所有未確認的 CRITICAL alerts

```sql
SELECT alert_time, monitor_type, metric_name, current_value, threshold, message
FROM alerts
WHERE severity = 'CRITICAL'
  AND is_acknowledged = FALSE
ORDER BY alert_time DESC;
```

### 3. 查詢某個 metric 的時序（給 Grafana panel）

```sql
SELECT metric_time, metric_value
FROM monitoring_metrics
WHERE monitor_type = 'alpha'
  AND metric_name = 'alpha_value_psi'
  AND dimension = 'wq001'
  AND metric_time >= $__timeFrom()
  AND metric_time <= $__timeTo()
ORDER BY metric_time;
```

### 4. DolphinDB — 取某標的某段時間的 alpha matrix

```dolphindb
alpha_table = loadTable("dfs://darams_alpha", "alpha_features")
result = select security_id, tradetime, alpha_id, alpha_value
         from alpha_table
         where tradetime between 2024.01.01 : 2024.12.31
           and security_id in ['2330','2317','2454']
           and alpha_id in ['wq001','wq002','wq014']
```

### 5. 找出當前 production 模型

```sql
SELECT model_id, trained_at, holdout_metrics
FROM model_registry
WHERE status = 'production'
ORDER BY trained_at DESC
LIMIT 1;
```

---

## 附錄：連線設定

預設連線參數（`.env` / `src/config/settings.py`）：

| 服務 | Host | Port | 預設帳密 |
|------|------|------|----------|
| PostgreSQL | localhost | **5433** | postgres / postgres |
| DolphinDB | localhost | 8848 | admin / 123456 |
| Redis | localhost | 6379 | （無密碼） |
| Grafana | localhost | 3000 | admin / admin |

> PostgreSQL 使用 5433 而非預設 5432，是為了避開本機已安裝的 PostgreSQL。
