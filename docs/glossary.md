# DARAMS 專有名詞表

> 更新日期：2026-04-16  
> 本文件收錄 DARAMS 專案中常見的量化金融、機器學習與系統架構專有名詞，依字母與中文分類排列。

---

## 目錄

1. [Alpha 因子相關](#1-alpha-因子相關)
2. [監控指標相關](#2-監控指標相關)
3. [機器學習相關](#3-機器學習相關)
4. [市場 Regime 相關](#4-市場-regime-相關)
5. [系統架構相關](#5-系統架構相關)
6. [交易執行相關](#6-交易執行相關)
7. [資料庫與基礎設施相關](#7-資料庫與基礎設施相關)

---

## 1. Alpha 因子相關

### Alpha / Alpha 因子
量化交易中，用來預測資產未來超額報酬的數學訊號。DARAMS 使用 WorldQuant 101 Alpha（WQ101）作為因子來源，共 101 個純價量因子公式。

### WQ101 / WorldQuant 101 Alpha
由 WorldQuant 於論文《101 Formulaic Alphas》公開的 101 個 Alpha 因子公式，以價格、成交量等市場資料計算，不依賴基本面或財務報表。DARAMS 實作其中 15 個純價量因子（wq001–wq015）。

### Alpha 退化（Alpha Decay）
Alpha 因子的預測能力隨時間下降的現象。市場結構改變、套利壓力、制度性變化都可能導致退化。DARAMS 的核心研究問題之一。

### Alpha Feature Universe
所有可用 Alpha 因子的集合，作為下游模型的輸入特徵空間。DARAMS 以 WQ101 的輸出作為 feature universe。

### Effective Alpha（有效 Alpha）
通過 IC 篩選標準（`|rank-IC| >= 0.01` 且 `coverage >= 0.80`）的 Alpha 子集。WP2 分析結果為 10/15 個入選。

### IC（Information Coefficient，資訊係數）
Alpha 因子預測值與實際未來報酬之間的 Pearson 相關係數，衡量因子預測準確度。範圍 [-1, 1]，絕對值越大代表預測力越強。

### Rank-IC（排名資訊係數）
使用 Spearman 排名相關計算的 IC，對極端值更穩健。量化研究中比 IC 更常用。

### Rolling IC（滾動資訊係數）
在滑動時間窗口（如 60 日）上計算的 IC 序列，用來觀察因子預測能力的時序穩定性。

### Hit Rate（命中率）
Alpha 因子預測方向正確的比率（預測漲而實際漲，或預測跌而實際跌）。在小股票池（如 10 檔）中噪聲較大，DARAMS WP2 中不作為篩選條件。

### Coverage（覆蓋率）
Alpha 因子在所有股票 × 交易日組合中，非 NaN 值的比率。Coverage 低代表因子對部分股票或時段無法計算。

### Alpha Panel Matrix
DolphinDB 中 Alpha 計算的輸入格式：行為交易日、列為股票的二維矩陣（panel matrix）。每個 WQAlpha 函式接收獨立的 panel matrix，而非字典。

### Long Format（長格式）
將 panel matrix 展開為 `(security_id, tradetime, alpha_id, alpha_value)` 四欄的資料表格式。DARAMS 的 `alpha_features` 表採用此格式儲存。

---

## 2. 監控指標相關

### KS-test（Kolmogorov-Smirnov 檢定）
非參數統計檢定，比較兩個分布是否相同。DARAMS 用於偵測 Alpha 因子分布是否相對基準期發生顯著漂移。p-value 越低代表分布差異越顯著。

### PSI（Population Stability Index，族群穩定指數）
量化兩個分布差異程度的指標，常用於偵測資料或模型輸入分布的漂移。計算方式為分組後的 KL 散度加總。  
判斷標準：
- `PSI < 0.10`：無顯著漂移
- `0.10 ≤ PSI < 0.25`：輕微漂移（WARN）
- `PSI ≥ 0.25`：嚴重漂移（CRITICAL）

### ECE（Expected Calibration Error，期望校準誤差）
衡量模型預測機率與實際頻率之間的差距。ECE 越低代表模型校準越好（預測 60% 信心的樣本，實際上約 60% 正確）。

### Drift（漂移）
統計意義上，資料或模型的分布相對基準期發生顯著變化的現象。DARAMS 監控三類漂移：資料漂移、Alpha 分布漂移、模型校準漂移。

### Concept Drift（概念漂移）
目標變數（如未來報酬）與輸入特徵之間的關係發生變化。與資料漂移（輸入分布改變）不同，概念漂移代表模型學到的規律本身失效。

### Monitoring Metrics（監控指標）
所有監控計算結果統一寫入 PostgreSQL 的 `monitoring_metrics` 表，schema 為 `(metric_time, monitor_type, metric_name, metric_value, dimension, window_size)`。

### Alert（警報）
當監控指標超過閾值時觸發的通知。DARAMS 有四個監控層各自定義警報規則，統一由 `AlertManager` 管理並寫入資料庫。

### Data Monitor（資料監控）
第九層監控的第一子模組，負責偵測原始市場資料的品質問題：缺失率、異常價格、成交量異常、分布漂移（KS + PSI）。

### Alpha Monitor（Alpha 監控）
第九層監控的第二子模組，負責偵測 Alpha 因子的訊號品質退化：rolling IC、因子周轉率、因子值分布 PSI。

### Model Monitor（模型監控）
第九層監控的第三子模組，負責偵測 Meta Model 的預測品質退化：方向準確率、ECE 校準誤差、預測值分布 KS 檢定。

### Strategy Monitor（策略監控）
第九層監控的第四子模組，負責偵測整體策略績效：rolling Sharpe、最大回撤、實際報酬 vs 預期報酬偏差。

---

## 3. 機器學習相關

### Meta Model（元模型）
在 Alpha 因子之上學習如何聚合多個 Alpha 訊號的模型。DARAMS 使用 XGBoost 回歸模型（`XGBRegressor`）預測未來 5 日報酬，輸出供 Portfolio 層排序。

### XGBoost（Extreme Gradient Boosting）
梯度提升決策樹演算法，DARAMS WP3 採用其回歸版本（`XGBRegressor`）作為 Meta Model，輸出連續預測值（非分類機率）。

### Purged Expanding-Window CV（清除式擴張窗口交叉驗證）
時序資料專用的交叉驗證方法。訓練集與驗證集之間插入 `purge_days`（清除期）間隔，防止 label leakage（標籤洩漏）。DARAMS 使用 3-fold，`purge_days=5`。

### Look-ahead Bias（前視偏差）
在模型訓練或回測中不當使用了未來資訊，導致績效虛高。DARAMS 以 `signal_time` 與 `label_available_at` 嚴格分離避免此問題。

### Delayed Label（延遲標籤）
Label 只有在 `signal_time + horizon + buffer` 之後才可用。DARAMS 使用 5 日 horizon + 1 日 buffer，確保不使用未來資料訓練模型。

### Feature Importance（特徵重要性）
模型中每個輸入特徵對預測結果的貢獻度。DARAMS 使用 XGBoost 的 Gain-based importance，自動對應回 Alpha id。

### Shadow Evaluation（影子評估）
在不影響正式生產系統的情況下，並行評估候選新模型的機制。DARAMS 的 `ShadowEvaluator` 比較現有模型與候選模型的 holdout IC，決定是否 promote。

### Model Registry（模型登錄表）
記錄所有訓練過模型的元資料，包含：`model_id, trained_at, training_window, features_used, hyperparams, holdout_metrics, status, regime_fingerprint`。

---

## 4. 市場 Regime 相關

### Regime（市場情態）
市場在特定時期呈現的統計特性，如趨勢、波動率、相關性等。DARAMS 將 2022–2024 台股分為三個 regime：熊市、反彈、盤整。

### Regime Fingerprint（情態指紋）
描述特定 Regime 特徵的統計摘要向量（如 volatility, trend, correlation 等），用於比對歷史 regime 相似度，支援 Recurring Concept Pool 的選模邏輯。

### Bear Market（熊市）
市場整體下跌超過 20% 的時期。DARAMS 中對應 2022 年（升息循環、科技股殺估值）。

### Recovery（反彈）
市場從低點回升的時期。DARAMS 中對應 2023 年（AI 浪潮、資金回流）。

### Consolidation（盤整）
市場在高位橫盤、漲跌幅收窄的時期。DARAMS 中對應 2024 年（高點輪動整理）。

---

## 5. 系統架構相關

### DARAMS
Drift-aware Real-time Alpha Monitoring and Adaptation System 的縮寫。本專案全名。

### Pipeline（流水線）
一系列依序執行的資料處理步驟。DARAMS 有四條主要 pipeline：`daily_batch`、`monitoring`、`adaptation`、`label_update`。

### Adaptation（自適應）
在監控偵測到系統退化後，自動或半自動調整模型或策略的機制。DARAMS 實作三種 policy：
1. **Scheduled Retraining**：定期（如每 30 日）重新訓練
2. **Performance-triggered**：績效或漂移指標超閾值時觸發
3. **Recurring Concept Pool**：找出歷史相似 regime 期間的優質模型重新啟用

### Scheduled Retraining（排程重訓）
Policy 1：依固定時間間隔（如每月）重新訓練模型，不管目前績效是否退化。

### Performance-triggered Adaptation（績效觸發式自適應）
Policy 2：當監控指標（IC 退化、critical PSI alerts 累積）超過閾值時，自動觸發重新訓練流程。

### Recurring Concept Pool（循環概念池）
Policy 3：維護一個歷史模型池，當偵測到目前 regime 與某個歷史 regime 相似時，直接複用對應時期訓練的模型，無需重訓。

### Replay Pipeline（重播流水線）
用歷史 CSV 資料模擬串流推送，逐筆送入 DolphinDB streamEngineParser，驗證串流計算結果與批次計算結果的一致性。

### Expanding Window（擴張窗口）
每次計算納入截至當前的所有歷史資料（訓練集從頭擴張到當前點），模擬真實的 online 學習場景，不使用未來資料。

### Sliding Window（滑動窗口）
固定長度的時間窗口在時序資料上滑動，用於計算 rolling 統計量（如 rolling IC、rolling PSI）。

### Panel Matrix
行為時間、列為股票的二維矩陣格式，為 DolphinDB WQAlpha 函式的標準輸入格式。

### Standardized Bars
經過標準化處理後的 OHLCV 市場資料，統一欄位名稱、補算 VWAP 與市值，以 DolphinDB 分區表格式儲存。

---

## 6. 交易執行相關

### Paper Trading（模擬交易）
不使用真實資金、不實際下單的交易模擬，用於驗證策略邏輯與績效計算。DARAMS 的 Execution 層為 paper trading 引擎。

### Rebalance（再平衡）
依照最新訊號重新調整投資組合權重的操作。DARAMS 每個交易日執行一次。

### Target Weight（目標權重）
Portfolio 層輸出的每檔股票目標持有比例，由 Alpha signal 排序後等權或依訊號強度分配。

### Turnover（周轉率）
投資組合在一段期間內買賣換手的比率，高周轉率代表交易成本較高。Risk 層會對周轉率設定上限。

### Position Cap（部位上限）
單一股票最大持有比例的限制，用於控制集中度風險。DARAMS Risk 層預設為 20%。

### Drawdown（回撤）
從策略淨值的歷史高點下跌的幅度。Max Drawdown（最大回撤）代表整個回測期間最大的峰谷跌幅。

### Sharpe Ratio（夏普比率）
超額報酬（報酬 - 無風險利率）除以報酬標準差，衡量每單位風險獲得的超額報酬。值越高代表風險調整後績效越好。

### Profit Factor（獲利因子）
所有獲利交易的總利潤除以所有虧損交易的總虧損絕對值。> 1 代表系統整體獲利。

### IC-weighted Composite Signal（IC 加權合成訊號）
Rule-based Meta Signal 的方法：以各 Alpha 的歷史 IC 作為權重，加權合成最終交易訊號。

---

## 7. 資料庫與基礎設施相關

### DolphinDB
高效能時序資料庫與分析平台，原生支援 panel matrix 運算，DARAMS 用於 Alpha 計算（wq101alpha 模組）與時序資料儲存。不可替代。

### PostgreSQL
關聯式資料庫，DARAMS 用於儲存元資料（15 張表）：訂單、成交、部位、模型登錄、警報、監控指標等。本機佔用 5432 port，Docker 容器改用 5433。

### Redis
記憶體資料庫，DARAMS 用於快取即時狀態：當前部位、最新訊號、警報狀態。

### MLflow
機器學習實驗追蹤平台，用於記錄模型訓練的超參數、指標與 artifacts，支援模型版本比較。（DARAMS v3 規劃整合）

### Grafana
資料視覺化與監控 Dashboard 工具，DARAMS 連接 PostgreSQL 資料來源，展示四個監控面板（Data / Alpha / Model / Strategy）。

### APScheduler（Advanced Python Scheduler）
Python 輕量排程函式庫，DARAMS 用於觸發定期 pipeline 執行（daily batch、scheduled retrain 等），不使用 Airflow。

### Docker Compose
容器編排工具，DARAMS 以 `docker-compose.yml` 統一管理四個服務：DolphinDB、PostgreSQL、Redis、Grafana。

### Migration（資料庫遷移）
使用 SQL 腳本（`migrations/001_init_tables.sql`）建立或更新資料庫 schema，確保各環境的資料庫結構一致。

### Provisioning（自動佈建）
Grafana 的自動設定機制，透過 YAML 設定檔在容器啟動時自動載入 datasource 與 dashboard，無需手動操作。

### TSDB（Time Series Database）
DolphinDB 的時序資料儲存模式，需指定正確的 `sortColumns` 排序欄位（`security_id, tradetime`）。

### streamEngineParser
DolphinDB 內建的串流計算引擎，支援以 SQL 表達式定義串流計算邏輯，DARAMS 的 Replay Pipeline 用其驗證批次與串流的一致性。

---

*如有新增名詞，請依所屬類別補充至本文件。*
