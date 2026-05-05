# Drift-aware Real-time Alpha Monitoring and Adaptation System

## Proposal Report

### 動機與目標

量化交易模型常見的問題不是「一開始沒有訊號」，而是訊號會隨市場 regime、資料品質、交易成本與模型老化而逐步失效。本專案希望把這件事做成一個可重現、可監控、可比較的研究系統：以 WorldQuant 101 Alpha 作為 feature universe，建立從市場資料、alpha 計算、meta model、portfolio construction、risk management、paper execution、labeling、monitoring 到 adaptation 的完整流程。

本專案的研究目標不是追求單一最佳交易策略，而是回答三個問題：

1. WorldQuant 101 Alpha 在台股 TEJ 正式資料上是否仍具有可被模型利用的 OOS 訊號？
2. 當 alpha / model / strategy 退化時，能否用監控指標即時辨識？
3. scheduled retraining、performance-triggered adaptation、recurring concept reuse 等 adaptation 方法，能否在扣除真實交易成本後改善績效？

### 競品比較

| 類型 | 代表工具 | 優點 | 本專案差異 |
|---|---|---|---|
| 回測框架 | Backtrader、Zipline、QuantConnect | 回測與交易流程成熟 | 本專案重點放在 drift monitoring、delayed label 與 adaptation 實驗，而不是單純下單回測 |
| Alpha 函式庫 | WorldQuant 101 Alpha、alpha101 類套件 | 提供現成 alpha 公式 | 本專案將 WQ101 視為 feature engine，後續仍經過 meta model、portfolio、risk 與成本感知評估 |
| MLOps / 監控工具 | MLflow、Grafana | 方便追蹤模型與 dashboard | 本專案把 monitoring 指標直接接到 adaptation research flow，用來比較策略退化與重訓時機 |
| 一般量化 notebook | 個人研究 notebook | 快速探索 | 本專案保留模組化 pipeline、測試、資料來源 guardrail 與可重現實驗輸出 |

### 預期功能

- TEJ survivorship-correct 台股資料 ingestion 與 parquet 快取。
- Python pandas 版本 WorldQuant 101 Alpha 計算，DolphinDB 作為 streaming / real mode 備援。
- Alpha IC 分析與 IS-only effective alpha selection，避免 look-ahead bias。
- XGBoost meta model，將多個 alpha feature 聚合成股票排序訊號。
- QuantConnect 風格流程：Alpha → Portfolio → Risk → Execution。
- 四層監控：Data Monitor、Alpha Monitor、Model Monitor、Strategy Monitor。
- 三類 adaptation：scheduled retraining、performance-triggered retraining、recurring concept pool。
- WP9 adaptation A/B 實驗、成本敏感度分析、buy-and-hold benchmark 比較。
- yfinance 資料來源 guardrail：正式研究預設 TEJ，yfinance 僅保留 demo / 資料品質反例。

### 使用技術

| 類別 | 技術 |
|---|---|
| 語言 | Python 3.11 |
| 資料處理 | pandas、NumPy、PyArrow、parquet |
| 模型 | XGBoost |
| API | FastAPI |
| 監控與資料庫 | PostgreSQL、Redis、Grafana、DolphinDB |
| 實驗追蹤 | pipeline 輸出、CSV / Markdown reports，預留 MLflow |
| 測試 | pytest、pytest-asyncio |
| 部署 | Docker Compose |

### Prototype 預計可驗證內容

Prototype 階段希望驗證以下內容：

1. 市場資料可以被標準化成 alpha engine 可使用的 panel matrix。
2. WQ101 alpha 可以被批次計算並轉為 long format feature store。
3. Meta model 可以在 delayed label 設計下訓練，避免未來資訊外洩。
4. Portfolio / risk / execution 可以輸出每日 PnL 與交易成本拆解。
5. Monitoring 指標可以量化 alpha、model 與 strategy 的退化。
6. Adaptation 策略可以在同一段 OOS period 下做公平 A/B 比較。

---

## Prototype Report

### 目前進度

目前已完成端到端研究系統骨架與 WP9 adaptation 實驗主流程：

- 已建立 10 層模組化架構：ingestion、standardization、alpha engine、meta signal、portfolio、risk、execution、labeling、monitoring、adaptation。
- 已將正式研究資料來源統一為 TEJ parquet，包含 2018-01 至 2026-04 台股資料與下市股覆蓋。
- 已完成 yfinance guardrail，避免舊 yfinance 資料污染正式實驗。
- 已完成 Python WQ101 主路徑，讓預測與回測不再依賴 DolphinDB 大表。
- 已在 TEJ 上重做 IS-only alpha selection，目前使用 64 個 effective alphas。
- 已完成 WP9 成本感知實驗重設計，加入真實交易成本、buy-and-hold benchmark 與 horizon-aligned portfolio 診斷。
- 最新 WP9 結果顯示，`scheduled_20 + top_k=10 + rebalance_every=10 + train_window=500` 在 TEJ OOS、扣成本後累積報酬為 `+47.679%`，略勝等權 buy-and-hold benchmark 的 `+43.612%`，但 Sharpe 與最大回撤仍較弱。

### 遇到的困難

1. **yfinance 資料污染**

   舊 yfinance CSV 中曾出現股票 8476 的非物理性 split-adjustment artifact，導致舊 WP9 高累積報酬不可採信。已改為正式研究預設 TEJ，並要求 yfinance 必須透過 `--allow-yfinance` 明確解鎖。

2. **DolphinDB 大表 OOM**

   DolphinDB `alpha_features` 大表在本機環境容易因 TSDB metadata 與 redo log 造成 OOM。已新增 Python WQ101 + parquet alpha cache 作為預設離線研究路徑，DolphinDB 保留為 real mode / streaming 備援。

3. **交易成本吃掉 gross edge**

   Daily top-k 雖有 gross signal，但 turnover 過高，扣手續費、證交稅與滑價後績效轉差。最新實驗改為 5 日 target 搭配 10 日 rebalance，讓 turnover 降低後才出現可防守的正報酬候選。

4. **研究結果仍需整理成最終展示**

   目前已找到一組扣成本後仍為正報酬、且累積報酬略勝 buy-and-hold benchmark 的 horizon-aligned adaptation candidate。不過該策略的 Sharpe 與最大回撤仍弱於 benchmark，因此 Final Report 前還需要補完整穩健性檢驗、圖表與研究解讀，避免只用單一累積報酬作為結論。

### 下一步計畫

Final Report 前預計完成以下項目：

1. **完成 drift monitoring 到 adaptation 的閉環展示**

   將 Data Monitor、Alpha Monitor、Model Monitor、Strategy Monitor 的指標整理成同一條研究敘事：先偵測 alpha / model / strategy 退化，再由 adaptation policy 決定是否重訓或切換模型，呈現本專案「不是單純回測，而是可監控、可調適的量化研究系統」。

2. **補完整成本敏感度與統計檢定**

   對最佳候選執行 round-trip cost sweep、paired test、gross / cost / net return waterfall，確認策略在不同成本假設下是否仍能保留 edge。

3. **完成 benchmark 與風險面比較**

   除累積報酬外，同時比較 Sharpe、最大回撤、勝率、turnover、持股數與 buy-and-hold benchmark，將結論寫成「收益、成本、風險」三面向，而不是只看報酬率。

4. **完成 end-to-end 可重現 demo**

   整理一條從 TEJ 資料、alpha cache、meta model 訓練、portfolio construction、risk / execution、monitoring 到 adaptation 的可重現執行流程，讓使用者可以用固定指令重跑主要結果，並清楚知道每個輸出檔案代表什麼。

5. **補強 recurring concept reuse 的研究亮點**

   讓 `model_pool` 不只是另一種重訓策略，而是能展示「市場狀態相似時重用歷史模型」的研究概念；Final Report 中需說明 regime fingerprint、model reuse 條件，以及它與 scheduled / triggered adaptation 的差異。

6. **整理 Final Report 與展示材料**

   將 TEJ 資料處理、alpha selection、monitoring 指標、adaptation A/B、成本敏感度與最終研究結論整理成 notebook、圖表與 README Final Report 區塊，作為完整展示版本。

---

## Final Report

### 專案說明

<!-- Final Report 階段再補上完整專案說明 -->

### 使用方式

<!-- Final Report 階段再補上正式使用方式 -->
