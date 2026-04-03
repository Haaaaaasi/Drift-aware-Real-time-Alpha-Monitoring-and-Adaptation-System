# Drift-aware-Real-time-Alpha-Monitoring-and-Adaptation-System

## Proposal Report

### 動機與目標
金融市場具有高度動態與非平穩的特性，同一組有效的交易訊號，往往會因市場結構改變、波動環境轉換或投資人行為改變而逐漸失效。傳統量化交易研究常把重點放在尋找新的 alpha，但相對較少系統化討論既有 alpha 在真實市場中如何被監控、診斷與調整，因此我希望以「alpha 的生命週期管理」為核心，設計一個能持續觀察與適應市場變化的研究型量化交易系統。

本專題的核心想法是，先不從零發明交易因子，而是使用 DolphinDB 提供的 WorldQuant 101 Alpha 因子庫作為既有的 alpha feature universe，將重點放在後續的 alpha aggregation、訊號產生、績效監控、漂移偵測與自適應調整機制上。

相較於單純做一個回測策略，本專題更希望回答的問題是：當市場進入非平穩或 recurring concept drift 的環境時，既有 alpha 何時會退化、如何被偵測，以及是否能透過定期重估、績效觸發調整或模型池重用機制，降低 alpha 失效所造成的損失。

因此，本專題的目標可分為三點：

建立一個從市場資料進入、因子計算、訊號產生、投組建構、風險控管、模擬執行，到標籤回補與監控分析的完整量化研究 pipeline。

實作一套針對 alpha、模型與策略績效的 monitoring 機制，觀察其在動態市場中的穩定度與退化現象。

設計 adaptation 流程，比較無調整、固定週期調整、績效觸發調整與 recurring concept reuse 等方法在不同市場情境下的效果差異。



### 預期功能

本系統預計採用模組化架構，資料流將依序經過 Data Ingestion、Data Standardization、Alpha Computation、Alpha Aggregation、Portfolio Construction、Risk Management、Execution、Labeling、Monitoring 與 Adaptation 等層，使系統具備從研究到模擬交易的完整流程。

預計實作的主要功能如下：

市場資料蒐集與整理：接收歷史價格資料與即時或模擬的市場事件資料，統一欄位格式，建立可重現的標準化資料表。

Alpha 因子計算：使用 DolphinDB 的 wq101alpha 與 prepare101 模組批次或增量計算 WorldQuant 101 Alpha 因子，並將計算結果儲存為 alpha feature store。

訊號生成：將多個 alpha 特徵經過標準化、加權或機器學習模型整合，產生最終的交易訊號分數與方向，而不是直接將單一 alpha 當作下單依據。

投組與風控：根據訊號分數建立目標部位，並施加單一標的權重上限、總曝險限制、換手率限制與流動性過濾等風險約束。

模擬交易執行：先以 paper trading 為主，模擬訂單、成交、滑價與持倉變化，作為後續策略績效與監控分析的基礎。

延遲標籤回補：針對未來 \(T\) 期報酬建立 delayed labels，將訊號時間與標籤可用時間明確分離，以避免資料洩漏與前視偏誤。

多層監控面板：分別監控資料品質、alpha 穩定度、模型預測分布與策略績效，例如 rolling IC、prediction drift、Sharpe ratio 與 max drawdown 等指標。

自適應調整機制：當監控結果顯示策略或模型退化時，觸發固定排程調整、績效觸發調整或 recurring concept pool 重用機制，形成封閉式的研究迴圈。

### 使用技術
本專題將以 Python 為主要開發語言，負責資料處理、後端服務、排程、監控與策略邏輯實作，並搭配 FastAPI 建立查詢或控制介面，以提升整體模組化與可維護性。

Alpha 因子計算核心則以 DolphinDB 為主，使用其 wq101alpha 與 prepare101 模組作為既有因子引擎，並可配合 streamEngineParser 支援批次與流式計算整合。

資料儲存方面，系統預計採用分層設計：

DolphinDB：儲存標準化 K 線資料與 alpha features，作為時間序列與因子計算核心。

PostgreSQL：儲存結構化的 metadata，例如 orders、fills、positions、alerts、model registry 與 monitoring metrics。

Redis：作為快取與輕量訊息流元件，用於保存最新訊號、當前狀態與部分即時任務協調。

模型與實驗管理方面，預計使用 MLflow 管理模型版本、訓練參數與評估結果，方便後續比較不同 meta model 或 adaptation policy 的效果。

系統部署初期將以 Docker Compose 為主，整合 DolphinDB、PostgreSQL、Redis、Grafana 與 Python 服務

### 時程規劃
第 1–2 週：基礎建設與離線資料流程
建立 monorepo 專案結構與 Docker Compose 基礎環境。

建立 PostgreSQL 與 DolphinDB 的資料表與初始 schema。

實作 historical data loader，將歷史資料匯入 raw tables。

完成 standardization pipeline，將資料整理成可供 prepareData 使用的格式。

第 3-4 週：Alpha 計算與第一版訊號系統
實作 DolphinDB 批次 alpha 計算流程，先選 10–20 個 WorldQuant Alpha 因子作為 MVP。

建立 alpha feature store 與 alpha registry。

完成 rule-based composite signal 版本，利用標準化與 rolling IC 權重合成訊號。

建立第一版 portfolio construction、risk management 與 paper trading engine。

第5-6 週：評估與監控
建立 delayed label generator，計算 forward return 與 future direction。

實作 alpha metrics、model metrics、strategy metrics 三層評估模組。

建立基本監控面板，包含 rolling IC、rolling Sharpe、missing ratio 與 drawdown 等指標。

跑出第一份完整 backtest 與 monitoring report。

第 7-8 週：進階 adaptation 與系統強化
加入 performance-triggered adaptation 機制。

規劃 recurring concept pool / ECPF-like reuse 架構，作為進階研究擴充方向。

視進度加入 streaming ingestion 與 incremental alpha computation。

整理實驗結果、撰寫文件與製作最終展示版本。

### 與課程的關聯
<!-- 你的專題可能涉及哪些資料結構或演算法概念？為什麼？ -->
首先，在 資料結構 層面，系統需要設計多種表格與索引結構來支援時間序列資料、alpha features、交易紀錄、監控指標與模型版本管理，例如以 (security_id, tradetime) 為核心鍵值建立可查詢、可追溯、可重播的資料儲存結構。

此外，若進一步實作 recurring concept pool，還會涉及 model pool、regime fingerprint 與歷史最佳權重的管理，本質上也是一種帶有檢索與比對需求的結構化資料設計。

在 演算法 層面，本專題至少會涉及以下概念：

排序與選擇問題：例如根據 signal score 選出 top-k 標的，或依照 alpha 表現篩選有效因子。

滑動視窗與時間序列計算：例如 rolling IC、rolling Sharpe、forward return、lookback window 與 drift metrics 的計算。

圖形化或關聯分析：例如 alpha correlation drift、regime fingerprint 比對與相似概念檢索。

最佳化與限制處理：例如 portfolio target 經過 risk constraints 修正後的部位配置。
---

## Prototype Report

### 目前進度
<!-- 完成了什麼 -->

### 遇到的困難
<!-- 遇到什麼問題、如何解決或打算如何解決 -->

### 下一步計畫
<!-- 接下來要做什麼 -->

### 與課程的關聯
<!-- 到目前為止，你的實作中哪些部分與課程內容有關？關係是什麼？ -->

---

## Final Report

### 專案說明
<!-- 完整描述你的專案做了什麼 -->

### 使用方式
<!-- 如何編譯、執行、使用你的程式 -->

### 與課程的關聯總結
<!-- 總結你的專題與進階程式設計及資料結構課程之間的關聯 -->
