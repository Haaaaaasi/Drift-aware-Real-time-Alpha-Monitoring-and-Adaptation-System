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

DARAMS（Drift-aware Real-time Alpha Monitoring and Adaptation System）是一個以台股日頻資料為研究對象的量化研究系統，重點不是尋找單一最高報酬策略，而是建立一條可以被重現、監控與比較的 alpha lifecycle：從資料標準化、WorldQuant 101 Alpha 特徵計算、meta model 訊號生成、portfolio construction、risk / execution、delayed labeling，到 drift monitoring 與 adaptation。

本專案的核心設計是把 WorldQuant 101 Alpha 視為 **feature engine**，而不是直接交易訊號。所有 alpha 會先經過 IC 篩選、point-in-time feature snapshot 與 meta model 聚合，再進入投組與風控流程。這樣可以避免把單一 alpha 的短期績效誤認為穩定策略，也能讓後續監控與 adaptation 有明確的比較基準。

正式研究路徑使用 TEJ survivorship-correct parquet 作為資料來源，並以 Python pandas 版本 WQ101 作為預設 alpha engine。DolphinDB 相關程式介面僅作為可選 real mode 連接，不是離線研究與每日預測的必要依賴。為避免 look-ahead bias，正式 alpha universe 使用 IS-only selection，且 OOS 實驗期間固定為 2024-07-01 至 2026-04-30。

Final Report 階段的 reviewer-facing baseline 採用：

- 資料來源：`data/tw_stocks_tej.parquet`
- Alpha 清單：`reports/alpha_ic_analysis/effective_alphas.json`
- Portfolio：`turnover_aware_topk`
- Execution：`next_vwap` 為主、`next_open` 為輔
- Baseline strategy：`scheduled_20`

在 frozen OOS 設定下，`scheduled_20` 於 `next_vwap` execution 的累積報酬為 `22.337%`、Sharpe 為 `0.587`、最大回撤為 `-41.907%`；於 `next_open` execution 的累積報酬為 `36.606%`、Sharpe 為 `0.771`、最大回撤為 `-36.374%`。結果顯示 adaptation baseline 明顯優於 no-adapt 與一般等權 benchmark，但相對 liquidity-filtered benchmark 的優勢較小，且 drawdown 仍偏深。因此本專案的最終結論是：drift-aware adaptation 在此資料與成本假設下具有可觀察的改善效果，但仍需謹慎處理交易成本、流動性與風險暴露，不應解讀為已可直接上線的交易策略。

### 使用方式

本 repo 保留正式展示與可執行主流程；大型資料、研究中間產物、舊測試與 agent memory 不納入 Git。若要重現正式流程，請先準備 Python 3.11 與 TEJ 資料檔。

#### 1. 安裝環境

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
```

如需使用 PostgreSQL、Redis、Grafana 或 API，請先建立本機環境變數：

```powershell
Copy-Item .env.example .env
```

#### 2. 準備 TEJ 資料

正式路徑預設讀取：

```text
data/tw_stocks_tej.parquet
data/tw_stocks_tej_universe.parquet
```

若手上是 TEJ 匯出的原始 CSV，可使用保留的 ingestion script 轉成正式 parquet：

```powershell
python scripts/ingest_tej_csv.py --input OHLSV20182022.csv OHLSV202320260502.csv
```

Alpha selection 的正式清單已保留在：

```text
reports/alpha_ic_analysis/effective_alphas.json
```

這個檔案是 runtime 會讀取的必要設定，不屬於一般 report artifact。

#### 3. 執行離線研究 pipeline

先確認 CLI 入口可正常載入：

```powershell
python -m pipelines.daily_batch_pipeline --help
python -m pipelines.predict_next_day --help
python -m pipelines.live_daily_runner --help
```

不依賴外部資料的快速 smoke run：

```powershell
python -m pipelines.daily_batch_pipeline --synthetic --start 2024-01-01 --end 2024-01-15
```

使用 TEJ parquet 與 Python WQ101 的預設正式路徑：

```powershell
python -m pipelines.daily_batch_pipeline --data-source tej
```

使用 XGBoost meta model：

```powershell
python -m pipelines.daily_batch_pipeline --data-source tej --signal-method ml_meta
```

指定期間：

```powershell
python -m pipelines.daily_batch_pipeline --data-source tej --start 2024-07-01 --end 2026-04-30
```

#### 4. 產生下一交易日目標持股

預設使用 TEJ parquet、Python WQ101 與 `effective_alphas.json`。這是正式推論流程，會讀取 TEJ parquet、alpha cache 並訓練當次 meta model；首次執行或 cache miss 時可能需要數分鐘以上，不適合作為快速 smoke test。

```powershell
python -m pipelines.predict_next_day --data-source tej --top-k 10
```

指定 as-of date：

```powershell
python -m pipelines.predict_next_day --data-source tej --as-of 2026-04-30 --top-k 10
```

輸出會寫入：

```text
reports/predictions/
```

#### 5. 每日 live workflow

Live workflow 需要本機已有 TEJ parquet、alpha cache，且 `predict-only` 模式需要既有 production model artifact。若只是檢查入口，請先使用 `python -m pipelines.live_daily_runner --help`。

若要把每日 TEJ CSV append 到正式 parquet，並接著產生 live recommendation：

```powershell
python -m pipelines.live_daily_runner --tej-input TEJ_YYYYMMDD.csv
```

只檢查 append，不執行 live pipeline：

```powershell
python -m pipelines.live_daily_runner --tej-input TEJ_YYYYMMDD.csv --dry-run-ingest
```

已有 production artifact 時，只用既有資料跑 predict-only：

```powershell
python -m pipelines.live_daily_runner --mode predict-only --production-artifact artifacts/models/<model_id>
```

#### 6. 啟動 API 與 dashboard 服務

```powershell
docker compose up -d postgres redis grafana
python main.py api
```

API 啟動後可開啟：

```text
http://127.0.0.1:8000/live
```

若要直接用 Docker Compose 啟動 API service：

```powershell
docker compose up -d api
```

#### 7. 注意事項

- 正式研究請使用 `--data-source tej`；yfinance CSV 僅保留 demo / 反例用途，必須明確加上 `--allow-yfinance` 才能使用。
- DolphinDB 不是預設離線研究依賴，只有在指定 real mode 或 `--alpha-source dolphindb` 時才需要。
- `reports/alpha_ic_analysis/effective_alphas.json` 是必要設定檔，請勿刪除或移動。
- `memory/`、`tests/`、`notebooks/`、大型 reports 與一次性診斷腳本已從正式 repo 中排除，以維持乾淨展示狀態。
