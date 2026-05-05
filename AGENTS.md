# DARAMS — Codex 專案指引文件

本文件供 Codex（或任何 AI 編碼代理人）快速理解本專案的定位、架構、目前進度與待辦事項，以便在後續對話中能正確延續開發而不重複或誤解已有設計。

> **語言規範**：本專案所有文件（包含 AGENTS.md、docs/、notebooks/、reports/ 及任何新增文件）請一律使用**繁體中文**撰寫。程式碼內的變數名稱、函式名稱維持英文，但註解、說明文字、commit message 說明段落請使用繁體中文。

> **接手記憶規範**：舊 Claude Code memory 已搬入 `memory/`。每次新代理人接手時，必須先讀 `memory/MEMORY.md`，再依任務讀相關條目（例如 `project_yfinance_8476_artifact.md`、`project_tej_is_alpha_selection.md`、`project_model_pool_phase_b1.md`）。若產生會影響研究結論、預設流程、資料品質判斷或 reviewer 防守的新發現，請同步更新 `memory/MEMORY.md` 與對應細項檔，維持原 Claude memory 的使用方式。

---

## 一、專案定位與核心目標

### 專案名稱

Drift-aware Real-time Alpha Monitoring and Adaptation System（DARAMS）

### 一句話描述

以 WorldQuant 101 Alpha（**Python pandas 實作為主**，DolphinDB 為線上 streaming 備援）作為現成 alpha feature universe，研究其在非平穩市場下的失效、監控與自適應處理的模組化量化交易研究系統。

### 專案不是什麼

- 不是一個「找最強交易策略」的回測腳本
- 不是把 wq101alpha 的輸出直接當 final trading signal
- 不是高頻交易系統
- 不是只會「定期重訓」的 MLOps 範例

### 專案是什麼

- 一個 10 層模組化的量化研究系統，資料流遵循 QuantConnect 風格：Alpha → Portfolio → Risk → Execution
- 研究核心放在 **Monitoring**（四層監控）與 **Adaptation**（三種策略）
- 強調 delayed labels、look-ahead bias prevention、model/alpha versioning、reproducibility
- 可作為研究所申請作品與 GitHub 展示專案

### 三大研究問題

1. WorldQuant 101 Alpha 因子在不同市場 regime 下能否穩定產生訊號？
2. 如何系統化監控 alpha / model / strategy 的退化現象？
3. Adaptation（scheduled / performance-triggered / recurring concept reuse）能否量化改善績效？

---

## 二、系統架構總覽

### 資料流

```
Market Data → Standardization → Alpha Engine (Python wq101_python，DolphinDB 為 streaming 備援)
  → Alpha Aggregation / Meta Model → Portfolio Construction → Risk Management
  → Execution (Paper) → Labeling → Monitoring → Adaptation → (loop back)
```

### 十層模組

| Layer | 目錄 | 責任 |
|-------|------|------|
| 1. Data Ingestion | `src/ingestion/` | 原始市場資料接入（CSV / Shioaji） |
| 2. Standardization | `src/standardization/` | 欄位映射至 panel matrix 格式 |
| 3. Alpha Computation | `src/alpha_engine/` + `dolphindb/` | WQ101 批次計算（panel matrix → long format） |
| 4. Meta Signal | `src/meta_signal/` | Alpha 聚合（rule-based / ML / regime） |
| 5. Portfolio | `src/portfolio/` | Signal → target weights |
| 6. Risk | `src/risk/` | 風控約束（position cap, exposure, turnover, DD halt） |
| 7. Execution | `src/execution/` | Paper trading / order management / reconciliation |
| 8. Labeling | `src/labeling/` | Delayed label 產生 + 三層評估 |
| 9. Monitoring | `src/monitoring/` | Data / Alpha / Model / Strategy 四大 monitor |
| 10. Adaptation | `src/adaptation/` | Scheduled / Performance-triggered / Recurring concept pool |

### 支援模組

| 模組 | 目錄 | 說明 |
|------|------|------|
| Config | `src/config/` | Pydantic Settings + constants enums |
| Common | `src/common/` | DB connections, logging, metrics, time utils |
| API | `src/api/` | FastAPI with 4 route groups |
| Pipelines | `pipelines/` | 每日批次 / 監控 / Adaptation / Label 更新 |

---

## 三、技術選型

| 角色 | 技術 | 說明 |
|------|------|------|
| Alpha Engine | Python wq101_python（主）/ DolphinDB（備援） | 預測與離線回測走 Python pandas + parquet 快取（不依賴 Docker）；DolphinDB streamEngineParser 保留供線上 streaming |
| Backend | Python 3.11 + FastAPI | 主開發語言 |
| Time-series DB | DolphinDB（real mode）/ parquet cache（python_wq101 mode） | DolphinDB 含 standardized_bars / alpha_features，但因 53.8M rows OOM 已非預設路徑 |
| Metadata DB | PostgreSQL | 15 張表：orders, fills, positions, model_registry, alerts... |
| Cache | Redis | current positions, latest signals, alert state |
| Experiment Tracking | MLflow | model registry, experiment comparison |
| Scheduling | APScheduler | 輕量排程，不用 Airflow |
| Message Queue | Redis Streams | MVP 足夠 |
| Dashboard | Grafana | 連接 PostgreSQL + DolphinDB |
| Deployment | docker-compose | DolphinDB + PostgreSQL + Redis + Grafana + Python |
| Testing | pytest + pytest-asyncio | 102 tests currently passing |

---

## 四、目錄結構

```
DARAMS/
├── docker-compose.yml          # DolphinDB + PostgreSQL + Redis + Grafana（DolphinDB 已非預設路徑）
├── pyproject.toml              # Python 依賴 (hatchling build)
├── .env.example                # 環境變數範本
├── main.py                     # CLI entry: api / backtest / monitor / adapt
│
├── dolphindb/                  # 僅 real mode 使用，預測/回測已改走 Python WQ101
│   ├── scripts/                # setup_database.dos, load_wq101alpha.dos, setup_stream_engine.dos, compute_ic_summary.dos
│   ├── modules/                # alpha_batch.dos, alpha_stream.dos, data_ingestion.dos, standardization.dos
│   └── queries/                # alpha_query.dos, monitoring_query.dos
│
├── src/
│   ├── config/                 # settings.py (Pydantic), constants.py (enums)
│   ├── common/                 # db.py, logging.py, metrics.py, time_utils.py
│   ├── ingestion/              # historical_loader.py, shioaji_stream.py (mock), replay.py
│   ├── standardization/        # schema_mapper.py, calendar.py, quality_check.py
│   ├── alpha_engine/           # ★ wq101_python.py（純 Python 101 alphas，主路徑）
│   │                           # ★ alpha_cache.py（parquet 快取，冷啟動 5–10 分）
│   │                           # dolphindb_client.py, batch_compute.py, stream_compute.py, alpha_registry.py
│   ├── meta_signal/            # rule_based.py, ml_meta_model.py (XGBRegressor), regime_ensemble.py, signal_generator.py
│   ├── portfolio/              # constructor.py
│   ├── risk/                   # risk_manager.py
│   ├── execution/              # paper_engine.py, order_manager.py, reconciliation.py
│   ├── labeling/               # label_generator.py, evaluator.py
│   ├── monitoring/             # data_monitor.py, alpha_monitor.py, model_monitor.py, strategy_monitor.py, alert_manager.py
│   ├── adaptation/             # scheduler.py, performance_trigger.py, recurring_concept.py
│   │                           # shadow_evaluator.py, model_registry.py
│   │                           # ★ model_pool_strategy.py（WP11，第 5 適應策略 Recurring Concept Pool）
│   └── api/                    # app.py, schemas.py, routes/{monitoring,signals,adaptation,backtest}.py
│
├── pipelines/                  # daily_batch_pipeline.py, monitoring_pipeline.py, adaptation_pipeline.py, label_update_pipeline.py
│                               # ★ simulate_recent.py（5 策略 walk-forward 模擬器）
│                               # ★ ab_experiment.py（5 策略 A/B 框架）
│                               # ★ predict_next_day.py（每日預測，預設 --alpha-source python_wq101）
│                               # ★ replay_pipeline.py（streaming 一致性驗證）
│
├── configs/                    # alpha_config.yaml（v3_effective_alphas + v3_effective_alphas_is）
│                               # monitoring_config.yaml, adaptation_config.yaml, risk_config.yaml, pipeline_config.yaml
├── migrations/                 # 001_init_tables.sql (15 PostgreSQL tables + seed data)
├── scripts/                    # download_tw_stocks.py, generate_report.py, validate_infrastructure.py
│                               # seed_security_master.py, backfill_alpha.py, export_results.py
│                               # ingest_csv_to_dolphindb.py, run_is_oos_validation.py, demo_monitoring_2026.py
│                               # ★ ingest_tej_csv.py（TEJ Pro 還原股價 → parquet，含期間下市股）
├── data/
│   ├── tw_stocks_ohlcv.csv          # 1083 檔台股 2022-2026（yfinance，無下市股）
│   ├── tw_stocks_tej.parquet        # ★ 1105 檔（含 51 檔已下市），TEJ 還原股價 2018-01 → 2026-04
│   ├── tw_stocks_tej_universe.parquet  # ★ 每檔 (first_date, last_date, is_active_at_end)
│   └── alpha_cache/wq101_alphas.parquet  # ★ Python WQ101 全 101 alphas 快取（snappy）
├── reports/                    # backtest PNG + TXT, alpha_ic_analysis/, drift_experiment/,
│                               # adaptation_ab/, adaptation_evaluation/, predictions/
├── memory/                     # ★ 舊 Claude memory 搬入；接手前先讀 MEMORY.md，重大發現要同步更新
├── tests/
│   ├── unit/                   # test_standardization, test_meta_signal, test_risk, test_monitoring,
│   │                           # test_drift_metrics, test_ml_meta_model, test_wq101_python,
│   │                           # test_alpha_cache, test_alpha_ic_split
│   └── integration/            # test_pipeline_batch, test_adaptation_loop, test_replay_pipeline,
│                               # test_ab_experiment, test_model_pool_strategy
├── dashboards/                 # data_monitor.json, alpha_monitor.json, model_monitor.json,
│                               # strategy_monitor.json, home.json + provisioning/
├── notebooks/                  # 01_alpha_ic_analysis.py（支援 --train-end IS/OOS split）
│                               # 02_drift_detection_experiment.py
│                               # 03_adaptation_evaluation.py（regime-stratified + paired t-test）
└── docs/                       # architecture.md, assignment.md, database_schema.md,
                                # glossary.md, grafana_setup.md
```

---

## 五、目前進度（截至 2026-04-27）

### MVP v1 — 端到端骨架 ✅

Monorepo 骨架、PostgreSQL schema（15 表）、DolphinDB 分區表與 alpha_batch/stream 腳本、十層 Python 模組、FastAPI（4 route groups）、Pipeline 排程器（synthetic / CSV / real 三模式）、Config YAML、台股 1083 檔資料下載器、Backtest 報告產生器、Docker 基礎設施全通、CSV → DolphinDB ingestion、Real pipeline 完整跑通。詳見 git log（commit `f7963fd` 之前的歷史）。

### MVP v2 — Monitoring 與 drift 驗證

| WP | 標題 | 狀態 | 備註 |
|----|------|------|------|
| WP1 | PSI / calibration_error drift metrics 補齊 | ✅ | 集中於 `src/common/metrics.py`；data/alpha/model monitor 整合；12 unit tests |
| WP2 | Per-alpha IC 分析 notebook | ✅ | `notebooks/01_alpha_ic_analysis.py`；篩選 `\|rank_ic\| >= 0.01 AND coverage >= 0.80` |
| WP3 | XGBoost MLMetaModel | ✅ | `src/meta_signal/ml_meta_model.py`，purged expanding-window CV；9 unit tests |
| WP4 | Drift detection 實驗 notebook | ✅ | `notebooks/02_drift_detection_experiment.py`，三 regime × 三窗口 |
| WP5 | Performance-triggered adaptation 整合測試 | ✅ | `tests/integration/test_adaptation_loop.py`（11 tests）+ `pipelines/adaptation_pipeline.py --csv` |
| WP6 | Grafana 4 面板 dashboard | ✅ | `dashboards/` + provisioning auto-load |
| WP7 | Replay streaming 串接 | ✅ | `pipelines/replay_pipeline.py` + 14 integration tests |
| Gap Fix | 模組間膠水層 | ✅ | persist_metrics 寫 PG、`--signal-method ml_meta` flag、adaptation_pipeline 寫 model_registry |

### MVP v3 — 研究所申請展示版

| WP | 標題 | 狀態 | 備註 |
|----|------|------|------|
| WP9 | Adaptation A/B 實驗（5 策略） | ✅ **Phase A 重跑完成（結論翻轉）** | 修正 #2 窗口拆分 + #4 完整成本後重跑 baseline + 4 段 cost sweep；scheduled_20 在所有成本場景排第一（rank_std=0），no_adapt 永遠墊底；推翻原 4-strategy 結論。輸出：`reports/adaptation_ab/ab_20220601_20241231_top10_phaseA_{baseline,sweep}/`、`reports/adaptation_evaluation/` |
| WP10 | Alpha universe 擴充（DolphinDB） | ✅（已被 #34 取代） | 101 alphas × 556 stocks × 2022-2026 寫入 DolphinDB（53.8M rows）；技術細節見 §11 |
| WP11 | Model Pool 第五策略 | ✅ | `src/adaptation/model_pool_strategy.py` + ab_experiment 5 策略；fake pool 整合測試（8 tests）；DB 不可用時降級成 triggered；Phase A 新增 `shadow_warmup_days` + dual-train（shadow 用 stricter cutoff、live 用完整資料） |
| #34 | Pure Python WQ101 Migration | ✅ | DolphinDB OOM → 純 Python port；`src/alpha_engine/wq101_python.py` + `alpha_cache.py`；19 新 unit tests；冷啟動 5–10 分；end-to-end `predict_next_day` 命令尚未實跑驗證 |
| Phase A | 方法學修正（P0★ #2 + #4）| ✅ 2026-04-27 完成 | 程式碼修改 + 21 新測試 + 端到端重跑 + memory 更新；詳見 §七 |
| TEJ Ingestion / yfinance 隔離 | Survivorship-correct 資料源 | ✅ 2026-05-04 完成 | `scripts/ingest_tej_csv.py` 合併 2 份 TEJ CSV → `data/tw_stocks_tej.parquet`（1105 檔含 51 下市，2018-01 → 2026-04，40.8 MB）；正式 pipeline 預設 `--data-source tej`；已加 yfinance guard（需 `--allow-yfinance` 才能使用污染 CSV）、per-source alpha cache 與 cache manifest；下市規則見 §十二 |
| WP9 Cost-aware Redesign | 成本感知目標函數與低換手持股流程 | ✅ 介面與測試完成（2026-05-04） | 新增 `turnover_aware_topk`、`previous_weights` turnover cap、`net_return_proxy` holdout diagnostics、`--benchmark ew_buy_hold_universe`；尚未執行 TEJ Phase 1 grid / Phase 2 full sweep |

### 四條執行路徑

| 模式 | 指令 | 外部依賴 | 狀態 |
|------|------|----------|------|
| 純合成 | `python -m pipelines.daily_batch_pipeline --synthetic` | 無 | ✅ 可用 |
| CSV + Python WQ101 | `python -m pipelines.daily_batch_pipeline --data-source csv --allow-yfinance` | yfinance | ⚠️ 僅 demo / 反例；預設會被 guard 擋下（8476 split-adjustment 污染 + survivorship bias） |
| **TEJ + Python WQ101**（survivorship-correct） | `python -m pipelines.simulate_recent --data-source tej` / `python -m pipelines.predict_next_day --data-source tej` | TEJ Pro 帳號 | ✅ 可用，含 51 檔下市股 |
| 完整 Real Pipeline | `python -m pipelines.daily_batch_pipeline` | docker-compose up -d | ⚠️ 因 alpha_features 啟動 OOM 常無法跑通，見 §11 #8 |
| 每日預測（Python WQ101） | `python -m pipelines.predict_next_day --data-source tej --alpha-source python_wq101` | 無 | ✅ **預設路徑**，不需 DolphinDB；正式研究一律用 TEJ |

### 測試狀態

```
136 passed
```

Phase A 新增 21 個測試：`test_simulate_recent_cost.py`（10 unit）、`test_rolling_ic_window.py`（7 unit）、`test_ab_experiment.py` cost sweep cases（4 integration）、`test_model_pool_strategy.py` shadow warmup cases（額外 integration）。

---

## 六、MVP 路線圖（已完成項目摘要）

| 階段 | 範圍 | 狀態 |
|------|------|------|
| MVP v1（第 1–8 週） | 端到端最小 demo（CSV / Real / synthetic 三條 pipeline、第一份 backtest report） | ✅ 全部完成 |
| MVP v2（第 9–16 週） | Monitoring 與 drift 驗證 — WP1–WP7 + Gap Fix | ✅ 全部完成 |
| MVP v3（第 17–24 週） | Adaptation A/B（WP9）、WQ101 全集（WP10/#34）、Model Pool（WP11） | ✅ **Phase A 重跑完成（2026-04-27）**；P0★ #2 + #4 已修；結論翻轉 |

未開始 / 延期：

- WP8（HMM regime detection）
- WP12（config versioning + data snapshot）
- WP13（研究報告 notebook 整合 WP4 + WP9 + WP11）
- Shioaji streaming 真實連線、MLflow tracking 整合

---

## 七、待辦事項（按優先順序）

### P0★ — 方法學與流程修正（影響既有研究結論可信度，優先於新功能）

> 以下六項由 2026-04-26 流程審視提出。**Phase A（2026-04-27）已完成 #2 + #4，並重跑 WP9**；
> #1 程式碼完成、IS 篩選跑通；#3 / #5 / #6 留到 Phase B。

- [~] **#1 修正 alpha 篩選的 look-ahead bias（WP10 / WP2）— TEJ 重做完成（2026-05-03）**
      `configs/alpha_config.yaml` 的 `v3_effective_alphas` 是用 2022-2026 全期 IC 篩出來，再回頭用在同一段歷史的 backtest 與 A/B 實驗——違反原則 #2「signal_time 與 label_available_at 必須分離」。
      - [x] `notebooks/01_alpha_ic_analysis.py` 新增 `--train-end YYYY-MM-DD` 參數；helper 化（`compute_alpha_row` / `split_panel_by_time` / `build_oos_validation` / `emit_selection_outputs`）；新增 `effective_alphas_oos_validation.csv`
      - [x] `tests/unit/test_alpha_ic_split.py`（10 tests）
      - [x] `configs/alpha_config.yaml` 為 `v3_effective_alphas` 加 LOOK-AHEAD BIAS WARNING；新增 `v3_effective_alphas_is` 區塊
      - [x] **舊跑（yfinance, IS 2022-01→2024-06, 200 top-by-rowcount, ~2.5y IS）**：52/101 通過，6 sign-flip，OOS mean rank_ic +0.0184。⚠️ 此 universe 排除任何 2024-06 前下市股 → 重新引入 survivorship bias
      - [x] **TEJ 重做（2026-05-03, IS 2018-01→2024-06, 200 random survivorship-correct, ~6.5y IS）**：64/101 通過，7 sign-flip（wq024, wq098, wq079, wq012, wq052, wq023, wq046），OOS mean rank_ic +0.0087；41 alpha 與舊跑重疊、23 個新增（IS 變長後通過閾值）、11 個被剔除。輸出覆寫 `reports/alpha_ic_analysis/effective_alphas.json`，舊版備份在 `effective_alphas_yfinance_backup_20260503.json`
      - [x] `scripts/run_is_oos_validation.py` 改寫：加 `--data-source {csv, tej}` flag、universe 改為「IS 期間 ≥ min_is_days 交易天數」（survivorship-correct 池）、cache 用 pyarrow 逐 alpha 串流（每 alpha 0.06s，101 alphas 共 43s）
      - [~] **WP9 TEJ baseline 已啟動（2026-05-04）**：用 TEJ + 新 `effective_alphas.json` 跑完五策略 baseline，五策略全為負績效，暫定排序 `model_pool > scheduled_60 > triggered > none > scheduled_20`；輸出見 `reports/adaptation_ab/ab_20220601_20241231_top10_tej_effective64_baseline/comparison_consolidated.csv`。下一步仍需補 TEJ cost sweep（0 / 0.2 / 0.4 / 0.6）與 `notebooks/03_adaptation_evaluation.py`。
      - [x] **成本感知實驗流程第一版已落地（2026-05-04）**：`simulate_recent` / `ab_experiment` 已支援 `--portfolio-method turnover_aware_topk`、`--rebalance-every`、`--entry-rank`、`--exit-rank`、`--max-turnover`、`--min-holding-days`、`--objective net_return_proxy`、`--benchmark ew_buy_hold_universe`；測試 `33 passed`。下一步是實際跑 TEJ Phase 1 小矩陣。
      - [ ] **後續可選**：做「IS-selected vs look-ahead 版本」對照圖加入論文附錄，量化 selection bias 影響

- [x] **#2 拆開 adaptation trigger 與 shadow evaluation 的窗口（WP5 / WP11）— Phase A 已完成**
      `_decide_retrain` 用 rolling IC 觸發，`ShadowEvaluator` 又在類似的近期窗口比較候選——會偏好「剛好擬合最近 noise」的新模型；WP9 結論可能是此 selection bias 的副作用。
      - [x] `pipelines/simulate_recent.py`：`_compute_rolling_ic` / `_compute_rolling_sharpe` 改用 calendar-day 雙邊界 `[t-trigger_window_days, t-trigger_eval_gap_days]`，預設 `[t-60, t-20]`
      - [x] `src/adaptation/model_pool_strategy.py`：新增 `shadow_warmup_days=5`，shadow 候選訓練 cutoff 推到 `shadow_cutoff_start - warmup`，shadow 選中新模型後再用完整資料重訓 live 版（dual-train）
      - [x] 6 個新 CLI flag（`--trigger-window-days` / `--trigger-eval-gap` / `--shadow-warmup-days` 等）+ `experiment_summary.md` 紀錄窗口配置
      - [x] 7 個 unit tests（`test_rolling_ic_window.py`）
      - [x] WP9 5 策略 A/B 重跑：**所有 adaptation 策略勝過 no-adapt**（與舊結論相反）

- [ ] **#3 monitoring 與 adaptation 形成真正的閉環（Phase B）**
      目前 `alert_mgr.persist_metrics()` 只寫 PostgreSQL 給 Grafana，`PerformanceTriggeredAdapter.check_trigger()` 卻直接讀記憶體裡的 rolling IC——兩條路徑各自為政，違反原則 #4「Adaptation 在 Monitoring 之後觸發」的精神。
      - `src/adaptation/performance_trigger.py`：新增從 `monitoring_metrics` / `alerts` 表查詢的路徑（保留記憶體 fallback）
      - 觸發條件改為「最近 N 個 monitor 週期內 CRITICAL alert 數 ≥ 閾值」or「rolling IC trend 由 monitor 表計算」
      - 整合測試 `test_adaptation_loop.py` 新增「DB-driven trigger」case

- [x] **#4 回測加入交易成本與滑點 — Phase A 已完成**
      WP9 報告未扣手續費/證交稅/滑點；台股單邊成本約 0.4%（手續費 0.1425% × 折扣 + 證交稅 0.3%），高換手策略可能在加入成本後完全反轉排序。
      - [x] `src/execution/paper_engine.py`：commission 0.000926/side（折扣後）、新增 tax 0.003/sell-side
      - [x] `configs/risk_config.yaml`：補 tax_rate 欄位、修正 commission_rate
      - [x] `pipelines/simulate_recent.py`：`_compute_costs` helper、daily_pnl 拆四欄（gross / commission / tax / slippage / net）；新增 `--round-trip-cost-pct` 覆寫模式供 sensitivity sweep
      - [x] `pipelines/ab_experiment.py`：`--cost-sweep` 新模式，4 段成本場景（0/0.2/0.4/0.6%）；產出 `cost_sensitivity.csv` + `cost_sensitivity.png`
      - [x] `notebooks/03_adaptation_evaluation.py`：新增 `--cost-sweep-dir` 模式，產出 rank stability 與 fig4
      - [x] 10 個 unit tests（`test_simulate_recent_cost.py`）+ 4 個 integration tests
      - [x] **結論穩健**：5 策略排名在 4 段成本場景下 rank_std=0（100% 一致），no_adapt 永遠墊底

- [ ] **#5 統一三條 pipeline 路徑，消除 drift（Phase B）**
      `daily_batch_pipeline.py` 的 synthetic / csv / real 三個模式各自實作 alpha 計算與 signal 流程，每次改 alpha engine 或 signal generator 都要同步三處。
      - 抽出共用的 `core_pipeline(data_source: DataSource, alpha_engine: AlphaEngine, ...)`
      - 三模式只差在 `DataSource` 與 `AlphaEngine` 的注入
      - 整合測試確保三模式對相同輸入產生一致 signal（容許 numerical tolerance）

- [ ] **#6 Model pool 改為跨 run 持久化（WP11，Phase B）**
      目前 `ModelPoolController._models_by_id` 是 process-local dict，跨 run 不可重用，違背 recurring concept pool 的核心動機。
      - `src/adaptation/model_pool_strategy.py`：新增 `persist_dir` 參數，模型 instance 用 `joblib.dump` / pickle 序列化到磁碟（或 MLflow artifact）
      - `MLMetaModel` 補上 `save(path)` / `load(path)` 方法（XGBoost 原生支援）
      - 跨 run 整合測試 + 在 `regime_pool` 表新增 `model_artifact_path` 欄位

### Phase A 重跑結論（2026-04-27，period 2022-06 → 2024-12，IS-selected 52 alphas，含完整成本）

**Baseline（baseline cost ~46 bps/day total）**：
| Strategy | Sharpe | n_retrains | Cum Ret % |
|---|---|---|---|
| **scheduled_20** | **3.757** | 32 | 15754 |
| triggered | 3.630 | 12 | 15269 |
| model_pool | 3.518 | 11 | 12925 |
| scheduled_60 | 3.514 | 11 | 10938 |
| none | 3.158 | 1 | 7625 |

**Cost sensitivity sweep（4 段 round-trip cost 0/0.2/0.4/0.6%）**：5 策略 mean_rank std=0（100% 一致），完整排序 `scheduled_20 > triggered > model_pool > scheduled_60 > none`。

**Paired t-test vs none**：4 個 adaptation 策略 mean_daily_excess_ret 都正向（0.0005～0.0011），p_one_sided 都 > 5%（最接近 triggered p=0.096），direction 對但 power 不足。

**對照舊 4-strategy WP9 結論**：原版「no_adapt 勝出」結論被識別為 selection bias（trigger/shadow 窗口重疊）+ 忽略交易成本所致，Phase A 修正後翻轉。詳見 memory `project_wp9_finding.md`。

### P1 — 短期 ✅ MVP v1/v2 全部完成

WP1–WP7 + 真實資料下載、Docker 基礎設施、DolphinDB ingestion、Grafana provisioning 等均已完成，詳見 git log 與 §五。

### P2 — 中期（待辦）

- [ ] 實作 Shioaji streaming 真實連線
- [ ] 接入 MLflow tracking，記錄每次 retrain 的 experiment

### P3 — 長期（MVP v3 研究亮點）

- [ ] 實作 HMM-based regime identification（WP8）
- [ ] 在真實 PostgreSQL 上以長期歷史執行 model_pool A/B 實驗，驗證 shadow guard 是否讓 model_pool 在 recurring regime 顯著優於 triggered（需 docker-compose up）
- [ ] 產出研究 notebook：`notebooks/04_model_pool_evaluation.py`（regime-stratified 比較 + paired t-test vs triggered）
- [ ] 撰寫 docs/adaptation_design.md
- [ ] 準備研究所申請用的展示材料

---

## 八、關鍵設計決策（不可違反）

在後續開發中，以下設計原則必須維持：

1. **wq101alpha 是 feature engine，不是 final signal**。所有 alpha 輸出必須經過 Layer 4（Meta Signal）聚合後才能產生交易訊號。

2. **signal_time 與 label_available_at 必須分離**。Label 只能在 `signal_time + horizon + buffer` 之後才可用於訓練，任何違反此規則的程式碼都可能引入 look-ahead bias。

3. **Monitoring 分四層**，不可合併：Data Monitor、Alpha Monitor、Model Monitor、Strategy Monitor 各自獨立計算 metrics、各自觸發 alerts。

4. **Adaptation 在 Monitoring 之後觸發**，不可混為一層。Adaptation 有三種 policy，不是只有 scheduled retrain。

5. **alpha_features 的 schema 是 (security_id, tradetime, alpha_id, alpha_value)**，long format，不是 wide format。

6. **model_registry 必須記錄**：model_id, trained_at, training_window, features_used, hyperparams, holdout_metrics, status, regime_fingerprint。

7. **所有 monitoring metrics 寫入統一的 monitoring_metrics 表**，schema 為 (metric_time, monitor_type, metric_name, metric_value, dimension, window_size)。

---

## 九、開發環境

### 本機環境

- OS: Windows 10
- Python: 3.11.9（透過 `py -3.11` 取得，虛擬環境在 `.venv/`）
- 套件管理: pip + hatchling editable install
- Git remote: `origin` → `https://github.com/Haaaaaasi/Drift-aware-Real-time-Alpha-Monitoring-and-Adaptation-System.git`

> **Codex 執行注意**：原 `.venv` 存在且可用，但 Codex 沙盒內直接啟動 `.venv\Scripts\python.exe` 可能因 base Python 目錄 ACL 顯示 `Unable to create process` / `存取被拒`。遇到此狀況時，請以非沙盒權限執行專案 `.venv\Scripts\python.exe`，不要改用本機 Anaconda。細節見 `memory/feedback_venv_acl.md`。

### 啟動虛擬環境

```powershell
.\.venv\Scripts\Activate.ps1
```

### 常用指令

```powershell
# 跑全部測試
pytest -q

# 下載台股資料
python scripts/download_tw_stocks.py                           # 所有上市股票（~1083 檔）
python scripts/download_tw_stocks.py --include-otc             # 上市 + 上櫃
python scripts/download_tw_stocks.py --append                  # 增量更新
python scripts/download_tw_stocks.py --tickers 2330 2317 2454  # 指定標的

# Pipeline 三種模式
python -m pipelines.daily_batch_pipeline --synthetic
python -m pipelines.daily_batch_pipeline                      # 預設 TEJ + Python WQ101
python -m pipelines.daily_batch_pipeline --real               # 完整 real（需 Docker）

# 產生 backtest 報告
python scripts/generate_report.py --csv data/tw_stocks_tej.parquet

# 驗證基礎設施
python scripts/validate_infrastructure.py --skip-dolphindb    # 快速檢查 PG + Redis
python scripts/validate_infrastructure.py --run-migrations    # 自動執行 migration SQL

# 啟動基礎設施
docker-compose up -d

# 啟動 API
python main.py api
```

---

## 十、程式碼慣例

- Python 模組：每個模組都有 `__init__.py`，使用 `from src.xxx import Yyy` 格式
- Logging：統一使用 `src/common/logging.py` 的 `get_logger()`（structlog）
- DB 連線：統一透過 `src/common/db.py`（`get_pg_connection()`, `get_dolphindb()`, `get_redis()`）
- Config：透過 `src/config/settings.py` 的 `get_settings()`（Pydantic Settings, 讀 `.env`）
- Constants：所有 enum 定義在 `src/config/constants.py`
- Metrics 計算：通用計算函式在 `src/common/metrics.py`（IC, Sharpe, drawdown, KS-test, etc.）
- 測試：`tests/unit/` 放單元測試，`tests/integration/` 放整合測試
- DolphinDB 腳本：`.dos` 副檔名，放在 `dolphindb/` 底下
- YAML 設定：放在 `configs/` 底下

---

## 十一、已知限制與風險

> 標記 `[DolphinDB-only]` 的項目僅在 real mode 觸發；預設 Python WQ101 路徑不適用。

### 通用 / Python 路徑

1. **wq101alpha 部分 alpha 在台股的適用性未知**：已建立 TEJ IS-only 篩選流程（`scripts/run_is_oos_validation.py --data-source tej`），101 alphas 中 64 個通過 IS 篩選，7 個出現 IS→OOS sign flip（wq024, wq098, wq079, wq012, wq052, wq023, wq046）。正式 single source of truth 是 `reports/alpha_ic_analysis/effective_alphas.json`；舊 yfinance 52-alpha 結果只保留為 deprecated 對照。
2. **yfinance 8476 split-adjustment 資料汙染**：`data/tw_stocks_ohlcv.csv` 中 stock 8476 有非物理性 ±50%/100% 日報酬，已證實會解釋舊 WP9/PhaseA/PhaseB 的極高累積報酬。正式研究、論文與展示一律使用 `--data-source tej`；yfinance 只保留 demo 或資料品質反例。
3. **Recurring concept pool 在短歷史（< 2 年）下可能樣本不足**，放在 MVP v3
4. **Shioaji streaming 是 mock 狀態**，需要帳號才能接真實資料
5. **單人開發風險**：容易在某一層花太多時間，應嚴格遵守 MVP scope
6. **本機 PostgreSQL 衝突**：本機已安裝 PostgreSQL 佔用 5432，Docker 容器改用 5433 port；`.env` 與 `settings.py` 已對應修改
7. **WQ101 論文公式 float window**：Kakushadze 2016 論文中許多算子的 window 參數是浮點數（如 `_delta(vwap, 4.72775)`）；pandas `rolling()` 要求整數，直接傳 float 會報 `slice indices must be integers`。修復方式：`_w(d) = max(1, int(round(d)))` helper，所有算子套用（`src/alpha_engine/wq101_python.py` 已修正）

### DolphinDB-only（real mode 才會碰到）

7. **[DolphinDB-only] 社區版**有連線數與記憶體限制，需先確認 license
8. **[DolphinDB-only] alpha_features 啟動 OOM**：`alpha_features` 表（53.8M rows）TSDB chunk metadata 在容器啟動時需超過 4GB，主機僅剩 1GB 可用——容器無法正常啟動。**緩解（已實作）**：`src/alpha_engine/wq101_python.py` + `alpha_cache.py` 提供純 Python 路徑，預測流程透過 `--alpha-source python_wq101` 完全脫離 DolphinDB
9. **[DolphinDB-only] 分區數限制**：預設 `maxPartitionNumPerQuery=65536`，已調整為 500000；VALUE 日分區 × HASH 50 共 ~200k 分區，大查詢需加 WHERE 時間範圍
10. **[DolphinDB-only] wq101alpha API**：`prepare101` 模組不存在；各 WQAlpha 函式接收獨立 panel matrix（column-major），不是字典；`matrixToLongTable` 使用 `flatten` + `stretch/take` 轉 long format
11. **[DolphinDB-only] TSDB 寫入緩衝 OOM**：寫入大量 alpha 時，TSDB write cache 跨 alpha 累積導致 OOM；必須在每次 `db.append!()` 後立即呼叫 `flushTSDBCache()` 釋放緩衝。若 OOM 異常重啟，TSDBRedo redo logs（可達 500MB+）會在下次啟動時重播再次觸發 OOM——需先停容器、用 alpine 刪除 `/data/ddb/server/data/local8848/log/TSDBRedo/*.log`，再重啟
12. **[DolphinDB-only] BOOL 型 alpha**：wq061、wq075、wq079、wq095 回傳 BOOL matrix（非 DOUBLE），寫入 `alpha_features.alpha_value`（DOUBLE）前需在 `alpha_batch.dos` 加 `if (typestr(mat) != "FAST DOUBLE MATRIX") { mat = double(mat) }` 轉型
13. **[DolphinDB-only] Python SDK `isClosed()`**：較新版 SDK 不存在 `isConnected` 屬性，連線狀態檢查須使用 `session.isClosed()`（`src/common/db.py` 已修正）
14. **[DolphinDB-only] per-alpha IC 計算效能**：`alpha_id` 不是分區鍵，每次 `where alpha_id = 'wqXXX'` 需掃描全部 ~200k 分區，101 個 alpha 約需 4-6 小時（`compute_ic_summary.dos`）；為避免 Python 端 OOM，IC 計算完整在 DolphinDB 端執行

---

## 十二、資料來源與 Survivorship 規則（2026-05-02 新增）

### 兩條資料路徑並存

| 來源 | 路徑 | 範圍 | 適用情境 |
|------|------|------|----------|
| `csv` | `data/tw_stocks_ohlcv.csv` | 1083 檔 TWSE 上市，**僅當前在市**（survivorship-biased） | 快速 demo / 教學示範 / 不需要 reviewer 防守的探索性研究 |
| `tej` | `data/tw_stocks_tej.parquet` | 1105 檔（含 51 檔已下市），TEJ Pro 還原股價 2018-01 → 2026-04 | **正式研究 / 論文 / 研究所申請展示**——必須使用 |

兩條路徑的 alpha 還原比例驗證一致（2330 抽 6 日 ratio = 1.0001），代表 yfinance `auto_adjust=True` 與 TEJ 還原權息結果在台股大型股上幾乎相同；切換到 TEJ 的 **核心收益是下市股覆蓋，不是價格還原品質**。

### CLI 切換

```bash
# Survivorship-correct simulate / predict
python -m pipelines.simulate_recent --data-source tej --start 2022-06-01 --end 2024-12-31 --strategy scheduled
python -m pipelines.predict_next_day --data-source tej --as-of 2026-04-30
```

`--data-source` 與 `--csv` 同時提供時 `--csv` 優先（可指定任意路徑）；省略 `--csv` 則依 `--data-source` 取對應預設路徑。
所有支援 `--data-source` 的正式 pipeline 預設值皆為 `tej`。自 2026-05-04 起，已知 yfinance 路徑 `data/tw_stocks_ohlcv.csv` 預設會被 guard 擋下；只有同時明確指定 `--data-source csv`（或 `--csv data/tw_stocks_ohlcv.csv`）且加上 `--allow-yfinance`，才允許作為 demo / 反例使用。

### 下市股處理規則（**保守一致版**，不需區分下市原因）

```
規則：每檔下市股，從下市日往前回推到「最後一筆 OHLCV」，當天報酬計為 0
（即不交易），下市日後該股退出 universe。
```

實作方式：**完全藉由現有 NaN 處理邏輯隱式實現，不需新增程式碼**：

1. **下市日當天**：`next_close` 為 NaN（沒有 t+1 bar）→ `next_return` NaN → `simulate_recent.py` 第 580 行 `if not np.isnan(r)` 跳過該股 → 該股當日對 portfolio 報酬貢獻為 0
2. **下市日之後**：該股不在 `alpha_panel[tradetime == t]` → 不會被 portfolio constructor 選中 → 自動退出 universe

### 為什麼不區分下市原因

理論上應該分「合併下市」（往往 +0~30% 溢價）與「終止上市」（往往 -50~-100% 暴跌），但：

* TEJ universe roster 提供下市日期但**未提供下市原因欄位**（要從另一張公司事件表查）
* sim 期間（2022-06 → 2024-12）TEJ 偵測到的 implicit 下市僅 12 檔，工程量不值得單獨處理
* 統一規則造成的方向性偏誤：對「合併下市」少抓溢價 → 績效略低估（保守）；對「終止上市」少抓暴跌 → 績效略高估。**兩個方向部分抵銷**，淨偏誤遠小於完全排除下市股
* 對 reviewer 是 defensible 的選擇：「we apply a conservative stop-trading rule for delisted stocks (last close → exit), which biases reported Sharpe slightly conservative for merger cases」

### 何時應重做下市原因？

只有在以下情境：

* 論文需要做 sensitivity analysis 量化下市原因影響
* sim universe 擴大到含 TPEX 上櫃（下市/下櫃約 800+ 檔，量級不可忽略）
* 業界級實盤策略，需要嚴格估計 tail risk

正常研究所申請展示**不需要做**——目前的保守規則已足夠通過 reviewer 防守。

### Alpha cache 注意事項

2026-05-04 起，Python WQ101 cache 已依資料源分流：

| 資料源 | cache 路徑 | 用途 |
|--------|------------|------|
| `tej` | `data/alpha_cache/wq101_alphas.parquet` | 正式研究預設；沿用舊檔名避免重算 1GB+ TEJ cache |
| `csv` | `data/alpha_cache/wq101_alphas_csv.parquet` | yfinance demo / 對照專用 |

請勿再讓 yfinance 與 TEJ 共用同一份 alpha cache；兩者 `security_id` 高度重疊，會造成污染資料的 alpha 被誤用到 TEJ。長期若要進一步避免 look-ahead，可改為「每次重訓視窗即時算 alpha」，但短期先以 per-data-source cache 隔離。

每份 alpha cache 旁都必須有 sidecar manifest（`<cache>.manifest.json`），至少記錄 `data_source`、`alpha_engine`、rows、股票數、alpha 數與日期範圍；讀取時若指定 `expected_data_source` 會強制驗證。現有 TEJ cache 已於 2026-05-04 驗證並寫入 manifest：1105 檔、101 alphas、200,032,808 rows、2018-01-02 → 2026-04-30。

若懷疑 TEJ cache 仍由舊資料產生，正式重跑前應先備份或刪除 `data/alpha_cache/wq101_alphas.parquet` 並強制重算。

### TEJ ingestion 重新匯入流程

```bash
# 1. 從 TEJ Pro 下載新版 OHLCV（UTF-16 LE / tab，含期間下市股）
# 2. 把 CSV 放到根目錄
# 3. 重跑 ingestion
python scripts/ingest_tej_csv.py
# → 輸出 data/tw_stocks_tej.parquet + data/tw_stocks_tej_universe.parquet
# → 自動過濾 ETF / 權證 / TDR（保留 4 碼純數字普通股）
# → 自動將成交量從千股換算為股
```

### 已知小問題

* yfinance CSV 把 `security_id` 存成 int 導致 ETF 0050 變成 50（前導零被吃）。TEJ ingestion 已修，但若使用 `--data-source csv` 仍會碰到。建議在切到正式研究時統一使用 TEJ。
