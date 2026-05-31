# DARAMS 過時功能清理計畫

建立日期：2026-05-25

本文件用來定義 DARAMS 目前哪些功能屬於正式主線、哪些應降級為實驗/封存、哪些可以在確認後移除。目標不是把歷史痕跡抹掉，而是讓日常開發、測試、文件與 reviewer-facing claim 只圍繞目前可信的主線。

## 1. 目前正式主線

正式主線已收斂為：

```text
TEJ survivorship-correct data
→ Python WQ101 / bar-aligned alpha cache
→ incumbent_55 + rolling_topk20_w126_pen10
→ scheduled_20 XGBoost meta model
→ turnover_aware_topk portfolio
→ T+1 execution: next_vwap primary, next_open secondary
→ live daily operating layer / web console
```

任何功能若不能服務這條主線，或只能重現已 deprecated 的研究歷史，預設應降級、封存或移除。

## 2. 清理原則

- 不刪除研究結論的唯一證據。正式結果應保留 summary / bundle / config / memory 條目；大量中間輸出可視為 generated artifact。
- 不保留會讓新使用者誤用的舊入口。舊資料源、舊成交假設、舊高報酬結論必須加警告或移出主流程。
- 不讓 smoke / backfill / stale experiment 污染正式狀態。Grafana、API、README 與 docs 預設只呈現 official / frozen 主線。
- 任何破壞性刪除前先確認：至少要有 `git status`、被替代入口、目標測試，以及必要時的 memory 記錄。

## 3. 保留清單

這些是目前不可拔的核心：

| 類別 | 路徑 / 功能 | 原因 |
|---|---|---|
| Data guard | `src/config/data_sources.py`、yfinance guard | 防止回到 yfinance survivorship / 8476 污染路徑 |
| Alpha cache | `src/alpha_engine/alpha_cache.py` | bar alignment 是正式資料品質防線 |
| Alpha selector | `src/alpha_selection/`、`configs/frozen_alpha_selector_20260517.yaml` | 現行 incumbent selector |
| Label / execution | label maturity、`next_vwap` / `next_open` | 防 look-ahead 與成交假設防守 |
| Live layer | `pipelines/daily_online_pipeline.py`、`pipelines/live_daily_runner.py`、`src/live/`、`src/api/routes/live.py` | 目前日常操作入口 |
| Final robustness | `docs/final_robustness_holdout_protocol.md`、`reports/adaptation_ab/final_robustness_20260518/` | reviewer-facing bundle |
| Memory | `memory/` | 接手脈絡與研究決策紀錄 |

## 4. 降級為實驗或失敗附錄

這些不應出現在預設流程，但可保留作研究附錄：

| 功能 | 現況 | 整理作法 |
|---|---|---|
| `model_pool` / recurring concept reuse | frozen selector 下輸給 `scheduled_20`，不能作主 claim | 從 README 主線淡出；文件標為 failure / ablation appendix；未來若重啟，必須以 frozen selector + prospective holdout 重新驗證 |
| admission gate / all_valid alpha expansion | 實驗顯示新增 alpha 整批邊際貢獻不穩 | 保留 `src/alpha_selection` 支援，但 docs 標為暫停，不當預設開關 |
| DolphinDB real mode | streaming 備援，但非預設且曾有 OOM 風險 | 從日常 quickstart 拔掉；保留於 architecture 的 optional appendix |
| yfinance CSV demo | 只適合反例與 demo | CLI 需 `--allow-yfinance`；文件只保留警告與 reproduction note |
| WP9 舊 Phase A / stale-code matrices | 已被 TEJ / cache-aligned / frozen selector 結論取代 | report summary 可封存；不可作正式 claim |

## 5. 可封存或移除候選

第一批只做封存與 ignore，不直接刪除：

| 路徑 / 型態 | 建議 | 驗證條件 |
|---|---|---|
| `reports/adaptation_ab/ab_*/` 新增大量 run directories | 視為 generated artifact，預設不進 git | `.gitignore` 已加入規則；正式 bundle 另行保留 |
| `reports/live_smoke*/`、`artifacts/live_smoke_models/` | smoke-only，本機輸出 | 不影響 official live run 查詢 |
| `reports/adaptation_ab/model_pool_*_runs/` | 實驗中間輸出 | 保留 failure summary markdown / memory |
| stale matrix summary，例如 `model_pool_selector_matrix_20260507` | 封存為 deprecated appendix | 文件內明確標 stale-code，不被 README 引用 |
| root `OHLSV*.csv` | TEJ 原始輸入，通常不進 repo | 轉成 `data/tw_stocks_tej.parquet` 後由 data ignore 管理 |

第二批才考慮真的刪除或移動：

| 路徑 / 功能 | 建議 | 前置條件 |
|---|---|---|
| 舊 workflow sweep scripts | 合併成少數正式 workflow，其他移到 `archive/experiments/` 或刪除 | 已有對應 summary、memory、測試不引用 |
| `daily_batch_pipeline` 的 legacy csv / real 分支 | 抽出共用核心後再刪舊分支 | synthetic / TEJ / live 入口測試通過 |
| DolphinDB-specific ingestion docs in quickstart | 移到 appendix | README 已改以 TEJ + Python WQ101 為預設 |

## 6. 建議執行順序

### Phase 0：降低噪音

- 已完成：`.gitignore` 忽略新增 generated experiment outputs。
- 驗證：`git status` 不再被 800+ 個新 report directories 淹沒。

### Phase 1：文件與入口收斂

- README 只保留正式主線 quickstart。
- `docs/architecture.md` 補一段「deprecated / optional paths」。
- `docs/live_daily_operating_layer.md` 保持 live 主入口，明確寫 `model_pool` 不進 live。
- 所有舊 yfinance / Phase A / stale-code 表述加 deprecated 標記。

驗證：

```powershell
git status --short
```

並人工檢查 README 是否仍把舊路徑寫成推薦流程。

### Phase 2：共用化重複工具

新增 `src/common/research_report.py`，集中：

- `markdown_table`
- `paired_t_one_sided`
- `block_bootstrap_mean`
- `read_json`
- `write_json`

優先替換 workflow scripts 內重複 helper，不碰研究參數。

驗證：

```powershell
ruff check src scripts --select F401,F841,E711,E712
python -m compileall -q src scripts
pytest tests/unit/test_research_report.py -q
```

### Phase 3：封存實驗腳本

把只重現舊結論、且不再服務正式主線的 scripts 移到 `archive/experiments/`，或在確認後刪除。

候選：

- `scripts/run_turnover_oos_workflow.py`
- `scripts/run_exit_discipline_oos.py`
- `scripts/run_rolling_topk_validation_workflow.py`
- `scripts/run_temporal_holdout_2026_workflow.py`
- `scripts/summarize_model_pool_selector_matrix.py`

這些腳本若仍需留作 reviewer appendix，應只留一個 README 指向正式 summary，而非讓每個 sweep script 都像 production tool。

驗證：

```powershell
pytest tests/unit/test_alpha_selection_rolling_topk.py tests/unit/test_frozen_alpha_selector.py -q
pytest tests/unit/test_live_portfolio_service.py tests/unit/test_live_execution_service.py -q
```

### Phase 4：移除舊 pipeline 分支

這是風險最高的一段，必須最後做。

- 先把 TEJ / Python WQ101 / selector / portfolio 的共用流程抽成核心 service。
- 再讓 `daily_batch_pipeline`、`simulate_recent`、`daily_online_pipeline` 只做 orchestration。
- 最後移除 legacy csv / DolphinDB real branch 的日常入口。

驗證至少包含：

```powershell
pytest tests/unit/test_data_source_guard.py tests/unit/test_alpha_cache.py -q
pytest tests/integration/test_pipeline_batch.py -q
```

## 7. 刪除前檢查表

任何刪除 PR 都必須回答：

- 這個功能是否仍在 README / docs / API / Grafana / tests 被引用？
- 是否有正式 summary 或 memory 保留其研究結論？
- 是否會讓 yfinance / close-to-close / stale alpha cache 路徑重新變得容易誤用？
- 是否已跑該功能鄰近測試？
- 是否需要更新 `memory/MEMORY.md`？

## 8. 目前建議的下一刀

下一個實作 PR 建議只做 Phase 2：

1. 新增 `src/common/research_report.py`。
2. 替換 3 到 4 個 workflow script 的 markdown / bootstrap 重複 helper。
3. 新增 `tests/unit/test_research_report.py`。

這一刀不動研究流程，卻能明顯減少重複程式碼，風險比直接刪 pipeline 分支低很多。
