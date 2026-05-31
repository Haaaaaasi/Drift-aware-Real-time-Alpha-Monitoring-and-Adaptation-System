# Live Daily Operating Layer（2026-05-19）

本條記錄 DARAMS live daily operating layer 的第一版與 P0.5 收尾。這條主線目標是把系統從「研究回測」推進到「每天可以跑、可以看、可以追溯」的正式作業層。

## 已完成

- `MLMetaModel.save_artifact()` / `MLMetaModel.load_artifact()`：保存 XGBoost `model.json` 與 `manifest.json`。
- `manifest.json` 記錄 `model_id`、`feature_columns`、`feature_columns_hash`、`selector_snapshot_hash`、`frozen_config_hash`、training window、label horizon、purge days。
- `migrations/004_live_daily_operating_layer.sql`：新增 `daily_live_runs`、`alpha_selection_snapshots`、`alpha_selection_scores`、`portfolio_snapshots`、`trade_recommendations`，並替 `meta_signals` / `portfolio_targets` 補 nullable `run_id`。
- `migrations/005_live_run_purpose_freshness.sql`：替 `daily_live_runs` 補 `run_purpose`、`is_official`、`data_max_date`、`data_lag_days`、`data_freshness_status`，並新增 official run index。
- `src/portfolio/live_service.py`：抽出 live 可重用的 `turnover_aware_topk + max_turnover + tail_cleanup` portfolio service。
- `src/live/operational_store.py`：統一寫入 / 查詢 live run、portfolio snapshot、trade recommendations、alpha snapshots；預設 previous portfolio 只讀前一個 official run，避免 smoke run 污染 production state。
- `pipelines/daily_online_pipeline.py`：新增 `auto` / `predict-only` / `train-only` CLI，並支援 `--run-purpose`、`--official`、資料新鮮度標記。
- `src/api/routes/live.py`：新增 live state API；預設 `official_only=true`。
- `dashboards/live_ops.json`：新增 Grafana Live Ops dashboard；面板預設只看 latest official run，Production Model Features 只顯示 artifact 實際 selected features。
- `docs/live_daily_operating_layer.md`：整理設計、CLI、API、Grafana 與最新正式 run 狀態。

## 設計決策

- 交易用 feature set 必須綁定 production model artifact，不可每天重新選 alpha 後餵給舊模型。
- Diagnostic selector 可以每天計算並顯示在 UI，但只作監控 / 研究診斷，不改變 production prediction schema。
- 正式 live baseline 使用 `configs/frozen_alpha_selector_20260517.yaml`：`incumbent_55 + rolling_topk20_w126_pen10 + scheduled_20 + turnover_aware_topk`。
- `model_pool` 暫不進 live 主流程；目前視為 frozen selector challenge 的 failure / ablation appendix。
- `trade_recommendations` 是決策紀錄，和未來 `orders` / `fills` 執行紀錄分開。
- Grafana / API 預設只看 `is_official = true` 的 production run，smoke / backfill 必須顯式查詢。

## 最新正式 run

- `run_id`：`dd1b4b6c-06f8-480c-8554-767cbec9836a`
- `as_of_date`：2026-04-30
- `production_model_id`：`ml_xgb_e4ebe834`
- `artifact_path`：`artifacts/models/ml_xgb_e4ebe834`
- `feature_columns_hash`：`c56638fb00633481c32a42f232a93fd00a406cc950bc76c684f5e20830c7c7c3`
- `n_feature_alphas`：20
- `n_recommendations`：10
- `data_freshness_status`：`STALE`
- `data_lag_days`：19

目前 TEJ / alpha cache 最新日期是 2026-04-30，而今天是 2026-05-19，所以畫面是正式 pipeline 輸出的 latest official state，但不是當日新鮮資料。若要變成真正日常使用，需要接上 2026-05-01 之後的 TEJ 增量資料。

修正前的一筆 official run `f56c1cb6-f7cf-42de-91a9-d7364c05c2b9` 已標為 `SUPERSEDED` 並改成非 official，因為它是在 previous-state filter 修正前產生，可能把 smoke holdings 當成前一日正式持倉。

## 驗證

- `py_compile` passed：`src/live/operational_store.py`、`pipelines/daily_online_pipeline.py`、`src/api/routes/live.py`、`src/meta_signal/ml_meta_model.py`、`src/portfolio/live_service.py`。
- Unit tests passed：`tests/unit/test_ml_meta_model.py tests/unit/test_live_portfolio_service.py -q`，共 `12 passed`。
- Smoke：
  - `train-only --as-of 2024-08-30 --no-db` 成功產生 artifact `artifacts/live_smoke_models/ml_xgb_d2690a06`。
  - `predict-only --as-of 2024-09-02 --production-artifact artifacts/live_smoke_models/ml_xgb_d2690a06 --no-db` 成功產出 10 筆 trade recommendations。
  - DB-persisted smoke run `19eeb477-018d-4fb7-9a10-d95b9db03883` 成功寫入 live tables。
  - 最新 official production run 成功寫入 `daily_live_runs`、`trade_recommendations`、`portfolio_snapshots` 與 alpha snapshot。

## 待辦

- 接上 TEJ 2026-05-01 之後增量資料，讓 `data_freshness_status` 從 `STALE` 變成 `FRESH`。
- 補 scheduled retrain 的 shadow / closure promotion gate，避免 auto retrain 後直接 promote 未驗證模型。
- 下一步 UI 可做「一鍵 approve/export recommendation」或更完整的 Web UI；Grafana 目前已足夠做 operational read-only dashboard。

## P1 每日接續 runner（2026-05-20）

- 新增 `src/ingestion/tej_daily_append.py`：可安全把每日 TEJ CSV append 到 `data/tw_stocks_tej.parquet`，並重建 `data/tw_stocks_tej_universe.parquet`。
- 新增 `scripts/append_tej_daily.py`：提供 dry-run / backup / append CLI。同 `(security_id, datetime)` 會採用新檔案的值，適合處理 TEJ 修正檔。
- 新增 `pipelines/live_daily_runner.py`：串接 `append TEJ daily -> daily_online_pipeline -> official live run -> Grafana/API`。
- 驗證：
  - `py_compile` passed：`src/ingestion/tej_daily_append.py`、`scripts/append_tej_daily.py`、`pipelines/live_daily_runner.py`。
  - `tests/unit/test_tej_daily_append.py -q`：`3 passed`。
  - CLI help passed：`scripts/append_tej_daily.py --help`、`python -m pipelines.live_daily_runner --help`。
- Dry-run 現有 `OHLSV202320260502.csv` 結果：原始檔最新仍是 2026-04-30，`added_keys=0`，所以目前 workspace 尚未包含 2026-05-01 TEJ daily rows。

## Read-only Web Console（2026-05-20）

- 新增 `/api/v1/live/console` 聚合 endpoint，回傳最新 run、recommendation summary、trade recommendations、holdings、production/diagnostic alphas、run history。
- 新增 FastAPI HTML route `/live`，由 `src/api/static/live_console.html` 提供 read-only operation console。
- Console 面板包含 current run、下一交易日買賣建議、當前目標持倉、Production / Diagnostic alpha 配置、Live Run History。
- FastAPI server 已於本機 `http://127.0.0.1:8000/live` 啟動驗證。
- 驗證：
  - `py_compile` passed：`src/api/app.py`、`src/api/routes/live.py`。
  - TestClient：`GET /live` 回 200；`GET /api/v1/live/console` 回 200，最新 run `dd1b4b6c-06f8-480c-8554-767cbec9836a`，recommendations=10，holdings=10，production_alphas=20。
  - 本機 HTTP smoke：`/health`、`/live`、`/api/v1/live/console` 皆可用。

## Recommendation approve / export（2026-05-20）

- 新增 `POST /api/v1/live/recommendations/status`：可把 latest 或指定 run 的 recommendations 從 `PENDING` 更新為 `APPROVED` / `CANCELED`，也可用 `security_ids` / `actions` 過濾。
- 新增 `POST /api/v1/live/recommendations/export`：匯出 `APPROVED` recommendations 到 `reports/live_exports/`，欄位包含 `order_side`、`quantity`、`last_price`、`notional` 與原始 recommendation 欄位；預設匯出後標成 `EXPORTED`。
- 新增 `GET /api/v1/live/recommendations/export/file?path=...` 下載匯出 CSV。
- Web console 新增 `核准 Pending`、`取消 Pending`、`匯出 Approved` 三個操作按鈕。
- 驗證：用暫時 smoke run 測試 `PENDING -> APPROVED -> EXPORTED`，2 筆 recommendations 成功匯出並標記；測完刪除暫時 run，不影響 official run。
- 本機 FastAPI 已重啟，`http://127.0.0.1:8000/live` 載入最新 UI；用不存在的 `security_id` 對 official run 測 status endpoint，`updated_count=0`，確認不誤改 official data。

## Dockerized API / Live Console（2026-05-22）

- 新增 `Dockerfile` 與 `.dockerignore`，避免 build context 帶入 `.venv`、data、reports、artifacts 等大型檔案。
- `docker-compose.yml` 新增 `api` service：
  - build context：repo root。
  - port：`8000:8000`。
  - command：`python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000`。
  - volume：`.:/app`，讓 `/live`、`reports/live_exports/` 與程式碼使用同一份 workspace。
  - container env：`POSTGRES_HOST=postgres`、`POSTGRES_PORT=5432`、`REDIS_HOST=redis`、`DOLPHINDB_HOST=dolphindb`。
- 已執行 `docker compose up -d --build api`，`darams-api` 正常啟動。
- 驗證：
  - `docker compose config --services` 包含 `api`。
  - `docker compose ps api` 顯示 `darams-api` up，port `0.0.0.0:8000->8000/tcp`。
  - `GET http://127.0.0.1:8000/health` 回 `ok`。
  - `GET http://127.0.0.1:8000/live` 回 200。
  - `GET http://127.0.0.1:8000/api/v1/live/console` 回 latest official run `dd1b4b6c-06f8-480c-8554-767cbec9836a`。

## Security name lookup（2026-05-22）

- `/api/v1/live/console`、`/api/v1/live/recommendations/latest`、`/api/v1/live/holdings/latest` 現在會替 `security_id` 補 `security_name`。
- 名稱來源：優先從 repo root 的 `OHLSV*.csv` 第一欄「代碼 名稱」建立 cached lookup；fallback 使用 `security_master`。
- Web console 的 recommendations / holdings 表新增「名稱」欄；export CSV 也新增 `security_name` 欄。
- Docker API 已重啟。HTTP smoke 顯示 2303/2330 的名稱實際 Unicode 為 `\u806f\u96fb` / `\u53f0\u7a4d\u96fb`，也就是「聯電」/「台積電」。PowerShell 直接顯示可能因 console encoding 看起來像 mojibake，但瀏覽器會正常顯示。

## Live cumulative return card（2026-05-22）

- `/api/v1/live/console` 新增 `performance` payload。
- Web console 新增 `Live Cum Return` 卡片，holdings 表新增 `Unrealized PnL` 欄。
- 計算規則：若 latest official run 的 `portfolio_snapshots.unrealized_pnl` 有資料，則用 `sum(unrealized_pnl) / capital` 顯示累積報酬率；若沒有成交 / 未實現損益資料，回傳 `status=UNAVAILABLE` 並顯示 `N/A`。
- 目前 official run 的 `pnl_rows=0`，所以 UI 會顯示 `N/A`。這是刻意防守，避免把尚未成交的 recommendation 當成真實 live performance。

## Live execution / PnL / adaptation event closure（2026-05-25）

- 新增 `migrations/006_live_execution_accounting.sql`：
  - `live_accounts` seed `paper_main`。
  - `live_market_prices` 保存 accounting price，必填 `price_source` / `adjustment_mode`，避免把 alpha 用 TEJ 還原價和會計價混在一起。
  - `live_position_snapshots` 保存 account-aware position state。
  - `live_account_snapshots` 以 `(account_id, as_of_date)` 作每日 official accounting state，`run_id` 僅作來源追溯。
  - `orders` / `fills` 擴充 `account_id`、`run_id`、`recommendation_id`、broker id、tax、fees、price metadata、raw payload。
- 新增 `src/live/execution_service.py` 與 `scripts/import_live_fills.py`：
  - approved recommendation → orders。
  - paper orders → fills。
  - broker/manual fills CSV normalize，缺 `recommendation_id` 時用 `(run_id, security_id, side)` 對應。
  - 平均成本 reconciliation，產生 position snapshots 與 account snapshot；SELL realized PnL 扣 fees/tax，account equity 由 cash + market value 計算。
- `/api/v1/live/execution/paper-fill` 可對 latest 或指定 run 產生 paper fills 並 reconcile。
- `/api/v1/live/account/snapshot/latest` 回傳最新 account snapshot。
- `/api/v1/live/console` 的 performance 現在優先讀 `live_account_snapshots`；若 migration/資料尚未存在才 fallback 舊的 `portfolio_snapshots.unrealized_pnl` 防守路徑。
- 新增 `src/monitoring/live_pnl_monitor.py`，從 account snapshots / orders / fills / recommendations 產生 scoped strategy metrics：daily/cumulative return、rolling Sharpe、max drawdown、fill rate、slippage、cost bps、tracking error。
- 新增 `migrations/007_adaptation_events.sql`：
  - `monitoring_metrics` / `alerts` 補 `run_id`、`account_id`、`model_id`、`strategy_id`、metadata。
  - 新增 `adaptation_events`。
  - `alerts.adaptation_event_id` 指向觸發它的 event；`triggered_adaptation` 保留為 denormalized flag。
- `PerformanceTriggeredAdapter.check_trigger_from_db()` 現在回傳 `TriggerDecision`，但保留 `triggered, reason = ...` tuple unpack 相容性。
- DB trigger 只吃未 acknowledge、未綁 event、且 monitor_type 屬於 `model` / `strategy` 的 CRITICAL alerts，並依 `account_id` / `model_id` scoped。
- 20 個交易日 cooldown 中會建立 `SKIPPED_COOLDOWN` event，不 promote。
- 新增 `evaluate_shadow_gate_and_promote()`：candidate 需 IC 不劣於 current，且 `topk_net_return` 至少改善 0.005，才呼叫 `ModelRegistryManager.promote_model()`；否則 event 設為 `REJECTED`。
- `daily_online_pipeline` 查 production artifact 時改為優先讀 `model_registry.status='production'`，再 fallback `production.json` pointer。
- 驗證：
  - `py_compile` passed：live execution/API/monitoring/adaptation 相關檔案。
  - 目標測試 `28 passed`：`test_live_execution_service.py`、`test_live_pnl_monitor.py`、`test_performance_trigger_events.py`、`test_live_portfolio_service.py`、`test_adaptation_loop.py`、`test_monitoring.py`、`test_shadow_evaluator_proxy.py`。

## Live PnL monitor auto wiring（2026-05-25）

- `LiveExecutionService.reconcile_run()` 現在完成以下步驟後會自動呼叫 `emit_live_pnl_metrics()`：
  - 寫入 `live_market_prices`。
  - 寫入 `live_position_snapshots`。
  - 寫入 `live_account_snapshots`。
  - 將有 fill 對應的 recommendations 標為 `EXECUTED`。
- `emit_live_pnl_metrics()` 會讀取：
  - 截至當日的 `live_account_snapshots`。
  - 當次 run 的 `orders`。
  - 當次 run 的 `fills`。
  - 當次 run 的 `trade_recommendations`。
- 然後用 `LivePnLMonitor.run()` 產生 strategy metrics，並呼叫：
  - `AlertManager.persist_metrics(metrics)`
  - `AlertManager.fire_alerts(metrics)`
- metrics 會帶上 `account_id`、`run_id`、`model_id`、`strategy_id`，使後續 `PerformanceTriggeredAdapter.check_trigger_from_db()` 能 scoped 讀取。
- 若 monitoring schema 尚未套用、PostgreSQL 不可用或 alerts 寫入失敗，只會記錄 `live_pnl_monitoring_emit_failed` warning，不讓 accounting reconciliation 失敗。
- `LiveExecutionService.__init__()` 新增 `emit_monitoring_metrics=True` 參數；測試或特殊重算可設為 false。
- 補測試：`test_emit_live_pnl_metrics_persists_metrics_and_alerts()`，確認 reconciliation 後處理會 persist metrics 並 fire alerts。
- 驗證：
  - `py_compile` passed：`src/live/execution_service.py`、`src/monitoring/live_pnl_monitor.py`、`src/monitoring/alert_manager.py`、`src/adaptation/performance_trigger.py`、`src/api/routes/live.py`、`scripts/import_live_fills.py`。
  - 目標測試 `31 passed`。
## Live execution DB smoke / trigger_type 修正（2026-05-25）

- PostgreSQL container 已實際套用 `migrations/006_live_execution_accounting.sql` 與 `migrations/007_adaptation_events.sql`，沒有破壞既有 live tables。
- 第一條隔離 smoke 使用 `paper_smoke` / run `23021713-621f-4939-abfe-668a45c88790`，驗證 recommendation → order → fill → position/account snapshot → monitoring_metrics → alerts 可落 DB；`daily_return=-0.0545457913` 觸發 `daily_return` CRITICAL alert。
- Smoke 發現 `_infer_trigger_type()` 會因為 `Critical alerts...` 內含字串 `ic` 而誤判成 `rolling_ic`；已改為先判斷 `critical alerts`，並將 IC 判斷收斂到 `rolling ic` / 獨立 `ic` token。
- 第二條隔離 smoke 使用 `paper_smoke_fix` / run `20f966d6-92bf-4e8d-8325-04ac4e3f543a`，確認新的 `adaptation_events.trigger_type='critical_alerts'`，且 alerts 已更新 `adaptation_event_id` 與 `triggered_adaptation=true`。
- 驗證：`py_compile` 通過；`tests/unit/test_performance_trigger_events.py -q` 為 5 passed；live execution/PnL/monitoring/adaptation 目標測試為 32 passed。

## 台股下單單位防線（2026-05-26）

- 新增 `src/live/order_units.py`，正式把內部 canonical quantity 定義為「股數」，並集中處理台股股數到 Shioaji order unit 的轉換。
- 轉換規則：1 張 = 1000 股；`Common` 的 Shioaji `quantity` 是張數；`IntradayOdd` 的 Shioaji `quantity` 是股數。
- 例：500 股 → `IntradayOdd quantity=500`；2000 股 → `Common quantity=2`；2500 股 → 拆成兩腿 `Common quantity=2` 與 `IntradayOdd quantity=500`。
- `LiveExecutionService.build_orders_from_recommendations()` 現在會把 recommendation 拆成 Shioaji-compatible legs，但 DB `orders.quantity` 仍保存該 leg 的實際股數，確保 paper fills / accounting 不混入張數單位；Shioaji 對應欄位寫入 `raw_payload`。
- recommendation export CSV 現在輸出 `quantity` / `quantity_unit` / `shioaji_order_lot` / `shioaji_quantity` / `shioaji_quantity_unit`；Web console 顯示 `shioaji_order_plan` 摘要。
- 文件已更新 `docs/live_daily_operating_layer.md` 的「台股下單單位」段落。
- 驗證：`py_compile` 通過；`tests/unit/test_tw_order_units.py tests/unit/test_live_execution_service.py tests/unit/test_live_portfolio_service.py -q` 為 15 passed。

## 2026-05 完整本機驗收（2026-05-29）

- 使用 repo root 的 `OHLSV20260529055726.csv` 先跑 append dry-run：incoming raw 21,240 rows、普通股 21,120 rows，日期範圍 `2026-05-04`→`2026-05-29`，和既有 TEJ parquet（截至 `2026-04-30`）沒有 overlap，`added_keys=21120`。
- 正式執行 `python -m pipelines.live_daily_runner --tej-input OHLSV20260529055726.csv`：
  - `data/tw_stocks_tej.parquet` 更新為 2,067,750 rows，`data/tw_stocks_tej_universe.parquet` 同步重建。
  - 備份已寫入 `data/backups/tej_daily/tw_stocks_tej_20260529_180305.parquet` 與 `data/backups/tej_daily/tw_stocks_tej_universe_20260529_180305.parquet`。
  - TEJ alpha cache 增量計算 20 個五月交易日，`data/alpha_cache/wq101_alphas.parquet.manifest.json` 更新為 rows 202,073,516、n_securities 1107、n_alphas 101、end `2026-05-29`。
  - 本機 live run `301eb66f-741c-4df2-a1ee-1df24f2d30e5` 完成，輸出在 `reports/live/301eb66f-741c-4df2-a1ee-1df24f2d30e5/`；manifest 顯示 as-of `2026-05-29`、`data_freshness_status=FRESH`、lag 0、`retrain_action=scheduled_or_initial_retrain`。
  - 新 production artifact 為 `artifacts/models/ml_xgb_a5bb9308`，20 個 feature alphas，holdout rank_ic 約 0.03999，feature hash `20f09c4f3b5ec963c4c10b2b9b2903e5ff9f4fca72da9ca35a0adf9651916227`。
  - `trade_recommendations.csv` 產出 10 筆 `PENDING` BUY recommendations（2308、2357、2382、3533、5258、6442、6446、6669、8039、8996）。
- 限制：當下 Docker Desktop / PostgreSQL 未啟動，`127.0.0.1:5433` connection refused，因此 model registry promote、previous official holdings lookup 與 live operational DB persist 都失敗；本次只能視為「本機資料、cache、artifact、CSV 輸出驗收完成」，Grafana/API latest official 尚未補登。DB 起來後可用已更新的五月資料重跑 `pipelines.live_daily_runner` 或 `pipelines.daily_online_pipeline` 完成 official persist。
- 驗證：`tests/unit/test_tej_daily_append.py tests/unit/test_live_portfolio_service.py tests/unit/test_ml_meta_model.py -q` 為 15 passed。

## 2026-05 DB/Grafana official persist 補登（2026-05-29）

- Docker Desktop 開啟後確認 compose 服務：`darams-postgres`、`darams-redis`、`darams-grafana`、`darams-api` 均為 Up；`scripts/validate_infrastructure.py --skip-dolphindb` 顯示 PostgreSQL / Redis OK。
- 第一次重跑 `python -m pipelines.daily_online_pipeline --mode auto --as-of 2026-05-29 --official --run-purpose production` 時，在 `alpha_cache._align_to_bar_keys()` 的全量 pandas merge 上 OOM（109,178,493 rows，需額外配置約 833 MiB join indexer）。
- 已修正 `src/alpha_engine/alpha_cache.py`：對齊 bars key 時改為依 `alpha_id` 分批 semi-join，建立單一 boolean keep mask，避免一次性大型 join indexer。語意維持「只保留當次 bars 實際存在的 `(security_id, tradetime)` alpha rows」。
- 修正後重跑成功：
  - official live run：`3f9a72ca-8365-4223-8de2-a74c118e9267`
  - as-of：`2026-05-29`
  - `data_freshness_status=FRESH`、lag 0
  - production model：`ml_xgb_d04bf2d8`，已寫入 `model_registry` 並 promote 為 `production`
  - run artifacts：`reports/live/3f9a72ca-8365-4223-8de2-a74c118e9267/`
  - DB 寫入驗證：`daily_live_runs=1`、`alpha_selection_snapshots=2`、`alpha_selection_scores=110`、`meta_signals=1056`、`portfolio_targets=19`、`portfolio_snapshots=19`、`trade_recommendations=19`
  - recommendations action summary：BUY 9、HOLD 1、REDUCE 9；因成功讀到前一筆 official holdings，turnover cap 生效為 25%，和前一次 DB 不通時的全 BUY 本機輸出不同。
- API 驗證：`GET http://127.0.0.1:8000/api/v1/live/console` 回傳 latest official run `3f9a72ca-8365-4223-8de2-a74c118e9267`、as-of `2026-05-29`、`FRESH`、model `ml_xgb_d04bf2d8`，holdings 19。
- 驗證：
  - `tests/unit/test_alpha_cache.py -q`：9 passed。
  - `py_compile src/alpha_engine/alpha_cache.py pipelines/daily_online_pipeline.py src/live/operational_store.py`：通過。

## Live Console target M2M 報酬 fallback（2026-05-29）

- 使用者詢問 2026-05 驗收報酬率與前端未顯示原因。原因是 Live Console 的 `Live Cum Return` 原本只接受真實 execution / accounting 資料：
  - 優先讀 `live_account_snapshots` 的 equity curve。
  - 若沒有 account snapshot，才看 `portfolio_snapshots.unrealized_pnl`。
  - 目前 2026-05-29 official run 尚未匯入 fills / reconcile，所以 `unrealized_pnl` 為空，前端依防守邏輯顯示 N/A。
- 已補 `src/api/routes/live.py`：在沒有真實 PnL 時，用上一筆 official target holdings 與本次 as-of 價格估算 target mark-to-market，回傳 `status=ESTIMATED`、`basis=target_portfolio_mark_to_market`，並附 message 明確標示「非真實 execution PnL」。
- 已補 `src/api/static/live_console.html`：`performance.status === "ESTIMATED"` 時也顯示 `Live Cum Return`，meta 文字用 API message，避免誤當真實成交績效。
- 2026-05 驗收估算：
  - period：2026-04-30 close → 2026-05-29 close
  - previous official run：`dd1b4b6c-06f8-480c-8554-767cbec9836a`
  - latest official run：`3f9a72ca-8365-4223-8de2-a74c118e9267`
  - weighted target return：約 +12.520155%
  - rounded-shares capital return：約 +12.518184%
  - estimated unrealized PnL：約 +1,251,818.4（以 capital 10,000,000 計）
- 驗證：`py_compile src/api/routes/live.py` 通過；`docker compose restart api` 後，`GET /api/v1/live/console` 回傳 `performance.status=ESTIMATED`、`cumulative_return=0.12518184`。
