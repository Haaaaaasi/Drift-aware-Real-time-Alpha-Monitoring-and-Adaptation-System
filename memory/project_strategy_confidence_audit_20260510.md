---
name: 策略信心稽核（2026-05-10）
description: 架構 / 研究策略審視後的主要漏洞與修正優先順序；結論是不可宣稱 100% 信心
type: project
originSessionId: codex-2026-05-10
---

# 策略信心稽核（2026-05-10）

## 核心結論

目前不能、也不應宣稱對 DARAMS 研究策略有「事實上的 100% 信心」。比較精準的說法是：

> 系統工程骨架與若干方法學防線已經比早期穩很多，但正式研究結論仍有幾個會影響 reviewer-facing claims 的高風險漏洞。主線策略 `scheduled_20 + top_k=10 + rebalance_every=10 + train_window_days=500` 只能描述為「在 TEJ、正式成本、既有探索期間內優於 no-adapt 且累積報酬略勝 equal-weight benchmark」，不可描述為已統計顯著、風險調整後勝過 benchmark、或可部署交易策略。

## 新發現的高優先風險

1. **TEJ 路徑缺真實產業分類與市值資料**
   - `pipelines/daily_batch_pipeline.py` 目前對 parquet/CSV 都用 `cap = close * 1_000_000` 與 `indclass = hash(security_id) % 10 + 1`。
   - Python WQ101 中多個 alpha 使用 `indclass` 或 `cap`；TEJ effective 64 alpha 內至少包含 `wq048, wq058, wq059, wq076, wq079, wq080, wq082, wq087, wq100` 等 industry-neutralized alpha。
   - Python `hash()` 受 hash randomization 影響，不同 process 的假產業分組可能不同；alpha cache manifest 目前未記錄 `indclass_source` 或 mapping hash。
   - 修正前，正式結果應標註「部分 industry-neutralized WQ101 alpha 使用 placeholder industry groups」，不可直接宣稱完整復刻 WQ101 industry-neutral logic。

2. **回測成交時點可能樂觀**
   - `simulate_recent` 目前語意是「T 日收盤產生訊號並在 T 收盤建倉，吃 close[T+1]/close[T]」。
   - 若 alpha 使用 T 日 close/vwap/high/low，真實 after-close 流程無法同時以 T close 成交。
   - 應新增保守 execution mode：`signal_at=t close -> fill_at=t+1 open/VWAP -> hold to next rebalance open/VWAP`，並把目前 close-to-close 模式標成 optimistic research proxy。

3. **下市日 `next_return = NaN -> 0` 的處理可能過度簡化**
   - 目前規則把最後一筆 OHLCV 後的報酬當 0 並隔日退出 universe。
   - 對合併下市可能保守，對終止上市 / 停牌 / 跌停流動性事件可能樂觀。
   - 應從 TEJ universe / 下市原因補終端報酬分類，或至少做 `0 / -50% / -100%` terminal shock sensitivity。

4. **策略參數已在同一段 OOS 上多輪探索**
   - `scheduled_20`、`top_k=10`、`rebalance_every=10`、`train_window_days=500` 是經多次診斷與矩陣搜尋後得到。
   - 現有 paired test 未達 5% 顯著，且 benchmark Sharpe / max drawdown 仍較佳。
   - 應凍結參數後使用 untouched final holdout、blocked bootstrap / SPA 或 Deflated Sharpe，並清楚揭露多重比較風險。

5. **docs 與 memory 中仍有 deprecated 結論**
   - `docs/architecture.md` 仍含 yfinance / DolphinDB 主路徑、52 alpha、Phase A 15,754% 等已被 TEJ / yfinance 8476 artifact 推翻的數字。
   - `configs/alpha_config.yaml` 的 `v3_effective_alphas_is` 註解仍是舊 yfinance 52-alpha 說明；正式 single source of truth 是 `reports/alpha_ic_analysis/effective_alphas.json` 的 TEJ 64 alpha。
   - Reviewer-facing 文件在清理前不可直接使用。

## 已驗證項目

- `pytest tests/unit -q`：186 passed，61 warnings。
- 護欄測試：`test_data_source_guard.py`、`test_label_maturity.py`、`test_simulate_recent_cost.py`、`test_model_pool_shadow_window.py` 共 29 passed。
- `tests/integration/test_adaptation_loop.py`：13 passed。

## 未完成驗證

- 完整 `pytest -q` 在 5 分鐘內未完成。
- `tests/integration/test_ab_experiment.py`、`tests/integration/test_model_pool_strategy.py`、`tests/integration/test_pipeline_batch.py + test_replay_pipeline.py` 分段也在 3 分鐘內未完成。
- 這代表慢速整合測試需要拆分成 fast smoke / slow nightly，否則無法作為每次研究變更的可靠回饋。

## 修正優先順序

P0：
- 取得或整理 TEJ 產業分類 / 市值資料，重算 alpha cache 與 TEJ effective alpha selection；若短期做不到，正式實驗先排除所有需要 `indclass` 或 `cap` 的 alpha，跑一次 ablation。
- 新增 T+1 open/VWAP fill mode，重跑 `scheduled_20` vs `none` vs benchmark。
- 清理 `docs/architecture.md` 與 `configs/alpha_config.yaml` 的 stale / deprecated 結論。

P1：
- 做 terminal delisting shock sensitivity。
- 凍結目前主線參數，建立 untouched final holdout 或 blocked bootstrap / SPA / Deflated Sharpe 評估。
- 拆整合測試標記：fast integration < 2 min；slow experiments 改 nightly。

P2：
- model_pool 補跨 run artifact persistence；否則 recurring concept pool 仍不是完整 recurring reuse。
- 將 triggered adaptation 的正式路徑改成 DB-driven monitoring alerts，而不是只吃 simulation memory rolling metrics。
## 2026-05-10 後續修補狀態

已先處理兩個最直接的方法學漏洞：

- 新增 `--exclude-indclass-cap-alphas`，把 TEJ IS-only 64 alphas 中需要 placeholder `indclass` 或 `cap` 的 9 個排除，短期使用 55 個純量價 alpha 做 reviewer-facing ablation。
- 新增 `--execution-price {close,next_open,next_vwap}`，讓 `simulate_recent` / `ab_experiment` / `ew_buy_hold_universe` benchmark 都可改用 T 日收盤訊號、T+1 open 或 VWAP 成交。

已完成文件整理與短窗 smoke。正式 full-period rerun 尚未完成，因此策略信心仍不能升級為 100%。下一步是固定 TEJ、55-alpha ablation、`none` / `scheduled_20` / benchmark，分別跑 `next_open` 與 `next_vwap` 的 2022-06-01 至 2024-12-31 全期間 A/B。

## 2026-05-11 rerun 後信心更新

正式 OOS（2024-07-01 → 2026-04-30）與 legacy comparability（2022-06-01 → 2024-12-31）的 no-indclass/cap + T+1 open/VWAP rerun 已完成，摘要在 `reports/adaptation_ab/no_indcap_execution_rerun_summary_20260511.md`。

結果顯示：

- 正式 OOS：`scheduled_20` 優於 `none`，但輸給 `ew_buy_hold_universe`。
- Legacy comparability：`scheduled_20` 輸給 `none`，也大幅輸給 `ew_buy_hold_universe`。

因此策略信心不能提高；相反地，原 `scheduled_20` 主策略 claim 必須下修。可防守說法是「scheduled adaptation 降低 no-adapt OOS 退化，但尚不足以勝過市場基準」。這解除了一個誇大結論風險，也把下一步焦點轉向 turnover、portfolio mapping、真實 `indclass` / `cap` 與 benchmark-aware reporting。

## 2026-05-11 信心迴圈與 cache 修補

新增稽核文件：`reports/adaptation_ab/strategy_confidence_loop_20260511.md`。

本輪再確認：不能宣稱 100% 信心；目前信心門檻改為「沒有已知 P0/P1 方法學缺陷且能穩定勝過 benchmark」，而非口頭 100%。已列出尚未關閉漏洞：portfolio turnover、`net_return_proxy` 仍偏診斷而非訓練 objective、`simulate_recent` 未把 current drawdown 傳給 `RiskManager`、下市終端報酬簡化、真實 indclass/cap 尚缺、selection stability / benchmark set / statistical test 尚不足。

同時修補一個可重現性漏洞：TEJ alpha cache manifest 顯示 200,032,808 rows，舊 `compute_with_cache()` 會全量讀 pandas 再切日期與 alpha，已在 `compat next_vwap scheduled_20` 原始 combined run 觸發 1.61 GiB allocation error。`src/alpha_engine/alpha_cache.py` 已新增 parquet filter 下推 `_read_cache_slice()`，在 manifest 日期覆蓋本次 bars 時直接用 `tradetime` + `alpha_id` 讀 slice。驗證：`pytest tests/unit/test_alpha_cache.py tests/unit/test_execution_alpha_universe.py -q` 為 15 passed；實際 TEJ cache smoke `2024-07-01→2024-07-10, wq001/wq002` 只讀 16,624 rows 並輸出 `cache_slice_read`。

## 2026-05-11 turnover-aware OOS rerun

依信心迴圈的下一步，已跑正式 OOS `turnover_aware_topk` 初始組合：TEJ、2024-07-01→2026-04-30、55 個 no-indclass/cap alpha、`none/scheduled_20/ew_buy_hold_universe`、`top_k=10`、`rebalance_every=10`、`entry_rank=20`、`exit_rank=40`、`max_turnover=0.25`、`min_holding_days=5`、`train_window_days=500`、`horizon_days=5`。

結果：
- `next_open`：`scheduled_20` cum +25.322%、Sharpe 0.610、max DD -35.407、avg cost 1.674 bps/day；勝 `none` +10.141% / 0.337 與 benchmark +10.618% / 0.403。
- `next_vwap`：`scheduled_20` cum +13.062%、Sharpe 0.412、max DD -40.282、avg cost 1.674 bps/day；勝 `none` -1.198% / 0.094 與 benchmark +4.178% / 0.220。

判讀：低換手 portfolio 後，scheduled_20 在兩個 T+1 execution 假設下都勝過 none 與 EW benchmark，顯示固定 top-k rerun 失敗主因很可能是 turnover/cost，而非 alpha edge 完全消失。但信心仍未達 100%：drawdown 仍明顯劣於 benchmark，且 turnover cap 使平均持倉擴散到約 60–104 檔，`top_k=10` 語意已變成緩慢換倉多檔組合；下一步需做 holdings concentration 診斷與 entry/exit/max_turnover/min_holding_days 小矩陣。輸出摘要：`reports/adaptation_ab/turnover_aware_oos_summary_20260511.md`。

## 2026-05-12 holdings concentration 與過夜矩陣啟動

已新增：
- `scripts/diagnose_holdings_concentration.py`
- `scripts/run_turnover_oos_workflow.py`

Holdings concentration 診斷已完成，輸出：`reports/adaptation_ab/holdings_concentration_20260512/`。重點：
- `scheduled_20` 平均實際持倉約 60.99 檔，effective holdings 約 35.74，top10 weight share 約 45.7%。
- `none` 平均實際持倉約 103.79 檔，effective holdings 約 40.91，top10 weight share 約 38.6%。
- `scheduled_20` 的 average negative-score weight 約 31.4%，代表不少權重留在 XGB 當下已不看好的股票。

判讀：低換手績效改善不是嚴格 top-10 策略，而是 XGB ranking + turnover cap 形成的 slow-rotation 多檔組合。這提高了「portfolio turnover 是主要漏洞」的可信度，但也新增 exit discipline 風險：需要 hard exit（signal<=0 / rank 極差 / 不可交易）、soft exit（過 min_holding_days 且 rank 跌出 exit_rank / 負貢獻）與 tail cleanup。

已啟動過夜 workflow：16 組 `next_vwap` matrix → top3 `next_open` confirmation → block bootstrap / paired test。log 與 PID 在 `reports/adaptation_ab/turnover_matrix_oos_20260512/logs/`；第一個未完成組合為 `entry=20, exit=40, max_turnover=0.25, min_holding_days=10`，初始 `entry=20, exit=40, max_turnover=0.25, min_holding_days=5` 會重用 2026-05-11 既有結果。

## 2026-05-12 過夜矩陣完成

過夜 workflow 已於 2026-05-12 04:45 完成，輸出在 `reports/adaptation_ab/turnover_matrix_oos_20260512/`。所有 Python 實驗程序已結束。

Next_vwap 16 組矩陣最佳區域為 `exit_rank=60`、`max_turnover=0.25`；`entry_rank=20/30` 與 `min_holding_days=5/10` 結果完全相同，表示目前 selection/turnover cap 下這兩個參數未形成有效差異。最佳組合：
- `e20_x60_t0p25_h10` / `e20_x60_t0p25_h5` / `e30_x60_t0p25_h10` / `e30_x60_t0p25_h5`：next_vwap `scheduled_20` cum +17.704%、Sharpe 0.501、max DD -42.056、相對 benchmark +13.526% cum / +0.281 Sharpe。
- `max_turnover=0.50` 版本較差：cum +15.461%、Sharpe 0.447、avg cost 3.051 bps/day；成本上升吃掉部分 edge。

Top3 next_open 複驗皆為同一績效：cum +31.478%、Sharpe 0.697、max DD -36.520、相對 benchmark +20.860% cum / +0.294 Sharpe。

Bootstrap / paired test：方向正，但統計顯著性仍不足。最佳 next_vwap vs benchmark mean daily excess +3.30 bps，95% block bootstrap CI 約 [-5.45, +11.60] bps，p_one_sided_boot 0.2154，paired t one-sided p 0.1664；next_open vs benchmark mean daily excess +4.78 bps，CI 約 [-3.41, +12.74] bps，p_one_sided_boot 0.119，paired t p 0.1571。

結論更新：低換手 scheduled_20 的績效與方向比固定 top-k 明顯更可防守，且在 next_vwap/next_open 都勝 benchmark；但仍不可宣稱統計顯著或 100% 信心。下一步應處理 exit discipline / holdings concentration（尤其 negative-score weight 與 long tail），並避免把多檔 slow-rotation 說成純 top-10 策略。
## 2026-05-12 exit discipline OOS 小矩陣

holdings concentration 顯示 `scheduled_20` 存在大量 residual / negative-score exposure 後，新增 exit discipline 實驗流程：
- `pipelines/simulate_recent.py` / `pipelines/ab_experiment.py` 已支援 hard exit 與 tail cleanup 參數，並輸出 exit 診斷欄位。
- `scripts/run_exit_discipline_oos.py` 固定上一輪最佳低換手參數 `entry=20, exit=60, max_turnover=0.25, min_holding_days=10`，比較 baseline、hard0、tail25、tail50。
- 測試：`py_compile` 通過；`pytest tests/unit/test_exit_discipline.py tests/unit/test_simulate_recent_cost.py tests/unit/test_execution_alpha_universe.py -q` 為 23 passed。

OOS 結果（2024-07-01 至 2026-04-30，TEJ、exclude indclass/cap、next_vwap）：
- `tail25`：scheduled_20 cum +18.680%、Sharpe 0.521、max DD -41.907%，小幅優於 baseline。
- `baseline`：scheduled_20 cum +17.704%、Sharpe 0.501、max DD -42.056%。
- `tail50`：cum +14.484%、Sharpe 0.440，門檻太重。
- `hard0`：cum -9.913%、Sharpe -0.390，明顯破壞策略，不應採用。

next_open confirmation：
- `tail25`：cum +32.532%、Sharpe 0.715，略優於 baseline +31.478% / 0.697。
- Bootstrap / paired test 仍未達 5%：tail25 next_vwap vs benchmark mean excess +3.48 bps/day，CI [-5.20, +11.62]，p_boot 0.203；next_open vs benchmark +4.95 bps/day，CI [-3.31, +12.78]，p_boot 0.109。

結論：不要用「滿持倉天數後 signal_score<=0 直接砍」當 hard exit；可以採 `tail_cleanup_weight=0.0025` 作為 portfolio hygiene，但不能宣稱顯著 alpha improvement。輸出：`reports/adaptation_ab/exit_discipline_oos_20260512/workflow_summary.md`。
