---
name: WP9 成本感知實驗流程重設計落地
description: 2026-05-04 已實作 turnover-aware long-only portfolio、net-return proxy 診斷指標、benchmark row 與 CLI flags；長時間 TEJ Phase 1/2 實驗尚未執行
type: project
originSessionId: codex-2026-05-04
---

2026-05-04 根據 TEJ baseline 全負績效診斷，已將 WP9 第一版成本感知實驗流程落地到程式介面。

## 已完成

- 新增 `turnover_aware_topk`：支援 entry/exit rank buffer、最短持有天數、沿用既有持股。
- `RiskManager.apply_constraints()` 新增 `previous_weights`，turnover cap 直接用目標權重差異計算，不再依賴 `quantity/current_weight` 混用。
- `simulate_recent` 新增低換手 portfolio CLI flags，並在 `daily_pnl.csv` 輸出 `rebalance_flag`、`held_from_prev_count`、`forced_sells_count`、`turnover_cap_applied`。
- `MLMetaModel` holdout metrics 新增 `gross_ic`、`net_ic_proxy`、`long_only_topk_net_return`、`turnover_proxy`。
- `ab_experiment` 支援同一組 portfolio/objective flags，並新增 `--benchmark ew_buy_hold_universe`。

## 驗證

已通過：

```powershell
.\.venv\Scripts\python.exe -m py_compile src\portfolio\constructor.py src\risk\risk_manager.py src\meta_signal\ml_meta_model.py pipelines\simulate_recent.py pipelines\ab_experiment.py tests\unit\test_turnover_aware_portfolio.py tests\unit\test_ml_meta_model.py tests\integration\test_ab_experiment.py

.\.venv\Scripts\python.exe -m pytest tests/unit/test_turnover_aware_portfolio.py tests/unit/test_risk.py tests/unit/test_ml_meta_model.py tests/integration/test_ab_experiment.py -q
```

結果：`33 passed`。CLI help 也已確認 `simulate_recent` / `ab_experiment` 都能載入新 flags。

## 尚未完成

- 尚未跑 TEJ Phase 1 小矩陣。
- 尚未用最佳 portfolio 設定重跑 Phase 2 五策略 + cost sweep。
- 尚未跑 `notebooks/03_adaptation_evaluation.py` 產出正式 paired test / figure。

建議下一步先跑短窗 smoke：

```powershell
.\.venv\Scripts\python.exe -m pipelines.ab_experiment --data-source tej --start 2022-06-01 --end 2022-08-31 --strategies none scheduled_60 --portfolio-method turnover_aware_topk --rebalance-every 5 --entry-rank 20 --exit-rank 40 --max-turnover 0.25 --min-holding-days 5 --objective net_return_proxy --benchmark ew_buy_hold_universe --run-tag costaware_smoke
```
