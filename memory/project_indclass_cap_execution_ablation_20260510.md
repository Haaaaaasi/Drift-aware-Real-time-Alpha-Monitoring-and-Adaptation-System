# Indclass/Cap Ablation 與 T+1 Execution 修補（2026-05-10）

## 背景

2026-05-10 的策略信心審查指出，TEJ 主線仍存在兩個必須先修的 P0 方法學漏洞：

1. `load_csv_data()` 仍用 placeholder 產生 `indclass = hash(security_id) % 10 + 1` 與 `cap = close * 1_000_000`。所有依賴 `indclass` 或 `cap` 的 WQ101 alpha 都可能混入人造結構訊號。
2. 舊 PnL 使用 close-to-close proxy，等同 T 日收盤後產生訊號卻可用 T 日收盤成交，對真實交易偏樂觀。

使用者決策：下市日 `next_return = 0` 的簡化先維持；`model_pool` 尚不宣稱完整 recurring concept reuse，等架構完全穩定後再處理。

## 本輪修補

### Alpha universe

- `src/config/constants.py`
  - 新增 `WQ101_CAP_ALPHA_IDS = ["wq056"]`。
  - 新增 `WQ101_INDCLASS_OR_CAP_ALPHA_IDS`。
  - `WQ101_PURE_PRICE_ALPHA_IDS` 改為同時排除 indclass 與 cap alpha，總數 82。
- `src/config/alpha_selection.py`
  - 新增 `exclude_indclass_cap_alpha_ids()`。
  - `load_effective_alpha_ids(..., exclude_indclass_cap=True)` 可直接輸出 no-indclass/cap 清單。
- `pipelines/simulate_recent.py` / `pipelines/ab_experiment.py`
  - 新增 `--exclude-indclass-cap-alphas`。
  - TEJ IS-only 64 alphas 在 ablation 後變成 55 個純量價 alpha。
  - 若手動 `--alpha-ids` 與 ablation 後交集為空，直接報錯，避免默默退回全量 alpha。

### Execution

- `pipelines/simulate_recent.py`
  - `_next_day_returns()` 新增 `execution_price`：
    - `close`：舊 close-to-close proxy。
    - `next_open`：T 日收盤訊號，T+1 open 進場，T+2 open 計算下一期報酬。
    - `next_vwap`：T 日收盤訊號，T+1 VWAP 進場，T+2 VWAP 計算下一期報酬。
  - `daily_pnl.csv` 與 `config.json` 記錄 `execution_price`。
- `pipelines/ab_experiment.py`
  - 新增 `--execution-price {close,next_open,next_vwap}`。
  - A/B 策略與 `ew_buy_hold_universe` benchmark 都套用同一 execution 假設。

### 文件

- `docs/architecture.md` 改寫為目前可防守架構狀態，移除舊 yfinance / DolphinDB 大表 / Phase A 超高報酬主張。
- `README.md` 改成目前主線、近期優先事項與 rerun 指令。
- `configs/alpha_config.yaml` 整理為 TEJ IS-only 64-alpha 清單，新增 55-alpha no-indclass/cap ablation 清單。

## 驗證

- `py_compile` 通過：
  - `src/config/constants.py`
  - `src/config/alpha_selection.py`
  - `pipelines/simulate_recent.py`
  - `pipelines/ab_experiment.py`
- YAML parse / count check 通過：
  - `v3_effective_alphas_is = 64`
  - `v3_effective_alphas_is_no_indclass_cap = 55`
  - `v3_effective_alphas_requires_indclass_or_cap = 9`
- Unit tests：
  - `tests/unit/test_execution_alpha_universe.py`
  - `tests/unit/test_simulate_recent_cost.py`
  - 結果：20 passed。
- CLI help：
  - `pipelines.simulate_recent --help` 與 `pipelines.ab_experiment --help` 都已顯示 `--execution-price` 與 `--exclude-indclass-cap-alphas`。

## Smoke rerun

### 單策略 smoke：next_open

指令摘要：

```powershell
python -m pipelines.simulate_recent --data-source tej --start 2024-01-02 --end 2024-01-31 --strategy none --top-k 10 --rebalance-every 10 --train-window-days 500 --exclude-indclass-cap-alphas --execution-price next_open --run-tag no_indcap_nextopen_smoke_20260510
```

重點結果：

- `n_features=55`
- 累積報酬 -0.755%
- Sharpe -0.552
- 輸出：`reports/simulations/sim_20240102_20240131_top10_none_no_indcap_nextopen_smoke_20260510/`

### 單策略 smoke：next_vwap

重點結果：

- `n_features=55`
- 累積報酬 -0.606%
- Sharpe -0.505
- 輸出：`reports/simulations/sim_20240102_20240131_top10_none_no_indcap_nextvwap_smoke_20260510/`

### A/B smoke：next_open

指令摘要：

```powershell
python -m pipelines.ab_experiment --data-source tej --start 2024-01-02 --end 2024-01-31 --strategies none scheduled_20 --top-k 10 --rebalance-every 10 --train-window-days 500 --benchmark ew_buy_hold_universe --exclude-indclass-cap-alphas --execution-price next_open --run-tag no_indcap_nextopen_ab_smoke_20260510
```

短窗結果：

| Strategy | Cum Ret % | Sharpe | n_retrains |
|---|---:|---:|---:|
| none | -0.755 | -0.552 | 1 |
| scheduled_20 | 0.472 | 0.429 | 2 |
| ew_buy_hold_universe | -0.138 | -0.136 | 0 |

輸出：`reports/adaptation_ab/ab_20240102_20240131_top10_no_indcap_nextopen_ab_smoke_20260510/`

## 判讀

本輪 smoke 只證明修補後的 pipeline 可跑通，不能視為正式策略結論。下一步正式 rerun 應固定：

- TEJ
- `--exclude-indclass-cap-alphas`
- `--execution-price next_open` 與 `next_vwap` 各跑一次
- `none` / `scheduled_20` / `ew_buy_hold_universe`
- full period 2022-06-01 至 2024-12-31

若 scheduled_20 在兩種 T+1 execution 下仍能防守，再進入 cost sweep、regime stress 與 paired test。

## 2026-05-11 full rerun 結果

報告：`reports/adaptation_ab/no_indcap_execution_rerun_summary_20260511.md`

共同設定：TEJ、55-alpha no-indclass/cap universe、`top_k=10`、`rebalance_every=10`、`train_window_days=500`、`none / scheduled_20 / ew_buy_hold_universe`。

### 正式 OOS（2024-07-01 → 2026-04-30）

`next_open`：

| Strategy | Cum Ret % | Sharpe | Max DD % |
|---|---:|---:|---:|
| none | -18.945 | -0.172 | -50.147 |
| scheduled_20 | 1.377 | 0.192 | -32.650 |
| ew_buy_hold_universe | 10.618 | 0.403 | -25.775 |

`next_vwap`：

| Strategy | Cum Ret % | Sharpe | Max DD % |
|---|---:|---:|---:|
| none | -21.758 | -0.371 | -52.618 |
| scheduled_20 | 0.761 | 0.153 | -33.265 |
| ew_buy_hold_universe | 4.178 | 0.220 | -29.334 |

### Legacy comparability（2022-06-01 → 2024-12-31）

`next_open`：

| Strategy | Cum Ret % | Sharpe | Max DD % |
|---|---:|---:|---:|
| none | -1.439 | 0.071 | -26.591 |
| scheduled_20 | -9.414 | -0.022 | -36.122 |
| ew_buy_hold_universe | 47.882 | 1.219 | -13.142 |

`next_vwap`：

| Strategy | Cum Ret % | Sharpe | Max DD % |
|---|---:|---:|---:|
| none | -0.644 | 0.068 | -22.606 |
| scheduled_20 | -4.983 | 0.019 | -31.087 |
| ew_buy_hold_universe | 38.505 | 1.108 | -14.020 |

`next_vwap` compat 原完整 A/B 在 scheduled_20 階段遇到 pandas 記憶體尖峰，改用 recovery run 補 scheduled_20 + benchmark；none 取原 run 已完成輸出。

結論：排除 placeholder `indclass` / `cap` 並改成 T+1 execution 後，`scheduled_20` 不再能作強主策略 claim。正式 OOS 中它優於 `none`，但輸給 buy-and-hold benchmark；compat window 中它甚至輸給 `none`。後續應改寫研究敘事為「adaptation 可降低 no-adapt 退化，但 scheduled retrain 尚不足以勝過市場基準」。

## 2026-05-11 turnover-aware OOS 補跑

為檢查上述失敗是否主要來自固定 top-k 高換手，已在正式 OOS 補跑低換手初始組合：

- `portfolio_method=turnover_aware_topk`
- `entry_rank=20`
- `exit_rank=40`
- `max_turnover=0.25`
- `min_holding_days=5`
- `rebalance_every=10`
- 其他設定同上：TEJ、55-alpha no-indclass/cap、`train_window_days=500`、`horizon_days=5`、`none / scheduled_20 / ew_buy_hold_universe`

`next_open`：

| Strategy | Cum Ret % | Sharpe | Max DD % | Avg Cost bps/day |
|---|---:|---:|---:|---:|
| none | 10.141 | 0.337 | -41.128 | 1.485 |
| scheduled_20 | 25.322 | 0.610 | -35.407 | 1.674 |
| ew_buy_hold_universe | 10.618 | 0.403 | -25.775 | 0.032 |

`next_vwap`：

| Strategy | Cum Ret % | Sharpe | Max DD % | Avg Cost bps/day |
|---|---:|---:|---:|---:|
| none | -1.198 | 0.094 | -43.798 | 1.485 |
| scheduled_20 | 13.062 | 0.412 | -40.282 | 1.674 |
| ew_buy_hold_universe | 4.178 | 0.220 | -29.334 | 0.032 |

更新判讀：低換手 portfolio 後，`scheduled_20` 在兩種 T+1 execution 下都重新勝過 `none` 與 benchmark，顯示固定 top-k 版本的主要漏洞是 turnover/cost。但這不是最終 100% 信心：`scheduled_20` drawdown 仍較 benchmark 差，且 turnover cap 讓平均持倉擴散到約 60–104 檔，需補 holdings concentration 診斷與參數矩陣。
