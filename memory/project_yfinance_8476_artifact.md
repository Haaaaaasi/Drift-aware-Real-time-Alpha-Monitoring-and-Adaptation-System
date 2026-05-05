---
name: yfinance 8476 split-adjustment 資料汙染（解釋 WP9/PhaseA/PhaseB 高 cum_ret）
description: 2026-05-03 sim 對照證實：yfinance baseline 35,486% 全是 stock 8476 還原股價錯誤所致；TEJ 同期 -77%
type: project
originSessionId: c3fdda4a-b406-4698-a2cb-72a01fd11a69
---
## 結論

WP9 Phase A baseline 「scheduled_20 cum_ret 15,754%」、Phase B model_pool 「11,290%」、TodayDiagnostic baseline 「35,486%」全部都是 **yfinance 對 stock 8476 套了部分還原股價** 的副作用，不是策略真有 alpha。

切到 TEJ（survivorship-correct + 乾淨還原股價）同條件 sim：cum_ret **-77.3%**、Sharpe **-2.19**、annualized **-44.7%**。

## 8476 yfinance 資料異常

`data/tw_stocks_ohlcv.csv` 中 stock 8476：
- 2022-01-04 ~ 2024-10-30 期間 207 天 |daily ret| > 20%（TWSE 漲跌停 ±10%，物理上不可能）
- 收盤價在 ~26 與 ~52 之間「日對日交替跳動」
- yf/tej 收盤價 ratio：400 天 = 1.00、270 天 = 2.00、195 天 ≈ 2.08
  → yfinance 對某些日期套了 2-for-1 還原股價，對相鄰日期沒套，產生人造 ±50%-100% 日報酬

TEJ 同檔股票同期 |ret|>20% 天數 = **0**，每天穩定在 ~26 附近。

## 為何單一錯誤股票就把 cum_ret 拉到 35,486%

* WQ101 alpha 偵測到 8476 規則性的高波動 → 該股 alpha 值大
* MLMetaModel（XGBoost）訓練時 forward return 也是 yfinance corrupt 版 → label 中 8476 ±50% 日報酬被學習
* 模型把 8476 signal_score 給很高（樣本日 0.66 vs 第二高 0.05）
* equal_weight_topk = 10 把 8476 列入 top-10，targat_weight 0.10
* 樣本期 631 天裡 **144 天（23.2%）** 都把 8476 排進前十
* 8476 被選中那 144 天的 mean next_return = +56.5%（fake）
* 一個 0.10 weight × +50% 跳動 = +5% 組合單日報酬
* 144 個這樣的日子 compounded 出 30,000%+ 額外報酬

### 反事實量化（yfinance baseline）

| Scenario | cum_ret | Sharpe |
|---|---|---|
| Original (8476 含 fake jumps) | **+35,486%** | +4.22 |
| 將 8476 next_ret 限制在 ±10% 內 | **-63.9%** | -1.53 |
| 將整個 universe next_ret 限制在 ±10% | **-65.1%** | -1.63 |
| 完全踢掉 8476，平均分配給其他 9 檔 | **-81.1%** | -2.60 |
| **TEJ 同條件 sim（乾淨資料）** | **-77.3%** | -2.19 |

8476 單檔解釋了 **99.9%** 的 yfinance baseline dollar return。

## 三個堆疊的 bias 來源

1. **Stock 8476 yfinance 還原股價錯誤**（最主要，~99.9%）
2. **Survivorship bias**：yfinance 1083 檔全部現存上市，TEJ 1105 檔含 51 檔下市；對 8476 不影響但對其他股票一般偏正
3. **Alpha selection 在汙染資料上做**：`reports/alpha_ic_analysis/effective_alphas.json` 是用 yfinance 跑 IC 篩出的 52 alphas（`source: csv_cache`），那 52 個被選中部分原因是它們 correlate 8476 的 fake forward return；用在乾淨資料上未必有 edge
4. **IS/OOS 視窗重疊**：IS 視窗 2022-01 → 2024-06，sim 視窗 2022-06 → 2024-12，~80% 重疊（CLAUDE.md §七 P0 #1 已知，影響預期較小）

## 後續行動（待執行）

- 所有 sim 永久切到 `--data-source tej`
- 砍 `data/alpha_cache/wq101_alphas.parquet` 任何用 yfinance 跑出的 cache（已砍過）
- 重做 IS-only alpha selection on TEJ：跑 `notebooks/01_alpha_ic_analysis.py --train-end 2024-06-30 --data-source tej`，覆寫 `effective_alphas.json`
- 重跑 Phase A WP9 5-strategy A/B on TEJ：`pipelines/ab_experiment.py --data-source tej`
- 把 PhaseA baseline / PhaseB model_pool diagnostic 的舊 reports 標 deprecated（檔名加 `_yfinance_corrupt` 或 README 警告）
- CLAUDE.md §十一 已知限制加一條：「yfinance 8476 split-adjustment 錯誤 — 使用 yfinance 路徑時須過濾 stock 8476，正式研究改用 TEJ」

## 數據

* yfinance baseline run dir: `reports/simulations/sim_20220601_20241231_top10_sched20_survivorship_csv/`
* TEJ sim run dir: `reports/simulations/sim_20220601_20241231_top10_sched20_survivorship_tej/`
* Date: 2026-05-03
* 條件：top_k=10、`scheduled_20`、`train_window_days=500`（rolling）、`effective_alphas.json` IS-selected 52 alphas、commission 0.000926/side、tax 0.003 sell-side、slippage 5 bps/side
* 注意：兩條 sim 用同一份 `effective_alphas.json`（yfinance-derived）；換成 TEJ-derived 後 TEJ 結果可能略有改變但不會翻轉到正報酬，因為模型在乾淨資料上 holdout rank_ic 跨重訓平均接近 0
