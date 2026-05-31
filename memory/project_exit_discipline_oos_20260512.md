---
name: Exit discipline OOS 實驗（2026-05-12）
description: 記錄 holdings concentration 後，針對 hard exit 與 tail cleanup 的 OOS 小矩陣、next_open confirmation 與 bootstrap 結論。
type: project
originSessionId: codex-2026-05-12
---

# Exit discipline OOS 實驗（2026-05-12）

## 背景

holdings concentration 診斷顯示，`turnover_aware_topk` 不是嚴格 top-10 portfolio，而是因 turnover cap 與 holding buffer 形成的 slow-rotation 多檔持股流程。`scheduled_20` 平均持股約 61 檔、effective holdings 約 35.7，且 negative-score weight 約 31.4%。因此補做 exit discipline 實驗，檢查「持股滿天數後，負訊號直接砍掉」是否真的改善。

## 實作

- `pipelines/simulate_recent.py` 新增 `--hard-exit-score-threshold`、`--hard-exit-min-holding-days`、`--tail-cleanup-weight`、`--renormalize-after-exit-cleanup`。
- `pipelines/ab_experiment.py` 接上同一組參數，並在 `comparison.csv` 輸出 `avg_hard_exit_count`、`avg_tail_exit_count`、`avg_exit_cleanup_weight`、`avg_negative_score_weight_after` 等診斷欄位。
- `scripts/run_exit_discipline_oos.py` 固定使用上一輪最佳低換手參數 `entry=20, exit=60, max_turnover=0.25, min_holding_days=10`，比較 `baseline`、`hard0`、`tail25`、`tail50`。
- 新增 `tests/unit/test_exit_discipline.py`，涵蓋 mature negative-score hard exit、tail cleanup 與 renormalization。

## OOS 結果

期間：2024-07-01 至 2026-04-30。資料：TEJ、exclude indclass/cap alpha、`scheduled_20` vs `none` vs `ew_buy_hold_universe`。

next_vwap：
- `tail25`：scheduled_20 cum +18.680%、Sharpe 0.521、max DD -41.907%、vs benchmark +14.502% cum / +0.301 Sharpe。
- `baseline`：scheduled_20 cum +17.704%、Sharpe 0.501、max DD -42.056%、vs benchmark +13.526% cum / +0.281 Sharpe。
- `tail50`：scheduled_20 cum +14.484%、Sharpe 0.440，門檻太重，低於 baseline。
- `hard0`：scheduled_20 cum -9.913%、Sharpe -0.390。直接砍 `signal_score <= 0` 的成熟持股會破壞策略，不應採用。

next_open confirmation：
- `tail25`：scheduled_20 cum +32.532%、Sharpe 0.715、max DD -36.374%、vs benchmark +21.914% cum / +0.312 Sharpe。
- `baseline`：scheduled_20 cum +31.478%、Sharpe 0.697、max DD -36.520%、vs benchmark +20.860% cum / +0.294 Sharpe。

Bootstrap / paired test：
- `tail25` next_vwap vs benchmark mean daily excess +3.48 bps，95% block bootstrap CI [-5.20, +11.62] bps，p_boot 0.203，paired t one-sided p 0.153。
- `tail25` next_open vs benchmark mean daily excess +4.95 bps，CI [-3.31, +12.78] bps，p_boot 0.109，paired t one-sided p 0.148。

## 結論

- 不應採用「持股滿天數後，`signal_score <= 0` 就直接砍掉」作為 hard exit。它降低 negative-score weight，但同步打掉後續會反彈或仍有組合價值的持股，且增加 turnover，OOS 大幅惡化。
- 可採用溫和 tail cleanup：持股滿 10 天後，權重低於 25 bps 的殘餘小倉位清掉。這在 next_vwap 與 next_open 都小幅優於 baseline，但改善幅度不大，統計檢定仍未達 5%。
- reviewer-facing claim 應寫成：tail cleanup 是 portfolio hygiene / operational cleanup，方向一致但 evidence 不足以宣稱顯著 alpha improvement。

## 輸出

- `reports/adaptation_ab/exit_discipline_oos_20260512/workflow_summary.md`
- `reports/adaptation_ab/exit_discipline_oos_20260512/next_vwap_ranked.csv`
- `reports/adaptation_ab/exit_discipline_oos_20260512/next_open_confirm_ranked.csv`
- `reports/adaptation_ab/exit_discipline_oos_20260512/bootstrap_paired_results.csv`
