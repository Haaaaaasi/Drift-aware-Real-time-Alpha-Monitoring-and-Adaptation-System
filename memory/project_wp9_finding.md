---
name: WP9 Phase-A 重跑後結論翻轉（Adaptation 勝過 no-adapt 且穩健）
description: DARAMS WP9 在 Phase A 修正（trigger/shadow 窗口拆分 + 完整成本拆三細項）後重跑 5 策略 A/B，結論從「no-adapt 勝出」翻轉為「all adaptation 策略勝過 no-adapt 且 cost-robust」
type: project
originSessionId: 5afd7a99-9c0c-4a38-9508-8258a96fd5e3
---
Phase A 在 2026-04-27 重跑 WP9（period 2022-06 → 2024-12，10 TW stocks，IS-selected 52 alphas）後，先前「no-adapt 勝出」結論不再成立。**這個翻轉是論文 RQ3 的關鍵發現**。

**Phase A 修正內容**：
1. P0★ #2：trigger window 從 `[t-20, t]` 拆成 `[t-60, t-20]`（calendar days），與 shadow eval `[t-30, t-10]` 完全不重疊；新增 `shadow_warmup_days=5` 讓 shadow 候選用 stricter cutoff 訓練（避免 IS leakage）。
2. P0★ #4：成本拆三細項 — commission 0.0926%/side、tax 0.3%/sell-side、slippage 5 bps/side。

**Baseline 結果（含 baseline 成本，~46 bps/day total）**：
| Strategy | Sharpe | n_retrains | Cum Ret % |
|---|---|---|---|
| scheduled_20 | **3.757** | 32 | 15754 |
| triggered | 3.630 | 12 | 15269 |
| model_pool | 3.518 | 11 | 12925 |
| scheduled_60 | 3.514 | 11 | 10938 |
| none | 3.158 | 1 | 7625 |

**Cost sensitivity sweep（4 場景 0/0.2/0.4/0.6% round-trip）**：5 策略排名在所有成本場景下 **rank_std=0**（100% 穩定）：
1. scheduled_20 → 2. triggered → 3. model_pool → 4. scheduled_60 → 5. **none always last**。

**Paired t-test vs none（one-sided）**：4 個 adaptation 策略 mean_daily_excess_ret 都正向（0.0005~0.0011），但 p_one_sided 都未達 5% 顯著（最接近的 triggered p=0.096）。所以 **direction 是對的，但 power 不足**。

**Why：** 原 WP9 的 no-adapt 勝出是兩個 bias 的複合產物：(a) trigger 用近期樣本判斷退化又馬上拿新模型在同段近期樣本做 shadow，等於 selection bias；(b) 沒扣交易成本時 high-turnover 策略表面看起來更糟。Phase A 拆窗口後 selection bias 消除，加成本後 turnover 弱項被定價，但所有 adaptation 策略仍勝出，且排序對成本不敏感。

**How to apply：**
- 論文裡寫 RQ3 結論時，**用 Phase A 重跑數據**，不要再引用 4-strategy 舊版的負面結論。
- 「no-adapt 勝出」要明確標為「修正前的 selection-bias-affected 結果」並對比修正後排序。
- 統計顯著性不足要誠實寫，並提數據量（631 trade days）為主要限制。
- Output artifacts：
  - baseline: `reports/adaptation_ab/ab_20220601_20241231_top10_phaseA_baseline/`（comparison.csv、experiment_summary.md）
  - sweep: `reports/adaptation_ab/ab_20220601_20241231_top10_phaseA_sweep/`（cost_sensitivity.csv、cost_sensitivity.png）
  - eval: `reports/adaptation_evaluation/`（regime_stratified.csv、paired_ttest.csv、cost_sensitivity_evaluation.md、fig1-4*.png）
- 舊 4-strategy run（`reports/adaptation_ab/ab_20220601_20241231_top10_v3/`）保留作為「pre-fix baseline」對照，不要刪。
