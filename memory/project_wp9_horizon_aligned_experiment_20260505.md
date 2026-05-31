name: WP9 5 日 horizon 對齊實驗
description: scheduled_20 + 10 日換倉在 TEJ 正式成本下首次略勝 buy-and-hold
date: 2026-05-05

# WP9 5 日 Horizon 對齊實驗

## 背景

WP9 signal diagnostics 顯示 XGBoost daily top-k 有 gross edge，但 turnover 約 0.87，成本後失效。本輪改測「5 日 forward-return 目標」與「5/10 日持有週期」對齊，並用 TEJ survivorship-correct universe、TEJ effective 64 alphas、真實台股成本評估。

## 已執行輸出

- 設計與結果摘要：`reports/adaptation_ab/wp9_horizon_aligned_experiment_plan_20260505.md`
- `horizon5_reb5_none_baseline`：`reports/adaptation_ab/ab_20220601_20241231_top10_horizon5_reb5_none_baseline/`
- `horizon5_reb5_none_tw500`：`reports/adaptation_ab/ab_20220601_20241231_top10_horizon5_reb5_none_tw500/`
- `horizon5_reb5_adapt_tw500`：`reports/adaptation_ab/ab_20220601_20241231_top10_horizon5_reb5_adapt_tw500/`
- `horizon5_reb10_sched20_tw500`：`reports/adaptation_ab/ab_20220601_20241231_top10_horizon5_reb10_sched20_tw500/`

## 主要結果

1. `none + rebalance_every=5` 失敗：
   - expanding train：cum -14.884%，gross 9.135 bps/day，cost 10.790 bps/day
   - rolling 500 train：cum -31.629%，gross 5.290 bps/day，cost 10.398 bps/day

2. `scheduled_20 + rebalance_every=5 + rolling500` 轉正但不夠：
   - cum +9.747%
   - gross 13.673 bps/day
   - cost 10.725 bps/day
   - net 2.948 bps/day
   - buy-and-hold benchmark +43.612%

3. `scheduled_20 + rebalance_every=10 + rolling500` 是目前第一個可防守候選：
   - cum +47.679%，略勝 buy-and-hold +43.612%
   - gross 13.209 bps/day
   - cost 5.633 bps/day
   - net 7.576 bps/day
   - turnover 0.099
   - Sharpe 0.721，仍低於 benchmark 1.054
   - max drawdown -22.972%，仍差於 benchmark -13.907%

4. 從已完成 daily PnL 重算 cost sensitivity：
   - 0.0% round-trip：+110.816%
   - 0.2% round-trip：+86.059%
   - 0.4% round-trip：+64.169%
   - 0.6% round-trip：+44.818%
   - 0.6% 下仍略高於 benchmark +43.612%，但 margin 很薄。

## 研究判讀

目前可以說：

> TEJ WQ101 + XGBoost 存在 gross signal；daily top-k 因 turnover 過高失效。當持股週期延長到 10 日換倉，並用 scheduled_20 維持模型新鮮度後，策略首次在真實成本下略勝 buy-and-hold，但風險調整後仍不如 benchmark。

不應說：

- alpha 完全太弱。
- turnover-aware residual 長尾版本已解決問題。
- adaptation 已明顯戰勝 benchmark。現在只是 cumulative return 略勝，Sharpe/MaxDD 仍落後。

## 下一步

- 優先測 `scheduled_20 + rebalance_every=10 + top_k=20/30`，看能否降低 max drawdown。
- 測 `rebalance_every=15/20`，看是否能維持 net edge 並進一步降成本。
- 若候選仍勝 benchmark，再跑正式五策略與正式 cost sweep。
## 2026-05-05 小矩陣補跑

輸出摘要：`reports/adaptation_ab/wp9_horizon_aligned_matrix_20260505.md`

本輪固定 TEJ、2022-06-01 至 2024-12-31、rolling 500 calendar days、正式 baseline cost、`equal_weight_topk`，補跑：

- `top_k=20 / rebalance_every=15 / scheduled_20`：cum +41.423%，Sharpe 0.747，max DD -22.106%，turnover 0.0649。
- `top_k=20 / rebalance_every=20 / scheduled_20`：cum +34.456%，Sharpe 0.632，max DD -22.739%，turnover 0.0510。
- `top_k=30 / rebalance_every=15 / scheduled_20`：cum +27.660%，Sharpe 0.593，max DD -21.175%，turnover 0.0642。
- `top_k=30 / rebalance_every=20 / scheduled_20`：cum +32.671%，Sharpe 0.645，max DD -21.090%，turnover 0.0504。
- `top_k=10 / rebalance_every=10 / model_pool`：cum -1.248%，Sharpe 0.109，max DD -34.157%，turnover 0.0968，`n_pool_reuses=0`、`n_pool_misses=23`。

判讀：

1. `scheduled_20 + top_k=10 + rebalance_every=10` 仍是唯一累積報酬勝過 `ew_buy_hold_universe` 的正式 TEJ 成本候選。
2. top20/top30 沒有改善，反而稀釋 gross edge；目前訊號主要集中在前 10 名。
3. rebalance 15/20 雖降低 turnover，但 gross return 下降更多，因此 net edge 未提升。
4. `model_pool` 本輪不能解讀為 recurring concept reuse 失敗，因 PostgreSQL registry 連線失敗，pool reuse 實際為 0；需先補 offline filesystem 或 in-memory pool backend 才能正式評估 WP11。

## 2026-05-06 正式五策略補跑（Docker / PostgreSQL 已開）

輸出摘要：`reports/adaptation_ab/wp9_horizon_aligned_formal_5strategy_20260506.md`

正式輸出：`reports/adaptation_ab/ab_20220601_20241231_top10_horizon5_reb10_formal_5strategy_tw500/`

本輪固定 TEJ、2022-06-01 至 2024-12-31、TEJ IS-only 64 alphas、`top_k=10`、`rebalance_every=10`、rolling 500 calendar days、正式 baseline cost、`ew_buy_hold_universe` benchmark，補跑五策略：

- `none`：cum +4.694%，Sharpe 0.193，max DD -30.702%，turnover 0.1003。
- `scheduled_20`：cum +47.679%，Sharpe 0.721，max DD -22.972%，turnover 0.0990。
- `scheduled_60`：cum -15.983%，Sharpe -0.149，max DD -39.759%，turnover 0.0990。
- `triggered`：cum -4.666%，Sharpe 0.063，max DD -27.400%，turnover 0.0965。
- `model_pool`：cum -25.525%，Sharpe -0.354，max DD -35.974%，turnover 0.1013，`pool_backend=postgres`，`n_pool_reuses=10`，`n_pool_misses=9`。
- `ew_buy_hold_universe`：cum +43.612%，Sharpe 1.054，max DD -13.907%，turnover 0.0016。

判讀：

1. `scheduled_20 + top_k=10 + rebalance_every=10 + rolling500` 仍是目前唯一在正式成本下同時勝過 `none` 與 buy-and-hold 累積報酬的候選。
2. horizon 對齊不是讓所有 adaptation 變好；`scheduled_60`、`triggered`、`model_pool` 都輸給 `none`。
3. `scheduled_20` 的累積報酬略勝 benchmark，但 Sharpe 與 max drawdown 仍落後，不可宣稱風險調整後勝過市場基準。
4. `model_pool` 本輪已連上 PostgreSQL 並實際發生 reuse，因此可排除「infrastructure fallback」作為主因；但不能只靠 formal A/B 直接宣稱 reuse 選錯，需補 event attribution 與 candidate-level proxy。
5. 下一步應優先對 `scheduled_20` 做正式 cost sweep 與 paired test；`model_pool` 則需要額外診斷 pool-hit days 的 shadow IC 與後續實際 PnL，確認是否為 shadow window 過擬合或 similarity threshold 太鬆。

## 2026-05-07 Model Pool failure diagnosis

輸出摘要：`reports/adaptation_ab/model_pool_failure_diagnosis_20260507.md`

正式診斷輸出：`reports/adaptation_ab/model_pool_failure_diagnosis_20260507/`

Diagnostic model_pool run：`reports/adaptation_ab/model_pool_diagnostic_runs/sim_20220601_20241231_top10_pool_horizon5_reb10_model_pool_diag_tw500/`

本輪只重跑單一 `model_pool`，固定 TEJ、2022-06-01 至 2024-12-31、TEJ IS-only 64 alphas、`top_k=10`、`rebalance_every=10`、rolling 500 calendar days、baseline cost、`similarity_threshold=0.5`、`pool_top_k=3`、`pool_regime_window=60`、`shadow_window=20`，並開啟 `--model-pool-diagnostics`。

驗證：

- `pool_backend=postgres`，`n_pool_reuses=10`、`n_pool_misses=9`，不是 fallback。
- `model_pool_decisions.csv` 共 68 列，對應 23 個 trigger event；每個 trigger 都有一列 `selected=True`。
- 候選角色分布：`current=23`、`new=23`、`reused=22`。
- 實際選中角色分布：`new=11`、`reused=10`、`current=2`。

診斷結果：

- Observed event attribution：selected reused 事件後 10 日平均 net return -1.24%，相對 `scheduled_20` 平均 excess -0.25%，proxy rank 平均 2.60。
- Candidate-level proxy：reused candidate 的 shadow 指標最高（mean shadow IC 0.0528、shadow Sharpe 1.4955），但下一持股週期 proxy 最弱（mean proxy net return -0.65%、proxy rank 2.68）。
- `candidate_similarity < 0.6` 的 reused candidate 共 13 筆，平均 proxy net return -0.44%。

判讀：

Evidence suggests：model_pool 在這個設定下的弱點集中在 shadow selector / recurring concept reuse 的「shadow 指標看起來好，但下一持股週期 net proxy 不好」。這能支持「目前 similarity threshold=0.5 可能偏鬆、shadow objective 與實際持股 PnL 對齊不足」的研究敘述；但 candidate proxy 不是完整反事實回測，因此不可寫成「已明確證明 reuse 選錯」。

下一步若要改善 model_pool，先測 `similarity_threshold=0.6/0.7`，並把 shadow selector 評分改得更接近 `net_return_proxy` 或 post-holding proxy，再與 current/new/reused candidate-level proxy 一起報告。

## 2026-05-07 Model Pool selector 修正

輸出摘要：`reports/adaptation_ab/model_pool_selector_fix_20260507.md`

本輪修正：

- `ShadowEvaluator` 新增 shadow-window top-k proxy 指標：`topk_gross_return`、`topk_net_return`、`topk_turnover`、`topk_n_days`。
- `ModelPoolController` 新增 `selection_metric`，預設仍為 `ic` 以保留舊結果可重現；可改用 `topk_net_return`。
- `simulate_recent` / `ab_experiment` 新增 `--model-pool-selection-metric {ic,hit_rate,sharpe,topk_gross_return,topk_net_return}`。
- `model_pool_decisions.csv` 新增 `selection_metric`、`selection_score`、`shadow_topk_*` 欄位，後續可直接比對 shadow selector 分數與 post-holding proxy。

短窗 smoke：

- 指令設定：TEJ、2024-06-01 至 2024-12-31、`top_k=10`、`rebalance_every=10`、`train_window_days=500`、`model_pool_selection_metric=topk_net_return`、`similarity_threshold=0.6`、baseline cost、diagnostics on。
- 輸出：`reports/adaptation_ab/model_pool_fix_smoke_runs/sim_20240601_20241231_top10_pool_topknet_thresh06_smoke/`
- 結果：`pool_backend=postgres`、`n_pool_reuses=0`、`n_pool_misses=4`、cum -15.154%、gross -4.180 bps/day、cost 5.545 bps/day、net -9.724 bps/day。

判讀：

修正成功讓 selector objective 與持股層 proxy 對齊，且短窗可跑通；但 smoke 未轉正，且 threshold 0.6 在此短窗沒有 reuse。下一步需跑完整 2022-06 至 2024-12 對照矩陣：`ic/0.5` baseline、`topk_net_return/0.5`、`topk_net_return/0.6`、可選 `topk_net_return/0.7`，再比較 `n_pool_reuses` 與 selected reused 的 post-10d proxy。
## 2026-05-07 Model Pool selector 對照矩陣完成

報告：`reports/adaptation_ab/model_pool_selector_matrix_20260507.md`

輸出目錄：`reports/adaptation_ab/model_pool_selector_matrix_20260507/`

主矩陣設定：TEJ、2022-06-01 至 2024-12-31、64 TEJ IS-only alphas、`model_pool`、`top_k=10`、`rebalance_every=10`、`train_window_days=500`、baseline cost、`pool_top_k=3`、`shadow_window=20`、diagnostics on。PostgreSQL pool 有 `detected_at >= session_start` 隔離，因此後跑 cell 不會讀到前一個 cell 的 regime entries。

主矩陣結果：
- `ic / threshold=0.5`：cum -25.525%、Sharpe -0.354、gross 2.156 bps/day、net -3.459 bps/day、turnover 0.1013、reuses 10、misses 7。
- `topk_net_return / threshold=0.5`：cum +49.206%、Sharpe 0.770、gross 13.113 bps/day、net 7.563 bps/day、turnover 0.0983、reuses 5、misses 7。
- `topk_net_return / threshold=0.6`：cum +8.681%、Sharpe 0.259、gross 8.068 bps/day、net 2.574 bps/day、turnover 0.0989、reuses 4、misses 8。
- `topk_net_return / threshold=0.7`：cum +15.750%、Sharpe 0.356、gross 9.190 bps/day、net 3.649 bps/day、turnover 0.0975、reuses 2、misses 9。

勝出設定延伸到 TEJ 目前末日：
- `topk_net_return / threshold=0.5`，2022-06-01 至 2026-04-30：cum +121.333%、annualized +23.488%、Sharpe 0.935、max DD -30.702%、gross 15.254 bps/day、net 9.751 bps/day、turnover 0.0997、reuses 5、misses 10。

研究解讀：evidence suggests 原 `model_pool` 失敗主因不是 PostgreSQL fallback，也不只是 similarity threshold 太寬，而是 shadow selector 用 IC 選模型時沒有對齊實際 top-k long-only portfolio 的 net return。改成 `topk_net_return` 後，同期間同成本下 `model_pool` 由 -25.525% 翻到 +49.206%。下一步應以 `topk_net_return / threshold=0.5` 作為候選正式 model_pool 設定，重跑正式五策略 A/B、benchmark comparison、cost sweep 與 paired test。

## 2026-05-08 current-code 正式五策略 A/B：topk selector matrix 被推翻

正式輸出：`reports/adaptation_ab/ab_20220601_20241231_top10_horizon5_reb10_formal_5strategy_tw500_topknet_t05/`

背景：2026-05-07 上午的 selector matrix 曾顯示 `model_pool(topk_net_return / threshold=0.5)` 可達 cum +49.206%。但該 matrix 完成後，`simulate_recent.py` 與 `model_pool_strategy.py` 在 2026-05-07 下午又有後續修改；2026-05-08 以 current code 重跑正式五策略 A/B 後，結果不再支持 matrix 結論。

固定設定：TEJ、2022-06-01 至 2024-12-31、TEJ IS-only 64 alphas、`top_k=10`、`rebalance_every=10`、rolling 500 calendar days、baseline cost、`ew_buy_hold_universe` benchmark、`model_pool_selection_metric=topk_net_return`、`similarity_threshold=0.5`、diagnostics on。

結果：

- `none`：cum +4.694%，Sharpe 0.193，max DD -30.702%，turnover 0.1003。
- `scheduled_20`：cum +47.679%，Sharpe 0.721，max DD -22.972%，turnover 0.0990。
- `scheduled_60`：cum -15.983%，Sharpe -0.149，max DD -39.759%，turnover 0.0990。
- `triggered`：cum -4.666%，Sharpe 0.063，max DD -27.400%，turnover 0.0965。
- `model_pool(topk_net_return / 0.5)`：cum -4.372%，Sharpe 0.043，max DD -30.058%，turnover 0.0962，`pool_backend=postgres`，`n_pool_reuses=3`，`n_pool_misses=9`。
- `ew_buy_hold_universe`：cum +43.612%，Sharpe 1.054，max DD -13.907%，turnover 0.0016。

關鍵比對：

- Formal A/B 的 `model_pool` config 與 stale matrix 的 `topknet_t05` config 逐項一致，但輸出不同。
- 差異來源較可能是 matrix run 使用的是 2026-05-07 上午的舊工作樹；formal A/B 使用 2026-05-07 下午後的 current code。檔案時間顯示 `simulate_recent.py` / `model_pool_strategy.py` 晚於 selector matrix run。
- 因此 `topk_net_return/0.5` matrix 的 +49.206% 不可再寫成正式績效，只能保留為 stale-code 診斷或開發過程記錄。

修正版研究結論：

1. `scheduled_20 + top_k=10 + rebalance_every=10 + rolling500` 仍是目前最可防守的 adaptation 設定；累積報酬略勝 benchmark，但 Sharpe / max drawdown 仍輸 benchmark。
2. `model_pool` 的 selector objective 改成 `topk_net_return` 後，在 current code 下只從原本 -25.525% 改善到 -4.372%，僅略優於 `triggered`，仍不能宣稱 recurring concept pool 已帶來正式優勢。
3. 下一步不應直接跑 cost sweep 來包裝 model_pool；應先把 model_pool 的可重現性、shadow window 樣本邊界與 decision path 狀態敏感性釐清。`scheduled_20` 的 cost sweep / paired test 仍可作為主策略防守實驗。

## 2026-05-08 current-code reproducibility check 與 scheduled_20 主線驗證

Model pool reproducibility check：

- 輸出：`reports/adaptation_ab/model_pool_repro_checks/sim_20220601_20241231_top10_pool_current_code_topknet_t05_repro_20260508/`
- 設定：TEJ、2022-06-01 至 2024-12-31、`top_k=10`、`rebalance_every=10`、`train_window_days=500`、`model_pool_selection_metric=topk_net_return`、`similarity_threshold=0.5`、diagnostics on。
- 結果完全重現 current-code formal A/B：cum -4.372%、Sharpe 0.043、max DD -30.058%、turnover 0.0962、selected role 分布 `current=13 / new=7 / reused=3`。
- 判讀：5/7 selector matrix 的 +49.206% 不是 current code 可重現結果；`reports/adaptation_ab/model_pool_selector_matrix_20260507.md` 已加 stale-code / deprecated banner。

Scheduled_20 cost sweep：

- 輸出：`reports/adaptation_ab/ab_20220601_20241231_top10_horizon5_reb10_sched20_cost_sweep_20260508/`
- 評估輸出：`reports/adaptation_evaluation/scheduled20_cost_sweep_20260508/`
- 策略：`none` vs `scheduled_20`，四段 round-trip cost 0 / 0.2 / 0.4 / 0.6。
- 因外層 cost-sweep 長跑在 `scheduled_20 cost=0.4` 前遇到 alpha cache 記憶體不足，完整 `cost_sensitivity.csv` 改由 `cost=0` daily PnL 的 `gross_return` 與 `turnover` 套用成本重算。這對 `none` / `scheduled_20` 等非 PnL-driven 策略等價，但不可套用到 `triggered` / `model_pool`。

| Cost % | none Cum Ret % | scheduled_20 Cum Ret % | scheduled_20 Sharpe |
|---:|---:|---:|---:|
| 0.0 | 46.931 | 110.816 | 1.263 |
| 0.2 | 29.446 | 86.059 | 1.073 |
| 0.4 | 14.014 | 64.169 | 0.883 |
| 0.6 | 0.397 | 44.818 | 0.692 |

Cost-sweep ranking stability：`scheduled_20` mean_rank 1.0、rank_std 0.0；`none` mean_rank 2.0、rank_std 0.0。

Paired test：

- 輸出：`reports/adaptation_evaluation/formal_5strategy_topknet_t05_20260508/paired_ttest.csv`
- `scheduled_20` vs `none`：mean daily excess +0.000600、t=1.065、p_two_sided=0.2872、p_one_sided=0.1436，方向正但未達 5% 顯著。

正式研究敘述更新：

1. 主線策略是 `scheduled_20 + top_k=10 + rebalance_every=10 + train_window_days=500`。
2. 可說：在 TEJ、正式成本與 0.6% round-trip 壓力測試下，`scheduled_20` 累積報酬仍優於 no-adapt，且成本排名穩定。
3. 不可說：`scheduled_20` 已在統計上顯著勝過 no-adapt，或風險調整後勝過 buy-and-hold；paired test 與 benchmark Sharpe 都不支持這種強敘述。

## 2026-05-09 Regime stress ??????

??????? experiment metadata ???? TEJ?IS-only 64 alphas?`top_k=10`?`rebalance_every=10`?`train_window_days=500`?baseline cost?????????????? 2022-H2 Bear?2022-07-01 ? 2022-12-31??2023 Recovery?2024 Consolidation????? `none`?`scheduled_20`?`scheduled_40`?`scheduled_60`???? `ew_buy_hold_universe` benchmark??????`reports/adaptation_ab/regime_stress_summary_20260509.md`?`reports/adaptation_ab/regime_stress_summary_20260509.csv`?`reports/adaptation_ab/regime_stress_paired_summary_20260509.csv`?

?????
- `simulate_recent.py` / `ab_experiment.py` ? `config.json` ?? `data_source`?`effective_alphas_path`?`effective_alphas_hash`?`pool_backend`?`git_sha`?`dirty_worktree`?
- `ab_experiment.py` ?? `scheduled_40` ???? regime stress ?? 20/40/60 ??????
- `notebooks/03_adaptation_evaluation.py` ?? A/B config ? benchmark path?? regime evaluation ? paired test ??? `ew_buy_hold_universe`?
- `experiment_summary.md` ????? shadow window ??????? shadow forward label ??? `label_available_at <= t` ????

???????

| Window | Best active | Cum Ret % | Sharpe | Benchmark Cum Ret % | Benchmark Sharpe | ???? |
|---|---|---:|---:|---:|---:|---|
| 2022-H2 Bear?strict? | none | -10.534 | -0.738 | 7.574 | 0.908 | scheduled strategies ?? none????? bear window ? adaptation ???? |
| 2023 Recovery | none | 5.535 | 0.415 | 33.046 | 3.038 | ? beta ? benchmark ?????scheduled adaptation ???? alpha |
| 2024 Consolidation | scheduled_40 | 39.329 | 1.377 | 11.659 | 0.796 | ??/????? scheduled adaptation ???? none ? benchmark |

??????? 2022-06-01 ? 2022-12-31 ? run ?? 6 ?????? WP4 ??? `2022-H2 Bear` ?????? regime stress ???? 2022-07-01 ?????

????????`scheduled_20` ?? 2022-06 ? 2024-12 ???? A/B ???????? regime stress ???????????????????????Final Report ???????? 20 ??????????bear market ? adaptation ????????????????????????? no-adapt ???????????? 2024 ??/?????2022-H2 ? 2023 ?? benchmark ? none ???

## 2026-05-10 Scheduled frequency full-period ??

?????? 2024 ??????? `scheduled_40` ????????????? full-period formal A/B ? scheduled frequency cost sweep?

???
- Formal A/B?`reports/adaptation_ab/ab_20220601_20241231_top10_formal_with_sched40_20260510/`
- Formal evaluation?`reports/adaptation_evaluation/formal_with_sched40_20260510/`
- Cost sweep?`reports/adaptation_ab/ab_20220601_20241231_top10_scheduled_frequency_cost_sweep_20260510/`
- Cost sweep evaluation?`reports/adaptation_evaluation/scheduled_frequency_cost_sweep_20260510/`
- ???`reports/adaptation_ab/scheduled_frequency_formal_summary_20260510.md`

Formal A/B?TEJ?2022-06-01 ? 2024-12-31?top_k=10?rebalance_every=10?train_window_days=500?baseline cost??

| Strategy | Cum Ret % | Sharpe | Max DD % | n_retrains |
|---|---:|---:|---:|---:|
| scheduled_20 | 47.679 | 0.721 | -22.972 | 32 |
| ew_buy_hold_universe | 43.612 | 1.054 | -13.907 | 0 |
| scheduled_40 | 11.222 | 0.293 | -29.219 | 16 |
| none | 4.694 | 0.193 | -30.702 | 1 |
| triggered | -4.666 | 0.063 | -27.400 | 24 |
| scheduled_60 | -15.983 | -0.149 | -39.759 | 11 |

Cost sweep?round-trip cost 0 / 0.2 / 0.4 / 0.6??

| Strategy | 0.0% | 0.2% | 0.4% | 0.6% | rank stability |
|---|---:|---:|---:|---:|---|
| scheduled_20 | 110.816 | 86.059 | 64.169 | 44.818 | rank 1 in all scenarios |
| scheduled_40 | 57.941 | 38.750 | 21.859 | 6.997 | rank 2 or 3 |
| none | 46.931 | 29.446 | 14.014 | 0.397 | rank 2 or 3 |
| scheduled_60 | 19.174 | 5.198 | -7.162 | -18.089 | rank 4 in all scenarios |

Pairwise tests?
- `scheduled_20` vs `none`?mean daily excess +6.005 bps?p_one_sided=0.1436?
- `scheduled_20` vs `scheduled_40`?mean daily excess +4.462 bps?p_one_sided=0.2034?
- `scheduled_20` vs `ew_buy_hold_universe`?mean daily excess +1.402 bps?p_one_sided=0.3836?

??????`scheduled_20` ???? full-period TEJ ?? A/B ??????????? active strategy?`scheduled_40` ? 2024 ?????????????? `scheduled_20`?Final Report ??????????????? no-adapt ????? 20 ??????????????? paired t-test ?? 5% ?????? benchmark ? Sharpe / drawdown ?????????????? alpha????????????????? adaptation evidence?
