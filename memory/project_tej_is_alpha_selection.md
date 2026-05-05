---
name: TEJ IS-only alpha selection (2026-05-03)
description: TEJ 重做 IS-only alpha 篩選結果，覆寫 effective_alphas.json，並提供 survivorship-correct universe；下游 WP9 需在此基礎上重跑
type: project
originSessionId: 971bce2c-ab3d-4150-9cca-f6f6ed6dd3d1
---
`reports/alpha_ic_analysis/effective_alphas.json` 在 2026-05-03 已被覆寫成 TEJ 版（survivorship-correct）。

**Why**：舊版本是 yfinance + 200 top-by-row-count 跑出來的，universe 自動排除任何 2024-06 之前下市的股票，等於重新引入 survivorship bias。yfinance 8476 artifact 事件後，所有以 yfinance 為基礎的研究結論都要在 TEJ 上重做。

**How to apply**：
- 任何下游 pipeline（`predict_next_day` / `simulate_recent` / `ab_experiment`）讀 `effective_alphas.json` 時拿到的就是 TEJ 版的 64 個 alpha，不需改 code。
- WP9 Phase-A 的結論用的是舊 effective_alphas（52 alphas），翻轉證據的「重做」清單裡此項已完成 → WP9 在 TEJ + 新 effective_alphas 重跑才會是正式結論。
- 想對照舊版（yfinance 52 alphas）：`reports/alpha_ic_analysis/effective_alphas_yfinance_backup_20260503.json`。
- 重跑命令：`python scripts/run_is_oos_validation.py --data-source tej --train-end 2024-06-30 --max-stocks 200 --min-is-days 252 --seed 42`（≈43s）。

**結果摘要**（200 隨機 stocks，含 8 檔 TEJ 期間下市；IS 2018-01→2024-06，6.5 年；OOS 2024-07→2026-04，1.8 年）：
- 64/101 alpha 通過 |rank_ic|≥0.01 + coverage≥0.80
- 7 個 IS→OOS sign-flip：wq024, wq098, wq079, wq012, wq052, wq023, wq046（下游 ML 模型若用這些要警覺）
- 41 alpha 與舊 yfinance 跑法重疊（穩定）；23 alpha 是 TEJ 新增（IS 變長後達標）；11 alpha 被剔除（短窗下表現是 2022-2024 noise）
- 全選的 mean OOS rank_ic +0.0087（舊版 +0.0184，舊版偏高的部分原因是短 IS overfit 到 high-IC 子集）
- decay = OOS rank_ic − IS rank_ic 平均 −0.0017（小，表示 IS 篩選對 OOS 一般化還可以）

**Universe 細節**：1083 stocks 有 IS bars（≤2024-06-30）→ 1057 通過 ≥252 IS 天門檻 → seed=42 隨機抽 200。`--max-stocks 0` 可改抽全部 1057（沒做，因為 200 random 已夠統計力且 seed 固定）。

**Runner 改寫亮點**（`scripts/run_is_oos_validation.py`）：
- 加 `--data-source {csv, tej}` 與 `--min-is-days` flag
- universe 從「全期 row count top-N」改成「IS 期間 ≥min_is_days 的合格池 + 隨機抽樣」，含期間下市股
- alpha cache 用 pyarrow predicate pushdown 逐 alpha 讀，每 alpha < 0.1s（cache 已按 alpha_id 排序，row group 統計可裁剪）
