---
name: Python WQ101 Migration (DolphinDB OOM fix)
description: 以純 Python pandas 取代 DolphinDB alpha engine，解決 OOM 問題；記錄實作狀態、已知 bug 修復與待完成驗證步驟
type: project
originSessionId: e563bab8-0bcd-4063-9145-e19824701a70
---
## 背景

DolphinDB 容器啟動 OOM：`alpha_features` 表（53.8M rows，101 alphas × 556 stocks × 2022-2026）的 TSDB chunk metadata 超過 4GB 上限，主機僅剩 1GB 可用記憶體。

**目標**：以純 Python（pandas 向量化）port WQ101 alpha engine，輸出 schema 與 DolphinDB 版本完全相容，透過 parquet 快取支援增量更新。

---

## 實作狀態（截至 2026-04-26）

### 已完成

| 項目 | 說明 |
|------|------|
| `src/alpha_engine/wq101_python.py` | 完整 WQ101 Python 實作；101 個 alpha 函式 + 算子庫 + `compute_wq101_alphas()` 入口 |
| `src/alpha_engine/alpha_cache.py` | parquet 快取（snappy）；`compute_with_cache()` 支援冷啟動 + 增量更新 |
| `pipelines/daily_batch_pipeline.py` | `compute_python_alphas()` 改為薄 wrapper，呼叫 `compute_wq101_alphas` / `compute_with_cache` |
| `pipelines/predict_next_day.py` | 新增 `--alpha-source python_wq101`（預設），移除舊 15-approx 路徑 |
| `pipelines/replay_pipeline.py` | `use_cache=False` 避免全局快取污染測試 |
| 19 個新單元測試 | `tests/unit/test_wq101_python.py`（14 tests）+ `tests/unit/test_alpha_cache.py`（5 tests），全部通過 |
| Bug fix — float window | 所有算子加 `_w(d) = max(1, int(round(d)))` helper，修復 17 個 float window 錯誤（wq064/066/068/069/070/073/081/084/085/087/091/093/095/096/097/098/099） |
| Bug fix — wq063 missing `low` | 函式簽名補 `low` 參數，dispatch table 補 `"low"` 面板 |
| 修復後驗證 | 101/101 alphas 成功（合成資料測試） |

### 進行中

- **全量快取重建**（背景任務 `bxcvy2wuy`，2026-04-26 啟動）：`force_recompute=True`，101 alphas × 1,083 stocks × 1,036 dates；前次（修復前）耗時 5.5 分鐘，83 alphas 成功

### 尚未完成

- 快取重建完成後：執行 end-to-end 預測驗證
  ```bash
  python -m pipelines.predict_next_day --csv data/tw_stocks_ohlcv.csv --as-of 2026-04-25 --top-k 20 --alpha-source python_wq101
  ```
  預期輸出：`reports/predictions/predict_20260425.csv`，含 ≤20 列 top-k 標的

---

## 關鍵設計決策

- **快取預設全 101 alphas**：subset 在讀取端做（`alpha_ids` 參數），未來切換 effective set 不需重算
- **增量更新**：新日期 + 252 日 lookback buffer（給 ts rolling windows 暖機）
- **快取 date-range 過濾**：`compute_with_cache` 回傳的結果嚴格限制在 `[bar_min_date, bar_max_date]`，避免測試間快取污染
- **indclass proxy**：`(hash(security_id) % 10) + 1`，與 DolphinDB 版本平價

**Why:** DolphinDB OOM 短期無法擴容；Python pandas 版本在 1,083 stocks × 4 年全量計算耗時約 5–10 分鐘，遠低於等待 DolphinDB 修復的成本。

**How to apply:** 後續所有 alpha 相關 pipeline（predict_next_day, daily_batch, replay）均應優先使用 `python_wq101` 路徑，只有在需要驗證 DolphinDB 一致性時才切換 `dolphindb` 模式。
