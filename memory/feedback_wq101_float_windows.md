---
name: WQ101 float window parameters
description: WQ101 論文中的算子 window 參數是浮點數，Python pandas rolling 需要整數；修復方式是加 _w() helper
type: feedback
originSessionId: e563bab8-0bcd-4063-9145-e19824701a70
---
WQ101 論文（Kakushadze 2016）的公式中，許多算子的 window 參數是浮點數（如 `_delta(vwap, 4.72775)`、`_ts_rank(..., 16.7411)`）。Python pandas 的 `rolling(d)` 與 `shift(d)` 要求整數，直接傳 float 會報 `slice indices must be integers` 或 `window must be an integer 0 or greater`。

**修復方式**：在 `src/alpha_engine/wq101_python.py` 的算子庫頂部加一個 `_w()` helper，所有算子呼叫前統一轉型：

```python
def _w(d) -> int:
    return max(1, int(round(d)))
```

每個算子（`_delay`, `_delta`, `_ts_sum`, `_ts_mean`, `_ts_min`, `_ts_max`, `_ts_stddev`, `_ts_rank`, `_ts_argmax`, `_ts_argmin`, `_ts_product`, `_correlation`, `_covariance`, `_decay_linear`）的 window 參數都用 `_w(d)` 包起來。

**Why:** 論文公式直接 port 時會有 18 個 alpha 失敗（wq064/066/068/069/070/073/081/084/085/087/091/093/095/096/097/098/099 + wq063），修復後 101/101 全部通過。

**How to apply:** 之後若新增 alpha 函式或算子，window 參數一律用 `_w(d)` 轉型，不要假設呼叫端會傳整數。
