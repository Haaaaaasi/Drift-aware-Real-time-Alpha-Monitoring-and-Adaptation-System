---
name: 只跑相關測試，不跑全套
description: 驗證程式碼修改時只跑受影響的測試，不無謂跑全套 pytest
type: feedback
originSessionId: c04df2b7-2a0b-41db-b3a9-535341097499
---
只跑與修改直接相關的測試，不跑全套 pytest。

**Why:** 全套測試（136 tests）需要 20+ 分鐘、消耗大量記憶體（WQ101 alpha 計算、integration pipelines），對使用者造成困擾。

**How to apply:** 修改某模組後，只跑對應的 unit/integration test 檔案。全套留給正式 merge 前才跑一次，且要先告知使用者。例如今天的 label maturity 修改只需要：
`pytest tests/unit/test_label_maturity.py tests/unit/test_rolling_ic_window.py tests/unit/test_simulate_recent_cost.py -q`
