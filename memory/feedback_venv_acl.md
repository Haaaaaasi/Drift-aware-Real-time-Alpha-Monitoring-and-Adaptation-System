---
name: 本機 .venv / Codex 沙盒 ACL 注意事項
description: 專案原有虛擬環境存在且可用，但 Codex 沙盒內直接啟動會因 base Python 目錄 ACL 被擋；不要改用 Anaconda
type: feedback
originSessionId: codex-2026-05-04
---

專案原有虛擬環境位於：

`C:\Users\chenp\Desktop\project\Drift-aware-Real-time-Alpha-Monitoring-and-Adaptation-System\.venv`

`pyvenv.cfg` 顯示它由 Python 3.11.9 建立，base interpreter 是：

`C:\Users\chenp\AppData\Local\Programs\Python\Python311\python.exe`

## 發現

`.venv` 本身存在、套件也在；使用非沙盒權限執行 `.venv\Scripts\python.exe` 可正常啟動，並可 import `structlog`、`pandas`、`xgboost` 等核心依賴。

但在 Codex 預設沙盒內直接執行 `.venv\Scripts\python.exe` 會失敗，訊息類似：

```text
Unable to create process using "C:\Users\chenp\AppData\Local\Programs\Python\Python311\python.exe" ...
```

直接執行或列出 base Python 目錄時會出現 Windows ACL 權限問題：

```text
存取被拒
```

這代表問題是 Codex 沙盒身分對 base Python 路徑沒有足夠權限，不是 `.venv` 或套件損壞。

## 使用方式

後續驗證、CLI、pytest 應使用專案原有 venv：

```powershell
& '.\.venv\Scripts\python.exe' -m pytest ...
& '.\.venv\Scripts\python.exe' -m pipelines.predict_next_day --help
```

若沙盒內啟動失敗，應以非沙盒權限重跑該 `.venv\Scripts\python.exe` 指令；不要改用本機 Anaconda `C:\Users\chenp\anaconda3\python.exe`，因為那不是專案環境，缺少部分依賴且版本是 Python 3.9。

## 已驗證

2026-05-04 使用原 `.venv` 成功驗證：

* Python 版本：3.11.9
* 核心依賴：`structlog`、`pandas`、`xgboost`
* `py_compile`：目前修改過的 pipeline / config / alpha cache 檔案通過
* CLI help：`predict_next_day`、`simulate_recent`、`ab_experiment`、`daily_batch_pipeline`、`adaptation_pipeline` 均可載入
