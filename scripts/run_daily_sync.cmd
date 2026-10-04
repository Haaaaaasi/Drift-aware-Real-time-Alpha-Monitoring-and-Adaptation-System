@echo off
rem 每日自動維護：證交所同步 + live run，再增量更新期交所台指期夜盤。由 Windows 工作排程器呼叫（見 register_daily_task.ps1）。
cd /d "%~dp0.."
if not exist "reports\live" mkdir "reports\live"
echo ===== %date% %time% start ===== >> "reports\live\daily_runner.log"
".venv\Scripts\python.exe" -m pipelines.live_daily_runner --sync-twse >> "reports\live\daily_runner.log" 2>&1
echo ===== %date% %time% exit %errorlevel% ===== >> "reports\live\daily_runner.log"
".venv\Scripts\python.exe" scripts\fetch_taifex_night.py >> "reports\live\daily_runner.log" 2>&1
echo ===== %date% %time% taifex night exit %errorlevel% ===== >> "reports\live\daily_runner.log"
