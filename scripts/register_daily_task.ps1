# 註冊 Windows 工作排程：平日 17:10 執行 scripts\run_daily_sync.cmd（證交所同步 + live run + 期交所台指期夜盤）。
# 錯過時段（電腦關機）會在下次可用時補跑；同步本身會補抓所有缺的交易日，所以漏跑幾天也能自我修復。
# 移除：Unregister-ScheduledTask -TaskName "DARAMS Daily Sync" -Confirm:$false
param(
    [string]$TaskName = "DARAMS Daily Sync",
    [string]$Time = "17:10"
)

$repo = Split-Path -Parent $PSScriptRoot
$cmd = Join-Path $repo "scripts\run_daily_sync.cmd"
if (-not (Test-Path $cmd)) { throw "not found: $cmd" }

$action = New-ScheduledTaskAction -Execute $cmd -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $Time
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 3) -RunOnlyIfNetworkAvailable
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName, State
(Get-ScheduledTaskInfo -TaskName $TaskName).NextRunTime
