# S7 pipeline watchdog — relaunches dead chains after machine sleep.
# Registered as a scheduled task (every 20 min) so it survives the login-
# session death that sleep causes. Idempotent: only acts when a python
# training process is absent AND a chain still has work to do.
$env:PYTHONPATH = "E:\DATA\vscode\FluxPhased"
$base = "E:\DATA\vscode\FluxPhased\experiments\array_face_s7\learning_repair"
$log = "$base\s7_watchdog.log"
$py = (Get-Process python -ErrorAction SilentlyContinue)

function Log([string]$m) {
  Add-Content $log "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] $m"
}

if ($py) { exit 0 }  # training is alive; nothing to do

$mainDone = Select-String -Path "$base\s7_chain.log" -Pattern "ALL SEEDS DONE" -Quiet -ErrorAction SilentlyContinue
$postDone = Select-String -Path "$base\s7_ablation.log" -Pattern "ALL POST-CHAIN DONE" -Quiet -ErrorAction SilentlyContinue

if ($postDone) { exit 0 }  # everything finished

if (-not $mainDone) {
  # main 3-seed chain still has work
  $running = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'File.*_run_s7_seeds.ps1' }
  if (-not $running) {
    Log "main chain dead - relaunching"
    Start-Process powershell -ArgumentList '-ExecutionPolicy','Bypass','-File','E:\DATA\vscode\FluxPhased\_run_s7_seeds.ps1' -WindowStyle Hidden
  }
} else {
  # main done; post-chain (ablation + seeds 02/03 -> 3000) should be running
  $running = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'File.*_run_s7_ablation.ps1' }
  if (-not $running) {
    Log "post-chain dead - relaunching"
    Start-Process powershell -ArgumentList '-ExecutionPolicy','Bypass','-File','E:\DATA\vscode\FluxPhased\_run_s7_ablation.ps1' -WindowStyle Hidden
  }
}
