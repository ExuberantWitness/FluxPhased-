# FluxMasterWatchdog — self-healing for the TAES master chain.
#
# Runs as a Scheduled Task every 5 minutes. Logic:
#   1. If ALL MASTER RUNS DONE is in master_chain.log -> disable myself (done).
#   2. If a python training process is alive -> healthy, exit.
#   3. Otherwise relaunch the master chain detached (it is idempotent:
#      artifact-based completion + max-iteration resume).
$base = "E:\DATA\vscode\FluxPhased\experiments\array_face_s7\learning_repair"
$repo = "E:\DATA\vscode\FluxPhased"
$wdlog = "$base\watchdog.log"

function Log([string]$m) {
  Add-Content $wdlog "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] $m"
}

# 1. done?
if ((Test-Path "$base\master_chain.log") -and
    (Select-String -Path "$base\master_chain.log" -Pattern "ALL MASTER RUNS DONE" -Quiet)) {
  Log "chain complete; disabling watchdog task"
  schtasks /Change /TN "FluxMasterWatchdog" /DISABLE | Out-Null
  exit 0
}

# 2. healthy? A python process alone is NOT health: a hung learner stays
# alive forever while writing nothing. Require fresh output: the newest
# train_metrics.jsonl, val_metrics.jsonl, or final_eval_run.log under the
# experiment trees must have been modified within the last 25 minutes.
# (Legit quiet gaps: a final eval writes its run log ~every 20 min and can
# run ~95 min for n=4; the driver's built-in validation appends
# val_metrics.jsonl per view. 25 min covers the longest legit gap.)
$py = Get-Process python -ErrorAction SilentlyContinue
if ($py) {
  $newest = Get-ChildItem "$repo\experiments" -Recurse -Include 'train_metrics.jsonl','val_metrics.jsonl','final_eval_run.log' -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($newest -and ((Get-Date) - $newest.LastWriteTime).TotalMinutes -lt 25) { exit 0 }
  Log ("python alive but outputs stale ({0}); killing hung learner" -f $newest.LastWriteTime)
  $py | Stop-Process -Force
  Start-Sleep -Seconds 5
}

# 3. relaunch
Log "no python running and chain incomplete -> relaunching master chain"
Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',"$repo\_run_master_chain.ps1" `
  -WindowStyle Hidden `
  -RedirectStandardOutput "$repo\_master_chain_stdout.log" `
  -RedirectStandardError "$repo\_master_chain_stderr.log"
Log "relaunch issued"
