# S7 three-seed replication chain (seeds 20260801, 20260802, 20260803), sequential.
#
# Completion criterion: max "iteration" in train_metrics.jsonl (robust to
# appended sessions; the resume-counter fix keeps iterations monotonic).
$env:PYTHONPATH = "E:\DATA\vscode\FluxPhased"
$env:PYTHONUNBUFFERED = "1"
$pyexe = "C:\Users\zhang\.conda\envs\fluxphased\python.exe"
Set-Location "E:\DATA\vscode\FluxPhased"
$base = "E:\DATA\vscode\FluxPhased\experiments\array_face_s7\learning_repair"
$chainlog = "$base\s7_chain.log"
$mutex = New-Object System.Threading.Mutex($false, "Global\FluxPhasedS7SeedChain")
if (-not $mutex.WaitOne(0)) {
  Add-Content $chainlog "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] DUPLICATE CHAIN EXIT"
  exit 0
}
try {

function Get-MaxIter([string]$path) {
  if (-not (Test-Path $path)) { return -1 }
  $last = Get-Content $path -Tail 1
  if ($last -match '"iteration":\s*(\d+)') { return [int]$Matches[1] }
  return -1
}

foreach ($seed in @(20260801, 20260802, 20260803)) {
  $out = "$base\s7_selfplay_output_seed${seed}"
  if (-not (Test-Path $out)) { New-Item -ItemType Directory -Path $out | Out-Null }
  for ($r = 1; $r -le 40; $r++) {
    $have = Get-MaxIter "$out\train_metrics.jsonl"
    if ($have -ge 999) { break }
    Add-Content $chainlog "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] RETRY $r seed $seed (max iter $have/999)"
    & $pyexe -u experiments\array_face_s7\learning_repair\run_s7_selfplay.py --seed $seed --resume --val-every 50 >> "$out\run.log" 2>&1
    Start-Sleep -Seconds 30
  }
  Add-Content $chainlog "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] DONE seed $seed"
}
Add-Content $chainlog "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] ALL SEEDS DONE"
} finally {
  $mutex.ReleaseMutex() | Out-Null
  $mutex.Dispose()
}
