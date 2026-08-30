# Greedy-stare counter-adaptation chain — queues behind the TAES chain.
#
# Reviewer-critical control: do jammers CO-TRAINED against the greedy
# mission-stare radar learn to punish it? (The self-play jammer teams could
# not: drop 0.0889 invariant.) One seed, 2000 iterations, jammer-only
# learning (radar side scripted). Validation rows log greedy_vs_jam_drop —
# the direct question metric — against the 0.0889 baseline.
$env:PYTHONPATH = "E:\DATA\vscode\FluxPhased"
$env:PYTHONUNBUFFERED = "1"
$pyexe = "C:/Users/zhang/.conda/envs/fluxphased/python.exe"
Set-Location "E:\DATA\vscode\FluxPhased"
$base = "E:\DATA\vscode\FluxPhased\experiments\array_face_s7\learning_repair"
$chainlog = "$base\greedycounter_chain.log"

function Log([string]$msg) {
  Add-Content $chainlog "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] $msg"
}
function Get-MaxIter([string]$path) {
  if (-not (Test-Path $path)) { return -1 }
  $last = Get-Content $path -Tail 1
  if ($last -match '"iteration":\s*(\d+)') { return [int]$Matches[1] }
  return -1
}

# wait for the main TAES chain (single GPU, serial runs)
while ($true) {
  if ((Test-Path "$base\taes_chain.log") -and
      (Select-String -Path "$base\taes_chain.log" -Pattern "ALL TAES RUNS DONE" -Quiet)) {
    break
  }
  Start-Sleep -Seconds 600
}
Log "TAES chain done; starting greedy-stare counter-adaptation"

$seed = 20260921
$out = "$base\s7_greedycounter_output_seed$seed"
if (-not (Test-Path $out)) { New-Item -ItemType Directory -Path $out | Out-Null }

for ($r = 1; $r -le 60; $r++) {
  $have = Get-MaxIter "$out\train_metrics.jsonl"
  if ($have -ge 1999) { break }
  if ($have -lt 999) {
    Log "GREEDYCOUNTER stage-A attempt $r (have=$have)"
    & $pyexe -u _run_s7_greedy_counter.py --seed $seed --resume `
      --iterations 1000 --val-every 50 *>> "$out\run.log"
  } else {
    Log "GREEDYCOUNTER stage-B attempt $r (have=$have)"
    & $pyexe -u _run_s7_greedy_counter.py --seed $seed --resume `
      --iterations 2000 --val-every 50 *>> "$out\run.log"
  }
}
Log "GREEDYCOUNTER training complete at iter $(Get-MaxIter "$out\train_metrics.jsonl")"
Log "ALL GREEDYCOUNTER DONE"
Write-Output "ALL GREEDYCOUNTER DONE"
