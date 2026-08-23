# S7 continuation run: seed 20260801 extended 1000 -> 2000 iters.
#
# Purpose: distinguish "offense not yet converged at the 1000-iter budget"
# from "late non-transitive cycling" (h2h rose 0.20 -> 0.26 while jam_vs_sweep
# was still climbing at iter 999). The continuation freezes entropy
# coefficients at coef_min (--anneal-done) so the only variable is MORE
# TRAINING, not a re-anneal.
#
# Default: START IMMEDIATELY (user reordered the queue 2026-08-24: stop seed
# 20260802, run the continuation first). Pass -WaitForChain for the original
# behavior (queue behind the 3-seed chain + final-eval chain with a 5h grace).
#
# After the continuation: relaunch the 3-seed chain detached so the GPU never
# idles (seed 20260801 skips via max-iter>=999; 20260802 resumes from its last
# checkpoint; 20260803 runs fresh).
param([switch]$WaitForChain)

$env:PYTHONPATH = "E:\DATA\vscode\FluxPhased"
$env:PYTHONUNBUFFERED = "1"
$pyexe = "C:\Users\zhang\.conda\envs\fluxphased\python.exe"
Set-Location "E:\DATA\vscode\FluxPhased"
$base = "E:\DATA\vscode\FluxPhased\experiments\array_face_s7\learning_repair"
$chainlog = "$base\s7_chain.log"
$contlog = "$base\s7_continue.log"
$src = "$base\s7_selfplay_output_seed20260801\selfplay_latest.pt"
$out = "$base\s7_continue_output_seed20260801"

function Get-MaxIter([string]$path) {
  if (-not (Test-Path $path)) { return -1 }
  $last = Get-Content $path -Tail 1
  if ($last -match '"iteration":\s*(\d+)') { return [int]$Matches[1] }
  return -1
}

if ($WaitForChain) {
  while (-not (Select-String -Path $chainlog -Pattern "ALL SEEDS DONE" -Quiet)) { Start-Sleep -Seconds 300 }
  Add-Content $contlog "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] main chain done; waiting for final evals"
  $deadline = (Get-Date).AddHours(5)
  while (-not (Select-String -Path $chainlog -Pattern "ALL EVALS DONE" -Quiet) -and ((Get-Date) -lt $deadline)) { Start-Sleep -Seconds 300 }
  Add-Content $contlog "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] evals done or grace expired; starting continuation"
} else {
  Add-Content $contlog "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] immediate start (queue reordered by user)"
}

# Seed the continuation dir with the iter-999 checkpoint (out_dir differs
# from the original so the 1000-iter final eval / checkpoint stay frozen)
if (-not (Test-Path $out)) { New-Item -ItemType Directory -Path $out | Out-Null }
if (-not (Test-Path "$out\selfplay_latest.pt")) { Copy-Item $src "$out\selfplay_latest.pt" }

for ($r = 1; $r -le 30; $r++) {
  $have = Get-MaxIter "$out\train_metrics.jsonl"
  if ($have -ge 1999) { break }
  Add-Content $contlog "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] RETRY $r continue (max iter $have/1999)"
  & $pyexe -u experiments\array_face_s7\learning_repair\run_s7_selfplay.py --seed 20260801 --resume --iterations 2000 --anneal-done --out-dir $out >> "$out\run.log" 2>&1
  Start-Sleep -Seconds 30
}
Add-Content $contlog "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] CONTINUATION DONE"

# GPU back to the multi-seed chain (02 resumes from checkpoint, 03 fresh)
Start-Process powershell -ArgumentList '-ExecutionPolicy','Bypass','-File','E:\DATA\vscode\FluxPhased\_run_s7_seeds.ps1' -WindowStyle Hidden
Add-Content $contlog "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] seeds chain relaunched (02 resume, 03 fresh)"
