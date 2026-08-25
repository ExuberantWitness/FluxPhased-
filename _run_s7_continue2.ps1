# S7 second-stage continuation: seed 20260801 extended 2000 -> 3000 iters.
#
# User call 2026-08-25: the 2000-iter curve is not flat enough point-to-point;
# add another 1000 steps to confirm the plateau (or catch a very slow grind).
# Entropy coefficients stay frozen at coef_min (--anneal-done probes the
# seeded checkpoint: window ends exactly at the 2000 resume point).
#
# Starts IMMEDIATELY (seed 20260803 was preempted again; its checkpoint 349 is
# preserved). After this stage the 3-seed chain relaunches detached (03
# resumes, 01/02 skip via max-iter criterion).
$env:PYTHONPATH = "E:\DATA\vscode\FluxPhased"
$env:PYTHONUNBUFFERED = "1"
$pyexe = "C:\Users\zhang\.conda\envs\fluxphased\python.exe"
Set-Location "E:\DATA\vscode\FluxPhased"
$base = "E:\DATA\vscode\FluxPhased\experiments\array_face_s7\learning_repair"
$cont2log = "$base\s7_continue2.log"
$src = "$base\s7_continue_output_seed20260801\selfplay_latest.pt"
$out = "$base\s7_continue2_output_seed20260801"

function Get-MaxIter([string]$path) {
  if (-not (Test-Path $path)) { return -1 }
  $last = Get-Content $path -Tail 1
  if ($last -match '"iteration":\s*(\d+)') { return [int]$Matches[1] }
  return -1
}

Add-Content $cont2log "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] immediate start (stage 2, user reorder)"

if (-not (Test-Path $out)) { New-Item -ItemType Directory -Path $out | Out-Null }
if (-not (Test-Path "$out\selfplay_latest.pt")) { Copy-Item $src "$out\selfplay_latest.pt" }

for ($r = 1; $r -le 30; $r++) {
  $have = Get-MaxIter "$out\train_metrics.jsonl"
  if ($have -ge 2999) { break }
  Add-Content $cont2log "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] RETRY $r continue2 (max iter $have/2999)"
  & $pyexe -u experiments\array_face_s7\learning_repair\run_s7_selfplay.py --seed 20260801 --resume --iterations 3000 --anneal-done --out-dir $out >> "$out\run.log" 2>&1
  Start-Sleep -Seconds 30
}
Add-Content $cont2log "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] CONTINUATION2 DONE"

Start-Process powershell -ArgumentList '-ExecutionPolicy','Bypass','-File','E:\DATA\vscode\FluxPhased\_run_s7_seeds.ps1' -WindowStyle Hidden
Add-Content $cont2log "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] seeds chain relaunched (03 resume)"
