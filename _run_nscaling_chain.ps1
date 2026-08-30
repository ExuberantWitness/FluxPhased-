# Attacker-count scaling chains (n = 3 / n = 4 jammers) — P1 of the TAES
# revision plan. Queues behind the greedy-counter chain (single GPU).
#
# Protocol: identical to the co-located ablation (two-stage 1000 + 1000
# anneal-frozen, same 63-token team budget split across n, radars fixed at
# +-20). n is the ONLY variable; the n=2 reference points are the published
# cross-fire (seed 20260801 @2000, q=0) and co-located (20260811 @2000)
# checkpoints. Pre-registered gate profiles: n_scaling_profiles.json (n=2
# anchor reproduces the published profile exactly; n=3/n=4 both PASS).
#
# Seeds avoid every running chain: n=3 -> 20261011-13, n=4 -> 20261021-23.
$env:PYTHONPATH = "E:\DATA\vscode\FluxPhased"
$env:PYTHONUNBUFFERED = "1"
$pyexe = "C:/Users/zhang/.conda/envs/fluxphased/python.exe"
Set-Location "E:\DATA\vscode\FluxPhased"
$base = "E:\DATA\vscode\FluxPhased\experiments\array_face_s7\learning_repair"
$chainlog = "$base\nscaling_chain.log"

function Log([string]$msg) {
  Add-Content $chainlog "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] $msg"
}
function Get-MaxIter([string]$path) {
  if (-not (Test-Path $path)) { return -1 }
  $last = Get-Content $path -Tail 1
  if ($last -match '"iteration":\s*(\d+)') { return [int]$Matches[1] }
  return -1
}

# serialize behind the greedy-counter chain (which itself queues behind the
# TAES chain). Test-Path guard: bare Select-String on a missing log returns
# an error result that skips the wait entirely (the 2026-08-30 incident).
while ($true) {
  if ((Test-Path "$base\greedycounter_chain.log") -and
      (Select-String -Path "$base\greedycounter_chain.log" -Pattern "ALL GREEDYCOUNTER DONE" -Quiet)) {
    break
  }
  Start-Sleep -Seconds 600
}
Log "greedy-counter chain done; starting attacker-count scaling"

function Run-Scaling {
  param([string]$seed, [string]$out, [int]$n, [string]$jaz, [string]$tag)
  if (-not (Test-Path $out)) { New-Item -ItemType Directory -Path $out | Out-Null }
  for ($r = 1; $r -le 60; $r++) {
    $have = Get-MaxIter "$out\train_metrics.jsonl"
    if ($have -ge 1999) { break }
    if ($have -lt 999) {
      Log "$tag stage-A attempt $r (have=$have)"
      & $pyexe -u experiments/array_face_s7/learning_repair/run_s7_selfplay.py `
        --seed $seed --resume --iterations 1000 --val-every 50 `
        --n-jammers $n --jammer-az $jaz --out-dir $out *>> "$out\run.log"
    } else {
      Log "$tag stage-B attempt $r (have=$have)"
      & $pyexe -u experiments/array_face_s7/learning_repair/run_s7_selfplay.py `
        --seed $seed --resume --iterations 2000 --anneal-done --val-every 50 `
        --n-jammers $n --jammer-az $jaz --out-dir $out *>> "$out\run.log"
    }
  }
  Log "$tag training complete at iter $(Get-MaxIter "$out\train_metrics.jsonl")"
  & $pyexe -u _s7_final_eval.py --seed $seed --out-dir $out `
    --n-jammers $n --jammer-az $jaz *>> "$out\final_eval_run.log"
  Log "$tag final eval done"
}

foreach ($sd in @(20261011, 20261012, 20261013)) {
  $out = "$base\s9_n3_output_seed$sd"
  if (-not (Test-Path "$out\final_eval.json")) {
    Run-Scaling -seed $sd -out $out -n 3 -jaz "+60,0,-60" -tag "N3-$sd"
  }
}
foreach ($sd in @(20261021, 20261022, 20261023)) {
  $out = "$base\s9_n4_output_seed$sd"
  if (-not (Test-Path "$out\final_eval.json")) {
    Run-Scaling -seed $sd -out $out -n 4 -jaz "+60,+20,-20,-60" -tag "N4-$sd"
  }
}

Log "ALL NSCALING DONE"
Write-Output "ALL NSCALING DONE"
