# S7 R3 mechanism ablation — co-located jammer pair vs separated radars.
#
# Pre-registered question: is the containment collapse (64% -> 20%) caused by
# CROSS-FIRE GEOMETRY (jammers at two bearings), or merely by doubling the
# attacker count?
#
# Design: identical to S7 (2 jammers vs 2 radars, snr=12, 63-token team
# budget, 2000 iterations - the converged budget from the continuation
# control) EXCEPT both jammers sit at +60 deg. The radar team stays at +-20.
#
# Pre-registered prediction (from the contestability sweep, 2026-08-25):
#   cross-fire profile  [0.837, 0.912, 0.989, 0.993, 0.855]  (3/5 contestable)
#   co-located profile  [0.921, 0.823, 0.989, 0.996, 0.991]  (2/5 contestable)
# With both jammers on one side, the far radar (rel -80 deg) is nearly
# unreachable, so the radar team can turn its Rx away from the single threat
# bearing while still serving missions - single-beam suppression structure
# returns. PREDICTION: converged neutralization >> 20.2% (defense recovers).
# FALSIFIER: neutralization ~= 20% despite co-location -> geometry is not the
# mechanism; attacker count alone does the damage.
#
# Ordering: queues behind the main 3-seed chain (ALL SEEDS DONE in
# s7_chain.log), then trains 1 seed (20260811) to 2000 iters, then runs the
# full-protocol final eval on CPU.
$env:PYTHONPATH = "E:\DATA\vscode\FluxPhased"
$env:PYTHONUNBUFFERED = "1"
$pyexe = "C:\Users\zhang\.conda\envs\fluxphased\python.exe"
Set-Location "E:\DATA\vscode\FluxPhased"
$base = "E:\DATA\vscode\FluxPhased\experiments\array_face_s7\learning_repair"
$ablog = "$base\s7_ablation.log"
$out = "$base\s7_ablation_output_seed20260811"
$seed = 20260811

function Get-MaxIter([string]$path) {
  if (-not (Test-Path $path)) { return -1 }
  $last = Get-Content $path -Tail 1
  if ($last -match '"iteration":\s*(\d+)') { return [int]$Matches[1] }
  return -1
}

# 1. wait for the main 3-seed chain
while (-not (Select-String -Path "$base\s7_chain.log" -Pattern "ALL SEEDS DONE" -Quiet)) { Start-Sleep -Seconds 300 }
Add-Content $ablog "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] main chain done; starting co-located ablation"

if (-not (Test-Path $out)) { New-Item -ItemType Directory -Path $out | Out-Null }

for ($r = 1; $r -le 40; $r++) {
  $have = Get-MaxIter "$out\train_metrics.jsonl"
  if ($have -ge 1999) { break }
  if ($have -lt 999) {
    # Stage A: fresh 1000 iters, normal anneal (identical schedule to the
    # cross-fire reference's first 1000)
    Add-Content $ablog "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] RETRY $r ablation stage-A (max iter $have/999)"
    & $pyexe -u experiments\array_face_s7\learning_repair\run_s7_selfplay.py --seed $seed --resume --iterations 1000 --jammer-az "+60,+60" --out-dir $out >> "$out\run.log" 2>&1
  } else {
    # Stage B: 1000 -> 2000 with frozen anneal (mirrors the cross-fire
    # continuation protocol exactly - the ONLY difference is the geometry)
    Add-Content $ablog "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] RETRY $r ablation stage-B (max iter $have/1999)"
    & $pyexe -u experiments\array_face_s7\learning_repair\run_s7_selfplay.py --seed $seed --resume --iterations 2000 --anneal-done --jammer-az "+60,+60" --out-dir $out >> "$out\run.log" 2>&1
  }
  Start-Sleep -Seconds 30
}
Add-Content $ablog "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] ABLATION TRAIN DONE"

# 2. full-protocol final eval (CPU: the GPU may already be free, but CPU keeps
#    the ablation self-contained and lets any follow-up use the GPU)
& $pyexe -u _s7_final_eval.py --seed $seed --device cpu --out-dir $out >> "$out\final_eval_run.log" 2>&1
Add-Content $ablog "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] ABLATION EVAL DONE"
Add-Content $ablog "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] ALL ABLATION DONE"

# 3. Multi-seed convergence replication: seeds 02/03 continued 1000 -> 3000.
#    Same protocol as seed 01's stage-2 (anneal pinned at the 1000 resume
#    point via --anneal-done; --iterations 3000 covers 1000..3000 in one
#    driver run). Fresh dirs keep the 1000-iter checkpoints frozen for the
#    replication table.
foreach ($cs in @(@(20260802, "s7_seed02_cont_output_seed20260802"), @(20260803, "s7_seed03_cont_output_seed20260803"))) {
  $cseed = $cs[0]; $cout = "$base\$($cs[1])"
  $csrc = "$base\s7_selfplay_output_seed$cseed\selfplay_latest.pt"
  if (-not (Test-Path $cout)) { New-Item -ItemType Directory -Path $cout | Out-Null }
  if (-not (Test-Path "$cout\selfplay_latest.pt")) { Copy-Item $csrc "$cout\selfplay_latest.pt" }
  for ($r = 1; $r -le 30; $r++) {
    $have = Get-MaxIter "$cout\train_metrics.jsonl"
    if ($have -ge 2999) { break }
    Add-Content $ablog "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] CONT seed $cseed (max iter $have/2999)"
    & $pyexe -u experiments\array_face_s7\learning_repair\run_s7_selfplay.py --seed $cseed --resume --iterations 3000 --anneal-done --out-dir $cout >> "$cout\run.log" 2>&1
    Start-Sleep -Seconds 30
  }
  & $pyexe -u _s7_final_eval.py --seed $cseed --device cpu --out-dir $cout >> "$cout\final_eval_run.log" 2>&1
  Add-Content $ablog "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] CONT DONE seed $cseed (3000 iters + eval)"
}
Add-Content $ablog "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] ALL POST-CHAIN DONE"
