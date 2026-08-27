# S7 R5-lite: opponent-class mixing — does mixing recover singleton robustness?
#
# Design: 3 mixing ratios x 1 seed, each 2000 iters (stage A 1000 normal
# anneal + stage B 1000 frozen anneal — identical protocol to the converged
# reference so the 0%-mixing comparison is seed 20260801's 2000-iter
# checkpoint eval). On singleton iterations (deterministic cycling by the
# fraction), jammer 1 is forced idle and the jammer side's update is SKIPPED:
# the jammer team learns purely from pair self-play while the radar team is
# league-trained across {pair, singleton} opponent classes.
#
#   mix=0.25  seed 20260821   (25% singleton exposure)
#   mix=0.50  seed 20260822
#   mix=0.75  seed 20260823
#   mix=0.00  reference = s7_continue_output_seed20260801/final_eval.json
#             (h2h 0.3282, j1_only 0.2532, neutralization 20.2% @ 2000 iters)
#
# Read-out per condition: full-protocol eval gives BOTH h2h (pair view) and
# j1_only (singleton view) on the same radar checkpoint -> the robustness
# frontier (h2h vs j1_only) as a function of mix fraction.
$env:PYTHONPATH = "E:\DATA\vscode\FluxPhased"
$env:PYTHONUNBUFFERED = "1"
$pyexe = "C:\Users\zhang\.conda\envs\fluxphased\python.exe"
Set-Location "E:\DATA\vscode\FluxPhased"
$base = "E:\DATA\vscode\FluxPhased\experiments\array_face_s7\learning_repair"
$r5log = "$base\s7_r5.log"

function Get-MaxIter([string]$path) {
  if (-not (Test-Path $path)) { return -1 }
  $last = Get-Content $path -Tail 1
  if ($last -match '"iteration":\s*(\d+)') { return [int]$Matches[1] }
  return -1
}

Add-Content $r5log "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] R5-lite starting"

foreach ($case in @(@(20260821, "0.25"), @(20260822, "0.5"), @(20260823, "0.75"))) {
  $cseed = $case[0]; $mix = $case[1]
  $out = "$base\s7_r5_mix$($mix -replace '\.','p')_output_seed$cseed"
  if (-not (Test-Path $out)) { New-Item -ItemType Directory -Path $out | Out-Null }
  for ($r = 1; $r -le 30; $r++) {
    $have = Get-MaxIter "$out\train_metrics.jsonl"
    if ($have -ge 1999) { break }
    if ($have -lt 999) {
      Add-Content $r5log "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] mix=$mix stage-A (max iter $have/999)"
      & $pyexe -u experiments\array_face_s7\learning_repair\run_s7_selfplay.py --seed $cseed --resume --iterations 1000 --val-every 50 --singleton-mix $mix --out-dir $out >> "$out\run.log" 2>&1
    } else {
      Add-Content $r5log "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] mix=$mix stage-B (max iter $have/1999)"
      & $pyexe -u experiments\array_face_s7\learning_repair\run_s7_selfplay.py --seed $cseed --resume --iterations 2000 --anneal-done --val-every 50 --singleton-mix $mix --out-dir $out >> "$out\run.log" 2>&1
    }
    Start-Sleep -Seconds 30
  }
  & $pyexe -u _s7_final_eval.py --seed $cseed --device cpu --out-dir $out >> "$out\final_eval_run.log" 2>&1
  Add-Content $r5log "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] mix=$mix DONE (2000 iters + eval)"
}
Add-Content $r5log "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] ALL R5 DONE"
