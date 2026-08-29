# TAES revision training chain — IPPO control, seed completion, SNR sweep.
#
# Fills the three training-dependent TAES reviewer gaps in priority order:
#   P0-a  IPPO algorithm control (3 seeds x 2000 iters) — "is the containment
#         collapse MAPPO-specific?"
#   P0-b  S6 third valid 12-dB seed (20260732, 1000 iters)
#   P0-c  co-located ablation seeds 2-3 (20260812/13, 2000 iters two-stage)
#   P1    SNR regime sweep 9/15 dB (20260911/12, 2000 iters two-stage)
#
# Every run self-heals: the retry loop checks the max train iteration and
# resumes from the atomic checkpoint after interruptions (machine sleep etc).
# Final evals run after each run completes. Logs to taes_chain.log;
# "ALL TAES RUNS DONE" marks completion.
$env:PYTHONPATH = "E:\DATA\vscode\FluxPhased"
$env:PYTHONUNBUFFERED = "1"
$pyexe = "C:/Users/zhang/.conda/envs/fluxphased/python.exe"
Set-Location "E:\DATA\vscode\FluxPhased"
$base = "E:\DATA\vscode\FluxPhased\experiments\array_face_s7\learning_repair"
$s6base = "E:\DATA\vscode\FluxPhased\experiments\array_face_s6\learning_repair"
$chainlog = "$base\taes_chain.log"

function Log([string]$msg) {
  Add-Content $chainlog "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] $msg"
}
function Get-MaxIter([string]$path) {
  if (-not (Test-Path $path)) { return -1 }
  $last = Get-Content $path -Tail 1
  if ($last -match '"iteration":\s*(\d+)') { return [int]$Matches[1] }
  return -1
}

# --- generic two-stage S7 runner (1000 + 1000 anneal-frozen continuation) ---
function Run-S7Stage {
  param([string]$seed, [string]$out, [string[]]$extraArgs, [string]$tag)
  if (-not (Test-Path $out)) { New-Item -ItemType Directory -Path $out | Out-Null }
  for ($r = 1; $r -le 60; $r++) {
    $have = Get-MaxIter "$out\train_metrics.jsonl"
    if ($have -ge 1999) { break }
    if ($have -lt 999) {
      Log "$tag stage-A attempt $r (have=$have)"
      & $pyexe -u experiments/array_face_s7/learning_repair/run_s7_selfplay.py `
        --seed $seed --resume --iterations 1000 --val-every 50 `
        --out-dir $out @extraArgs *>> "$out\run.log"
    } else {
      Log "$tag stage-B attempt $r (have=$have)"
      & $pyexe -u experiments/array_face_s7/learning_repair/run_s7_selfplay.py `
        --seed $seed --resume --iterations 2000 --anneal-done --val-every 50 `
        --out-dir $out @extraArgs *>> "$out\run.log"
    }
  }
  Log "$tag training complete at iter $(Get-MaxIter "$out\train_metrics.jsonl")"
}

# --- generic IPPO two-stage runner ---
function Run-IPPOStage {
  param([string]$seed, [string]$out, [string]$tag)
  if (-not (Test-Path $out)) { New-Item -ItemType Directory -Path $out | Out-Null }
  for ($r = 1; $r -le 60; $r++) {
    $have = Get-MaxIter "$out\train_metrics.jsonl"
    if ($have -ge 1999) { break }
    if ($have -lt 999) {
      Log "$tag stage-A attempt $r (have=$have)"
      & $pyexe -u _run_s7_ippo.py --seed $seed --resume --iterations 1000 `
        --val-every 50 --out-dir $out *>> "$out\run.log"
    } else {
      Log "$tag stage-B attempt $r (have=$have)"
      & $pyexe -u _run_s7_ippo.py --seed $seed --resume --iterations 2000 `
        --anneal-done --val-every 50 --out-dir $out *>> "$out\run.log"
    }
  }
  Log "$tag training complete at iter $(Get-MaxIter "$out\train_metrics.jsonl")"
}

Log "=== TAES chain start ==="

# 1. IPPO seed 1 (P0-a)
$ippo1 = "$base\s7_ippo_output_seed20260901"
if (-not (Test-Path "$ippo1\final_eval.json")) {
  Run-IPPOStage -seed 20260901 -out $ippo1 -tag "IPPO-20260901"
  & $pyexe -u _s7_ippo_final_eval.py --out-dir $ippo1 --seed 20260901 `
    *>> "$ippo1\final_eval_run.log"
  Log "IPPO-20260901 final eval done"
}

# 2. S6 third valid seed (P0-b)
$s6dir = "$s6base\s6_selfplay_output_seed20260732"
if (-not (Test-Path "$s6dir\final_eval.json")) {
  if (-not (Test-Path $s6dir)) { New-Item -ItemType Directory -Path $s6dir | Out-Null }
  for ($r = 1; $r -le 40; $r++) {
    $have = Get-MaxIter "$s6dir\train_metrics.jsonl"
    if ($have -ge 999) { break }
    Log "S6-20260732 attempt $r (have=$have)"
    & $pyexe -u experiments/array_face_s6/learning_repair/run_s6_selfplay.py `
      --seed 20260732 --resume --iterations 1000 *>> "$s6dir\run.log"
  }
  & $pyexe -u _s6_final_eval_seed.py --seed 20260732 *>> "$s6dir\final_eval_run.log"
  Log "S6-20260732 final eval done"
}

# 3-4. co-located ablation seeds 2-3 (P0-c)
foreach ($sd in @(20260812, 20260813)) {
  $out = "$base\s7_ablation_output_seed$sd"
  if (Test-Path "$out\final_eval.json") { continue }
  Run-S7Stage -seed $sd -out $out -extraArgs @('--jammer-az', '+60,+60') -tag "COLOC-$sd"
  & $pyexe -u _s7_final_eval.py --seed $sd --out-dir $out --jammer-az "+60,+60" `
    *>> "$out\final_eval_run.log"
  Log "COLOC-$sd final eval done"
}

# 5-6. IPPO seeds 2-3 (P0-a replication)
foreach ($sd in @(20260902, 20260903)) {
  $out = "$base\s7_ippo_output_seed$sd"
  if (Test-Path "$out\final_eval.json") { continue }
  Run-IPPOStage -seed $sd -out $out -tag "IPPO-$sd"
  & $pyexe -u _s7_ippo_final_eval.py --out-dir $out --seed $sd `
    *>> "$out\final_eval_run.log"
  Log "IPPO-$sd final eval done"
}

# 7-8. SNR sweep (P1)
foreach ($pair in @(@('20260911', '9'), @('20260912', '15'))) {
  $sd = $pair[0]; $snr = $pair[1]
  $out = "$base\s7_snr${snr}db_output_seed$sd"
  if (Test-Path "$out\final_eval.json") { continue }
  Run-S7Stage -seed $sd -out $out -extraArgs @('--baseline-snr-db', $snr) -tag "SNR$snr-$sd"
  & $pyexe -u _s7_final_eval.py --seed $sd --out-dir $out --baseline-snr-db $snr `
    *>> "$out\final_eval_run.log"
  Log "SNR$snr-$sd final eval done"
}

Log "ALL TAES RUNS DONE"
Write-Output "ALL TAES RUNS DONE"
