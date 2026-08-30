# MASTER chain — every remaining TAES-revision run in ONE sequential script.
#
# Replaces the three separate chains (TAES / greedy-counter / n-scaling):
# cross-chain log waits proved fragile (one wait let a follower start early
# and share the GPU all night). Completion inside this chain is detected by
# artifacts, not logs; every training self-heals via max-iteration resume.
#
# Single-flight guard: refuses to start if any python training is running.
# Launch DETACHED so it survives CLI session restarts:
#   powershell -NoProfile -ExecutionPolicy Bypass -File _run_master_chain.ps1
$env:PYTHONPATH = "E:\DATA\vscode\FluxPhased"
$env:PYTHONUNBUFFERED = "1"
$pyexe = "C:/Users/zhang/.conda/envs/fluxphased/python.exe"
Set-Location "E:\DATA\vscode\FluxPhased"
$base = "E:\DATA\vscode\FluxPhased\experiments\array_face_s7\learning_repair"
$s6base = "E:\DATA\vscode\FluxPhased\experiments\array_face_s6\learning_repair"
$chainlog = "$base\master_chain.log"

function Log([string]$msg) {
  Add-Content $chainlog "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] $msg"
}
function Get-MaxIter([string]$path) {
  if (-not (Test-Path $path)) { return -1 }
  $last = Get-Content $path -Tail 1
  if ($last -match '"iteration":\s*(\d+)') { return [int]$Matches[1] }
  return -1
}
function Done([string]$path) {
  # final_eval.json counts as done unless it is an explicit skip marker
  if (-not (Test-Path $path)) { return $false }
  $raw = Get-Content $path -Raw
  return $raw -notmatch '"skipped"\s*:\s*true'
}

# ---- single-flight guard: no concurrent training ----
$live = Get-Process python -ErrorAction SilentlyContinue
if ($live) {
  Log "REFUSED to start: python processes already running (PIDs $($live.Id -join ',')). Start me when the GPU is idle."
  Write-Output "REFUSED: python already running"
  exit 1
}
Log "=== MASTER chain start (single-flight OK) ==="

function Ensure-Dir([string]$d) {
  if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null }
}

# ---- two-stage S7-family runner (1000 fresh + 1000 anneal-frozen) ----
function Run-S7Stage {
  param([string]$seed, [string]$out, [string[]]$extraArgs, [string]$tag, [int]$nJ = 2)
  Ensure-Dir $out
  for ($r = 1; $r -le 90; $r++) {
    $have = Get-MaxIter "$out\train_metrics.jsonl"
    if ($have -ge 1999) { break }
    if ($have -lt 999) {
      Log "$tag stage-A attempt $r (have=$have)"
      & $pyexe -u experiments/array_face_s7/learning_repair/run_s7_selfplay.py `
        --seed $seed --resume --iterations 1000 --val-every 50 --n-jammers $nJ `
        --out-dir $out @extraArgs *>> "$out\run.log"
    } else {
      Log "$tag stage-B attempt $r (have=$have)"
      & $pyexe -u experiments/array_face_s7/learning_repair/run_s7_selfplay.py `
        --seed $seed --resume --iterations 2000 --anneal-done --val-every 50 --n-jammers $nJ `
        --out-dir $out @extraArgs *>> "$out\run.log"
    }
    if ($r -ge 30 -and (Get-MaxIter "$out\train_metrics.jsonl") -lt 0) {
      Log "$tag WARNING: no progress after $r attempts; continuing"
    }
  }
  Log "$tag training complete at iter $(Get-MaxIter "$out\train_metrics.jsonl")"
}

function Eval-S7 {
  param([string]$seed, [string]$out, [string[]]$extraArgs, [string]$tag)
  for ($r = 1; $r -le 5; $r++) {
    & $pyexe -u _s7_final_eval.py --seed $seed --out-dir $out @extraArgs `
      *>> "$out\final_eval_run.log"
    if (Done "$out\final_eval.json") { Log "$tag final eval done"; return }
    Log "$tag final eval attempt $r failed; retrying"
    Start-Sleep -Seconds 30
  }
  Log "$tag FINAL EVAL STILL MISSING after retries"
}

# ---- 1. S6 seed-3 eval recovery (training done, eval was killed) ----
if (-not (Done "$s6base\s6_selfplay_output_seed20260732\final_eval.json")) {
  for ($r = 1; $r -le 3; $r++) {
    & $pyexe -u _s6_final_eval_seed.py --seed 20260732 `
      *>> "$s6base\s6_selfplay_output_seed20260732\final_eval_run.log"
    if (Done "$s6base\s6_selfplay_output_seed20260732\final_eval.json") { break }
    Start-Sleep -Seconds 30
  }
  Log "S6-20260732 eval recovery done"
}

# ---- 2. n=3 seed 1 eval recovery (training done; eval crashed on the old
#         N_JAMMERS=2 hardcode, now fixed). Skipped if my manual recovery
#         already wrote final_eval.json. ----
$n3a = "$base\s9_n3_output_seed20261011"
if (-not (Done "$n3a\final_eval.json")) {
  Eval-S7 -seed 20261011 -out $n3a -extraArgs @('--n-jammers','3','--jammer-az','+60,0,-60') -tag "N3-1011"
}

# ---- 3. n=3 seeds 2-3 ----
foreach ($sd in @(20261012, 20261013)) {
  $out = "$base\s9_n3_output_seed$sd"
  if (Done "$out\final_eval.json") { continue }
  Run-S7Stage -seed $sd -out $out -nJ 3 -extraArgs @('--jammer-az','+60,0,-60') -tag "N3-$sd"
  Eval-S7 -seed $sd -out $out -extraArgs @('--n-jammers','3','--jammer-az','+60,0,-60') -tag "N3-$sd"
}

# ---- 4. co-located mechanism control seeds 2-3 ----
foreach ($sd in @(20260812, 20260813)) {
  $out = "$base\s7_ablation_output_seed$sd"
  if (Done "$out\final_eval.json") { continue }
  Run-S7Stage -seed $sd -out $out -nJ 2 -extraArgs @('--jammer-az','+60,+60') -tag "COLOC-$sd"
  Eval-S7 -seed $sd -out $out -extraArgs @('--jammer-az','+60,+60') -tag "COLOC-$sd"
}

# ---- 5. SNR retrained regimes ----
foreach ($pair in @(@('20260911','9'), @('20260912','15'))) {
  $sd = $pair[0]; $snr = $pair[1]
  $out = "$base\s7_snr${snr}db_output_seed$sd"
  if (Done "$out\final_eval.json") { continue }
  Run-S7Stage -seed $sd -out $out -nJ 2 -extraArgs @('--baseline-snr-db',$snr) -tag "SNR$snr-$sd"
  Eval-S7 -seed $sd -out $out -extraArgs @('--baseline-snr-db',$snr) -tag "SNR$snr-$sd"
}

# ---- 6. greedy-stare counter-adaptation (own driver) ----
$greed = "$base\s7_greedycounter_output_seed20260921"
if (-not (Test-Path "$greed\CHAIN_DONE")) {
  Ensure-Dir $greed
  for ($r = 1; $r -le 90; $r++) {
    $have = Get-MaxIter "$greed\train_metrics.jsonl"
    if ($have -ge 1999) { break }
    if ($have -lt 999) {
      Log "GREEDY stage-A attempt $r (have=$have)"
      & $pyexe -u _run_s7_greedy_counter.py --seed 20260921 --resume `
        --iterations 1000 --val-every 50 *>> "$greed\run.log"
    } else {
      Log "GREEDY stage-B attempt $r (have=$have)"
      & $pyexe -u _run_s7_greedy_counter.py --seed 20260921 --resume `
        --iterations 2000 --val-every 50 *>> "$greed\run.log"
    }
  }
  Set-Content "$greed\CHAIN_DONE" "done $(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')"
  Log "GREEDYCOUNTER training complete"
}

# ---- 7. n=4 seeds ----
foreach ($sd in @(20261021, 20261022, 20261023)) {
  $out = "$base\s9_n4_output_seed$sd"
  if (Done "$out\final_eval.json") { continue }
  Run-S7Stage -seed $sd -out $out -nJ 4 -extraArgs @('--jammer-az','+60,+20,-20,-60') -tag "N4-$sd"
  Eval-S7 -seed $sd -out $out -extraArgs @('--n-jammers','4','--jammer-az','+60,+20,-20,-60') -tag "N4-$sd"
}

Log "ALL MASTER RUNS DONE"
Write-Output "ALL MASTER RUNS DONE"
