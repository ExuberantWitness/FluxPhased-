# Strict n-scale supplementation chain for TAES.
#
# Produces the matched 2000-iteration terminal endpoint required for a clean
# n=2/3/4 comparison. All final evaluations use schema v2 metadata.
# Existing checkpoints are copied into fresh strict directories before resume;
# legacy outputs are never overwritten.
$env:PYTHONPATH = "E:\DATA\vscode\FluxPhased"
$env:PYTHONUNBUFFERED = "1"
$pyexe = "C:/Users/zhang/.conda/envs/fluxphased/python.exe"
Set-Location "E:\DATA\vscode\FluxPhased"
$base = "E:\DATA\vscode\FluxPhased\experiments\array_face_s7\learning_repair"
$chainlog = "$base\strict_nscale_chain.log"

function Log([string]$msg) {
  Add-Content $chainlog "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] $msg"
}
function Ensure-Dir([string]$d) {
  if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null }
}
function Get-MaxIter([string]$path) {
  if (-not (Test-Path $path)) { return -1 }
  $last = Get-Content $path -Tail 1
  if ($last -match '"iteration":\s*(\d+)') { return [int]$Matches[1] }
  return -1
}
function Run-Continuation {
  param([string]$src, [string]$out, [string]$seed, [int]$n, [string]$jaz)
  Ensure-Dir $out
  if (-not (Test-Path "$out\selfplay_latest.pt")) {
    Copy-Item "$src\selfplay_latest.pt" "$out\selfplay_latest.pt"
  }
  for ($r = 1; $r -le 8; $r++) {
    $have = Get-MaxIter "$out\train_metrics.jsonl"
    if ($have -ge 1999) { break }
    if ($have -lt 0) {
      $probe = & $pyexe -c "import torch; print(torch.load(r'$out/selfplay_latest.pt',map_location='cpu')['iteration'])"
      $have = [int]$probe[-1]
    }
    Log "N$n-$seed continuation attempt $r (have=$have)"
    & $pyexe -u experiments/array_face_s7/learning_repair/run_s7_selfplay.py `
      --seed $seed --resume --iterations 2000 --anneal-done --val-every 50 `
      --n-jammers $n --jammer-az $jaz --out-dir $out *>> "$out\run.log"
  }
  if ((Get-MaxIter "$out\train_metrics.jsonl") -lt 1999) {
    Log "N$n-$seed FAILED to reach terminal iter"
    throw "N$n-$seed continuation did not reach iter 1999"
  }
}
function Eval-Strict {
  param([string]$seed, [string]$out, [int]$n, [string]$jaz)
  Log "N$n-$seed final eval start"
  & $pyexe -u _s7_final_eval.py --seed $seed --n-jammers $n `
    --jammer-az $jaz --out-dir $out --output-name final_eval_v2.json `
    --device cpu *>> "$out\final_eval_v2_run.log"
  if (-not (Test-Path "$out\final_eval_v2.json")) {
    Log "N$n-$seed final eval missing"
    throw "N$n-$seed final eval missing"
  }
  Log "N$n-$seed final eval done"
}

Log "=== strict n-scale supplementation start ==="

# n=2 cross-fire, seed 01: existing 2000 checkpoint, clean schema-v2 eval.
$n2a = "$base\s7_strict_n2_output_seed20260801"
Ensure-Dir $n2a
if (-not (Test-Path "$n2a\final_eval_v2.json")) {
  Copy-Item "$base\s7_continue_output_seed20260801\selfplay_latest.pt" "$n2a\selfplay_latest.pt" -Force
  Eval-Strict -seed 20260801 -out $n2a -n 2 -jaz "+60,-60"
}

# n=2 cross-fire, seeds 02/03: 999 -> 1999, then eval.
foreach ($pair in @(
  @('20260802', "$base\s7_selfplay_output_seed20260802", "$base\s7_strict_n2_output_seed20260802"),
  @('20260803', "$base\s7_selfplay_output_seed20260803", "$base\s7_strict_n2_output_seed20260803")
)) {
  $sd = $pair[0]; $src = $pair[1]; $out = $pair[2]
  if (-not (Test-Path "$out\final_eval_v2.json")) {
    Run-Continuation -src $src -out $out -seed $sd -n 2 -jaz "+60,-60"
    Eval-Strict -seed $sd -out $out -n 2 -jaz "+60,-60"
  }
}

# n=3: all checkpoints are terminal; clean schema-v2 eval (especially 1011,
# whose first evaluation attempt had a K=2 floor bug in its log).
foreach ($sd in @(20261011, 20261012, 20261013)) {
  $out = "$base\s9_n3_output_seed$sd"
  if (-not (Test-Path "$out\final_eval_v2.json")) {
    Eval-Strict -seed $sd -out $out -n 3 -jaz "+60,0,-60"
  }
}

# n=4 seed 21 is terminal; seeds 22/23 require 1949 -> 1999 continuation.
$n4a = "$base\s9_n4_output_seed20261021"
if (-not (Test-Path "$n4a\final_eval_v2.json")) {
  Eval-Strict -seed 20261021 -out $n4a -n 4 -jaz "+60,+20,-20,-60"
}
foreach ($sd in @(20261022, 20261023)) {
  $src = "$base\s9_n4_output_seed$sd"
  $out = "$base\s9_strict_n4_output_seed$sd"
  if (-not (Test-Path "$out\final_eval_v2.json")) {
    Run-Continuation -src $src -out $out -seed $sd -n 4 -jaz "+60,+20,-20,-60"
    Eval-Strict -seed $sd -out $out -n 4 -jaz "+60,+20,-20,-60"
  }
}

Log "ALL STRICT NSCALE DONE"
Write-Output "ALL STRICT NSCALE DONE"
