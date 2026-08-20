$env:PYTHONPATH = "E:\DATA\vscode\FluxPhased"
$env:PYTHONUNBUFFERED = "1"
$pyexe = "C:\Users\zhang\.conda\envs\fluxphased\python.exe"
Set-Location "E:\DATA\vscode\FluxPhased"
$base = "E:\DATA\vscode\FluxPhased\experiments\array_face_s6\learning_repair"
$chainlog = "$base\s6_chain.log"

$seed = 20260729
$out = "$base\s6_selfplay_output_seed${seed}"
if (-not (Test-Path $out)) { New-Item -ItemType Directory -Path $out | Out-Null }
for ($r = 1; $r -le 30; $r++) {
  $lines = 0
  if (Test-Path "$out\train_metrics.jsonl") { $lines = (Get-Content "$out\train_metrics.jsonl" | Measure-Object -Line).Lines }
  if ($lines -ge 1000) { break }
  Add-Content $chainlog "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] RETRY $r seed $seed (have $lines/1000 iters)"
  & $pyexe -u experiments\array_face_s6\learning_repair\run_s6_selfplay.py --seed $seed --resume >> "$out\run.log" 2>&1
  Start-Sleep -Seconds 30
}
Add-Content $chainlog "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] DONE seed $seed"
