$env:PYTHONPATH = "E:\DATA\vscode\FluxPhased"
$env:PYTHONUNBUFFERED = "1"
$pyexe = "C:\Users\zhang\.conda\envs\fluxphased\python.exe"
Set-Location "E:\DATA\vscode\FluxPhased"
$base = "E:\DATA\vscode\FluxPhased\experiments\array_face_s5\learning_repair"

$tag = "shared"; $seed = 20260729
$out = "$base\s5_${tag}_output_seed${seed}"
if (-not (Test-Path $out)) { New-Item -ItemType Directory -Path $out | Out-Null }
for ($r = 1; $r -le 30; $r++) {
  $lines = 0
  if (Test-Path "$out\train_metrics.jsonl") { $lines = (Get-Content "$out\train_metrics.jsonl" | Measure-Object -Line).Lines }
  if ($lines -ge 1000) { break }
  Add-Content "$base\s5_chain.log" "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] RETRY $r $tag seed $seed (have $lines/1000)"
  & $pyexe -u experiments\array_face_s5\learning_repair\run_s5_ippo.py --seed $seed --resume --shared-budget >> "$out\run.log" 2>&1
  Start-Sleep -Seconds 30
}
Add-Content "$base\s5_chain.log" "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] DONE $tag seed $seed"
