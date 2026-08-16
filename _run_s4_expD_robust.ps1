$env:PYTHONPATH = "E:\DATA\vscode\FluxPhased"
$env:PYTHONUNBUFFERED = "1"
$pyexe = "C:\Users\zhang\.conda\envs\fluxphased\python.exe"
Set-Location "E:\DATA\vscode\FluxPhased"
$base = "E:\DATA\vscode\FluxPhased\experiments\array_face_s4\learning_repair"
$chainlog = "$base\expD_3seed_chain.log"

foreach ($seed in 20260730, 20260801) {
  $out = "$base\s4_ppo_output_seed${seed}_expD_1k"
  if (-not (Test-Path $out)) { New-Item -ItemType Directory -Path $out | Out-Null }
  for ($r = 1; $r -le 20; $r++) {
    $lines = 0
    if (Test-Path "$out\train_metrics.jsonl") { $lines = (Get-Content "$out\train_metrics.jsonl" | Measure-Object -Line).Lines }
    if ($lines -ge 1000) { break }
    Add-Content $chainlog "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] RETRY $r seed $seed (have $lines/1000 iters)"
    & $pyexe -u experiments\array_face_s4\learning_repair\run_s4_ppo.py --seed $seed --resume --iterations 1000 --shaping-mode average --shaping-coef 0.0 --no-beam-trunk --beam-anneal-frac 0.9 --outdir-tag expD_1k >> "$out\run.log" 2>&1
    Start-Sleep -Seconds 30
  }
  Add-Content $chainlog "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] DONE seed $seed"
}
Add-Content $chainlog "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] ALL SEEDS COMPLETE"
