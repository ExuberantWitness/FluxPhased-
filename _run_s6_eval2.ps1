$env:PYTHONPATH = "E:\DATA\vscode\FluxPhased"
$env:PYTHONUNBUFFERED = "1"
$pyexe = "C:\Users\zhang\.conda\envs\fluxphased\python.exe"
Set-Location "E:\DATA\vscode\FluxPhased"
$base = "E:\DATA\vscode\FluxPhased\experiments\array_face_s6\learning_repair"
foreach ($seed in @(20260730, 20260731)) {
  Add-Content "$base\s6_chain_seeds2.log" "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] EVAL seed $seed"
  & $pyexe -u _s6_final_eval_seed.py --seed $seed >> "$base\s6_selfplay_output_seed$seed\final_eval_run.log" 2>&1
  Add-Content "$base\s6_chain_seeds2.log" "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] EVAL DONE seed $seed"
}
Add-Content "$base\s6_chain_seeds2.log" "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] ALL EVALS DONE"
