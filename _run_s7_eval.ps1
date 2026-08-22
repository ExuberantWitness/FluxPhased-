"""S7 four-view final-eval chain (parameterized by _s7_final_eval.py)."""
$env:PYTHONPATH = "E:\DATA\vscode\FluxPhased"
$env:PYTHONUNBUFFERED = "1"
$pyexe = "C:\Users\zhang\.conda\envs\fluxphased\python.exe"
Set-Location "E:\DATA\vscode\FluxPhased"
$base = "E:\DATA\vscode\FluxPhased\experiments\array_face_s7\learning_repair"
foreach ($seed in @(20260801, 20260802, 20260803)) {
  Add-Content "$base\s7_chain.log" "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] EVAL seed $seed"
  & $pyexe -u _s7_final_eval.py --seed $seed >> "$base\s7_selfplay_output_seed$seed\final_eval_run.log" 2>&1
  Add-Content "$base\s7_chain.log" "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] EVAL DONE seed $seed"
}
Add-Content "$base\s7_chain.log" "[$(Get-Date -Format 'yyyy/MM/dd HH:mm:ss')] ALL EVALS DONE"
