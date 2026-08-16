@echo off
REM expD: NO shaping + NO beam trunk + extended beam entropy anneal (0.9)
setlocal
set PYTHONPATH=E:\DATA\vscode\FluxPhased
set PYTHONUNBUFFERED=1
set PYEXE=C:\Users\zhang\.conda\envs\fluxphased\python.exe
cd /d E:\DATA\vscode\FluxPhased
set OUTDIR=E:\DATA\vscode\FluxPhased\experiments\array_face_s4\learning_repair\s4_ppo_output_seed20260729_expD_anneal09
if not exist "%OUTDIR%" mkdir "%OUTDIR%"
"%PYEXE%" -u experiments\array_face_s4\learning_repair\run_s4_ppo.py --seed 20260729 --iterations 400 --shaping-mode average --shaping-coef 0.0 --no-beam-trunk --beam-anneal-frac 0.9 --outdir-tag expD_anneal09 > "%OUTDIR%\run.log" 2>&1
endlocal
