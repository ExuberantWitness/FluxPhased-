@echo off
REM Detached S2 PPO runner. Pass seed as %1.
REM Sets PYTHONPATH explicitly so run_s2_ppo.py's REPO computation is bypassed.
setlocal
set PYTHONPATH=E:\DATA\vscode\FluxPhased
set PYTHONUNBUFFERED=1
set PYEXE=C:\Users\zhang\.conda\envs\fluxphased\python.exe
cd /d E:\DATA\vscode\FluxPhased
set OUTDIR=E:\DATA\vscode\FluxPhased\experiments\array_face_s2\learning_repair\s2_ppo_output_amend02_seed%1
if not exist "%OUTDIR%" mkdir "%OUTDIR%"
"%PYEXE%" -u experiments\array_face_s2\learning_repair\run_s2_ppo.py --seed %1 > "%OUTDIR%\run.log" 2>&1
endlocal
