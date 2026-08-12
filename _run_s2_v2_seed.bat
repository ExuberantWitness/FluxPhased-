@echo off
REM Detached S2 PPO v2 runner. Args: %1=seed, %2=mode (amend03|amend02eq), %3=resume (1|0)
REM Sets PYTHONPATH explicitly so run_s2_ppo_v2.py's REPO computation is bypassed.
setlocal
set PYTHONPATH=E:\DATA\vscode\FluxPhased
set PYTHONUNBUFFERED=1
set PYEXE=C:\Users\zhang\.conda\envs\fluxphased\python.exe
cd /d E:\DATA\vscode\FluxPhased
set MODE=%2
if "%MODE%"=="" set MODE=amend03
set RESUME=%3
set OUTDIR=E:\DATA\vscode\FluxPhased\experiments\array_face_s2\learning_repair\s2_ppo_output_%MODE%_seed%1
if not exist "%OUTDIR%" mkdir "%OUTDIR%"
if "%RESUME%"=="1" (
  "%PYEXE%" -u experiments\array_face_s2\learning_repair\run_s2_ppo_v2.py --seed %1 --mode %MODE% --resume >> "%OUTDIR%\run.log" 2>&1
) else (
  "%PYEXE%" -u experiments\array_face_s2\learning_repair\run_s2_ppo_v2.py --seed %1 --mode %MODE% > "%OUTDIR%\run.log" 2>&1
)
endlocal
