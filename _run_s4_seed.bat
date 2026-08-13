@echo off
REM Detached S4 PPO runner. Args: %1=seed, %2=resume (1|0)
setlocal
set PYTHONPATH=E:\DATA\vscode\FluxPhased
set PYTHONUNBUFFERED=1
set PYEXE=C:\Users\zhang\.conda\envs\fluxphased\python.exe
cd /d E:\DATA\vscode\FluxPhased
set RESUME=%2
set OUTDIR=E:\DATA\vscode\FluxPhased\experiments\array_face_s4\learning_repair\s4_ppo_output_seed%1
if not exist "%OUTDIR%" mkdir "%OUTDIR%"
if "%RESUME%"=="1" (
  "%PYEXE%" -u experiments\array_face_s4\learning_repair\run_s4_ppo.py --seed %1 --resume >> "%OUTDIR%\run.log" 2>&1
) else (
  "%PYEXE%" -u experiments\array_face_s4\learning_repair\run_s4_ppo.py --seed %1 > "%OUTDIR%\run.log" 2>&1
)
endlocal
