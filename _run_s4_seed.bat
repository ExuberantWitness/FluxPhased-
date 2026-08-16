@echo off
REM Detached S4 PPO runner.
REM Args: %1=seed  %2=resume(1|0)  %3=iterations  %4=shaping-mode  %5=shaping-coef  %6=trunk(1|0)  %7=outdir-tag
setlocal
set PYTHONPATH=E:\DATA\vscode\FluxPhased
set PYTHONUNBUFFERED=1
set PYEXE=C:\Users\zhang\.conda\envs\fluxphased\python.exe
cd /d E:\DATA\vscode\FluxPhased
set RESUME=%2
set OUTDIR=E:\DATA\vscode\FluxPhased\experiments\array_face_s4\learning_repair\s4_ppo_output_seed%1_%7
if not exist "%OUTDIR%" mkdir "%OUTDIR%"
set TRUNKARG=
if "%6"=="0" set TRUNKARG=--no-beam-trunk
if "%RESUME%"=="1" (
  "%PYEXE%" -u experiments\array_face_s4\learning_repair\run_s4_ppo.py --seed %1 --resume --iterations %3 --shaping-mode %4 --shaping-coef %5 %TRUNKARG% --outdir-tag %7 >> "%OUTDIR%\run.log" 2>&1
) else (
  "%PYEXE%" -u experiments\array_face_s4\learning_repair\run_s4_ppo.py --seed %1 --iterations %3 --shaping-mode %4 --shaping-coef %5 %TRUNKARG% --outdir-tag %7 > "%OUTDIR%\run.log" 2>&1
)
endlocal
