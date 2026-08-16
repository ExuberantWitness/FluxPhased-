@echo off
REM Robust expD resume chain: auto-retry each seed until train_metrics reaches
REM 1000 iters (self-healing after silent crashes / OOM / system events).
setlocal EnableDelayedExpansion
set PYTHONPATH=E:\DATA\vscode\FluxPhased
set PYTHONUNBUFFERED=1
set PYEXE=C:\Users\zhang\.conda\envs\fluxphased\python.exe
cd /d E:\DATA\vscode\FluxPhased
set BASE=E:\DATA\vscode\FluxPhased\experiments\array_face_s4\learning_repair

call :run_seed 20260730
call :run_seed 20260801
echo [%date% %time%] ALL SEEDS COMPLETE >> "%BASE%\expD_3seed_chain.log"
goto :eof

:run_seed
set SEED=%1
set OUTDIR=%BASE%\s4_ppo_output_seed%SEED%_expD_1k
for /L %%R in (1,1,20) do (
  set LINES=0
  if exist "%OUTDIR%\train_metrics.jsonl" for /f %%C in ('type "%OUTDIR%\train_metrics.jsonl" ^| find /c /v ""') do set LINES=%%C
  if !LINES! GEQ 1000 goto :done_seed
  echo [%date% %time%] RETRY %%R seed %SEED% (have !LINES!/1000 iters) >> "%BASE%\expD_3seed_chain.log"
  "%PYEXE%" -u experiments\array_face_s4\learning_repair\run_s4_ppo.py --seed %SEED% --resume --iterations 1000 --shaping-mode average --shaping-coef 0.0 --no-beam-trunk --beam-anneal-frac 0.9 --outdir-tag expD_1k >> "%OUTDIR%\run.log" 2>&1
  timeout /t 30 /nobreak >nul
)
:done_seed
echo [%date% %time%] DONE seed %SEED% >> "%BASE%\expD_3seed_chain.log"
goto :eof
endlocal
