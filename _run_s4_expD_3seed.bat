@echo off
REM expD official S4 training: 3 seeds x 1000 iters, serial.
REM Config: NO shaping + NO beam trunk + beam entropy anneal 0.9
setlocal
set PYTHONPATH=E:\DATA\vscode\FluxPhased
set PYTHONUNBUFFERED=1
set PYEXE=C:\Users\zhang\.conda\envs\fluxphased\python.exe
cd /d E:\DATA\vscode\FluxPhased
set BASE=E:\DATA\vscode\FluxPhased\experiments\array_face_s4\learning_repair

for %%S in (20260729 20260730 20260801) do (
  echo [%date% %time%] START seed %%S >> "%BASE%\expD_3seed_chain.log"
  if not exist "%BASE%\s4_ppo_output_seed%%S_expD_1k" mkdir "%BASE%\s4_ppo_output_seed%%S_expD_1k"
  "%PYEXE%" -u experiments\array_face_s4\learning_repair\run_s4_ppo.py --seed %%S --iterations 1000 --shaping-mode average --shaping-coef 0.0 --no-beam-trunk --beam-anneal-frac 0.9 --outdir-tag expD_1k > "%BASE%\s4_ppo_output_seed%%S_expD_1k\run.log" 2>&1
  echo [%date% %time%] DONE seed %%S >> "%BASE%\expD_3seed_chain.log"
)
endlocal
