@echo off
REM Resume crashed expD seeds (30, 31) from checkpoint_latest.pt (iter 49).
REM Uses call :subroutine so %time% expands per-execution (not parse-time).
setlocal
set PYTHONPATH=E:\DATA\vscode\FluxPhased
set PYTHONUNBUFFERED=1
set PYEXE=C:\Users\zhang\.conda\envs\fluxphased\python.exe
cd /d E:\DATA\vscode\FluxPhased
set BASE=E:\DATA\vscode\FluxPhased\experiments\array_face_s4\learning_repair

call :run_seed 20260730
call :run_seed 20260801
goto :eof

:run_seed
echo [%date% %time%] RESUME seed %1 >> "%BASE%\expD_3seed_chain.log"
"%PYEXE%" -u experiments\array_face_s4\learning_repair\run_s4_ppo.py --seed %1 --resume --iterations 1000 --shaping-mode average --shaping-coef 0.0 --no-beam-trunk --beam-anneal-frac 0.9 --outdir-tag expD_1k >> "%BASE%\s4_ppo_output_seed%1_expD_1k\run.log" 2>&1
echo [%date% %time%] DONE seed %1 >> "%BASE%\expD_3seed_chain.log"
goto :eof
endlocal
