@echo off
REM Chained S4 ablation: 3 experiments x 400 iterations, serial on GPU.
REM   A: tx_only shaping  coef=0.003  trunk ON
REM   B: average shaping  coef=0.001  trunk ON
REM   C: average shaping  coef=0.01   trunk OFF
setlocal
cd /d E:\DATA\vscode\FluxPhased
set CHAINLOG=E:\DATA\vscode\FluxPhased\experiments\array_face_s4\learning_repair\ablation_chain.log

echo [%date% %time%] START expA (tx_only, coef=0.003, trunk=ON) >> "%CHAINLOG%"
call _run_s4_seed.bat 20260729 0 400 tx_only 0.003 1 expA_txonly
echo [%date% %time%] DONE expA >> "%CHAINLOG%"

echo [%date% %time%] START expB (average, coef=0.001, trunk=ON) >> "%CHAINLOG%"
call _run_s4_seed.bat 20260729 0 400 average 0.001 1 expB_lowcoef
echo [%date% %time%] DONE expB >> "%CHAINLOG%"

echo [%date% %time%] START expC (average, coef=0.01, trunk=OFF) >> "%CHAINLOG%"
call _run_s4_seed.bat 20260729 0 400 average 0.01 0 expC_notrunk
echo [%date% %time%] DONE expC >> "%CHAINLOG%"
endlocal
