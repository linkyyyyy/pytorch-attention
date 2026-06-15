@echo off
setlocal EnableDelayedExpansion

REM ============================================================================
REM run_cpu_int8_sweep.bat — CPU-INT8 methodological control (Option A: 21 configs)
REM
REM PURPOSE: precision control only — NOT a deployment path.
REM   precision_ratio   = cpu_FP32 / cpu_INT8   (same silicon)
REM   architecture_ratio = cpu_INT8 / npu_INT8  (matched QDQ graph)
REM
REM Reuses pre-built onnx_graphs\*_xint8.onnx — does NOT call Quark.
REM Does NOT touch original session 20260612_143854 artifacts.
REM
REM Run from a CLEAN standalone terminal (IDE/browser closed) on the tower.
REM ============================================================================

call conda activate ryzen-ai-1.6.0
if errorlevel 1 (
    echo [FAIL] Could not activate conda environment ryzen-ai-1.6.0
    exit /b 1
)

cd /d "%~dp0"

set PY=C:\ProgramData\miniconda3\envs\ryzen-ai-1.6.0\python.exe
set UPROF_CLI=AMDuProfCLI.exe
set DURATION=30
set WARMUP=5
set REPEATS=5
set COOLDOWN=5
set CORNER=avg

for /f %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set SESSION_TS=%%T
set SESSION_ID=%SESSION_TS%_cpu_int8
set UPROF_DIR=results\uprof\%SESSION_ID%
set OUTFILE=results\runs_%SESSION_ID%.csv
set PLAN=results\sweep_plan_%SESSION_ID%.csv
set ENRICHED=results\runs_%SESSION_ID%_enriched.csv

if not exist results mkdir results
if not exist results\uprof mkdir results\uprof
mkdir "%UPROF_DIR%"

echo [SESSION] %SESSION_ID%
echo [SESSION] CSV=%OUTFILE%
echo [SESSION] uProf=%UPROF_DIR%
echo [SESSION] corner=%CORNER%  duration=%DURATION%s  warmup=%WARMUP%s  repeats=%REPEATS%
echo.

echo %SESSION_ID%>results\last_cpu_int8_session.txt
echo %OUTFILE%>>results\last_cpu_int8_session.txt
echo %UPROF_DIR%>>results\last_cpu_int8_session.txt

echo [PLAN] Generating run matrix from registry...
%PY% operators.py --sweep-plan --corner %CORNER% --plan-out %PLAN%
if errorlevel 1 exit /b 1

echo.
echo ========================================================================
echo [PLAN] Option A: all operators in corner=%CORNER% plan x engine=cpu_int8
echo [PLAN] Expect 21 measure configs x %REPEATS% repeats + idle_cpu +
echo        dispatch_baseline_cpu_int8 ^(distinct from original dispatch_cpu^)
echo [PLAN] XINT8 graphs: onnx_graphs\*_xint8.onnx ^(no re-quantize^)
echo ========================================================================
echo Ctrl+C to abort, or press any key to start measurement...
pause >nul

set /a RUN_FAILS=0

echo.
echo [BASELINE] idle_cpu...
%UPROF_CLI% timechart --event power --interval 100 -o "%UPROF_DIR%\idle_cpu" "%PY%" harness.py --engine cpu --mode idle --duration %DURATION% --warmup %WARMUP% --repeats 1 --run-id idle_cpu --outfile %OUTFILE%
if errorlevel 1 set /a RUN_FAILS+=1
timeout /t %COOLDOWN% /nobreak >nul

echo [BASELINE] dispatch_baseline_cpu_int8...
%UPROF_CLI% timechart --event power --interval 100 -o "%UPROF_DIR%\dispatch_baseline_cpu_int8" "%PY%" harness.py --engine cpu_int8 --mode dispatch --duration %DURATION% --warmup %WARMUP% --repeats 1 --run-id dispatch_baseline_cpu_int8 --outfile %OUTFILE%
if errorlevel 1 set /a RUN_FAILS+=1
timeout /t %COOLDOWN% /nobreak >nul

echo.
echo [SWEEP] cpu_int8 measure matrix...
%PY% run_plan.py --plan %PLAN% --uprof-dir "%UPROF_DIR%" --outfile %OUTFILE% --python "%PY%" --uprof-cli %UPROF_CLI% --engines cpu_int8 --duration %DURATION% --warmup %WARMUP% --repeats %REPEATS% --cooldown %COOLDOWN%
if errorlevel 1 set /a RUN_FAILS+=1

echo.
if !RUN_FAILS! GTR 0 echo [WARN] !RUN_FAILS! stage(s) returned non-zero exit code
if !RUN_FAILS! EQU 0 echo [OK] Sweep completed without exit errors
echo [DONE] CSV:      %OUTFILE%
echo [DONE] uProf dir: %UPROF_DIR%
echo.
echo --- Post-process ^(run after sweep; separate enriched CSV^) ---
echo %PY% parse_energy.py --runs %OUTFILE% --uprof-dir %UPROF_DIR% --outfile %ENRICHED% --plot
echo %PY% analysis.py --runs %ENRICHED%
echo.
echo --- Step 2 idle floor check ---
echo Compare parse_energy [idle_mean_J] to original session 216.22 J
echo If delta ^> 2-3%% STOP before decomposition ratios.
echo.
echo --- Original session ^(unchanged^) ---
echo results\runs_20260612_143854.csv
echo results\uprof\20260612_143854\

endlocal
