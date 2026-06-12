@echo off
setlocal EnableDelayedExpansion

REM ============================================================================
REM run_session.bat — baselines + production-corner measure sweep
REM Standard window: 5 s warmup, 30 s measured loop
REM Shape selection: operators.py --sweep-plan (registry-driven avg corner)
REM ============================================================================

call conda activate ryzen-ai-1.6.0
if errorlevel 1 (
    echo [FAIL] conda env ryzen-ai-1.6.0
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
set ENGINES=cpu igpu npu
if not defined SKIP_PREP set SKIP_PREP=0

for /f %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set SESSION_TS=%%T
set UPROF_DIR=results\uprof\%SESSION_TS%
set OUTFILE=results\runs_%SESSION_TS%.csv
set PLAN=results\sweep_plan_%SESSION_TS%.csv

if not exist results mkdir results
if not exist results\uprof mkdir results\uprof
mkdir "%UPROF_DIR%"

echo [SESSION] %SESSION_TS%
echo [SESSION] CSV=%OUTFILE%  corner=%CORNER%  duration=%DURATION%s warmup=%WARMUP%s

echo.
echo [PLAN] Generating run matrix from registry...
%PY% operators.py --sweep-plan --corner %CORNER% --plan-out %PLAN%
if errorlevel 1 exit /b 1

echo.
echo ========================================================================
echo [PLAN] corner=%CORNER% ^(registry avg = N=197 SDPA; single-profile ops = index 0^)
echo [PLAN] Matrix above: operator x engine x shape_index x input_shape
echo [PLAN] Eyeball N=197 on SDPA/GEMM rows before continuing.
echo ========================================================================
echo Ctrl+C to abort, or press any key to start measurement...
pause >nul

set /a RUN_FAILS=0

if "%SKIP_PREP%"=="1" (
    echo.
    echo [PREP] skipped ^(SKIP_PREP=1^)
) else (
    echo.
    echo [PREP] NPU XINT8 quantize all operators...
    %PY% operators.py --quantize-xint8-all --sweep-corner-quantize --corner %CORNER%
    if errorlevel 1 exit /b 1
)

echo.
echo [BASELINE] idle...
%UPROF_CLI% timechart --event power --interval 100 -o "%UPROF_DIR%\idle_cpu" "%PY%" harness.py --engine cpu --mode idle --duration %DURATION% --warmup %WARMUP% --repeats 1 --run-id idle_cpu --outfile %OUTFILE%
if errorlevel 1 set /a RUN_FAILS+=1

for %%E in (%ENGINES%) do (
    echo [BASELINE] dispatch %%E...
    %UPROF_CLI% timechart --event power --interval 100 -o "%UPROF_DIR%\dispatch_baseline_%%E" "%PY%" harness.py --engine %%E --mode dispatch --duration %DURATION% --warmup %WARMUP% --repeats 1 --run-id dispatch_baseline_%%E --outfile %OUTFILE%
    if errorlevel 1 set /a RUN_FAILS+=1
    timeout /t %COOLDOWN% /nobreak >nul
)

%PY% run_plan.py --plan %PLAN% --uprof-dir "%UPROF_DIR%" --outfile %OUTFILE% --python "%PY%" --uprof-cli %UPROF_CLI% --engines cpu,igpu,npu --duration %DURATION% --warmup %WARMUP% --repeats %REPEATS% --cooldown %COOLDOWN%
if errorlevel 1 set /a RUN_FAILS+=1

echo.
if !RUN_FAILS! GTR 0 echo [WARN] !RUN_FAILS! runs returned non-zero exit code
if !RUN_FAILS! EQU 0 echo [OK] All runs completed without exit errors
echo [DONE] CSV:      %OUTFILE%
echo [DONE] uProf dir: %UPROF_DIR%
echo %OUTFILE%>results\last_session_csv.txt
echo %UPROF_DIR%>results\last_session_uprof.txt
echo.
echo --- Post-process ---
echo %PY% parse_energy.py --runs %OUTFILE% --uprof-dir %UPROF_DIR% --plot
echo %PY% analysis.py --runs results\runs_enriched.csv

endlocal

