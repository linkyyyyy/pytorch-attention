@echo off
setlocal EnableDelayedExpansion

REM ============================================================================
REM run_sweep.bat — Operator x engine sweep (registry-driven avg corner)
REM Standard window: 5 s warmup, 30 s measured loop, 5 repeats per harness call
REM ============================================================================

call conda activate ryzen-ai-1.6.0
if errorlevel 1 (
    echo [FAIL] Could not activate conda environment ryzen-ai-1.6.0
    exit /b 1
)

cd /d "%~dp0"

set PY=C:\ProgramData\miniconda3\envs\ryzen-ai-1.6.0\python.exe
set DURATION=30
set WARMUP=5
set REPEATS=5
set DEVICE_ID=0
set CORNER=avg
set ENGINES=cpu igpu npu
set OUTFILE=results\runs.csv
set UPROF_CLI=AMDuProfCLI.exe
set IGPU_VGM_MB=512
set BENCHMARK_IGPU_VGM_MB=%IGPU_VGM_MB%

for /f %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set SESSION_TS=%%T
set UPROF_SESSION_DIR=results\uprof\%SESSION_TS%
set PLAN=results\sweep_plan_%SESSION_TS%.csv

echo [SESSION] timestamp=%SESSION_TS%
echo [SESSION] uProf output dir=%UPROF_SESSION_DIR%

if not exist results mkdir results
if not exist results\uprof mkdir results\uprof
if not exist "%UPROF_SESSION_DIR%" mkdir "%UPROF_SESSION_DIR%"

echo.
echo [PLAN] Generating run matrix from registry...
%PY% operators.py --sweep-plan --corner %CORNER% --plan-out %PLAN%
if errorlevel 1 exit /b 1

echo.
echo Review the plan above. Ctrl+C to abort, or press any key to start sweep...
pause >nul

echo [SWEEP] corner=%CORNER% engines=%ENGINES% repeats=%REPEATS% duration=%DURATION%s warmup=%WARMUP%s

for /f "usebackq skip=1 tokens=1-8 delims=|" %%a in ("%PLAN%") do (
    for %%E in (%ENGINES%) do (
        echo.|findstr /C:"%%E" "%%h" >nul 2>&1
        if not errorlevel 1 (
            echo [SKIP] %%a shape_index=%%b engine=%%E ^(skip_engines=%%h^)
        ) else (
            set RUN_BASE=%%a_%%E_s%%b
            echo.
            echo [RUN] !RUN_BASE! repeats=%REPEATS% session=%SESSION_TS%
            %UPROF_CLI% timechart --event power --interval 100 -o "%UPROF_SESSION_DIR%\!RUN_BASE!" "%PY%" harness.py --operator %%a --engine %%E --duration %DURATION% --warmup %WARMUP% --repeats %REPEATS% --device-id %DEVICE_ID% --mode measure --shape-index %%b --tier %%e --block-id %%c --shape-class %%d --run-id !RUN_BASE! --outfile %OUTFILE%
            if errorlevel 1 (
                echo [WARN] run !RUN_BASE! returned non-zero exit code
            )
            timeout /t 5 /nobreak >nul
        )
    )
)

echo.
echo [SWEEP] complete. Session uProf dir: %UPROF_SESSION_DIR%
echo [SWEEP] CSV: %OUTFILE%
echo.
echo --- Baselines: run once per session before or after sweep (see run_session.bat) ---
echo %UPROF_CLI% timechart --event power --interval 100 -o "%UPROF_SESSION_DIR%\idle_cpu" "%PY%" harness.py --mode idle --engine cpu --duration %DURATION% --warmup %WARMUP% --repeats 1 --run-id idle_cpu --outfile %OUTFILE%
echo %UPROF_CLI% timechart --event power --interval 100 -o "%UPROF_SESSION_DIR%\dispatch_baseline_cpu" "%PY%" harness.py --mode dispatch --engine cpu --duration %DURATION% --warmup %WARMUP% --repeats 1 --run-id dispatch_baseline_cpu --outfile %OUTFILE%
echo --- Post-process ---
echo %PY% parse_energy.py --runs %OUTFILE% --uprof-dir %UPROF_SESSION_DIR% --plot
echo %PY% analysis.py --runs results\runs_enriched.csv

endlocal
