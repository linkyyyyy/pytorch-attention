@echo off
setlocal EnableDelayedExpansion

REM Short dry run: 1 repeat, 5s window, registry avg corner (NOT production sweep)

call conda activate ryzen-ai-1.6.0
if errorlevel 1 (
    echo [FAIL] Could not activate conda environment ryzen-ai-1.6.0
    exit /b 1
)

cd /d "%~dp0"

set PY=C:\ProgramData\miniconda3\envs\ryzen-ai-1.6.0\python.exe
set DURATION=5
set WARMUP=5
set REPEATS=1
set DEVICE_ID=0
set CORNER=avg
set ENGINES=cpu igpu npu
set UPROF_CLI=AMDuProfCLI.exe
set IGPU_VGM_MB=512
set BENCHMARK_IGPU_VGM_MB=%IGPU_VGM_MB%

for /f %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set SESSION_TS=%%T
set UPROF_SESSION_DIR=results\uprof\dry_%SESSION_TS%
set PLAN=results\sweep_plan_dry_%SESSION_TS%.csv
set OUTFILE=results\runs_dry_%SESSION_TS%.csv

echo [DRY] session=%SESSION_TS%
echo [DRY] CSV=%OUTFILE%
echo [DRY] uProf=%UPROF_SESSION_DIR%

if not exist results mkdir results
if not exist results\uprof mkdir results\uprof
if not exist "%UPROF_SESSION_DIR%" mkdir "%UPROF_SESSION_DIR%"

echo.
echo [PLAN] Generating run matrix from registry...
%PY% operators.py --sweep-plan --corner %CORNER% --plan-out %PLAN%
if errorlevel 1 exit /b 1

echo.
echo ========================================================================
echo [PLAN] corner=%CORNER% DRY: warmup=%WARMUP%s window=%DURATION%s repeats=%REPEATS%
echo [PLAN] Matrix above: operator x engine x shape_index x input_shape
echo ========================================================================
echo Auto-continue in 3s ^(plan verified separately^)...
choice /C Y /N /T 3 /D Y >nul

echo [DRY-SWEEP] starting measure loop...

%PY% run_plan.py --plan %PLAN% --uprof-dir "%UPROF_SESSION_DIR%" --outfile %OUTFILE% --python "%PY%" --uprof-cli %UPROF_CLI% --engines cpu,igpu,npu --duration %DURATION% --warmup %WARMUP% --repeats %REPEATS% --device-id %DEVICE_ID% --cooldown 2
if errorlevel 1 (
    echo [WARN] run_plan.py reported one or more failed measure runs
)

echo.
echo [DRY-SWEEP] complete CSV=%OUTFILE%
echo %OUTFILE%>results\last_dry_csv.txt
echo %UPROF_SESSION_DIR%>results\last_dry_uprof.txt

endlocal
