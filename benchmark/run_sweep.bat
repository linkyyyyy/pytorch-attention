@echo off
setlocal EnableDelayedExpansion

REM ============================================================================
REM run_sweep.bat — Operator energy sweep orchestrator
REM
REM uProf measures energy; harness.py drives the marked measurement window.
REM Analysis formula (post-uProf alignment):
REM   energy_per_op = (window_energy - dispatch_energy) / iterations
REM Idle baseline captures the static floor separately.
REM ============================================================================

call conda activate ryzen-ai-1.6.0
if errorlevel 1 (
    echo [FAIL] Could not activate conda environment ryzen-ai-1.6.0
    exit /b 1
)

cd /d "%~dp0"

REM --- Configuration (edit IGPU_VGM_MB to match BIOS Variable Graphics Memory) ---
set DURATION=30
set WARMUP=5
set REPEATS=5
set DEVICE_ID=0
set OUTFILE=results\runs.csv
set UPROF_CLI=AMDuProfCLI.exe
set IGPU_VGM_MB=512
set BENCHMARK_IGPU_VGM_MB=%IGPU_VGM_MB%

REM --- ONE locale-safe session timestamp (never use %%date%%%%time%% in filenames) ---
for /f %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set SESSION_TS=%%T
set UPROF_SESSION_DIR=results\upprof\%SESSION_TS%

echo [SESSION] timestamp=%SESSION_TS%
echo [SESSION] uProf output dir=%UPROF_SESSION_DIR%

if not exist results mkdir results
if not exist results\upprof mkdir results\upprof
if not exist "%UPROF_SESSION_DIR%" mkdir "%UPROF_SESSION_DIR%"
if errorlevel 1 (
    echo [FAIL] Could not create uProf session directory: %UPROF_SESSION_DIR%
    exit /b 1
)

REM ============================================================================
REM TODO: UPROF FLAGS — DO NOT GUESS
REM On the HX 370 tower, run:
REM   AMDuProfCLI.exe timechart --help
REM   AMDuProfCLI.exe --help
REM Verify child-launch syntax (standard trailing args: ... -- python harness.py ...)
REM and fill in the timechart flags below.
REM ============================================================================
REM TODO: UPROF CLOCK BASE — verify timestamp base of uProf CSV output.
REM   If timestamps are absolute system time: align directly to harness t_start/t_end.
REM   If elapsed-since-collection-start: record collection start wall-clock epoch
REM   (e.g. immediately before AMDuProfCLI returns) and add offset to uProf times
REM   before matching Python time.time() WINDOW_OPEN/WINDOW_CLOSE markers.
REM ============================================================================

set OPERATORS=patch_embed_conv2d downsample_conv2d ffn_gemm gelu layer_norm group_norm batch_norm residual_add qkv_proj_gemm attn_score_matmul xcit_cov_matmul softmax attn_value_matmul sra_conv2d avg_pool_token_mixer depthwise_conv2d
set ENGINES=cpu igpu

echo [SWEEP] starting operator x engine x repeat sweep
echo [SWEEP] operators=%OPERATORS%
echo [SWEEP] engines=%ENGINES% repeats=%REPEATS% duration=%DURATION%s warmup=%WARMUP%s

set /a LAST_R=%REPEATS% - 1

for %%O in (%OPERATORS%) do (
    for %%E in (%ENGINES%) do (
        for /L %%R in (0,1,!LAST_R!) do (
            set RUN_ID=%%O_%%E_r%%R
            echo.
            echo [RUN] !RUN_ID! session=%SESSION_TS%

            REM uProf parent-wrap: harness runs as child of AMDuProfCLI.
            REM TODO: insert verified timechart flags (from AMDuProfCLI timechart --help) before --output.
            %UPROF_CLI% timechart --output "%UPROF_SESSION_DIR%\!RUN_ID!.csv" -- python harness.py --operator %%O --engine %%E --duration %DURATION% --warmup %WARMUP% --repeats 1 --device-id %DEVICE_ID% --mode measure --shape-index 0 --run-id !RUN_ID! --outfile %OUTFILE%

            if errorlevel 1 (
                echo [WARN] run !RUN_ID! returned non-zero exit code
            )

            timeout /t 5 /nobreak >nul
        )
    )
)

echo.
echo [SWEEP] complete. Session uProf dir: %UPROF_SESSION_DIR%
echo [SWEEP] CSV: %OUTFILE%
echo.
echo --- Baseline templates (uncomment and run once per engine per session) ---
echo REM Idle baseline (static floor):
echo %UPROF_CLI% timechart ... --output "%UPROF_SESSION_DIR%\idle_cpu_r0.csv" -- python harness.py --mode idle --engine cpu --duration %DURATION% --warmup %WARMUP% --repeats 1 --run-id idle_cpu_r0 --outfile %OUTFILE%
echo.
echo REM Dispatch baseline (launch overhead; non-elidable Add graph):
echo %UPROF_CLI% timechart ... --output "%UPROF_SESSION_DIR%\dispatch_igpu_r0.csv" -- python harness.py --mode dispatch --engine igpu --duration %DURATION% --warmup %WARMUP% --repeats 1 --run-id dispatch_igpu_r0 --outfile %OUTFILE%

endlocal
