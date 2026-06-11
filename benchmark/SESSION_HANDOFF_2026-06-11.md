# Benchmark Session Handoff — 2026-06-11

Portable summary of today's harness / energy-pipeline work on the Ryzen AI 9 HX 370 tower. Feed this to Claude (or any agent) at the start of the next session.

---

## 1. What we built / fixed today

### Energy methodology (code — already merged)

| Component | Change |
|-----------|--------|
| **`harness.py`** | Idle is a real timed window (sleep loop, no ORT). `resolve_run_id()` appends `_r{repeat}` once; strips trailing `_r\d+` from `--run-id`. |
| **`parse_energy.py`** | Joins uProf → `window_energy_J`. Mean-subtract baselines: `idle_energy_J` = mean over **all** idle captures; `dispatch_energy_J` = mean per engine. No pre-subtraction. |
| **`analysis.py`** | Headline: `energy_per_op_J` = idle-subtracted. Secondary: `energy_per_op_dispatch`. `fillna` guards for missing baselines (0 subtract + `[WARN]`, no crash). |
| **`metadata.json` / `PROJECT_CONTEXT.md`** | Documented mean-baseline rationale, run-id suffix rule, idle headline metric. |

### Orchestration scripts

| Script | Purpose |
|--------|---------|
| **`run_session.bat`** | One session: NPU quantize prep → idle + dispatch baselines → all operators × cpu/igpu/npu → session CSV + uProf folder. |
| **`run_sweep.bat`** | Repeatable op×engine sweep with `REPEATS=5`; one uProf wrap per op×engine, harness runs all repeats internally. |

### Standard timing (both scripts now)

```
WARMUP=5 s     # not included in energy window
DURATION=30 s  # measured window ([WINDOW_OPEN]..[WINDOW_CLOSE])
REPEATS=5      # run_session / run_sweep
COOLDOWN=5 s   # between uProf wraps (run_session)
```

Plumbing session used 5 s / 2 s / 1 repeat temporarily; reverted to standard above.

---

## 2. Pipeline (always this order)

```
1. harness.py (+ AMDuProfCLI wrap)  →  results/runs_<SESSION>.csv
2. parse_energy.py                  →  results/runs_enriched.csv
3. analysis.py --runs runs_enriched.csv  →  results/analysis_out.csv
```

### `runs.csv` vs `runs_enriched.csv`

| File | Who writes it | Energy columns |
|------|---------------|----------------|
| **`runs.csv`** | Harness only | **Empty** — timing (`t_start_epoch`, `t_end_epoch`, `iterations_completed`) only |
| **`runs_enriched.csv`** | `parse_energy.py` | **`window_energy_J`**, **`idle_energy_J`**, **`dispatch_energy_J`** filled from uProf |

**Never run `analysis.py` on raw `runs.csv`** — you get `count=0` and NaN.

---

## 3. uProf launch syntax (critical — learned the hard way)

**Correct** (HX 370 confirmed):

```bat
AMDuProfCLI.exe timechart --event power --interval 100 -o "results\uprof\<name>" ^
  "C:\ProgramData\miniconda3\envs\ryzen-ai-1.6.0\python.exe" harness.py --engine cpu ...
```

**Wrong** — causes `ERROR: Launch application (harness.py) is not executable binary`:

```bat
AMDuProfCLI.exe ... -o outdir -- %PY% harness.py ...
```

uProf requires a real `.exe` as the child process. **No `--` before python.exe.**

### CMD batch gotchas

Inside `if (...)` blocks, **never put `(s)` in echo text** — e.g. `run(s)` closes the block and yields `was unexpected at this time`. Use `runs` instead.

---

## 4. Completed plumbing session (2026-06-11 ~17:06)

| Artifact | Path |
|----------|------|
| Session CSV | `results/runs_20260611_165420.csv` (**55 rows**) |
| uProf traces | `results/uprof/20260611_165420/` |
| Operators | 18 × 3 engines + idle + 3 dispatch baselines |

**`attn_block_fused` omitted** — NPU VAI EP crashes with `from_batch_size == to_batch_size (768 vs. 3136)`. Re-enable for SDPA Tier-2 work after isolated ops are stable.

### Post-process commands for that session

```bat
cd benchmark
conda activate ryzen-ai-1.6.0
python parse_energy.py --runs results\runs_20260611_165420.csv --uprof-dir results\uprof\20260611_165420 --plot
python analysis.py --runs results\runs_enriched.csv
```

Expect **~51 measure rows** in the per-operator table with numeric J/op values.

---

## 5. How to run next session

### Full session (baselines + all operators)

```bat
cd C:\Users\FPGA\Documents\pytorch-attention\benchmark
run_session.bat
```

Skip re-quantize if XINT8 graphs already exist:

```bat
set SKIP_PREP=1
run_session.bat
```

Post-process (paths also written to `results/last_session_csv.txt` and `last_session_uprof.txt`):

```bat
python parse_energy.py --runs results\runs_<SESSION>.csv --uprof-dir results\uprof\<SESSION> --plot
python analysis.py --runs results\runs_enriched.csv
```

### Single-op smoke (GEMM, all three engines)

```bat
set PY=C:\ProgramData\miniconda3\envs\ryzen-ai-1.6.0\python.exe

%PY% operators.py --quantize-xint8-all   REM once per session

%PY% harness.py --engine npu --prep-npu-quantize --mode measure --operator ffn_gemm --shape-index 1

AMDuProfCLI.exe timechart --event power --interval 100 -o results\uprof\ffn_gemm_cpu "%PY%" harness.py --engine cpu --mode measure --operator ffn_gemm --shape-index 1 --duration 30 --warmup 5 --repeats 1 --run-id ffn_gemm_cpu --outfile results\runs.csv

REM repeat for igpu, npu; then dispatch_baseline_{cpu,igpu,npu}; then idle_cpu once
```

### Repeat sweep only (no baselines in script)

```bat
run_sweep.bat
```

Then run baselines + post-process manually (templates printed at end of script).

---

## 6. Engines and prep

| Engine | Prep | Notes |
|--------|------|-------|
| **cpu** | None | FP32 ONNX |
| **igpu** | None | FP32, DirectML io_binding |
| **npu** | `operators.py --quantize-xint8-all` or per-op `--prep-npu-quantize` | XINT8 + VitisAI EP; compile before window |

**All three engines** are measured in every session script — not NPU-only.

---

## 7. Analysis metrics

| Metric | Formula | Use |
|--------|---------|-----|
| **Headline** `energy_per_op_J` | `(window_energy_J − idle_energy_J) / iterations` | Cross-engine comparison |
| **Secondary** `energy_per_op_dispatch` | `(window_energy_J − dispatch_energy_J) / iterations` | Per-engine kernel view |
| **Dispatch caveat** | NPU dispatch baseline runs CPU-only | Do not treat dispatch-subtract as NPU-isolated |

Baselines are **mean-subtracted across all captures** (not per-repeat paired) — idle/dispatch run in separate uProf sessions from measure rows.

`WINDOW_ENERGY_IS_RAW=True` — all subtraction in `analysis.py` only.

---

## 8. Production shape selection (registry-driven)

**Do not hardcode `shape_index=1` in batch files.** Use:

```bat
python operators.py --sweep-plan --corner avg
```

`profile_indices_for_corner()` in `operators.py`:

| Operator class | avg-corner rule |
|----------------|-----------------|
| SDPA block ops (ffn, gelu, qkv, …) | all profiles with `shape_class=avg` (qkv → 3 rows: q/k/v) |
| `attn_block_fused` | fused_block profile for avg (index 1) |
| Single-profile conv/pool | index `0` only |
| Missing corner | `[WARN]` skip — never OOR-crash |

`run_session.bat` / `run_sweep.bat` print the full matrix (operator × engine × shape_index × input_shape) and **pause** before any uProf wrap.

Block metadata on measure rows: `--block-id sdpa_avg --shape-class avg` from plan (harness also fills from profile when CLI empty).

### fusion_member column

Harness writes `fusion_member` on every measure row — sourced from shape profile (`fusion_member` key) or `OperatorEntry.fusion_member` default (`False`). Baselines/idle: `n/a`. Enables `analysis.py` fusion-gap without warnings.

---

## 9. Known issues / next steps

1. **Post-process plumbing session `20260611_165420`** — `parse_energy.py` + `analysis.py` still required before trusting any new production sweep numbers.
2. **`attn_block_fused` on NPU** — VAI EP `from_batch_size 768 vs 3136`; sweep plan marks `skip_engines=npu`. Isolated-op headline proceeds; **Tier-2 fusion measurement on NPU blocked** pending VAI investigation.
3. **Quark custom_ops.dll** — `Access is denied` during quantize; quantize still succeeds.
4. **Next production run:** `run_session.bat` (5 s warmup / 30 s window / 5 repeats); review `--sweep-plan` output before continuing past pause.

---

## 10. Key file map

```
benchmark/
  harness.py           # measurement window driver
  parse_energy.py      # uProf join → runs_enriched.csv
  analysis.py          # idle-subtract headline + fusion gap
  operators.py         # registry + --sweep-plan + --quantize-xint8-all
  run_session.bat      # baselines + full operator session
  run_sweep.bat        # op×engine×repeats sweep
  results/
    runs_<SESSION>.csv
    runs_enriched.csv
    uprof/<SESSION>/<run_id>/timechart.csv
    analysis_out.csv
  npu/path.py          # XINT8 + VitisAI session
```

---

## 11. Stopping point

- Plumbing sweep **completed** (last op: `depthwise_conv2d_npu`, session `20260611_165420`).
- Scripts updated to **standard 5 s warmup / 30 s window / 5 repeats**.
- Ready for: post-process plumbing data → review table → schedule full-energy 30 s session when machine time allows.
