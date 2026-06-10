# Benchmark Code Review Guide

A reading companion for `operators.py`, `harness.py`, and `run_sweep.bat` before you confirm end-to-end runs on the HX 370 tower.

**Related docs:** `BENCHMARK_WORKFLOW.md` (how to run), `directives/measurement_harness_spec.md` (protocol).

---

## 1. Big picture

```
operators.py          harness.py              run_sweep.bat
     |                     |                        |
  Build ONNX          Drive timed window         Loop operators
  graphs +            (warmup + measure)         x engines x repeats
  registry            Log CSV markers            uProf wraps harness
     |                     |                        |
     v                     v                        v
onnx_graphs/*.onnx    results/runs.csv         results/upprof/<SESSION>/
```

**Energy is NOT computed in Python.** uProf measures power; harness prints `WINDOW_OPEN` / `WINDOW_CLOSE` epoch timestamps for alignment. Post-run:

```
energy_per_op = (window_energy - dispatch_energy) / iterations
```

Idle baseline (`--mode idle`) captures static floor separately.

---

## 2. `operators.py` — graph factory & registry

### Purpose

Defines 16 transformer operators as minimal single-op ONNX graphs, exports them to `onnx_graphs/`, and returns feed metadata for the harness.

### Key constants

| Name | Value | Notes |
|---|---|---|
| `OPSET` | `20` | Used by `onnx.helper` builders |
| `FALLBACK_OPSET` | `19` | Only for `torch.onnx.export` retry |
| `H`, `N`, `HEAD_DIM` | 12, 197, 64 | Batched MHA shapes (ViT-B/16) |
| `ONNX_GRAPHS_DIR` | `benchmark/onnx_graphs/` | Gitignored `*.onnx` |
| `RNG` | seed `0` | Deterministic numpy feeds |

### Data structures

**`OperatorEntry`** (dataclass) — one registry row:

| Field | Meaning |
|---|---|
| `name` | Registry key (snake_case) |
| `build_fn(path, profile, opset) -> dict` | Writes ONNX file; returns `{"feeds": {...}, "output_names": [...]}` |
| `shape_profiles` | **List** of shape dicts; harness uses `--shape-index` (default 0) |
| `dtype` | `"float32"` |
| `cluster` | `"A"`, `"B1"`, or `"B2"` |

**Shape profile dict:**

```python
{
    "label": "vit_mha_qkt",           # human tag
    "input_shape": "12x197x64@...",   # CSV string
    "tensors": {"a_shape": [...], ...}  # passed to build_fn
}
```

### Function reference

#### Path & export helpers

| Function | What it does |
|---|---|
| `_onnx_path(name, shape_index)` | `name.onnx` if index 0; else `name_s{idx}.onnx` |
| `export_torch_module(...)` | `torch.onnx.export` at opset 20; on failure warns and retries opset 19. Returns actual opset used. |
| `_batched_matmul_output_shape(a, b)` | Computes MatMul output dims for batched 3D tensors |

#### ONNX helper builders (direct `onnx.helper`, no PyTorch)

| Function | ONNX op | Notable |
|---|---|---|
| `build_onnx_matmul_graph` | `MatMul` | Batched MHA matmuls; explicit output shape for checker |
| `build_onnx_softmax_graph` | `Softmax` | Axis from profile |
| `build_onnx_residual_add_graph` | `Add` | Two runtime inputs |
| `build_onnx_gelu_graph` | `Gelu` | **Single node** — bypasses PyTorch decomposition |
| `build_dispatch_baseline_graph` | `Add(input, const_one)` | **Runtime input + initializer** — survives `ORT_ENABLE_ALL` |

#### Per-operator `_build_*` functions

| Function | Build method | ONNX result | Default shape |
|---|---|---|---|
| `_build_patch_embed` | PyTorch Conv2d | Conv | `[1,3,224,224]` → 768×14×14 |
| `_build_downsample_conv` | PyTorch Conv2d | Conv | `[1,768,56,56]` stride-2 |
| `_build_ffn_gemm` | PyTorch Linear **bias=True** | **Gemm** | `[197,768]→[197,3072]` |
| `_build_gelu` | `build_onnx_gelu_graph` | Gelu | `[197,3072]` |
| `_build_layer_norm` | PyTorch LayerNorm | LayerNormalization | `[197,768]` |
| `_build_group_norm` | PyTorch GroupNorm **groups=1** | GroupNorm | `[1,768,14,14]` PoolFormer spec |
| `_build_batch_norm` | PyTorch BatchNorm2d | BatchNormalization | `[1,256,56,56]` |
| `_build_residual_add` | helper Add | Add | `[197,768]` + `[197,768]` |
| `_build_qkv_proj` | PyTorch Linear **bias=True** | **Gemm** | `[197,768]→[197,768]` |
| `_build_attn_score_matmul` | helper MatMul | MatMul | `[12,197,64]@[12,64,197]` |
| `_build_xcit_cov_matmul` | helper MatMul | MatMul | `[12,64,197]@[12,197,64]` |
| `_build_softmax` | helper Softmax | Softmax | profile 0: `[12,197,197]`; profile 1: `[12,64,64]` |
| `_build_attn_value_matmul` | helper MatMul | MatMul | `[12,197,197]@[12,197,64]` |
| `_build_sra_conv` | PyTorch Conv2d 8×8 stride 8 | Conv | `[1,768,56,56]` — **large .onnx (~144 MB)** |
| `_build_avg_pool` | PyTorch AvgPool2d | AveragePool | `[1,768,14,14]` |
| `_build_depthwise_conv` | PyTorch depthwise Conv | Conv | `[1,768,14,14]` groups=768 |

#### Registry API (used by harness)

| Function | What it does |
|---|---|
| `list_operators()` | All 16 registry keys |
| `get_entry(name)` | `OperatorEntry` or `KeyError` |
| `get_shape_profile(name, idx)` | One profile dict |
| `build_operator_graph(name, shape_index=0)` | Build/load ONNX; returns `(path, meta, entry)` |
| `export_all_graphs(all_shapes=False)` | CLI entry: export all ops + `dispatch_baseline.onnx` |

### Registry summary (16 operators)

| Cluster | Operators |
|---|---|
| **A** | `patch_embed_conv2d`, `downsample_conv2d`, `ffn_gemm`, `gelu`, `layer_norm`, `group_norm`, `batch_norm`, `residual_add` |
| **B1** | `qkv_proj_gemm`, `attn_score_matmul`, `xcit_cov_matmul`, `softmax`, `attn_value_matmul`, `sra_conv2d` |
| **B2** | `avg_pool_token_mixer`, `depthwise_conv2d` |

### Notable review points (`operators.py`)

1. **OpSet in CSV vs export:** Helper-built graphs use opset 20. PyTorch exports may silently land on 19 — harness reads **actual** opset from the `.onnx` file, not the `OPSET` constant.
2. **`sra_conv2d.onnx` is huge** — full 768×768 conv weights embedded. Regenerate locally; not in git.
3. **`softmax` has 2 profiles** — index 1 is XCiT `[12,64,64]`; sweep uses index 0 only.
4. **`dispatch_baseline.onnx`** — verify in Netron: one `Add`, inputs `input` (runtime) + `const_one` (initializer).
5. **Run export before first harness run:** `python operators.py` (add `--all-shapes` for `softmax_s1.onnx`).

---

## 3. `harness.py` — measurement loop

### Purpose

ONE shared loop for all operators, engines, and modes. Drives a duration-based inference window and appends timing rows to CSV. Does **not** integrate power.

### Key constants

| Name | Value | Logged to |
|---|---|---|
| `INTRA_OP_NUM_THREADS` | `12` | CSV, notes, metadata |
| `GRAPH_OPTIMIZATION_LEVEL_NAME` | `ORT_ENABLE_ALL` | CSV, notes |
| `COOLDOWN_S` | `5.0` | Between repeats |
| `DEFAULT_OUTFILE` | `results/runs.csv` | |
| `METADATA_PATH` | `results/metadata.json` | Written once per session |

### CLI flags (`parse_args`)

| Flag | Default | Purpose |
|---|---|---|
| `--operator` | — | Registry key (`measure` mode) |
| `--engine` | required | `cpu` or `igpu` |
| `--duration` | `30` | Timed window (seconds) |
| `--warmup` | `5` | Discarded warmup (seconds) |
| `--repeats` | `5` | Harness-internal repeats (sweep passes `1`) |
| `--device-id` | `0` | DML device (890M = 0) |
| `--mode` | `measure` | `measure` \| `idle` \| `dispatch` |
| `--shape-index` | `0` | `shape_profiles[n]` |
| `--outfile` | `results/runs.csv` | Append target |
| `--run-id` | `""` | Prefix for row IDs (set by batch file) |

### Function reference

#### Setup & metadata

| Function | What it does |
|---|---|
| `_igpu_vgm_mb()` | Reads `BENCHMARK_IGPU_VGM_MB` env (default `512`) |
| `_ensure_dispatch_baseline()` | Builds `dispatch_baseline.onnx` if missing |
| `_read_graph_opset(onnx_path)` | Reads real opset from ONNX file for CSV |
| `_write_metadata_once()` | Creates `metadata.json` on first run (ORT version, providers, constants) |

#### ORT session layer

| Function | What it does |
|---|---|
| `_make_session_options(engine, profiling)` | `ORT_ENABLE_ALL`, threads=12; DML: `mem_pattern=False`, `SEQUENTIAL`; optional profiling |
| `_create_session(path, engine, device_id, profiling)` | `CPUExecutionProvider` or `DmlExecutionProvider` |
| `_check_ep_placement(sess, engine)` | Parses profiling JSON; checks `item["args"]["provider"]`; **deletes** prof file after read |

#### iGPU I/O path (critical for measurement validity)

| Function | What it does |
|---|---|
| `probe_dml_ortvalue(device_id)` | Tests `OrtValue.ortvalue_from_numpy(arr, "dml", id)`; cached per process |
| `igpu_use_iobinding(device_id)` | Alias for probe — if False, fall back to `sess.run(feeds)` |
| `_create_io_binding(...)` | Binds inputs/outputs once. CPU: `bind_cpu_input`. iGPU: DML OrtValues. Returns `None` if iGPU fallback. |
| `_run_inference_iteration(...)` | **Core per-iteration call.** IOBinding path: `run_with_iobinding` + **`synchronize_outputs()`**. Fallback: `sess.run`. |

#### Measurement loop

| Function | What it does |
|---|---|
| `_run_profile_check(...)` | **Separate profiled session**, 1 iteration, **before warmup**. Never enters timed window. |
| `_duration_loop(execute_fn, duration_s)` | Loop until wall clock elapsed; count iterations |
| `_append_csv_row(outfile, row)` | Append one CSV row; write header if new file |
| `_run_repeat(...)` | **One repeat:** warmup loop → `WINDOW_OPEN` → measure loop → `WINDOW_CLOSE` → CSV row |

#### Entry points

| Function | What it does |
|---|---|
| `run_harness(args)` | Main orchestration (see flow below) |
| `parse_args()` | Argparse |
| `main()` | Banner + `run_harness` |

### `run_harness` flow (per invocation)

```
1. _write_metadata_once()
2. Resolve mode:
     measure  → build_operator_graph(), read csv_opset from .onnx
     dispatch → dispatch_baseline.onnx
     idle     → no ORT session
3. FOR repeat_idx in 0..repeats-1:
     a. IF not idle AND repeat_idx==0:
          _run_profile_check()  → sess_profile, 1 iter, EP check, prof file deleted
     b. sess_measure = _create_session(profiling=OFF)
     c. io_binding = _create_io_binding()  (or None if iGPU fallback)
     d. _run_repeat():
          - execute() used for BOTH warmup and timed window
          - idle: sleep(0.001)
          - else: _run_inference_iteration() with sync on IOBinding path
          - print WINDOW_OPEN (time.time epoch) / WINDOW_CLOSE
          - append CSV row (energy columns empty)
     e. cooldown 5s between repeats
```

### Console markers to look for

| Marker | Meaning |
|---|---|
| `[PROFILE_SESSION]` | EP placement check (profiling ON, 1 iter) |
| `[MEASURE_SESSION]` | Timed session (profiling OFF) |
| `[DML_IO]` | OrtValue probe result / inference path |
| `[EP_CHECK]` | `EP_OK` or `EP_FALLBACK` in notes |
| `[WARMUP_START]` / `[WARMUP_END]` | Discarded warmup window |
| `[WINDOW_OPEN] t_start=...` | **uProf alignment start** (epoch seconds) |
| `[WINDOW_CLOSE] t_end=... iterations=...` | **uProf alignment end** |
| `[COOLDOWN]` | 5s sleep between repeats |

### Three modes

| Mode | ORT? | Operator CSV field | Purpose |
|---|---|---|---|
| `measure` | Yes | Registry name | Actual operator energy window |
| `idle` | No (sleep) | `idle` | Static power floor |
| `dispatch` | Yes | `dispatch_baseline` | Launch overhead floor (non-elidable Add) |

### CSV columns filled by harness now

`run_id`, `operator`, `cluster`, `tier`, `block_id`, `shape_class`, `engine`, `device_id`, `shape_index`, `input_shape`, `dtype`, **`opset` (from file)**, `intra_op_num_threads`, `graph_optimization_level`, `igpu_vgm_mb`, `repeat_idx`, `warmup_s`, `window_s`, `iterations_completed`, `wall_time_s`, `t_start_epoch`, `t_end_epoch`, `mean_latency_ms`, `notes`

**Left empty for post-uProf analysis:** `idle_power_w`, `active_power_w`, `window_energy_J`, `idle_energy_J`, `energy_per_op_J`, `dispatch_energy_J`

### Notable review points (`harness.py`)

1. **Two sessions per run** — profiling never pollutes warmup/measure loops (`run_sweep.bat` always `--repeats 1`, so profile check runs every sweep iteration).
2. **`synchronize_outputs()`** — required on iGPU IOBinding path; without it iterations are over-counted (async DML queue).
3. **iGPU fallback** — if `"dml"` OrtValue fails, uses `sess.run(feeds)` (blocking). Probe: `python dml_ortvalue_smoke_test.py`.
4. **EP JSON path** — provider is at `item["args"]["provider"]`, not top-level.
5. **Profiling files deleted** — `onnxruntime_profile__*.json` removed after EP check.
6. **Energy columns intentionally blank** — filled offline after uProf alignment.

---

## 4. `run_sweep.bat` — sweep orchestrator

### Purpose

Activates conda, creates session directories, loops **16 operators × 2 engines × 5 repeats**, wraps each harness invocation with uProf as parent process.

### Sections (top to bottom)

| Lines | Section |
|---|---|
| 1–11 | Header comments (energy formula) |
| 13–17 | `conda activate ryzen-ai-1.6.0` — **must succeed** |
| 19 | `cd /d "%~dp0"` — run from `benchmark/` |
| 21–29 | Config: `DURATION=30`, `WARMUP=5`, `REPEATS=5`, `DEVICE_ID=0`, `IGPU_VGM_MB=512` |
| 31–44 | Session timestamp via PowerShell `Get-Date`; create `results\upprof\<SESSION_TS>\` |
| 46–59 | **TODO blocks:** uProf flags + clock-base verification |
| 61–62 | Operator list (16) and engines (`cpu igpu`) |
| 68–88 | Triple nested `for` loop with uProf parent-wrap |
| 90–106 | Echo baseline templates (not executed) |

### Nested loop structure

```bat
set /a LAST_R=%REPEATS% - 1    REM = 4 when REPEATS=5

for %%O in (16 operators) do
  for %%E in (cpu igpu) do
    for /L %%R in (0,1,!LAST_R!) do
      RUN_ID = %%O_%%E_r%%R        REM e.g. ffn_gemm_cpu_r0
      AMDuProfCLI timechart ... -- python harness.py ... --repeats 1 --run-id !RUN_ID!
      timeout 5s cooldown
```

**Total runs per sweep:** 16 × 2 × 5 = **160** harness invocations.

### uProf parent-wrap pattern

```bat
%UPROF_CLI% timechart --output "...\RUN_ID.csv" -- python harness.py ...
```

Harness runs as **child** of `AMDuProfCLI`. Collection lifetime = harness process lifetime.

**Still TODO on tower:** insert verified `timechart` flags before `--output`; confirm `--` child-launch syntax via `AMDuProfCLI.exe timechart --help`.

### Environment variables set

| Variable | Source | Consumed by |
|---|---|---|
| `BENCHMARK_IGPU_VGM_MB` | `IGPU_VGM_MB` in batch | `harness._igpu_vgm_mb()` |

### Baseline templates (echoed, not run)

| Template | Mode | Engine |
|---|---|---|
| `idle_cpu_r0` | `idle` | cpu |
| `idle_igpu_r0` | `idle` | igpu |
| `dispatch_cpu_r0` | `dispatch` | cpu |
| `dispatch_igpu_r0` | `dispatch` | igpu |

Run once per engine per session **before** analyzing operator energy.

### Notable review points (`run_sweep.bat`)

1. **`EnableDelayedExpansion`** required for `!RUN_ID!`, `!LAST_R!` inside nested loops.
2. **Never `%date%%time%`** in filenames — uses `SESSION_TS` folder + simple `op_engine_rN` IDs.
3. **`mkdir` before loops** — prevents uProf write failures.
4. **`--repeats 1`** per uProf bracket — harness handles one timed window per uProf trace.
5. **`--shape-index 0` only** — XCiT softmax profile not in sweep yet.
6. **uProf flags are placeholder** — current line may fail on tower until flags filled in.
7. **Loop order:** operator → engine → repeat (not interleaved). Consider manual reorder for thermal fairness.

---

## 5. How the three files connect

```
run_sweep.bat
    │
    ├─ conda activate ryzen-ai-1.6.0
    ├─ set BENCHMARK_IGPU_VGM_MB
    │
    └─ AMDuProfCLI ... -- python harness.py
                              │
                              ├─ import operators
                              ├─ build_operator_graph("ffn_gemm", 0)
                              │       └─ reads onnx_graphs/ffn_gemm.onnx
                              │
                              ├─ _run_profile_check()  [1 iter, profiling ON]
                              ├─ _create_session(profiling OFF)
                              ├─ _create_io_binding() or sess.run fallback
                              │
                              └─ warmup (5s) + WINDOW_OPEN + measure (30s) + WINDOW_CLOSE
                                      └─ append results/runs.csv
```

---

## 6. Pre-flight review checklist

### One-time setup

- [ ] `conda activate ryzen-ai-1.6.0`
- [ ] `python operators.py` — graphs in `onnx_graphs/`
- [ ] Netron: `gelu.onnx` = 1× Gelu; `dispatch_baseline.onnx` = Add(input, const)
- [ ] `python dml_ortvalue_smoke_test.py` — confirms IOBinding+dml path
- [ ] Edit `IGPU_VGM_MB` in `run_sweep.bat` to match BIOS

### Smoke tests (no uProf)

```cmd
python harness.py --operator ffn_gemm --engine cpu --duration 5 --warmup 2 --repeats 1 --run-id review_cpu
python harness.py --operator ffn_gemm --engine igpu --duration 5 --warmup 2 --repeats 1 --run-id review_igpu
python harness.py --mode dispatch --engine igpu --duration 5 --warmup 2 --repeats 1 --run-id review_dispatch
python harness.py --mode idle --engine cpu --duration 5 --warmup 2 --repeats 1 --run-id review_idle
```

### What to verify in smoke output

- [ ] `[PROFILE_SESSION]` before `[WARMUP_START]`
- [ ] `[MEASURE_SESSION]` with profiling off
- [ ] `[WINDOW_OPEN]` / `[WINDOW_CLOSE]` with sensible `iterations`
- [ ] iGPU: `[DML_IO] inference path=iobinding+dml`
- [ ] `[EP_CHECK] EP_OK: intended=...` (or documented fallback)
- [ ] Row in `results/runs.csv` with correct `opset` column
- [ ] No leftover `onnxruntime_profile__*.json` in `benchmark/`

### Before full sweep

- [ ] Fill uProf `timechart` flags in `run_sweep.bat`
- [ ] Verify uProf CSV timestamp base (absolute vs elapsed) — see TODO in batch file
- [ ] Run from **standalone CMD**, not Cursor terminal
- [ ] Close background apps; tower on AC

---

## 7. Auxiliary files (not covered above)

| File | Purpose |
|---|---|
| `dml_ortvalue_smoke_test.py` | Probes `OrtValue(..., "dml", 0)` on your ORT build |
| `BENCHMARK_WORKFLOW.md` | Step-by-step CMD workflow |
| `IMPLEMENTATION_BLUEPRINT.md` | Design history / decisions |
| `directives/measurement_harness_spec.md` | Authoritative protocol |

---

*Generated for pre-run code review. Update this file if function signatures or flow change materially.*
