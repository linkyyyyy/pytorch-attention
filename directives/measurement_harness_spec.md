# Measurement Harness Spec — CPU & iGPU Operator Energy Benchmarking

**Platform:** AMD Ryzen AI 9 HX 370 tower · **OS for this phase:** native Windows (CMD)
**Engines this phase:** Zen 5 CPU (`CPUExecutionProvider`) · Radeon 890M iGPU (`DmlExecutionProvider`)
**Operators:** see `operator_architecture_selection.md` (Clusters A / B1 / B2)
**Headline metric:** energy per operator (Joules), from AMD uProf running as a *separate process*.

> Cursor: you are encouraged to update this file as you build. Keep the protocol section
> authoritative — if you change a default (e.g. window duration), change it here too.

---

## 0. The one idea that drives every design choice

The Python script does **not** measure energy. uProf does, as a separate process.
The script's only jobs are to (a) drive a **clean, sustained, well-marked power window**
and (b) log exactly when that window opened/closed and how many iterations ran, so the
uProf power trace can be aligned to it.

**Headline formula (post-uProf alignment):**

```
energy_per_op = (window_energy − dispatch_energy) ÷ iterations
```

Idle baseline captures the static power floor separately. Dispatch baseline subtracts
runtime launch overhead (non-elidable minimal graph through the same loop).

Latency the loop records is a useful cross-check, never the headline number.

---

## 1. Why this is NOT one file per operator

The execution provider (CPU vs DirectML) is a **runtime argument**, not a reason to split
code. The same operator ONNX graph runs on both engines — you just pass a different provider.
Splitting per operator (or per engine) means copy-pasted measurement loops, and any drift
between copies silently biases results. Validity requires that **every number comes from
identical measurement code.**

Target structure:

```
benchmark/
  harness.py              # the ONE shared measurement loop + argparse + logging + baselines
  operators.py            # registry: name -> (build_fn, shape_profiles[], dtype, cluster)
  onnx_graphs/            # each op exported to .onnx for Netron inspection
  results/
    runs.csv              # one row per (operator x engine x repeat)
    metadata.json         # session header (ORT ver, VGM, threads, opset)
    upprof/<SESSION_TS>/  # uProf CSV per run_id
  run_sweep.bat           # CMD orchestrator: conda activate, uProf parent-wraps harness
  BENCHMARK_WORKFLOW.md   # user CMD workflow guide
```

One graph per operator (per shape profile), one loop for all of them, two engines selected by flag.

---

## 2. Environment

- **Single conda/venv, Windows.** Env name: `ryzen-ai-1.6.0`. Install **`onnxruntime-directml`**
  (NOT plain `onnxruntime` — they conflict). Provides both `CPUExecutionProvider` and
  `DmlExecutionProvider`.
- Verify: `python -c "import onnxruntime as ort; print(ort.get_available_providers())"`
  must list `DmlExecutionProvider`.
- Also: `torch`, `onnx`, `numpy`. PyTorch only needed to *build/export* graphs, not to run them.
- **Export ONNX graphs at opset 20** (global fallback to 19 if Gelu schema unavailable).
  Log chosen opset. Native `Gelu` only at opset 20 — warn if decomposed.
- AMD uProf 5.x installed (CLI: `AMDuProfCLI.exe`). GPU/driver up to date (DirectX 12 required).
- Record **iGPU VGM (Variable Graphics Memory)** BIOS allocation in metadata (placeholder default: 512 MB).

---

## 3. Session options (CPU and DML)

**Shared (all ORT sessions):**

```python
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # logged
so.intra_op_num_threads = 12  # fixed, logged — NOT default all-cores
```

**DirectML EP — mandatory:**

```python
so.enable_mem_pattern = False
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
providers=[("DmlExecutionProvider", {"device_id": 0})]
```

**EP placement check:** `so.enable_profiling = True` on first repeat; parse JSON; record
CPU fallback in `notes`. DML silently falls back unsupported ops to CPU.

---

## 4. The measurement loop (per run)

1. **Warm-up (~5 s, discarded).** DML compiles shaders on first execution.
2. **Open window:** `t_start = time.time()` (epoch). Print `[WINDOW_OPEN]` marker.
3. **Steady-state loop until ~30 s elapsed**, counting iterations (duration-based, never fixed count).
4. **Close window:** `t_end`, `iterations`, mean latency. Print `[WINDOW_CLOSE]` marker.
5. Cooldown ~5 s before next repeat.

---

## 5. Duration, not iteration count

Loop until target wall-clock duration elapses; count iterations as output.

- **Target window: ~30 s steady-state.**
- **Warm-up: ~5 s, discarded.**
- **Repeats: 5 per (operator × engine).** Report mean ± std dev.

---

## 6. Baselines — two of them

- **Idle baseline:** `--mode idle` — same loop with `sleep()`, no inference. Static floor.
- **Dispatch baseline:** `--mode dispatch` — minimal **non-elidable** graph (element-wise `Add`
  on `[1]` tensor: `Add(runtime_input, constant)` — NOT two initializers, or `ORT_ENABLE_ALL`
  constant-folds it). Measures launch/dispatch overhead through the identical loop.

Subtract dispatch energy in the headline formula. Idle baseline for static floor context.

---

## 7. uProf integration (the actual energy source)

uProf runs as parent process; `python harness.py` runs as **child** (`AMDuProfCLI ... -- python harness.py ...`).

- Verify flags against installed 5.x help (`AMDuProfCLI.exe timechart --help`).
- **Timestamp alignment (confirmed):** uProf `timechart.csv` = wall-clock `HH:MM:SS:ms` (local tz);
  harness markers = Unix epoch seconds. **Conversion required** — not the same unit. Prefer parsing
  uProf strings + session date + timezone (Europe/Athens) → epoch; avoid epoch→time-of-day primary.
- **Robust anchor:** log uProf `Profile Start Time` / launch epoch in `run_sweep.bat` so both traces
  share one reference (do not rely on independent clock matching).
- **TODO:** alignment parser — map `WINDOW_OPEN`/`WINDOW_CLOSE` epoch to uProf CSV rows for slicing.
- Integrate power (∫P dt) over aligned window, or read cumulative energy counter difference.
- Output per run to `results/upprof/<SESSION_TS>/<operator>_<engine>_r<N>.csv`.
- Session timestamp: ONE locale-safe value at sweep start (PowerShell `Get-Date -Format`); never `%date%%time%` in filenames.

`run_sweep.bat`: conda activate → create session dir → nested loops (operator × engine × repeat) → uProf-wrap harness per run.

---

## 8. Experimental controls (validity checklist)

- [ ] Tower on AC; single pinned Windows power plan
- [ ] Close background apps; measured runs from clean standalone CMD (not Cursor terminal)
- [ ] Same ORT build for CPU and iGPU (`onnxruntime-directml`)
- [ ] `intra_op_num_threads = 12`, logged
- [ ] `ORT_ENABLE_ALL`, logged
- [ ] VGM BIOS allocation recorded (`igpu_vgm_mb`)
- [ ] Warm-up discarded every run
- [ ] Cooldown between repeats; flag throttling in `notes`
- [ ] Consider interleaving run order (op A cpu, op A igpu, …)

---

## 9. Output schema (one CSV row per run)

```
run_id, operator, cluster, tier, block_id, shape_class, engine, device_id, shape_index,
input_shape, dtype, opset, intra_op_num_threads, graph_optimization_level, igpu_vgm_mb,
repeat_idx, warmup_s, window_s, iterations_completed, wall_time_s, t_start_epoch, t_end_epoch,
mean_latency_ms, idle_power_w, active_power_w, window_energy_J, idle_energy_J,
energy_per_op_J, dispatch_energy_J, notes
```

Energy/power columns filled post-uProf. `cluster` (A / B1 / B2 / baseline) on every row.

**Block-context columns** (between `cluster` and `engine`):

| Column | Measure-mode values | Baseline values |
|---|---|---|
| `tier` | `isolated`, `fused_block` | `n/a` |
| `block_id` | free-form string (e.g. `vit_avg`); empty if not block-scoped | `n/a` |
| `shape_class` | `small`, `avg`, `large`, or empty | `n/a` |

Harness flags: `--tier`, `--block-id`, `--shape-class`. Full schema + example rows in
`benchmark/results/metadata.json` → `csv_schema`.

---

## 10. Operators to cover (from operator_architecture_selection.md)

Registry in `benchmark/operators.py`. Each entry has a **list of shape profiles** (v1 uses index 0).

- **Cluster A:** patch-embed Conv2D, downsampling Conv2D, FFN GEMM, GELU, LayerNorm,
  GroupNorm (`num_groups=1`), BatchNorm, residual Add.
- **Cluster B1:** Q/K/V projection GEMM, batched MHA QKᵀ matmul (`[12,197,64]@[12,64,197]`),
  batched XCiT cross-cov matmul (`[12,64,197]@[12,197,64]`), Softmax (profiles: `[12,197,197]` and
  XCiT `[12,64,64]`), batched attention·V matmul, SRA Conv2D.
- **Cluster B2:** average pooling, depthwise Conv2D.

*(Enrichment excluded v1: shift, token-mixing MLP.)*

Each measured on **both** CPU and iGPU this phase; NPU added later by swapping the EP.
