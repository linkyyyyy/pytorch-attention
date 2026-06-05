# Benchmark Workflow — CMD Operator Energy Measurement

Run all measured workloads from a **clean standalone CMD window** on the HX 370 tower — not from the Cursor IDE terminal. Close background apps before collecting data.

---

## 1. Prerequisites

| Item | Check |
|---|---|
| Conda env | `conda activate ryzen-ai-1.6.0` |
| ORT package | `onnxruntime-directml` (CPU + DML EPs in one env) |
| Verify EPs | `python -c "import onnxruntime as ort; print(ort.get_available_providers())"` → must include `DmlExecutionProvider` |
| uProf | `AMDuProfCLI.exe` on PATH (AMD uProf 5.x) |
| Power | Tower on AC; single pinned Windows power plan |
| BIOS VGM | Note Variable Graphics Memory allocation (default placeholder: **512 MB**) |

Set VGM to match BIOS before a session:

```cmd
set BENCHMARK_IGPU_VGM_MB=512
```

Or edit `IGPU_VGM_MB` in `run_sweep.bat`.

---

## 2. Headline energy formula

Python does **not** compute energy. uProf does. After aligning the uProf trace to harness `WINDOW_OPEN` / `WINDOW_CLOSE` markers:

```
energy_per_op = (window_energy - dispatch_energy) / iterations
```

- **Dispatch baseline** (`--mode dispatch`): non-elidable `Add(input, constant)` graph — measures runtime launch overhead. Subtract this from the active window to isolate compute.
- **Idle baseline** (`--mode idle`): same-length sleep window — captures the static power floor. Use for context; dispatch subtraction is the headline correction for per-op energy.

Harness prints `t_start` / `t_end` as `time.time()` epoch seconds for alignment.

### uProf clock base (TODO — verify on tower)

Before your first real session, inspect a uProf CSV and determine whether timestamps are:

1. **Absolute system time** — align directly to harness `t_start` / `t_end`, or
2. **Elapsed since collection start** — record the collection start wall-clock epoch immediately before/after `AMDuProfCLI` launches, then add that offset to uProf timestamps before matching Python markers.

Document your finding in `results/metadata.json` (`upref_version` / notes).

---

## 3. One-time graph export

```cmd
cd C:\path\to\pytorch-attention\benchmark
conda activate ryzen-ai-1.6.0
python operators.py
```

Exports 16 operator graphs + `dispatch_baseline.onnx` to `onnx_graphs/`.

**Verify in Netron:**

- `gelu.onnx` — exactly one `Gelu` node (warns if decomposed at opset &lt; 20)
- `dispatch_baseline.onnx` — live `Add` node with **runtime input** `input` + initializer `const_one` (not two initializers)
- MHA matmuls — batched shapes e.g. `[12,197,64] @ [12,64,197]`

Export all softmax profiles (including XCiT `[12,64,64]`):

```cmd
python operators.py --all-shapes
```

---

## 4. Smoke test (no uProf)

```cmd
python harness.py --operator ffn_gemm --engine cpu --duration 5 --warmup 2 --repeats 1 --run-id smoke_cpu
python harness.py --operator ffn_gemm --engine igpu --duration 5 --warmup 2 --repeats 1 --run-id smoke_igpu
```

Look for `[WINDOW_OPEN]` / `[WINDOW_CLOSE]` markers and a row in `results/runs.csv`.

---

## 5. Baselines (once per engine per session)

Run under uProf using the same parent-wrap pattern as the sweep (see `run_sweep.bat` templates at the end).

**Idle baseline (static floor):**

```cmd
AMDuProfCLI.exe timechart --output results\upprof\SESSION\idle_cpu_r0.csv -- python harness.py --mode idle --engine cpu --duration 30 --warmup 5 --repeats 1 --run-id idle_cpu_r0
```

**Dispatch baseline (launch overhead):**

```cmd
AMDuProfCLI.exe timechart --output results\upprof\SESSION\dispatch_igpu_r0.csv -- python harness.py --mode dispatch --engine igpu --duration 30 --warmup 5 --repeats 1 --run-id dispatch_igpu_r0
```

Repeat for each engine (`cpu`, `igpu`).

---

## 6. Full operator sweep

```cmd
cd benchmark
run_sweep.bat
```

- Activates `ryzen-ai-1.6.0`
- Creates `results\upprof\<SESSION_TS>\` (locale-safe timestamp via PowerShell)
- Per-run IDs: `<operator>_<engine>_r<repeat>` (no illegal filename characters)
- uProf wraps `python harness.py` as child process
- **TODO:** fill verified `AMDuProfCLI timechart` flags in `run_sweep.bat` after `AMDuProfCLI.exe timechart --help`

---

## 7. Manual per-operator runs

**CPU:**

```cmd
python harness.py --operator attn_score_matmul --engine cpu --duration 30 --warmup 5 --repeats 1 --shape-index 0 --run-id attn_score_matmul_cpu_r0
```

**iGPU:**

```cmd
python harness.py --operator attn_score_matmul --engine igpu --duration 30 --warmup 5 --repeats 1 --shape-index 0 --run-id attn_score_matmul_igpu_r0
```

**XCiT softmax profile (shape index 1):**

```cmd
python harness.py --operator softmax --engine cpu --shape-index 1 --duration 30 --warmup 5 --repeats 1 --run-id softmax_xcit_cpu_r0
```

### All 16 operators

`patch_embed_conv2d`, `downsample_conv2d`, `ffn_gemm`, `gelu`, `layer_norm`, `group_norm`, `batch_norm`, `residual_add`, `qkv_proj_gemm`, `attn_score_matmul`, `xcit_cov_matmul`, `softmax`, `attn_value_matmul`, `sra_conv2d`, `avg_pool_token_mixer`, `depthwise_conv2d`

---

## 8. Session metadata

First harness invocation writes `results/metadata.json`:

- ORT version, opset, `intra_op_num_threads` (= 12), `graph_optimization_level` (= ORT_ENABLE_ALL)
- `igpu_vgm_mb`, conda env name
- Fill `upref_version` and `gpu_driver_version` manually after the session

CSV columns per run include the same constants plus timing fields. Energy columns are filled during post-uProf analysis.

---

## 9. NPU (later)

Add `--engine npu` with `VitisAIExecutionProvider` in `harness.py`. Same loop, same graphs (INT8 via Quark when precision policy is locked).

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| DML session error | Ensure `enable_mem_pattern=False` and `ORT_SEQUENTIAL` (harness sets both) |
| Op runs on CPU unexpectedly | Check `notes` column for `EP_FALLBACK`; inspect ORT profiling JSON on repeat 0 |
| Dispatch baseline shows 0 iterations / no power delta | Open `dispatch_baseline.onnx` in Netron — `Add` must have runtime `input`, not two initializers |
| Gelu warning on export | Opset fell back to 19; native `Gelu` requires opset 20 |
| uProf I/O crash | Ensure `results\upprof\<SESSION_TS>\` exists before sweep (`run_sweep.bat` creates it) |
| Illegal filename in uProf output | Do not use `%date%%time%`; use session folder + `op_engine_rN` run IDs |

---

## 11. Recommended run order

Interleave engines to reduce thermal bias:

```
op_A cpu → op_A igpu → op_B cpu → op_B igpu → ...
```

`run_sweep.bat` currently loops operator → engine → repeat. Reorder manually or edit the batch loops if interleaving is preferred.
