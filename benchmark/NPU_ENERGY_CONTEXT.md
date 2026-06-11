# NPU Energy Measurement — Full Context for Next Steps

> **Audience:** Claude / supervisor review (Chris).  
> **Platform:** AMD Ryzen AI 9 HX 370 · conda `ryzen-ai-1.6.0` · uProf 5.3.518.0 · ORT 1.23.0.dev20250928  
> **Last updated:** 2026-06-11 (after first end-to-end NPU uProf smoke)

This document captures what was built, what the first NPU energy smoke actually measured, how that differs from CPU/iGPU, and what must be decided before a trustworthy three-engine comparison.

---

## 1. Pipeline overview (one loop, three EP branches)

All engines share **one** harness measurement loop (`benchmark/harness.py`):

```
energy_per_op = (window_energy − dispatch_energy) / iterations
```

| Component | Role |
|-----------|------|
| `harness.py` | Drives warmup + timed window; prints `[WINDOW_OPEN]` / `[WINDOW_CLOSE]`; writes `results/runs.csv` with `t_start_epoch`, `t_end_epoch` |
| uProf `timechart` | Parent-wraps harness; package power → `results/uprof/<run_id>/timechart.csv` |
| `parse_energy.py` | Joins uProf CSV + `runs.csv`; fills `window_energy_J`, `dispatch_energy_J`, `energy_per_op_J` → `results/runs_enriched.csv` |
| `benchmark/npu/` | Offline Quark XINT8 quant + VitisAI session (compile **before** window; not inside timed loop) |

**NPU branch only supplies:** XINT8 quant (offline) → VitisAI session → synchronous `sess.run()`. No second measurement loop. No DirectML `synchronize_outputs()`.

**uProf output flag:** use **`-o`**, not `--output` (invalid on AMDuProfCLI 5.3).

---

## 2. First NPU energy smoke — treat as plumbing test only

### Protocol used (not publication-grade)

| Parameter | Value |
|-----------|--------|
| Operator | `ffn_gemm` |
| Engine | NPU (XINT8, VitisAI EP) |
| `--shape-index` | **0** → **N=49** (sdpa_small), not headline N=197 |
| `--repeats` | 1 |
| `--warmup` / `--duration` | 3 s / 10 s |
| Environment | Editor/agents may have been open |

### Cleaned results (`results/runs_enriched.csv`, 2 rows)

| run_id | input_shape | iters | mean_latency_ms | window_energy_J | dispatch_energy_J | energy_per_op_J |
|--------|-------------|-------|-----------------|-----------------|-------------------|-----------------|
| `ffn_gemm_npu_smoke_r0` | **49×768@768×3072** | 43,572 | 0.230 | **204.03** | **175.87** | **6.46×10⁻⁴** (~0.65 mJ/op) |
| `dispatch_baseline_npu_r0_r0` | dispatch Add [1] | 945,916 | 0.0105 | **175.87** | — | baseline row |

**Sanity check:** (204.03 − 175.87) / 43572 ≈ 6.46×10⁻⁴ J/op ✓

### What worked

- End-to-end: quant → VitisAI compile → harness window → uProf CSV → `parse_energy.py` join
- `runs.csv` header includes `t_start_epoch` / `t_end_epoch` (after clean re-run)
- NPU **ffn_gemm** partitioned to NPU: **5** NPU ops, **2** VITIS_EP_CPU fallback (Q/DQ boundary on CPU)
- INT8 vs FP32 relative L2 ≈ **0.017–0.018** (finite output)

### What is wrong or incomplete for a paper

1. **Wrong shape for headline comparison** — N=49 (EfficientFormer-small corner), not N=197 (ViT avg).
2. **No CPU/iGPU energy at N=49** — cannot compare three engines at this shape.
3. **NPU dispatch baseline did not land on NPU** — VAI reported CPU-only for XINT8 Add graph (see §5).
4. **Precision asymmetry** — NPU XINT8 vs CPU/iGPU FP32 (see §7).
5. **Single repeat, 10 s window** — not `run_sweep.bat` defaults (5×30 s).
6. **Idle baseline not subtracted in code** — columns exist but `parse_energy.py` ignores them (see §4).

---

## 3. What N=49 means (common confusion)

**N=49 is sequence length (tokens)** in the SDPA **small** DSE corner — **not** repeats, **not** Quark calibration count.

| shape_index | block_id | N | shape_class | ffn_gemm shape |
|-------------|----------|---|-------------|----------------|
| **0** | sdpa_small | **49** | small | 49×768@768×3072 ← smoke used this |
| **1** | sdpa_avg | **197** | avg | 197×768@768×3072 ← headline / prior CPU-iGPU latencies |
| **2** | sdpa_large | 3136 | large | 3136×768@768×3072 |

- **Repeats:** smoke used `--repeats 1` → `repeat_idx=0`.
- **Calibration:** Quark uses **n=16** synthetic tensors per graph (operator-energy unit, not task accuracy) — unrelated to N.

---

## 4. Baseline model — idle vs dispatch (what we have vs paper framing)

### Implemented today

| Mode | Harness behavior | Engine-specific? | Used in `parse_energy.py`? |
|------|------------------|------------------|----------------------------|
| **measure** | Real operator on chosen EP | Yes | `window_energy_J` |
| **dispatch** | `dispatch_baseline.onnx` (Add) on **that** EP | Yes | Subtracted as `dispatch_energy_J` |
| **idle** | `time.sleep(0.001)` — **no ORT** | Label only; same host sleep | **No** — `idle_energy_J` empty |

**Headline formula (live in code):**

```
energy_per_op = (window_energy − dispatch_energy) / iterations
```

Spec also mentions idle as “static floor captured separately,” but **idle is not wired into the energy join yet**. For a paper that foregrounds idle:

- Run `--mode idle` per session (or per engine for symmetry).
- Extend `parse_energy.py` to fill/subtract `idle_energy_J` for reporting **active power** and static floor tables.

### CPU / iGPU / NPU — how dispatch baselines resolve

| Engine | Dispatch baseline | On-engine? |
|--------|-------------------|------------|
| **CPU** | `Add` on `CPUExecutionProvider` | **Yes** — CPU ORT launch + trivial op |
| **iGPU** | Same graph on `DmlExecutionProvider` | **Yes** — DML launch + trivial op |
| **NPU** | XINT8 `Add` on `VitisAIExecutionProvider` | **No in smoke** — VAI partitioned **CPU-only** (1 CPU + 5 VITIS_EP_CPU) |

**Idle** for all engines: effectively **common host floor** (sleep loop). `--engine` on idle is organizational metadata, not a different hardware path.

### Implication for three-engine comparison

- **CPU and iGPU** dispatch baselines are **on-engine** launch overhead through the same EP as the measure run.
- **NPU dispatch** in the smoke is **not** comparable — it measures mostly CPU-side VAI session overhead, not NPU kernel dispatch.
- **Do not** interpret NPU `energy_per_op` as fully isolated NPU rail energy: uProf is **package-level** only (socket0-package-power). All engines share the die; isolation is via **baseline subtraction + controlled EP**, not a dedicated NPU power counter.

### Recommended paper framing (idle + dispatch)

1. **Keep dispatch subtraction** for “real op vs trivial op on same EP path.”
2. **Add idle** for exposition: static floor W, active energy = window − idle (requires parser work).
3. **Document NPU dispatch anomaly** explicitly until a VAI-eligible dispatch graph exists or we accept CPU-side VAI overhead as the NPU “dispatch” term.

---

## 5. NPU-specific measurement differences (why Claude’s “NPU gap” matters)

### 5.1 Package power, not NPU rail

uProf on HX 370 exposes **socket0-package-power** and core power only. No iGPU/NPU rail. Every engine’s “energy” is **whole-package J during execution**, minus baselines.

### 5.2 Precision path

| Engine | Precision in smoke | Notes |
|--------|-------------------|--------|
| CPU | FP32 | Native ORT CPU EP |
| iGPU | FP32 | DirectML |
| NPU | **XINT8** | Quark QDQ + VitisAI; boundary Q/DQ on CPU |

This is **deployment-realistic**, not precision-matched. Must be stated in methods. Optional: iGPU INT8 via DirectML for a matched column.

### 5.3 Offline quant + compile outside window

Before `[WINDOW_OPEN]`:

1. Quark XINT8 quant (if `_xint8.onnx` missing) — **not timed**
2. `InferenceSession` creation — **NPU compile** — **not timed**
3. Warmup loop — discarded
4. Timed window — **only this is integrated with uProf**

First run in a session is much slower; uProf still wraps the whole child process (includes compile unless pre-warmed in a prior harness invocation).

### 5.4 Dispatch baseline on NPU (smoke finding)

```
[Vitis AI EP] No. of Operators :   CPU     1 VITIS_EP_CPU     5
```

The minimal Add graph **stays on CPU**. So `dispatch_energy_J ≈ 176 J` for NPU mostly reflects **VAI/CPU session churn**, not NPU matmul dispatch. The measure run’s NPU GEMM **did** use NPU (5 ops on NPU).

**Consequence:** `(window − dispatch)` for NPU may **under-** or **over-** correct relative to CPU/iGPU where dispatch is on-engine. This is the main **NPU energy methodology outlier** — not a parser bug.

### 5.5 Graph / partition count: 11 vs 5 NPU ops

| Source | Input | NPU ops | Notes |
|--------|-------|---------|-------|
| Early standalone validate (old script) | `[1×197×768]` 3D | **11** | Pre-registry export |
| Current `npu.gemm_validate` + harness smoke | `[49×768]` 2D, shape_index 0 | **5** | Registry `ffn_gemm.onnx` + Quark Gemm |

**Not** “validated graph A, measured graph B” at the same shape today — both use `operators.build_operator_graph("ffn_gemm", shape_index=0)`. The **11→5** gap is **N=197 vs N=49**, 3D vs 2D export, and Quark fusion to Gemm.

**Before headline run:** re-validate with **`shape_index=1`** (N=197) and confirm partition line matches harness.

---

## 6. CPU / iGPU comparison data available today

| Engine | N=197 latency (FP32) | uProf energy at N=197 | uProf energy at N=49 |
|--------|----------------------|------------------------|----------------------|
| CPU | **1.073 ms/iter** (tower smoke) | **Not in repo** | **No** |
| iGPU | **0.716 ms/iter** (tower smoke) | toy ~457 J / 10 s (not in current enriched CSV) | **No** |
| NPU | — | **No** | **0.65 mJ/op net** (plumbing smoke only) |

**No fair three-engine J/op table exists yet** at any single shape.

---

## 7. Precision policy — decide before testing

**Primary framing (recommended if unchanged):** “Best realistic deployment per engine” — XINT8 NPU vs FP32 CPU/iGPU. Legitimate; must be explicit.

**Secondary (optional):** precision-matched points — e.g. iGPU INT8 via DirectML.

Chris will ask; decide in methods before the real sweep.

---

## 8. Operational bugs fixed during bring-up (for context)

| Issue | Resolution |
|-------|------------|
| uProf `--output` invalid | Use **`-o results\uprof\<run_name>`** |
| `parse_energy` couldn't find CSVs | Match `-o` folder names; strip `_r0` suffix from run_id |
| Stale `runs.csv` header | Clean file before first run of a session |
| `--run-id dispatch_baseline_npu_r0` | Harness appends `_r0` → `dispatch_baseline_npu_r0_r0`; use **`--run-id dispatch_baseline_npu`** instead |
| Quark custom_ops.dll warning | Non-fatal on tower |

---

## 9. Trustworthy next-run protocol (recommended)

### Pre-decisions

- [ ] **shape_index = 1** (N=197) for headline, unless DSE sweep intentionally includes 0/1/2
- [ ] **Precision framing** written in methods (deployment-realistic vs matched)
- [ ] **Idle:** run `--mode idle` per engine; wire or manually report idle J (parser TODO)
- [ ] **NPU dispatch:** document CPU-only partition or find NPU-eligible dispatch graph

### Measurement (clean standalone terminal; editor closed)

Per engine (cpu, igpu, npu), same operator and shape:

```
--operator ffn_gemm --shape-index 1 --duration 30 --warmup 5 --repeats 5
```

Plus per engine:

- `--mode dispatch --run-id dispatch_baseline_<engine>`  (no `_r0` in run-id)
- `--mode idle --run-id idle_<engine>`

uProf wrap each run:

```
AMDuProfCLI.exe timechart --event power --interval 100 -o results\uprof\<run_name> ...
```

### Verification before trusting J/op

1. `parse_energy.py --uprof-dir results\uprof --plot`
2. Inspect PNGs: `t_start` / `t_end` lines on power plateau
3. Confirm `runs.csv` has `t_start_epoch`, `t_end_epoch`, correct `engine`, `input_shape`
4. One dispatch row per engine in enriched CSV (dedupe duplicates)
5. Re-run `python -m npu.gemm_validate` with `SHAPE_INDEX=1`; partition should match harness

### Analysis

- `parse_energy.py` → `runs_enriched.csv`
- `analysis.py` (note `WINDOW_ENERGY_IS_RAW=True` — dispatch subtracted in analysis if not pre-netted)

---

## 10. Suggested next steps (ordered)

1. **Fix `npu/gemm_validate.py`** default to `SHAPE_INDEX=1` for headline alignment (or document both).
2. **Run idle baselines** + decide parser change for idle subtraction/reporting.
3. **Headline triple-engine session:** ffn_gemm @ shape_index 1, CPU + iGPU + NPU, 5×30 s, dispatch + idle each.
4. **Document NPU dispatch CPU-only** in paper; discuss with Chris whether to accept or replace baseline.
5. **Precision statement** in `PROJECT_CONTEXT.md` / methods draft.
6. **Optional:** harness run-id helper — don’t double-append `_r0` if base already ends with `_r\d+`.

---

## 11. Key file paths

| Path | Purpose |
|------|---------|
| `benchmark/harness.py` | Shared measurement loop + NPU branch |
| `benchmark/npu/path.py` | Quark quant, VitisAI session, partition capture |
| `benchmark/npu/gemm_validate.py` | Plumbing test (no uProf) |
| `benchmark/parse_energy.py` | uProf ↔ runs.csv join |
| `benchmark/operators.py` | Registry; shape_index → N=49/197/3136 |
| `benchmark/results/runs.csv` | Harness timing + epoch bounds |
| `benchmark/results/runs_enriched.csv` | Energy columns |
| `benchmark/results/metadata.json` | `npu_path` config block |
| `PROJECT_CONTEXT.md` | Project-wide context |

---

## 12. One-paragraph summary for Claude

We completed the first **package-level** NPU energy plumbing test for `ffn_gemm` at **N=49** (accidental `shape_index=0`): **~0.65 mJ/op net** after subtracting a **dispatch baseline that did not actually run on the NPU** (CPU-only VAI partition). CPU/iGPU dispatch baselines are **on-engine**; idle exists in the harness but **is not subtracted in code**. There is **no** three-engine energy comparison at any single shape yet. The 11→5 NPU op count difference was **shape/export path**, not validate-vs-harness mismatch at the same graph. Before a real run: **shape_index 1**, **5×30 s**, **three engines + dispatch + idle**, **plot-verify uProf alignment**, **explicit precision framing**, and **honest NPU dispatch caveat**. The NPU energy story is methodology-heavy — the outlier is dispatch baseline behavior and package-level measurement, not the join math once `runs.csv` schema is clean.
