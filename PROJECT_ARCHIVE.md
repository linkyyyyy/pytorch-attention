# PROJECT_ARCHIVE.md

> **Purpose.** Dated session log and historical progress notes for this internship.
> Extracted from `PROJECT_CONTEXT.md` on **2026-06-22** at the start of **Phase 2**
> (Radeon R9700 fourth engine). Intended for **Hermes Agent** and other tools that need
> full chronological context across sessions.
>
> **Not the live source of truth.** For current facts, tasks, and methodology, use
> `PROJECT_CONTEXT.md`. When the two disagree, `PROJECT_CONTEXT.md` wins.

---

## Phase 1 — HX 370 harness & operator sweep (2026-06-02 → 2026-06-15)

- **2026-06-02 (Day 1):** uProf responsive via CLI. ONNX Runtime confirmed exposing CPU + DirectML +
  Vitis AI EPs in one `ryzen-ai-1.6.0` conda env (single-environment rule satisfied). Tooling decided: Cursor now,
  Claude Code idle-installed, Hermes deferred. This context file created.
  - GEMM plumbing test PASSED on CPU EP. Full pipeline validated end-to-end
    (PyTorch 2.7.1+cpu -> ONNX opset 17 -> ORT 1.23.0.dev -> inference -> timing).
    Numerical check MATCH (max abs diff 1.43e-6). All three EPs visible in the
    `ryzen-ai-1.6.0` env (Vitis AI, DirectML, CPU). Model-selection scope confirmed:
    3-4 from EACH list, 6-8 total.
  - Full model survey completed (all 15 attention modules + 18 ViTs read and
    analyzed). Proposed shortlist drafted — **pending three decisions for Chris**
    (see §4 in PROJECT_CONTEXT). Key correction vs initial draft: ViT vs PoolFormer is NOT a clean
    control (topology differs); PvT-Tiny vs PoolFormer-12 is the correct pair.
    Quantization policy gap identified as the most important unresolved methodological
    decision.

- **2026-06-03 (Day 2):** Architecture and NPU foundations research day — reading only, no code.
  - **Transformer attention architecture:** Traced the full operator chain: QKV projections (three
    linear/GEMM ops) → scaled dot-product attention (QKᵀ matmul → scale → softmax → AV matmul) →
    output projection. Understood why attention is O(N²·d) and how that maps to two distinct matmul
    shapes: [N×d]·[d×N] for the QKᵀ product, then [N×N]·[N×d] for the AV product.
  - **NPU dataflow architectures:** Studied how spatial dataflow arrays (systolic arrays and their
    variants) work: data tiles flow through a 2D PE mesh; the array's efficiency depends on the matmul
    being *stationary enough* for the tiled operands to fill the PEs. Understood that CPU SIMD, iGPU
    SIMT warps, and NPU systolic arrays all see the *same operation* but have fundamentally different
    utilization curves as a function of matrix shape and sequence length.
  - **Why static shapes and INT8 are vital for Vitis NPU (EOD question answered):**
    The Vitis AI compiler AOT-compiles an ONNX graph into a fixed instruction schedule for the XDNA
    array. Dynamic shapes would require runtime recompilation — the compiler does not support that.
    INT8 is required because the XDNA MAC array is an 8-bit integer array; FP32/FP16 ops fall back
    to the CPU and lose the NPU's energy advantage entirely. Together: a model must have fixed input
    dimensions AND be quantized to INT8 *before* the compiler sees it, or it will not run on the NPU.

- **2026-06-04/5 (Day 3/4):** Architecture deep-dive — all five shortlisted models read; operator roles
  and NPU implications documented. Reference: `operator_architecture_selection.md` (Opus-generated
  baseline) for the cluster framework; this entry adds the per-model and NPU-specific detail.

  **Cluster A — structural operators (shared across all five models, same NPU behavior everywhere)**

  | Operator | Where in workflow | How computed | NPU behavior |
  |---|---|---|---|
  | Patch-embedding Conv2D | Stem: image → token sequence | Strided conv, e.g. 16×16 kernel, produces [N×C] tokens | Mapped to NPU conv engine if INT8; static spatial dims required |
  | Downsampling Conv2D | Between pyramid stages (PvT, EfficientFormer only) | Strided conv halves H,W | Same as above |
  | Linear/GEMM (FFN) | After every token-mixer block | Two matmuls: [N×C]·[C×4C] then [N×4C]·[4C×C] | NPU's primary target — square-ish shapes fill systolic array well |
  | GELU | Inside FFN | Element-wise approximation (tanh or erf formula) | Likely falls back to CPU; no MAC array benefit |
  | LayerNorm / GroupNorm / BatchNorm | After each sub-block | Running mean+variance over C (or group) dimension | Reduction op; partial CPU fallback expected |
  | Residual add | Skip connections throughout | Element-wise add | Trivially CPU/DMA; not a bottleneck |

  **Cluster B1 — token-mixer (attention-matrix) operators — the key NPU study axis**

  Each model's attention differs in *what matrix is built*, *its shape*, and *how many tokens feed it*:

  - **ViT** (isotropic, 197 tokens = 196 patches + CLS):
    - Q/K/V projections: three [197×C]·[C×C] GEMMs per layer. For ViT-B/16: C=768, head_dim=64, 12 heads.
    - QKᵀ score matmul: [197×64]·[64×197] → **[197×197]** attention matrix per head. Aspect ratio ≈ 1:1 but N=197 is modest — systolic array fills reasonably.
    - Softmax over 197-wide rows — element-wise reduction.
    - AV weighted-sum: [197×197]·[197×64] → [197×64]. Consumes the N×N matrix. Tall operand.
    - *NPU note:* Both matmuls are square-ish. 197 is small enough that PE utilization is not guaranteed — the array may be underused unless the batch tiles well.

  - **XCiT** (isotropic; cross-covariance attention):
    - Instead of QKᵀ over tokens, computes KᵀQ over channels → attention map is **[C×C]**, not [N×N].
    - Score matmul shape: [64×197]·[197×64] → **[64×64]** per head (XCiT-nano: head_dim=64). Tiny square.
    - AV equivalent: [64×64]·[64×197] → [64×197]. Transpose back to token space.
    - Complexity flips from O(N²·C) to O(C²·N): cheaper at large N, but the [64×64] matrix may underutilize a large systolic array — too small to tile across all PEs.
    - Also has a **depthwise Conv2D LPI block** (Cluster B2 below) for local spatial mixing.

  - **PvT** (hierarchical; Spatial-Reduction Attention):
    - Before forming QKᵀ, a **strided Conv2D** shrinks K and V spatially (e.g. 8× reduction at stage 1).
    - Attention matrix is then [N_q × N_kv_reduced] — significantly smaller than full N×N.
    - QKᵀ shape example (stage 1, 56×56 input, 8× SR): [3136 × 64]·[64 × 49] → [3136×49]. Very rectangular — tall-and-thin; systolic array rows load well but columns are sparse.
    - *NPU note:* The SR conv is itself an NPU-eligible INT8 conv. The attention matrix is non-square, which may affect utilization differently than ViT's.

  - **PoolFormer** (hierarchical; no attention matrix at all):
    - Token mixer is **average pooling** over a local window — no QKV, no matrix product.
    - Only Cluster A ops (Conv2D stem, FFN GEMM, GroupNorm, residual add) remain.
    - *NPU note:* Pooling is a reduction, not a MAC-array workload. Expected to run mostly on CPU/iGPU. This model is the **control**: any energy delta vs PvT is attributable to the token mixer alone.

  - **EfficientFormer** (hierarchical; conv/pooling stages + late MHSA):
    - Stages 1–3: Conv2D blocks + average pooling token mixers on 4D feature maps (no attention).
    - Stage 4 only: full MHSA, but on a **49-token sequence** (7×7 spatial grid, flattened).
    - MHSA shapes at stage 4: QKᵀ is [49×32]·[32×49] → [49×49] per head. Very small — 49×49 matrix.
    - *NPU note:* The [49×49] attention matrix is tiny. Systolic array will be severely underutilized unless the NPU tiles across batch or heads. This tests NPU behavior at minimal sequence length.

  **Cluster B2 — token-mixer (non-attention) operators**

  | Operator | Where | Models | How computed | NPU behavior |
  |---|---|---|---|---|
  | Average pooling (token mixer) | Every block | PoolFormer, EfficientFormer stages 1–3 | Local window mean over H×W | Reduction; likely CPU or iGPU; no matmul benefit |
  | Depthwise Conv2D (LPI) | After cross-covariance block | XCiT | Per-channel 3×3 conv (no cross-channel mixing) | NPU conv engine can handle if INT8 and static shape |

  **Key NPU contrasts across models (the measurement hypothesis):**
  - Cluster A ops cost the same on all engines regardless of model — any energy gap between models is B1/B2.
  - ViT and XCiT are the cleanest NPU matmul-shape experiment: same isotropic topology, same FLOP order of magnitude, but [197×197] vs [64×64] attention matrices → tests whether matrix *shape* affects NPU utilization.
  - PvT's SR conv before attention and PoolFormer's pooling-only mixer test whether NPU conv engines outperform attention for mixing.
  - EfficientFormer's 49-token MHSA is the edge case: can the NPU profitably accelerate a 49×49 matmul, or does dispatch overhead dominate?

  **Files created (Day 3/4 — Cursor-generated benchmark harness):**

  | File | Purpose |
  |---|---|
  | `benchmark/operators.py` | Operator registry + SDPA block configs + isolated/fused ONNX export (`attn_block_fused`) |
  | `benchmark/analysis.py` | Fusion-gap analysis on `runs.csv` → `analysis_out.csv`; `--demo` synthetic fixture |
  | `directives/attention_block_dimensions.md` | Per-architecture N/D extraction + SDPA corner rationale |
  | `benchmark/harness.py` | Single shared measurement loop — argparse, ORT session setup, WINDOW_OPEN/CLOSE markers, CSV append, EP placement verification |
  | `benchmark/run_plan.py` | Parses pipe-delimited sweep plan; drives per-operator uProf+harness loop (replaces broken CMD `for /f`) |
  | `benchmark/run_session.bat` | Session wrapper: baselines (direct uProf) + `run_plan.py` measure matrix |
  | `benchmark/run_sweep.bat` | Full sweep entry: `conda activate ryzen-ai-1.6.0` → `run_session.bat` |
  | `benchmark/BENCHMARK_WORKFLOW.md` | User-facing CMD workflow guide (prerequisites, export, smoke test, baselines, sweep, troubleshooting) |
  | `benchmark/IMPLEMENTATION_BLUEPRINT.md` | Design spec: registry schema, harness CLI, dispatch baseline rationale, CSV schema, uProf parent-wrap pattern, locked defaults |
  | `benchmark/onnx_graphs/` | 16 pre-exported `.onnx` operator graphs (+ `dispatch_baseline.onnx`) for Netron inspection and ORT sessions |
  | `benchmark/parse_energy.py` | Join uProf timechart + `runs.csv` → `runs_enriched.csv` (`window_energy_J`, `idle_energy_J`, `dispatch_energy_J`); `--test-toy` + `--plot` |
  | `benchmark/results/` | Empty placeholder — `runs.csv` and `uprof/<SESSION_TS>/` populate during measurement |
  | `directives/measurement_harness_spec.md` | Protocol spec: measurement philosophy, session options, loop structure, baselines, uProf integration, CSV schema, validity checklist |
  | `directives/operator_architecture_selection.md` | Architecture & operator selection rationale (Opus-generated; authoritative cluster definitions) |

- **2026-06-05 (idle headline + run-id hygiene):**
  - **Idle:** `--mode idle` uses the same warmup/window timer as measure/dispatch (sleep loop body, no ORT). `parse_energy.py` joins idle uProf window → `idle_energy_J` on measure rows (by `repeat_idx`).
  - **Headline:** `energy_per_op_J` = idle-subtracted in `analysis.py`; `energy_per_op_dispatch` retained as secondary (NPU dispatch CPU-fallback caveat).
  - **Run-id:** harness `resolve_run_id()` appends `_r{repeat_idx}` once; strips trailing `_r\d+` from `--run-id`.

- **2026-06-08 (Day 5 — tower verification COMPLETE):** Full harness + uProf smoke test passed on HX 370 in `ryzen-ai-1.6.0`. Reference artifacts in `benchmark/uprof_smoke/`.

  **Tower verification checklist — all confirmed:**
  - [x] **Conda env:** `ryzen-ai-1.6.0`
  - [x] **ORT:** `1.23.0.dev20250928`; providers: `CPUExecutionProvider`, `DmlExecutionProvider`, `VitisAIExecutionProvider`
  - [x] **uProf:** `5.3.518.0`; child-launch via full interpreter path `C:\ProgramData\miniconda3\envs\ryzen-ai-1.6.0\python.exe` (not PATH alias)
  - [x] **uProf timestamp format:** wall-clock `HH:MM:SS:ms` (local tz), **not** epoch — conversion required to align with harness epoch markers (see PROJECT_CONTEXT §3)
  - [x] **Power scope:** `timechart --list` — Power counters only at **[Socket, Core]**; no per-rail iGPU/NPU
  - [x] **GPU driver (890M):** `32.0.21030.0`
  - [x] **DML IOBinding:** `OrtValue.ortvalue_from_numpy(arr, "dml", 0)` OK (`dml_ortvalue_smoke_test.py` PASS)
  - [x] **GPU sync:** `io_binding.synchronize_outputs()` after each `run_with_iobinding` — iGPU `ffn_gemm` = **0.716 ms/iter**, ~**6,982 iters/5 s** (physically sane; no async over-count)
  - [x] **EP placement:** `EP_OK` on all four smoke runs

  **Smoke-test reference latencies (`ffn_gemm`, shape `197×768@768×3072`, FP32, opset 20):**

  | Run | Engine | Mean latency |
  |---|---|---|
  | `ffn_gemm` | CPU | **1.073 ms**/iter |
  | `ffn_gemm` | iGPU | **0.716 ms**/iter |
  | `dispatch_baseline` | iGPU | **0.076 ms**/iter |

  Session constants recorded in `benchmark/results/metadata.json` (includes `csv_schema` with
  `tier` / `block_id` / `shape_class` allowed values and example rows).

- **2026-06-08 (Week 2, Day 1 — close-out):** All tower-verification items cleared.

  **Status (confirmed on HX 370):**
  - `synchronize_outputs()` on iGPU IOBinding — `ffn_gemm` ~**0.72 ms/iter**, ~**13,560 iters/10 s**, no async over-count
  - DML IOBinding probe passes; **EP_OK** on all smoke runs
  - uProf launch confirmed: trailing positional target, absolute `python.exe` path (no `--`), `--event power --interval 100`

  **uProf CSV schema (ground truth: `benchmark/uprof_toy/`):**
  - Preamble → `PROFILE RECORDS` section; data header `RecordId,Timestamp,socket0-package-power,...`
  - `Timestamp`: `HH:MM:SS:ms` (colon before ms, e.g. `16:36:25:532`)
  - Power column: **`socket0-package-power`** (W) — integrate this; ignore `core0-power`..`core11-power`
  - Session date from preamble `Profile Start Time:` (e.g. `Jun-08-2026_16-36-25`); local tz **Europe/Athens**

  **`run_sweep.bat` fixes:** launch line uses `--event power --interval 100` + absolute `python.exe` path (no `--`); `upprof`→`uprof` session-dir typo corrected.

  **Window bounds (design decision):** Option **(b)** — `harness.py` writes `t_start_epoch` / `t_end_epoch` columns to `runs.csv`. `parse_energy.py` reads these directly; falls back to per-run `.log` parsing if columns are absent.

  **`parse_energy.py`:** Deterministic post-processor — `parse_uprof_csv()`, trapezoidal `integrate_power()`; joins `runs.csv` to uProf CSVs by `run_id`; fills `window_energy_J`, joins `idle_energy_J` and `dispatch_energy_J`; dedups baseline rows. Subtraction stays in `analysis.py`.

  **Toy verification (`toy_igpu`, `AMDuProf-python-Timechart_Jun-08-2026_16-36-25/timechart.csv`):**
  - Window: `t_start=1780925790.173200`, `t_end=1780925800.172677`, `iterations=13560`
  - `window_energy_J` ≈ **457 J** (mean package power ~46 W × ~10 s); `energy_per_op_J` ≈ **3.37×10⁻² J/op**

  **GUI note:** `timechart` output is **CSV-only** (no `.uprof` DB). `collect` is a CPU profiler, not the package-power path.

- **2026-06-10 (analysis layer — config extraction + fusion-gap pipeline):**
  - **`directives/attention_block_dimensions.md`:** N/D/head dims pulled from repo model files; proposed small/avg/large corners documented.
  - **`operators.py`:** `SDPA_BLOCK_CONFIGS` (three corners); isolated-op shapes derived from config (`fusion_member` tagging); `attn_block_fused` Tier-2 graphs.
  - **`runs.csv` schema:** `tier`, `block_id`, `shape_class`, `fusion_member`. Documented in `metadata.json` → `csv_schema`.
  - **`analysis.py`:** Built and validated on synthetic data (`python analysis.py --demo`): gap arithmetic confirmed **22 → 16 mJ = 6 mJ / 27.27%**.
  - **`WINDOW_ENERGY_IS_RAW=True` confirmed** on production session `20260612_143854`.

- **2026-06-11 (NPU harness branch — code only, not measured yet):**
  - **`benchmark/npu/`:** NPU package — `path.py`, `gemm_validate.py`.
  - **`harness.py`:** Third EP branch `--engine npu`; offline quantize → session → `sess.run()`.
  - **`operators.py`:** `xint8_onnx_path()`; `--quantize-xint8-all`.
  - **`run_sweep.bat`:** `ENGINES=cpu igpu npu`; idle + dispatch baselines per session.
  - **Validated plumbing:** `ffn_gemm` XINT8 → VitisAI EP (`benchmark/npu/gemm_validate.py`).
  - **Cache rule:** delete `benchmark/.vaip_cache/` after NPU driver or VitisAI EP version change.

- **2026-06-11 (fusion_member + registry-driven sweep plan):**
  - **`harness.py`:** emits `fusion_member` column on measure rows.
  - **`operators.py`:** `profile_indices_for_corner()`, `--sweep-plan --corner avg`; `SESSION_MEASURE_OPERATORS` + `SWEEP_SKIP_ENGINES` (attn_block_fused NPU).

- **2026-06-12 (production sweep — measurement complete):**
  - **Session:** `benchmark/results/runs_20260612_143854.csv` + `benchmark/results/uprof/20260612_143854/`.
  - **Row counts:** 62 measure rows (cpu **21** / igpu **21** / npu **20**) + **4 baselines**.
  - **Infrastructure:** measure loop via `run_plan.py` (supersedes CMD `for /f` plan parser).
  - **Known issues:**
    - **`attn_block_fused` omitted on NPU** — VitisAI EP crash. Tier-2 fusion gap on NPU **not measurable**.
    - **NPU dispatch baseline runs CPU-fallback (`VITIS_EP_CPU`)** — use idle-subtracted headline only.

- **2026-06-15 (operator sweep + cpu_INT8 control — COMPLETE):**
  Production sweep session `20260612_143854` measured, post-processed, trust-validated.
  cpu_INT8 control `20260615_145614_cpu_int8` measured + decomposition validated.
  NPU GEMM/conv wins are **architectural** (`architecture_ratio` 3–14×). Sign-divergence on
  `softmax`, `depthwise_conv2d`. Trust carry-forwards documented in PROJECT_CONTEXT §7.
  Idle +51% cross-session resolved (per-session subtraction). **Durable summary: PROJECT_CONTEXT §7.**

---

## Phase 2 — Radeon R9700 fourth engine (2026-06-22 →)

- **2026-06-22 (Phase 2 Day 1 — R9700 4th engine; planning + code, REMOTE/off-tower):** Stood up the
  R9700 path as additive code on `RadeonR9700` (off `main`). Decisions: Option A (Ubuntu+ROCm), ROCm EP
  (operator isolation), FP32, r9700 in `attn_block_fused`. Staged (not committed): harness `r9700` engine
  (ROCm EP, device-resident IOBinding via `probe_rocm_ortvalue`, output sync, placement check; npu imports
  made lazy so harness loads on Linux); new `gpu_power.py` amd-smi sampler (power + SMU energy accumulator,
  drift-corrected, signal-safe); `run_plan.py` amd-smi backend (background sampler brackets unwrapped
  harness; uProf path unchanged; `--uprof-dir` made optional for amdsmi); `parse_energy.py` amd-smi join
  (per-window counter-delta vs integration, wrap guard, gfx_busy placement gate; uProf numerics untouched);
  passing `test_parse_energy_amdsmi.py`. Remote validation: parse layer green on synthetic traces
  (counter-delta == integration at constant power; both fallbacks + wrap guard + busy gate fire); four files
  syntax-clean and Linux-importable; run_plan amd-smi dry-run correct (fused block included). Tower-gated next:
  ROCm EP presence, energy-counter resolution fix in `gpu_power.read_sample`, device_id alignment, IOBinding
  device-string probe, then the sweep. **Durable runbook: `benchmark/RADEON_R9700_RUNBOOK.md`.**
