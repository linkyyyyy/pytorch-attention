# PROJECT_CONTEXT.md

> **Purpose of this file.** Portable, tool-agnostic context for an 8-week research internship
> (started **June 1, 2026**). Paste into any AI agent at session start. **Live source of truth**
> for current facts and tasks. Dated session log and Phase 1 detail: **`PROJECT_ARCHIVE.md`**
> (for Hermes / historical lookup). When in doubt, this file wins over an agent's own memory.
---

## 1. ROLE & PURPOSE (instructions for the AI reading this)

You are a research assistant supporting a summer internship (8 weeks, started **June 1, 2026**)
at a lab in Greece, supervised by **Christoforos Kachris**. Your job is to handle well-scoped,
often monotonous tasks: drafting code, summarizing papers, generating boilerplate scripts,
organizing notes, and answering setup/tooling questions.

Be precise. Cite the project facts below rather than guessing. **Flag when a task needs hardware
the user does not currently have access to.** Respect the measurement-integrity rules in §6 —
they are the most common place where well-meaning suggestions go wrong.

---

## 2. PROJECT GOAL

Run AI-operator and model benchmarks on an **AMD Ryzen AI 9 HX 370** (CPU, iGPU, NPU) and —
in **Phase 2** — a discrete **AMD Radeon AI PRO R9700**, performing a **design space exploration
(DSE) in terms of energy efficiency** across compute engines.
- Energy is measured as **E = P · t** (power × execution time).
- The deliverable is a **short, publishable paper**.
- The core comparison must be **fair**: hold the runtime constant (ONNX Runtime) so that
  *only the hardware engine varies*. See §6.
- The intent is experimental: measure energy behavior of operators/models common to LLMs
  and Vision Transformers, and report **which engine wins for which computational pattern, and why.**

The headline result should be about **diversity of computational profile across engines**, not
a single narrow finding like "the NPU likes matmuls."

---

## 3. HARDWARE FACTS (important — do not conflate the three machines)

| Machine | Role | Details |
|---|---|---|
| **HX 370 tower** (lab-provided) | **HX 370 measurements** | AMD Ryzen AI 9 HX 370: Zen 5 CPU + Radeon 890M iGPU + XDNA NPU. Windows-based. Phase 1 operator sweep complete (§7). |
| **Personal laptop** (2024 ROG Strix G16) | **Development only** | Windows 11, NVIDIA RTX 4070. The CUDA/NVIDIA GPU is **irrelevant to this project** — it serves a separate CUDA-learning effort. Never used for measurement. |
| **R9700 tower** (lab-provided, Phase 2) | **R9700 measurements only** | Host: Intel i7-12700, 32 GB, Ubuntu 24.04, Python 3.12.3. GPU: Radeon AI PRO R9700 (**gfx1201** / RDNA4), 32 GB GDDR6, ~300 W TBP, **COMPUTE** profile (`rocm-smi --setprofile 4`). Stack: ROCm 7.1.52802, PyTorch 2.13.0.dev+rocm7.1, onnxruntime-migraphx 1.23.1, MIGraphX 2.14.0. **Do not change ROCm/OS versions.** |

The AMD Ryzen AI Software SDK is **Windows-based**.

### Measurement Scope Notes

**Confirmed on tower (2026-06-08):** `AMDuProfCLI timechart --list` exposes Power counters
only at **[Socket, Core]** granularity (`socket0-package-power`, `coreN-power`). No per-rail
iGPU or NPU power counters exist on this platform.

**Timestamp alignment (units differ — conversion required):** uProf `timechart.csv` uses
**wall-clock time-of-day** (`HH:MM:SS:ms`, local tz). Harness `WINDOW_OPEN`/`WINDOW_CLOSE` uses
**Unix epoch seconds** (`time.time()`). These are not the same unit; do not assume direct match.

- **Preferred conversion:** parse uProf `HH:MM:SS:ms` + known session date + **Europe/Athens**
  offset → epoch; slice CSV in epoch space. Avoid epoch→time-of-day as the primary path (no date
  on uProf strings; midnight/DST ambiguity).
- **Window bounds in `runs.csv`:** `t_start_epoch` / `t_end_epoch` from harness `time.time()` at
  `WINDOW_OPEN`/`WINDOW_CLOSE` — primary join key for `parse_energy.py`.
- **Block-context columns in `runs.csv`** (after `cluster`, before `shape_index`):
  - `tier` — `isolated` (single operator) or `fused_block` (whole attention-core graph); `n/a` for
    idle/dispatch baselines.
  - `block_id` — SDPA corner id: `sdpa_small`, `sdpa_avg`, `sdpa_large`; empty when not
    block-scoped; `n/a` for baselines.
  - `shape_class` — DSE corner: `small`, `avg`, or `large`; empty when unspecified; `n/a` for baselines.
  - `fusion_member` — `True` / `False` / `n/a` (baselines). Harness writes from `operators.py` registry
    shape profiles (attention-core ops = `True`; LN/FFN/residual context = `False`; single-profile conv
    ops = entry default `False`). Generalizes to per-block via profile tags later.
  Allowed values and example rows: `benchmark/results/metadata.json` → `csv_schema`.
  **Conda env (all tower work):** `ryzen-ai-1.6.0`.
- **Robust anchor (TODO):** capture uProf `Profile Start Time` and/or log epoch at `AMDuProfCLI`
  launch in `run_sweep.bat` so both traces share one reference instead of matching independent clocks.
- Reference smoke captures: `benchmark/uprof_smoke/`.

**Individual iGPU or NPU rail isolation is unavailable** on the HX 370: CPU, Radeon 890M iGPU,
and XDNA NPU share the same die and power delivery. All energy figures are **package-level power
during engine execution**, isolated via baseline subtraction, not a dedicated engine rail.

**Headline metric (cross-engine):** `energy_per_op_J = (window_energy_J − idle_energy_J) /
iterations`. Idle is a **common host floor** — same timed window as measure/dispatch, no ORT work
(sleep loop only); `--engine` on idle runs is metadata only. Package-level uProf has no per-rail
counter, so marginal energy above the shared idle floor is the defensible cross-engine comparison.

**Secondary metric (per-engine kernel attribution):** `energy_per_op_dispatch = (window_energy_J −
dispatch_energy_J) / iterations`. **Caveat:** NPU dispatch baseline runs CPU-only (not
dispatch-isolated on NPU). After production join, expect ~9 negative NPU `dispatch_energy_J`
values — known/expected; do not use dispatch-subtracted headline for NPU cross-engine claims.

`window_energy_J` stays **RAW** upstream (`WINDOW_ENERGY_IS_RAW=True`); all subtraction is in
`analysis.py`. Harness appends `_r{repeat_idx}` to base `--run-id` exactly once (strips trailing
`_r\d+` if caller already included it).

Per-engine comparison relies on **controlled execution** (one ORT execution provider per run) and
**baseline subtraction**, not separate hardware power rails. State this explicitly in the paper
methodology.

### R9700 Measurement Scope Notes (Phase 2 — updated 2026-06-23)

- **Runtime (Chris signed off):** engine `r9700` uses the **MIGraphX Python API directly**, not ORT.
  ROCm EP was removed from ORT 1.23+; the wheel's only GPU EP is MIGraphX, which fails on gfx1201
  (`std::get: wrong index for variant` on every ORT path). Same ONNX graphs/compiler → kernel-level
  comparability; runtime deviation documented in methodology.
- **Execution:** `compile(get_target("gpu"), offload_copy=False)` + one `to_gpu` upload (resident
  inputs). `run()` enqueues async (~4 µs) vs ~117 µs kernel — harness loops `sync_every` (default
  **64**, `--sync-every`) then `gpu_sync()`; injected `execute_fn` + `iter_multiplier` in shared
  `_run_repeat` (CSV schema unchanged). Day-1 validated: gfx_busy ~100%, ~242 W active (idle ~33 W).
- **Power (Branch B):** discrete card invisible to uProf; `gpu_power.py` amd-smi sampler brackets
  each harness run. Energy accumulator **dead** on gfx1201 (`energy_accumulator=0`) → trapezoidal
  integration of `socket_power` (watts, no /1000; `current_socket_power='N/A'`). Branch-A selector
  fixed 2026-06-23: `parse_energy` requires `e_end > e_start`; `gpu_power` leaves `energy_uj` blank
  when accumulator is 0 (no fake `"0.0"`). Re-enrich validate trace → expect ~7 kJ,
  `window_energy_method=trapezoid`.
- **Cross-instrument caveat:** HX 370 on uProf package (Windows/DML/Vitis) vs R9700 board telemetry
  (TBP-class, Linux/ROCm). Idle-subtracted within-engine deltas are clean; absolute cross-machine
  comparisons carry the caveat.
- **Net metric:** same headline formula as §3 on the discrete rail. **Idle + dispatch baselines
  still TODO.** Transfer baseline (Chris Option 2) likely **moot** — `offload_copy=False` residency
  = one H2D, no per-iter transfer; confirm with Chris before building one.
- **Sweep correctness (`operators.py`, 2026-06-23):** `shape_index` 0=small (N=49) / 1=avg (N=197) /
  2=large (N=3136); Phase-1 headline corner = **avg**. Day-1 validation used index 0 (small) —
  mechanism-valid, not the headline corner. Default `operators.py` builds index 0 only; run
  `operators.py --all-shapes` before sweep. Avg index is **not** uniformly 1 (XCiT etc. have own
  profiles) — drive sweep via `profile_indices_for_corner(op, 'avg')`, never hardcode `--shape-index 1`.
- **Paper caveats:** MIGraphX-direct runtime; gfx1201 kernel immaturity (ecosystem-wide); no
  instantaneous power (socket integration OK for Tier 1 steady-state, weak for Tier 2 transients).
- **Code:** branch `RadeonR9700`; `harness.py` (migraphx_direct), `gpu_power.py`, smoke/diag scripts.
  Runbook: `benchmark/RADEON_R9700_RUNBOOK.md` (gates partially superseded by Day-1 decisions).

---

## 4. OPERATORS / WORKLOADS IN SCOPE

### Operator-level (~8–12 ops)
GEMM/matmul, scaled dot-product attention, softmax, LayerNorm and RMSNorm, activations
(ReLU / GELU / SiLU), RoPE, embedding/gather. (F1–F4 in the supervisor's original diagram
were illustrative examples, not a ceiling.)

### SDPA fusion-gap microbenchmark (Track 1 — primary sweep axis)

**Two-tier measurement design:**
- **Tier 1 (isolated):** each attention-core operator microbenchmarked as its own ONNX graph
  (`tier=isolated`). Shapes derived from one shared `SdpaBlockConfig` per corner in
  `benchmark/operators.py` — isolated and fused graphs stay in lockstep.
- **Tier 2 (fused_block):** whole attention core as one graph (`attn_block_fused`,
  `tier=fused_block`) — Q/K/V projections → reshape → scaled dot-product attention → output
  projection, single `[N, D]` in / `[N, D]` out.

**Fusion gap (per `engine`, `block_id`):**
```
gap_J = sum(energy_per_op_J for isolated rows where fusion_member=True)
        − energy_per_op_J for the fused_block row
```
Implemented in `benchmark/analysis.py`. Only **five attention-core op families** are fusion
members (`fusion_member=True`): `qkv_proj_gemm` (q, k, v — three profiles), `attn_score_matmul`,
`softmax`, `attn_value_matmul`, `out_proj_gemm`. **LN / FFN / GELU / residual** are measured at
the same `(N, D)` shapes but tagged `fusion_member=False` — context ops, excluded from the gap sum.

**Fused graph status:** Tier 2 is currently the **unfused ONNX op chain** exported from a PyTorch
`nn.Module` (`export_torch_module`, opset 20 / fallback 19). A `# TODO(tower)` in `operators.py`
marks a future ORT `com.microsoft.MultiHeadAttention` / `Attention` contrib-op swap — **not a
one-line change** (input-layout plumbing differs); must verify **DirectML + Vitis AI** on the tower
before adopting.

**SDPA shape corners (locked in `SDPA_BLOCK_CONFIGS`):**

| `block_id` | `shape_class` | N | D | heads | head_dim |
|---|---|---:|---:|---:|---:|
| `sdpa_small` | small | 49 | 768 | 12 | 64 |
| `sdpa_avg` | avg | 197 | 768 | 12 | 64 |
| `sdpa_large` | large | 3136 | 768 | 12 | 64 |

**D held constant (768) so energy varies with N (and N² attention scaling), not embed-dim sweep.**
Only **avg** matches a verbatim repo block geometry (canonical ViT-B/16: 197 tokens, D=768,
12 heads, head_dim=64). **small** (N=49) and **large** (N=3136) are controlled DSE points — real
sequence lengths from the repo pyramid (EfficientFormer MHSA / PvT stage 1) with standardized D/heads.

**`shape_index` in `operators.py`:** 0=small / 1=avg / 2=large (Phase-1 pin = **avg**). Default build
emits index 0 only — use `operators.py --all-shapes` before sweep. Per-operator avg index varies
(e.g. XCiT); use `profile_indices_for_corner(op, 'avg')`, not a global `--shape-index 1`.
Note: repo `VisionTransformer()` default is **4 heads / head_dim=192** — deliberately overridden to
canonical 12/64 for comparable SDPA geometry. See `directives/attention_block_dimensions.md`.

**Track 2 (real mixer ops — not blocking SDPA sweep):** XCiT cross-covariance, PvT SRA conv,
PoolFormer pool, depthwise LPI still carry **placeholder shapes** in the registry (e.g. XCiT profiles
use fabricated 12/64/197 instead of real `xcit_nano` 4-head/head_dim-32/N-196; conv ops hardcode
768 channels vs real per-stage dims). Correct before Track-2 measurement.

### Model-level (from the benchmark repo — see §9)
The supervisor's guidance: use AI to identify the most *complex* models, then pick
**3–4 from each of the two lists** in the repo — **Attention Mechanisms** and **Vision
Transformers** — chosen for *meaningful diversity*, not just raw complexity.

**Selection principle:** complexity **and** diversity of computational profile. Avoid picking
several models that are heavy in the *same* way (all GEMM-dominated) — they'd tell the same story.

Frame light modules as **overhead probes** (measuring each engine's fixed dispatch/DMA cost) and
heavy modules as **compute efficiency probes**. That distinction can anchor a section of the paper.

**Proposed shortlist — PENDING three decisions for supervisor (see below):**

*Attention Mechanisms (4 proposed):*
- **ECALayer** — O(C) overhead probe. 5 params. Fixed dispatch cost dominates; NPU may be *slower*
  than CPU here, which is itself a finding about NPU launch overhead.
- **CBAM** — O(C + HW) dual-path lightweight. Channel + spatial, representative mid-weight module.
- **DoubleAttention (A2Net)** — O(C²·N) two-stage cross-attention via batched matmuls. The only
  module in the list with explicit bmm-based attention. Bridges attention-mechanisms and ViT workloads.
- **DANet-PAM** — O((HW)²) full spatial self-attention. Heaviest module by far (compute probe).
  Requires a fixed (H,W) at export time — recommend 56×56 to mimic ResNet stage-3 feature maps.

  Note: params column is misleading for these modules. What matters is the attention complexity
  class (O(C), O(HW), O((HW)²)) at the chosen input resolution — not raw parameter count.

*Vision Transformers (5 proposed — nudges scope, confirm with supervisor):*
- **ViT** (~86M params, ~17 GFLOPs) — isotropic baseline. Full quadratic attention on 197 tokens.
- **PvT-Tiny** (~13M, ~1.9 GFLOPs) — hierarchical pyramid + spatial-reduction attention.
- **PoolFormer-12** (~12M, ~1.8 GFLOPs) — same hierarchical pyramid as PvT, pooling instead of
  attention. **PvT vs PoolFormer is the clean control** (one variable: token mixer). Do NOT use
  ViT vs PoolFormer — they differ in both token mixer AND topology (isotropic vs pyramid).
- **XCiT-nano-12-p16** (~3M, ~0.56 GFLOPs) — isotropic, cross-covariance attention O(C²·N) not
  O(N²·C). Different matmul layout from ViT. Very ONNX-friendly (no windows, no dynamic shapes).
- **EfficientFormer-L1** (~12M, ~1.3 GFLOPs) — pooling stages 1-3, MHSA only at 7×7 (49 tokens).
  Tests NPU behavior on attention at very short sequence length.

**Three decisions needed from supervisor before shortlist is final:**

1. **PoolFormer control pair (architecture correctness):** ViT is isotropic; PoolFormer is a
   hierarchical pyramid. They differ in two variables. The clean control is PvT-Tiny vs
   PoolFormer-12 (same topology, different mixer). This requires 5 ViT-list models total.
   Confirm scope or drop one.

2. **Quantization/precision policy:** Main operator sweep uses **FP32 on cpu/igpu, XINT8 on npu**
   (see §7). A **cpu_INT8 control** sweep decomposes that cross-precision gap — do not treat
   headline cpu-vs-npu cells as pure architecture without checking `precision_ratio`. For model-level
   work, still choose explicitly: INT8-all-EPs vs native-best with precision reported.

3. **Attention module execution context:** Run modules on synthetic feature map tensors directly
   (operator-level, analogous to the GEMM test) OR embed them in a ResNet-50 backbone and
   benchmark the full model. Synthetic is cleaner for isolation; embedded is closer to real use.
   This also affects whether energy readings are above noise for the lightweight modules (ECA etc.).

> **Resolved:** originally 6-8; proposed set is now 9 pending decision #1
> confirm whether 5 from ViT list is acceptable given the topology-control argument.

---

## 5. TOOLCHAIN

```
PyTorch
  → export to ONNX (or TorchScript)
  → quantize with AMD Quark (INT8 / BF16)
  → run via ONNX Runtime (ORT) execution providers:
        • CPU EP            → Zen 5 CPU
        • DirectML EP       → Radeon 890M iGPU   (DmlExecutionProvider)
        • Vitis AI EP       → XDNA NPU           (VitisAIExecutionProvider)
  → power telemetry via AMD uProf (AMDuProfCLI.exe), paired with an external wall-meter
```

- **Single-environment rule:** drive all three EPs from inside the **one** Ryzen AI conda
  environment (e.g. `ryzen-ai-1.6.0`). Do **not** install a separate stock `onnxruntime-directml`
  into a different environment — that would measure the iGPU on Microsoft's ORT build and the NPU
  on AMD's ORT build (different kernels = unfair comparison). Verify providers with:
  ```python
  import onnxruntime as ort
  print(ort.__version__)
  print(ort.get_available_providers())
  # want: CPUExecutionProvider, DmlExecutionProvider, VitisAIExecutionProvider
  ```
  **STATUS: confirmed working — all three EPs available in one environment.**
- **Ollama / llama.cpp:** a *discovery tool only* — used to see which operators matter in real
  LLMs. **Not a measurement runtime.** Mixing runtimes breaks comparison fairness.
- DirectML note for the eventual methodology section: DirectML is in sustained engineering
  (still supported; Microsoft has moved new feature development to WinML). Fine for a stable
  benchmarking study — just an honest one-sentence footnote, not a reason to switch tools.
- **Sweep orchestration (2026-06-12):** the per-operator measure matrix is driven by
  `benchmark/run_plan.py` (Python), **not** the old CMD `for /f delims=|` loop.
  `run_session.bat` / `run_sweep.bat` / `run_dry_sweep.bat` call `run_plan.py` for the
  operator×engine matrix; **baselines** (idle + dispatch) stay as direct uProf-wrapped
  harness calls in `run_session.bat`. Reason: CMD `for /f` collapses consecutive `|`
  delimiters and cannot emit empty tokens — plan rows with empty `block_id`/`shape_class`
  (8 single-profile ops) shifted tokens and passed `input_shape` into `--tier`, silently
  dropping those ops. `run_plan.py` parses the pipe plan with Python `csv` (correct
  empty-field handling) and does skip-engine checks in Python.

---

## 6. MEASUREMENT INTEGRITY (the rules that matter most)

1. **The agent is never in the measurement loop.** AI editors/agents (Cursor, Claude Code, Hermes)
   are for *authoring and orchestration*. The benchmark is a *separate native Windows process* in
   the `ryzen-ai-1.6.0` conda env, with uProf polling it. Where the agent generates its text has **zero**
   bearing on recorded wattage or latency, as long as the benchmark runs natively. (WSL2 vs native
   Windows for an agent is a *convenience* question, not a telemetry one.)
2. **Develop in one place, measure in a quiet room.** Write and debug in Cursor freely. For the
   *timed runs*, launch the benchmark from a **clean standalone PowerShell/conda prompt** with the
   editor (and other background CPU/GPU load) closed — Electron apps, language servers, and an
   editor's own AI calls share the package power rail and add noise.
3. **Hold the runtime constant.** Only the hardware engine varies across the CPU/iGPU/NPU columns
   (same ORT build, same model, same input). See §5.
4. **Hold precision explicit across engines.** Measured operator sweep: **cpu/igpu = FP32,
   npu = Quark XINT8** (`onnx_graphs/*_xint8.onnx`). Cross-engine headline gaps confound precision
   and architecture — use §7 decomposition (`precision_ratio`, `architecture_ratio`) before
   attributing NPU wins to XDNA2 alone. Do not rank deployment on `original_gap` where
   `precision_ratio` departs far from 1.
5. **Verify each model exports AND compiles before building the harness around it.** A model that
   exports to ONNX successfully may still fail the Vitis AI compiler (unsupported ops, dynamic
   shapes, control flow). Run `onnxruntime.InferenceSession(path, providers=["VitisAIExecutionProvider"])`
   on each shortlisted model before committing to it. Discover export failures early.
6. **R9700 is a deliberate runtime exception (Phase 2).** HX 370 engines share one ORT build; R9700
   uses MIGraphX Python API direct because ORT's MIGraphX EP is broken on gfx1201. Same ONNX graphs
   and compiler preserve kernel comparability; document the runtime deviation. Power is Branch B
   (socket integration, no instantaneous reading). Transfer baseline must match `sync_every` cadence.

---

## 7. OPERATOR ENERGY SWEEP — MEASURED FINDINGS

Full production sweep **done** on HX 370. Headline metric: idle-subtracted `energy_per_op_J_mean`
in `benchmark/results/analysis_out.csv`. Env: `ryzen-ai-1.6.0`, `cwd: benchmark/`.

### Sessions (both measured + post-process validated)

| Session | Engines | Idle floor (30 s) | Key artifacts |
|---|---|---|---|
| `20260612_143854` | cpu FP32, igpu FP32, npu XINT8 | **216.22 J** | `results/runs_20260612_143854.csv`, `results/analysis_out.csv`, `results/runs_enriched.csv` |
| `20260615_145614_cpu_int8` | cpu_INT8 (same `_xint8.onnx`, CPU EP) | **326.49 J** | `results/runs_20260615_145614_cpu_int8.csv`, `results/analysis_out_20260615_145614_cpu_int8.csv` |

21 operator configs × 5 repeats + baselines per session. uProf under `results/uprof/<session_id>/`.

### cpu_INT8 control — purpose (not a deployment path)

Methodological control to split the FP32-cpu vs INT8-npu headline gap:

| ratio | formula | isolates |
|---|---|---|
| `precision_ratio` | cpu_FP32 / cpu_INT8 | 8-bit vs 32-bit on **same silicon** |
| `architecture_ratio` | cpu_INT8 / npu_INT8 | engine at **matched QDQ graph** |
| `original_gap` | cpu_FP32 / npu_INT8 | prior cross-engine headline |

Sanity: `precision_ratio × architecture_ratio = original_gap` — identity = 1.000 by algebra (**consistency check only**; see idle-drift caveat below).

### Headline decomposition result

On **GEMMs and heavy convs** (`ffn_gemm`, `qkv_proj_gemm`, `out_proj_gemm`, `downsample_conv2d`,
`patch_embed_conv2d`, `depthwise_conv2d`): **`precision_ratio` < 1 on all** (INT8 *penalizes* CPU via
Q/DQ overhead) and **`architecture_ratio` ≈ 3–14×** (NPU wins at matched INT8). The NPU advantage on
these ops is **architectural (XDNA2)**, not a quantization artifact in the headline grid.

Cheap memory-bound ops (`softmax`, `gelu`, `residual_add`): `precision_ratio` ≈ **0.04–0.28**
(`avg_pool_token_mixer` excluded — LOW-TRUST, see trust table below) — QDQ-on-CPU is catastrophic;
quantizing them for a CPU path is energetically irrational.

### Sign-divergence (key insight decomposition surfaced)

**Headline winner ≠ best INT8 engine** when `original_gap` < 1 but `architecture_ratio` > 1:

| op | original_gap | architecture_ratio | meaning |
|---|---:|---:|---|
| `softmax[s1]` | 0.22 | 5.75 | FP32-cpu beats npu_INT8; if quantizing, NPU runs INT8 far better than cpu_INT8 |
| `depthwise_conv2d` | 0.52 | 3.15 | same pattern |

"Which engine wins at measured precision" and "which engine is better at matched INT8" are **different
decisions** — the headline `ENGINE_SELECTION` grid hid this; decomposition exposes it.

### Trust carry-forwards (apply to all future analysis)

| cell | status | action |
|---|---|---|
| `sra_conv2d × npu` | **UNTRUSTED** | Gate 2: CPU-class ~45 W, throughput collapse. Exclude from NPU ranking. |
| `attn_block_fused × npu` | **N/A** | Tier-2 fused graph not measured on NPU. |
| `avg_pool_token_mixer × cpu_int8` | **LOW-TRUST** | CV 20.7% on control session; flag, do not roll into range summaries. |

Orig session Gate 1: **0/62** LOW-TRUST on main sweep repeats.

### Idle-drift caveat (cross-session)

Raw idle differed **+51%** (216 J vs 327 J) between sessions. Resolved: **static-offset** character
(steady ~45 W active plateaus, clean inter-window drops) + **per-session** `(window − idle) / iters`.
**Identity = 1.000 is a consistency check only** (`cpu_INT8` cancels algebraically) — not validity
evidence. Future sweeps: capture idle in the **same** session as measures.

### Step 4 (operator→engine mapping)

**Complete (2026-06-17, validated).** Mechanistic routing on avg corner. Artifacts:
`benchmark/STEP4_OPERATOR_ENGINE_MAPPING.md`, `benchmark/step4_operator_engine_mapping.py`.
Full method and sign-divergence findings summarized in prior context — see archive if needed.

---

## 8. TIMELINE (supervisor's draft plan; started June 1, 2026)

| Phase | Duration | Notes |
|---|---|---|
| Getting familiar with NPU programming | 2 weeks | Done (GEMM plumbing + NPU branch). |
| Preparing the testbench for power evaluation on LLM functions | 1 week | Done (harness, operators, run_plan, three-engine EPs). |
| Measurements | 1 week | **Done (2026-06-12)** — three-engine production sweep captured. |
| Refinement and validation | 1 week | **Done (2026-06-17)** — trust validation, cpu_INT8 decomposition, Step 4 mapping. |
| Preparation of the report | 2 weeks | In progress (HX 370 numbers validated). |
| **Phase 2 — R9700 fourth engine** | ~1 week | **← current phase.** Day 1 (2026-06-23): MIGraphX-direct path + Branch B power validated. Sweep pending. |

**Operator ordering tip:** GEMM first as a *pipeline plumbing test* (simplest op, proves the loop),
then move immediately to **scaled-dot-product attention** as the first *real* operator — that's what
the repo and supervisor pointed at. Don't let the easy op (raw FP32 matmul) become a headline result;
an un-fused single GEMM can make the NPU look bad for boring reasons.

---

## 9. KEY LINKS

**Related work (goal is similar to these, but for NPUs):**
- https://arxiv.org/html/2409.04941v1
- https://arxiv.org/abs/2410.00907
- https://ieeexplore.ieee.org/document/6120962

**Benchmark repo (source of the model lists — Attention Mechanisms + Vision Transformers):**
- https://github.com/changzy00/pytorch-attention

**AMD Ryzen AI Software SDK:**
- https://www.amd.com/en/developer/resources/ryzen-ai-software.html

---

## 10. SUPERVISOR'S EMAILS (verbatim)

**Email 1 — overall plan:**
> First weeks you get familiar with programming NPUs and then the next couple of weeks you will
> start programming and getting measurements on the NPUs. I am sending you some related work.
> (The goal of the internship will be similar to these papers but for the NPUs)
> https://arxiv.org/html/2409.04941v1
> https://arxiv.org/abs/2410.00907
> https://ieeexplore.ieee.org/document/6120962
> Draft time plan — 2 weeks: getting familiar with NPUs programming; 1 week: Preparing the testbench
> for the power evaluation on LLMs functions; 1 week: Measurements; 1 week: refinement and validation;
> 2 weeks: preparation of the report.

**Email 2 — prep before week one:**
> The goal will be to run the following benchmarks on an AMD Ryzen AI 9 HX 370 that has CPU, GPU and
> NPUs and perform a design space exploration in terms of energy efficiency.
> https://github.com/changzy00/pytorch-attention
> The CPU and GPU will be easy to run. But for the NPU you will need to export the PyTorch attention
> models to ONNX or TorchScript, and then compile them using the AMD Ryzen AI Software SDK (which
> utilizes Vitis AI / ONNX Runtime execution providers). so check the following to get familiar with
> the SDK
> https://www.amd.com/en/developer/resources/ryzen-ai-software.html

---

## 11. TOOLING DECISIONS (current)

- **Cursor** — primary authoring environment for now (free until **June 28, 2026**). Reads the cloned
  `pytorch-attention` repo; used to draft export scripts.
- **Claude Code** — installed on the APU PC proactively, kept idle until a stronger multi-file agentic
  loop is needed. When used, runs in Cursor's integrated terminal (a clean, common combo).
- **Hermes Agent** — **deferred.** Revisit for cross-session tracking. Route sessions through
  Hermes when added; seed with `PROJECT_CONTEXT.md` + `PROJECT_ARCHIVE.md`. Hermes does not
  retroactively ingest past work not run through it.

---

## 12. CURRENT STATUS & TASKS

**Phase:** Phase 2 — R9700 (`RadeonR9700` branch).
**Phase 1:** HX 370 operator sweep **complete** — measured findings in §7; Step 4 mapping done (2026-06-17).
**Last update:** 2026-06-23 — Day 2: EP pivot (MIGraphX-direct) + Branch B power validated end-to-end.

**Dated session log (2026-06-02 → present):** `PROJECT_ARCHIVE.md` — use for Hermes and historical lookup.

### Phase 2 — R9700 Day 2 (2026-06-23)

**Decisions:** ROCm EP dead in ORT 1.23.1; MIGraphX EP broken on gfx1201 via ORT → **MIGraphX Python
API direct** (Chris signed off). Power: **Branch B** (`socket_power` integration; accumulator dead).
Committed on `RadeonR9700`: `harness.py`, `gpu_power.py`, `migraphx_smoke.py`, `migraphx_measure_one.py`,
`smoke_pathb.sh`, `validate_r9700_integrated.sh`, `diag_r9700.sh`, `mgx_core_test.sh`.

**Analysis fix (staged 2026-06-23, off-tower):** `parse_energy.amdsmi_window_energy` Branch-A guard
`e_end >= e_start` → `e_end > e_start`; `gpu_power.read_sample` skips `energy_uj` when accumulator==0.
Re-enrich validate trace → expect ~7 kJ, `window_energy_method=trapezoid`.

**Open (next session):**
- [off-tower] Re-enrich today's validate trace with staged parse/gpu_power fixes
- [tower] `operators.py --all-shapes`; sweep via `profile_indices_for_corner(op,'avg')` (not hardcoded index 1)
- [tower] r9700 idle + dispatch baselines (transfer baseline likely moot — confirm w/ Chris)
- [tower] 21-op sweep (MIGraphX-direct, **avg** corner)

### HX 370 / paper (backlog)

- [ ] Paper drafting from validated §7 numbers (scaffold; divergence section draftable first)
- [ ] F2 figure from validated mapping (`figures/F2_decomposition.png`) if not yet rendered
- [ ] Chris sign-off on model-level precision policy (§4)
- [ ] PoolFormer vs PvT control pair (architecture selection; §4)
- [ ] Track-2 real-mixer shape fixes (§4); full ViT shortlist scope (§4)
- [ ] `attn_block_fused×npu` remediation (VitisAI crash — fusion gap on NPU unmeasurable)
- [ ] Optional: re-measure `avg_pool_token_mixer×cpu_int8` (LOW-TRUST, CV 20.7%)
- [ ] small/large SDPA corners (deferred DSE expansion)

