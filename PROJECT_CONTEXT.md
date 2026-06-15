# PROJECT_CONTEXT.md

> **Purpose of this file.** This is a portable, tool-agnostic context document for an
> 8-week research internship. Paste it into any AI agent (Cursor, Claude Code, Gemini,
> Hermes, etc.) at the start of a session so the agent has full project context without
> re-explanation. Keep it updated as the single source of truth. When in doubt, this
> file wins over an agent's own memory.

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

Run AI-operator and model benchmarks on an **AMD Ryzen AI 9 HX 370** (which has a CPU, an
integrated GPU, and an NPU) and perform a **design space exploration (DSE) in terms of energy
efficiency** across all three compute engines.

- Energy is measured as **E = P · t** (power × execution time).
- The deliverable is a **short, publishable paper**.
- The core comparison must be **fair**: hold the runtime constant (ONNX Runtime) so that
  *only the hardware engine varies*. See §6.
- The intent is experimental: measure energy behavior of operators/models common to LLMs
  and Vision Transformers, and report **which engine wins for which computational pattern, and why.**

The headline result should be about **diversity of computational profile across engines**, not
a single narrow finding like "the NPU likes matmuls."

---

## 3. HARDWARE FACTS (important — do not conflate the two machines)

| Machine | Role | Details |
|---|---|---|
| **HX 370 tower** (lab-provided) | **All measurements** | AMD Ryzen AI 9 HX 370: Zen 5 CPU + Radeon 890M iGPU + XDNA NPU. Windows-based. This is where every benchmark runs. |
| **Personal laptop** (2024 ROG Strix G16) | **Development only** | Windows 11, NVIDIA RTX 4070. The CUDA/NVIDIA GPU is **irrelevant to this project** — it serves a separate CUDA-learning effort. Never used for measurement. |

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
dispatch-isolated on NPU).

`window_energy_J` stays **RAW** upstream (`WINDOW_ENERGY_IS_RAW=True`); all subtraction is in
`analysis.py`. Harness appends `_r{repeat_idx}` to base `--run-id` exactly once (strips trailing
`_r\d+` if caller already included it).

Per-engine comparison relies on **controlled execution** (one ORT execution provider per run) and
**baseline subtraction**, not separate hardware power rails. State this explicitly in the paper
methodology.

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

### Step 4 scoping (not yet executed)

Operator→engine mapping is **not** done. When executed: rank GEMM/conv NPU candidates on
`architecture_ratio` / matched-INT8 energy; rank FP32-deployment ops on FP32 grid; do not map on
`original_gap` where `precision_ratio` ≪ 1 or ≫ 1.

---

## 8. TIMELINE (supervisor's draft plan; started June 1, 2026)

| Phase | Duration | Notes |
|---|---|---|
| Getting familiar with NPU programming | 2 weeks | **← current phase.** Good Week-1 deliverable: a trivial GEMM "plumbing test" — PyTorch matmul → ONNX export → ORT session on CPU EP → clean timing harness — to prove the whole export-and-measure loop end to end before adding quantization/real ops. |
| Preparing the testbench for power evaluation on LLM functions | 1 week | Lock model selection (§4), build the measurement harness. |
| Measurements | 1 week | |
| Refinement and validation | 1 week | |
| Preparation of the report | 2 weeks | The short paper. |

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
- **Hermes Agent** — **deferred.** Revisit once real benchmarking/measurement is underway and a richer
  cross-session context tracker is genuinely needed (and once it's less brand-new on Windows). Note:
  Hermes memory is a local FTS5-indexed SQLite log of sessions *run through it* — it does **not**
  retroactively ingest past work, so installing later loses nothing that wasn't routed through it.
  This file is the real fix for "re-explaining context," and seeds Hermes well whenever it's added.

---

## 12. CURRENT STATUS / CHANGELOG

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
    (see §4). Key correction vs initial draft: ViT vs PoolFormer is NOT a clean
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
  | `benchmark/run_sweep.bat` | CMD orchestrator: `conda activate ryzen-ai-1.6.0` → create session dir → uProf parent-wraps harness per (operator × engine × repeat) |
  | `benchmark/BENCHMARK_WORKFLOW.md` | User-facing CMD workflow guide (prerequisites, export, smoke test, baselines, sweep, troubleshooting) |
  | `benchmark/IMPLEMENTATION_BLUEPRINT.md` | Design spec: registry schema, harness CLI, dispatch baseline rationale, CSV schema, uProf parent-wrap pattern, locked defaults |
  | `benchmark/onnx_graphs/` | 16 pre-exported `.onnx` operator graphs (+ `dispatch_baseline.onnx`) for Netron inspection and ORT sessions |
  | `benchmark/parse_energy.py` | Join uProf timechart + `runs.csv` → `runs_enriched.csv` (`window_energy_J`, `idle_energy_J`, `dispatch_energy_J`); `--test-toy` + `--plot` |
  | `benchmark/results/` | Empty placeholder — `runs.csv` and `uprof/<SESSION_TS>/` populate during measurement |
  | `directives/measurement_harness_spec.md` | Protocol spec: measurement philosophy, session options, loop structure, baselines, uProf integration, CSV schema, validity checklist |
  | `directives/operator_architecture_selection.md` | Architecture & operator selection rationale (Opus-generated; authoritative cluster definitions) |

- **2026-06-08 (Day 5 — tower verification COMPLETE):** Full harness + uProf smoke test passed on HX 370 in `ryzen-ai-1.6.0`. Reference artifacts in `benchmark/uprof_smoke/`.

  **Tower verification checklist — all confirmed:**
  - [x] **Conda env:** `ryzen-ai-1.6.0`
  - [x] **ORT:** `1.23.0.dev20250928`; providers: `CPUExecutionProvider`, `DmlExecutionProvider`, `VitisAIExecutionProvider`
  - [x] **uProf:** `5.3.518.0`; child-launch via full interpreter path `C:\ProgramData\miniconda3\envs\ryzen-ai-1.6.0\python.exe` (not PATH alias)
  - [x] **uProf timestamp format:** wall-clock `HH:MM:SS:ms` (local tz), **not** epoch — conversion required to align with harness epoch markers (see §3)
  - [x] **Power scope:** `timechart --list` — Power counters only at **[Socket, Core]**; no per-rail iGPU/NPU (see §3 Measurement Scope Notes)
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
  `tier` / `block_id` / `shape_class` allowed values and example rows). Harness code under
  `benchmark/` (`operators.py`, `harness.py`, `run_sweep.bat`, `parse_energy.py`, `onnx_graphs/`, `results/`).

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

  **Window bounds (design decision):** Option **(b)** — `harness.py` now writes `t_start_epoch` / `t_end_epoch` columns to `runs.csv` (alongside stdout `[WINDOW_OPEN]`/`[WINDOW_CLOSE]` markers). `parse_energy.py` reads these directly; falls back to per-run `.log` parsing if columns are absent (option (a) compatibility).

  **`parse_energy.py`:** Deterministic post-processor — `parse_uprof_csv()`, trapezoidal `integrate_power()`; joins `runs.csv` to uProf CSVs by `run_id`; fills `window_energy_J`, joins `idle_energy_J` (common floor by `repeat_idx`) and `dispatch_energy_J` (per engine); dedups baseline rows. Subtraction stays in `analysis.py`. `--plot` saves per-run PNG with `t_start`/`t_end` vertical markers (requires `matplotlib`).

  **Toy verification (`toy_igpu`, `AMDuProf-python-Timechart_Jun-08-2026_16-36-25/timechart.csv`):**
  - Window: `t_start=1780925790.173200`, `t_end=1780925800.172677`, `iterations=13560`
  - `window_energy_J` ≈ **457 J** (mean package power ~46 W × ~10 s); `energy_per_op_J` ≈ **3.37×10⁻² J/op** (dispatch baseline 0 for toy)

  **GUI note:** `timechart` output is **CSV-only** (no `.uprof` DB; confirmed in `timechart --help`). `collect` is a CPU profiler, not the package-power path. CSV→`.uprof` conversion is impossible. GUI viewing of power data not pursued — visual verification via `parse_energy.py --plot` PNGs instead. uProf-GUI practice deferred to a future CPU/GPU-profiling exercise where `collect`/`gputrace` is the correct tool.

- **2026-06-10 (analysis layer — config extraction + fusion-gap pipeline):**
  - **`directives/attention_block_dimensions.md`:** N/D/head dims pulled from repo model files;
    proposed small/avg/large corners documented.
  - **`operators.py`:** `SDPA_BLOCK_CONFIGS` (three corners); isolated-op shapes derived from config
    (`fusion_member` tagging); `attn_block_fused` Tier-2 graphs (unfused ONNX chain).
  - **`runs.csv` schema:** `tier`, `block_id`, `shape_class`, `fusion_member` (+ harness CLI
    `--tier` / `--block-id` / `--shape-class`). Documented in `metadata.json` → `csv_schema`.
  - **`analysis.py`:** Built and validated on synthetic data (`python analysis.py --demo`):
    gap arithmetic confirmed **22 → 16 mJ = 6 mJ / 27.27%**; dispatch-baseline dedupe, repeat
    aggregation (mean ± std), `fusion_member` filtering, incomplete/NaN guard all verified.
  - **Unresolved assumption (must clear before trusting sweep output):** `WINDOW_ENERGY_IS_RAW=True`
    in `analysis.py` — assumes `window_energy_J` is gross (dispatch still inside). Whether the
    harness / `parse_energy.py` actually writes raw vs dispatch-netted `window_energy_J` is
    **unconfirmed**. Verify against **one real uProf-joined row** before any full sweep analysis.

- **2026-06-11 (NPU harness branch — code only, not measured yet):**
  - **`benchmark/npu/`:** NPU package — `path.py` (Quark XINT8, `make_npu_session()`, partition capture), `gemm_validate.py` (standalone plumbing test). Run: `cd benchmark && python -m npu.gemm_validate`.
  - **`harness.py`:** Third EP branch `--engine npu` reuses the **same** `_run_repeat` / `_duration_loop` path as CPU/iGPU. NPU supplies only: offline quantize → session (compile) → `sess.run()` (no `synchronize_outputs()`). Partition summary → `notes` + `results/partitions/{run_id}.json`. Q/DQ boundary CPU fallback recorded in notes.
  - **`operators.py`:** `xint8_onnx_path()`; `python operators.py --quantize-xint8-all` for batch offline prep.
  - **`run_sweep.bat`:** `ENGINES=cpu igpu npu`; capture idle + dispatch baselines per session. Pass base `--run-id` without `_r0` (harness suffixes repeats).
  - **Validated plumbing (pre-harness):** `ffn_gemm` XINT8 → VitisAI EP, ops_NPU=11 / VITIS_EP_CPU=2, relative L2 ≈ 0.017 vs FP32 CPU (`benchmark/npu/gemm_validate.py`).
  - **Cache rule:** delete `benchmark/.vaip_cache/` after NPU driver or VitisAI EP version change.
  - **Calibration policy:** synthetic n=16 distribution-matched tensors per graph (operator energy, not task accuracy) — documented in `metadata.json` → `npu_path` section.

- **2026-06-11 (fusion_member + registry-driven sweep plan):**
  - **`harness.py`:** emits `fusion_member` column on measure rows (from profile / entry default).
  - **`operators.py`:** `profile_indices_for_corner()`, `--sweep-plan --corner avg` prints/writes pipe-delimited plan; `SESSION_MEASURE_OPERATORS` + `SWEEP_SKIP_ENGINES` (attn_block_fused NPU).
  - **`run_session.bat` / `run_sweep.bat`:** pre-launch plan matrix + pause; per-op `shape_index` from registry (avg corner: SDPA N=197; q/k/v all three; single-profile ops index 0); `[WARN]` skip when corner missing.
  - **Known:** `attn_block_fused` NPU VAI EP crash (768 vs 3136 batch) — Tier-2 blocked on NPU; plumbing session `20260611_165420` needs post-process before trusting production sweep.

- **2026-06-05 (idle headline + run-id hygiene):**
  - **Idle:** `--mode idle` uses the same warmup/window timer as measure/dispatch (sleep loop body, no ORT). `parse_energy.py` joins idle uProf window → `idle_energy_J` on measure rows (by `repeat_idx`).
  - **Headline:** `energy_per_op_J` = idle-subtracted in `analysis.py`; `energy_per_op_dispatch` retained as secondary (NPU dispatch CPU-fallback caveat).
  - **Run-id:** harness `resolve_run_id()` appends `_r{repeat_idx}` once; strips trailing `_r\d+` from `--run-id`.

- **2026-06-15 (operator sweep + cpu_INT8 control — COMPLETE):** Production sweep session
  `20260612_143854` (cpu/igpu FP32, npu XINT8; idle 216.22 J) measured, post-processed, trust-validated.
  cpu_INT8 methodological control `20260615_145614_cpu_int8` (idle 326.49 J) measured + Step 2 validated.
  Decomposition complete: NPU GEMM/conv wins are **architectural** (`architecture_ratio` 3–14×,
  `precision_ratio` < 1 on all GEMMs). Sign-divergence on `softmax`, `depthwise_conv2d` (headline
  winner ≠ best INT8 engine). Trust carry-forwards: `sra_conv2d×npu` UNTRUSTED, `attn_block_fused×npu`
  N/A, `avg_pool_token_mixer×cpu_int8` LOW-TRUST. Idle +51% cross-session resolved (static offset +
  per-session subtraction; identity=1.000 is consistency-only). **Durable summary: §7.** Step 4
  (operator→engine mapping) not started. Handoff `.md` files in `benchmark/` are disposable scaffolding.

- **Open — pending supervisor input:**
  - **PoolFormer vs PvT control pair** (architecture selection).
  - **Model-level precision policy** for full ViT benchmarks (operator sweep used FP32 cpu/igpu + INT8 npu; see §7).

- **Deferred:** Track-2 real-mixer shape fixes (§4); Christoforos check-in; full 18-model shortlist scope
  decisions (§4); Step 4 operator→engine mapping (§7).