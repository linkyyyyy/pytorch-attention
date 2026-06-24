# Methodology & limitations — R9700 Phase 2 addendum

> Draft paragraphs for the paper. Session `20260624_133311`, avg corner (N=197, D=768).
> Cross-engine tables: `benchmark/results/r9700_vs_hx370_comparison.csv`.

## Measurement stack (four deltas vs HX 370 Phase 1)

Phase 1 measured CPU, iGPU, and NPU on an AMD Ryzen AI 9 HX 370 APU using a single ONNX Runtime
build (CPU / DirectML / Vitis AI execution providers) with package-level power from AMD uProf on
Windows. Phase 2 adds a discrete **AMD Radeon AI PRO R9700** (gfx1201, 32 GB GDDR6) on a separate
Ubuntu 24.04 host. Four deliberate asymmetries apply to any cross-tower comparison:

1. **Instrument:** HX 370 reports socket/package power (uProf); R9700 reports board-level
   `socket_power` via amd-smi (TBP-class: VRAM, VRM, fans; driver-estimated). Within-engine
   idle-subtracted energy-per-operation is internally consistent on each platform; absolute
   joules-per-op across instruments are not directly comparable without a common reference meter.

2. **Runtime:** HX 370 engines share one ORT session path; R9700 uses the **MIGraphX Python API**
   directly (`compile(..., offload_copy=False)`, resident GPU buffers, amortized `gpu_sync` every
   64 enqueues). ROCm EP is unavailable on the installed ORT+MIGraphX wheel for gfx1201. Operator
   graphs are identical; kernel dispatch path is not.

3. **Energy integration (Branch B):** On gfx1201 the SMU energy accumulator reads zero for the
   session duration. All R9700 window energies use **trapezoidal integration of `socket_power`**
   (`window_energy_method = trapz_power_w:no_counter` on every measure row). Phase 1 HX 370
   windows use uProf package-power integration (`uprof_socket0_trapz`). Counter-delta (Branch A)
   was not available on R9700.

4. **Idle baseline:** R9700 uses the **session mean** of four in-session idle windows (open, two
   mid-session, close) as the common floor (285.43 J for a 30 s window). Open-to-close idle drift
   on the discrete rail was **+57.4%** (225 J → 354 J), larger than the HX 370 cross-session
   idle drift (+51% between separate sessions). Headline `energy_per_op_J = (window_energy_J −
   idle_mean) / iterations` does not pair each repeat with its temporally adjacent idle capture.

## Headline metric (unchanged formula)

Across both phases we report idle-subtracted energy per operation:

`energy_per_op_J = (window_energy_J − idle_energy_J) / iterations_completed`

Secondary dispatch attribution uses `(window_energy_J − dispatch_energy_J) / iterations`. On R9700
the dispatch baseline runs on the GPU (MIGraphX); unlike the HX 370 NPU path (CPU-fallback dispatch),
dispatch-subtraction is valid for the r9700 engine.

## Stack-level efficiency (preliminary, avg corner only — session `20260624_133311`)

**Working hypothesis (not yet reviewer-proof):** on memory-class operators, the R9700 + MIGraphX stack
may show lower energy-per-op than the HX 370 iGPU + DirectML stack, consistent with discrete GDDR6
vs shared LPDDR5x — but this session is **avg corner only**; large/small corners are not yet measured.

**Metric decomposition (required for cross-engine claims):**

| Metric | R9700 formula | HX 370 Step4 (iGPU) |
|---|---|---|
| Headline (idle-subtracted) | `(window − idle_mean) / iters` | same (idle-subtracted FP32) |
| Dispatch-subtracted | `(window − dispatch_gpu) / iters` | **not in Step4 sheet** — iGPU remains idle-subtracted |

Cross-engine ratios therefore mix subtraction policies unless explicitly labeled. For R9700-internal
attribution, dispatch-subtraction is valid (GPU dispatch baseline). Comparing R9700 dispatch-subtracted
to HX370 idle-subtracted iGPU is **indicative** and must be stated as such.

**Idle-subtracted vs iGPU (memory cluster, 9 ops):** 7/9 lower on R9700. Exceptions:
**depthwise_conv2d** and **group_norm** (higher than iGPU on idle metric).

**Dispatch-subtracted vs iGPU (same cluster, asymmetric baseline):** 9/9 lower on R9700 — the two
idle-metric exceptions flip when GPU dispatch overhead is removed from the R9700 numerator. This
supports a **stack-level efficiency** reading (MIGraphX + discrete memory subsystem) rather than a
pure DRAM-bandwidth claim until large-corner sweeps confirm shape scaling.

**Do not claim** universal GDDR6 bandwidth dominance from this session alone. Cross-instrument,
runtime, corner-coverage, and subtraction-policy caveats apply to all ratio claims.

## Limitations (explicit)

- Cross-tower J/op rankings are **indicative**, not metrologically tied to a single power rail.
- R9700 and HX 370 differ in OS, driver stack, and compiler (MIGraphX vs DirectML/Vitis).
- NPU (XINT8) is excluded from the FP32 R9700-vs-GPU slice; Phase 1 precision decomposition still
  governs cpu-vs-npu claims.
- Single session; thermal drift visible in idle ramp (+57% open→close).
- `sra_conv2d` and other heavy convs show R9700 **higher** than HX 370 cpu/iGPU — investigate
  MIGraphX utilization and shape tags before drawing compute-bound conclusions.
