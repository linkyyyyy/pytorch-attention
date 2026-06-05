# Measurement Harness Spec — CPU & iGPU Operator Energy Benchmarking

**Platform:** AMD Ryzen AI 9 HX 370 tower · **OS for this phase:** native Windows (PowerShell)
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
uProf power trace can be aligned to it. Energy-per-op = (window energy − idle energy) ÷ iterations.

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
  harness.py          # the ONE shared measurement loop + argparse + logging + baseline
  operators.py        # registry: operator name -> (build_onnx_graph_fn, input_shape, dtype, cluster)
  onnx_graphs/        # optional: each op exported to .onnx for Netron inspection
  results/            # CSV output, one row per (operator x engine x repeat)
  run_sweep.ps1       # thin orchestrator: loops operators x engines x repeats, brackets uProf
```

One graph per operator (in the registry), one loop for all of them, two engines selected by flag.

---

## 2. Environment

- **Single conda/venv, Windows.** Install **`onnxruntime-directml`** (NOT plain `onnxruntime`
  — they conflict). This one package provides *both* `CPUExecutionProvider` and
  `DmlExecutionProvider`, so both engines run from one environment (good for comparability).
- Verify: `python -c "import onnxruntime as ort; print(ort.get_available_providers())"`
  must list `DmlExecutionProvider`.
- Also: `torch`, `onnx`, `numpy`. PyTorch only needed to *build/export* graphs, not to run them.
- **Export all ONNX graphs at opset 20** (DML EP ceiling; also what the NPU/Vitis path wants).
- AMD uProf 5.x installed (CLI: `AMDuProfCLI.exe`). GPU/driver up to date (DirectX 12 required).

---

## 3. DirectML EP — mandatory session options

The DML EP errors out unless these are set:

```python
so = ort.SessionOptions()
so.enable_mem_pattern = False                       # required by DML EP
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL # required by DML EP
sess = ort.InferenceSession(
    onnx_path, so,
    providers=[("DmlExecutionProvider", {"device_id": 0})],  # 890M is the only DX12 GPU on the tower -> 0; verify
)
```

For the CPU EP, **fix and log `intra_op_num_threads`** (default = all physical cores).
Energy scales with thread count, so it must be a documented constant across runs.

**Confirm the op actually ran where you think.** Set `so.enable_profiling = True`, run once,
open the profiling JSON, and check the node was placed on the intended EP and *not* silently
decomposed or fed back to CPU. DML falls back unsupported ops to CPU without warning — that
fallback is itself a result worth recording.

---

## 4. The measurement loop (per run)

1. **Warm-up (~5 s, discarded).** DML compiles shaders on first execution; first calls are
   wildly unrepresentative. Loop the inference for ~5 s and throw the timings away.
2. **Open window:** record `t_start` (epoch, `time.time()`) and a high-resolution
   `perf_counter()`. Print a clear marker line and write it to the log.
3. **Steady-state loop until ~30 s elapsed**, counting completed iterations. Do **not** fix
   the iteration count (see §5).
4. **Close window:** record `t_end`, completed `iterations`, and mean latency
   (`window_seconds / iterations`).
5. Sleep a few seconds (let clocks settle) before the next repeat.

---

## 5. Duration, not iteration count

A fixed iteration count is a trap: a softmax call is microseconds and a big conv is
milliseconds, so 10 k iterations gives a 0.3 s window for one op and a 5-minute window for
another. **Loop until a target wall-clock duration elapses and count iterations.** This
auto-normalizes across operators and gives a clean per-op divisor.

- **Target window: ~30 s steady-state.** Long enough for hundreds of uProf samples and to
  average out frequency ripple / dispatch jitter; short enough to limit thermal drift.
- **Warm-up: ~5 s, discarded.**
- **Repeats: 3–5 per (operator × engine).** Report **mean ± standard deviation** — reviewers
  want error bars. Discard outliers only with a logged reason (e.g. detected throttle).

---

## 6. Baselines — two of them

Total package power includes large static draw that swamps cheap ops, plus fixed per-call
dispatch overhead. Subtract both:

- **Idle baseline:** a same-length (~30 s) window with the process just `sleep()`-ing, no
  inference. Capture once per session (per engine, since an active DML context draws differently).
  `energy_per_op = (window_energy − idle_energy) / iterations`.
- **Dispatch baseline (recommended):** an `Identity`/near-no-op graph measured through the
  *exact same loop*. Subtracting it isolates compute from launch/dispatch cost. For cheap ops
  (softmax, GELU) the raw number is dispatch-dominated — that's a real finding, but only
  visible if you measure the dispatch floor.

---

## 7. uProf integration (the actual energy source)

uProf runs as a separate process and brackets the script's window.

- **CLI collection** (preferred for scripting): use `AMDuProfCLI` timechart/power collection
  for a fixed duration that brackets the window, output to CSV. Exact flags vary by uProf
  version — verify against the installed 5.x help (`AMDuProfCLI.exe timechart --help`).
- **Alignment:** the script's `t_start`/`t_end` markers map the script window onto the uProf
  timeline. Integrate power (∫P dt) between markers, OR — cleaner if available — read uProf's
  **cumulative energy counter** at the two window boundaries and take the difference.
- Capture **package/SoC power** at minimum; on this APU, CPU and iGPU share one die/rail, so
  per-engine isolation comes from the **idle-vs-active delta**, not from a separate iGPU rail.
- Log uProf's sampling interval; report it in the methodology.

`run_sweep.ps1` should: start uProf collection → run `harness.py` for one (op, engine, repeat)
→ stop collection → save the CSV tagged with the same run id the harness logged.

---

## 8. Experimental controls (validity checklist)

Run these the same way for **every** measurement, CPU and iGPU alike:

- [ ] Tower on AC; never on battery.
- [ ] **Single Windows power plan, pinned** (pick one — e.g. a fixed High-Performance plan —
      and never change it mid-study). Disable adaptive brightness / sleep / USB selective suspend.
- [ ] **Close background apps**, browsers, OneDrive/Dropbox sync, Windows Update; disable Wi-Fi
      if a run doesn't need it. Cursor's own IDE terminal counts — prefer a clean standalone
      PowerShell for the actual measured runs.
- [ ] **Same OS + same package** for CPU and iGPU (both via `onnxruntime-directml` on Windows).
- [ ] **Thermal control:** cooldown between repeats; monitor die temp (uProf or HWiNFO); watch
      for throttling and discard+flag any throttled window. Consider **interleaving** run order
      (op A cpu, op A igpu, op B cpu …) so slow thermal drift doesn't bias one engine.
- [ ] Fixed CPU `intra_op_num_threads`, logged.
- [ ] Warm-up discarded every run.
- [ ] Record driver version, uProf version, ORT version, opset — once, in the results header.

---

## 9. Output schema (one CSV row per run)

```
run_id, operator, cluster, engine, device_id, input_shape, dtype, opset,
repeat_idx, warmup_s, window_s, iterations_completed, wall_time_s, mean_latency_ms,
idle_power_w, active_power_w, window_energy_J, idle_energy_J,
energy_per_op_J, dispatch_energy_J, notes
```

Keep `cluster` (A / B1 / B2 from the selection doc) on every row so the analysis can group by
operator role without a re-join. This CSV is the input to the analysis/plots later.

---

## 10. Operators to cover (from operator_architecture_selection.md)

Build a registry entry for each, at the characteristic shapes already discussed:

- **Cluster A (global/structural):** Conv2D (patch-embed, strided), Conv2D (downsampling),
  Linear/GEMM (FFN), GELU, LayerNorm, GroupNorm, BatchNorm, residual Add.
- **Cluster B1 (attention-matrix):** Q/K/V+output projection GEMM, QKᵀ score matmul,
  cross-covariance matmul (XCiT), Softmax, attention·V matmul, spatial-reduction Conv2D (SRA).
- **Cluster B2 (non-attention mixers):** average pooling, depthwise Conv2D.
  *(Enrichment, only if approved: shift, token-mixing MLP.)*

Each measured on **both** CPU and iGPU this phase; NPU added later by swapping the EP.
