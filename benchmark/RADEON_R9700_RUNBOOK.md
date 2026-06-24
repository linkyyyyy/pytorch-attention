# Radeon R9700 — tower execution runbook (Phase 2)
Goal: tomorrow is execution, not decisions. Do steps in order; each gate must pass
before the next. Measurement-integrity: CLOSE Cursor/agents and use a clean standalone
terminal for every step that captures power (7, 9, 10). Source the ROCm env first.

## 0. Gate zero — ROCm EP present
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
  - Must list ROCMExecutionProvider. If absent (AMD's Radeon wheel is MIGraphX-centric),
    STOP: either install an ORT build with ROCm EP, or accept MIGraphX EP for the
    ISOLATED tier (single nodes don't fuse) and flag fused_block as non-comparable.

## 1. Versions (record into results/metadata.json notes)
cat /opt/rocm/.info/version ; python -c "import onnxruntime as ort; print(ort.__version__)"

## 2. Host + device identity (closes the device_id gate)
rocminfo | grep -E "Name:|gfx"        # find the gfx1201 agent
amd-smi list                          # note R9700 BDF + index
export ROCR_VISIBLE_DEVICES=<R9700_IDX>   # expose ONLY the R9700 => it is device 0
                                          # everywhere (ORT EP, OrtValue, amdsmi handle)
# re-run step 0 under the mask to confirm one device.

## 3. amdsmi energy verification + the resolution fix (decides counter vs integration)
python -c "import amdsmi; amdsmi.amdsmi_init(); \
h=amdsmi.amdsmi_get_processor_handles()[0]; print(amdsmi.amdsmi_get_energy_count(h)); \
print(amdsmi.amdsmi_get_power_info(h))"
  - Inspect keys. If energy returns {energy_accumulator, counter_resolution, ...}:
    FIX gpu_power.py:read_sample to set energy_uj = accumulator * counter_resolution
    (NOT accumulator alone). Confirm power_info key/unit for power_w too.
  - If energy API missing/zero -> Branch B (integration); that's fine, expected on RDNA.

## 4. Sampler smoke (5 s, GPU idle)
python gpu_power.py --out /tmp/smoke.csv --interval 0.1 --device-id 0   (Ctrl-C after ~5s)
  - power_w populated? gfx_busy moves? energy_uj populated yes/no = your A/B verdict.

## 5. Build FP32 graphs
python operators.py        # populates onnx_graphs/ ; confirm *.onnx exist

## 6. Parse-layer regression (no GPU needed)
pytest benchmark/tests/test_parse_energy_amdsmi.py     # must pass

## 7. Single-op live smoke (real GPU) — the IOBinding + placement gate
# clean terminal. backgrounded sampler + one short op:
python gpu_power.py --out results/gpu_power/smoke_ffn_gemm_r9700_s0.csv --device-id 0 & SP=$!
sleep 0.5
python harness.py --operator ffn_gemm --engine r9700 --mode measure --shape-index 0 \
  --duration 5 --warmup 2 --repeats 1 --device-id 0 --run-id smoke_ffn_gemm_r9700_s0 \
  --outfile results/runs_smoke.csv
sleep 0.5 ; kill $SP
  - REQUIRE in harness log: "[ROCM_IO] ... OrtValue device 'X' OK" (NOT HOST-FEED-FALLBACK)
    and "[EP_CHECK] ... EP_OK intended=ROCMExecutionProvider".
  - python parse_energy.py --power-backend amdsmi --runs results/runs_smoke.csv \
      --gpu-power-dir results/gpu_power --outfile results/runs_smoke_enriched.csv
    Confirm gfx_busy_mean_pct is HIGH (not <5) and no placement WARN.
  - If HOST-FEED-FALLBACK appears: fix the device string in probe_rocm_ortvalue BEFORE
    trusting any memory-bound row. This is the central-hypothesis tripwire.

## 8. Dry-run the sweep
python run_plan.py --plan <PLAN_CSV> --engines r9700 --power-backend amdsmi \
  --gpu-power-dir results/gpu_power --outfile results/runs_r9700.csv \
  --python <VENV_PY> --device-id 0 --dry-run
  - Eyeball the per-job sampler + (unwrapped) harness commands.

## 9. Baselines (run_plan does MEASURE rows only — capture these manually)
for MODE in idle dispatch; do
  RID=${MODE/dispatch/dispatch_baseline}_r9700
  python gpu_power.py --out results/gpu_power/$RID.csv --device-id 0 & SP=$!
  sleep 0.5
  python harness.py --engine r9700 --mode $MODE --duration 30 --warmup 5 --repeats 5 \
    --run-id $RID --outfile results/runs_r9700.csv --device-id 0
  sleep 0.5 ; kill $SP
done

## 10. The sweep (Cursor CLOSED, clean terminal)
# drop --dry-run from step 8. ~21 rows x 5 repeats x ~40 s ≈ plan for the wall time.

## 11. Enrich
python parse_energy.py --power-backend amdsmi --runs results/runs_r9700.csv \
  --gpu-power-dir results/gpu_power --outfile results/runs_r9700_enriched.csv

## 12. Gate review before trusting numbers
- No [WARN] gfx_busy placement warnings on measure rows (`gfx_busy_mean_pct` ≥ 90).
- **Method-string gate (gfx1201 / Branch B):** every measure row's `window_energy_method` must
  **start with `trapz_power_w`** (e.g. `trapz_power_w:counter_nonmonotonic`). **FAIL** if any row
  uses `counter_delta_uj` or other counter_delta path — the energy accumulator is dead on gfx1201.
- `sweep_r9700_avg.sh` runs this gate automatically after enrich (`validate_enriched_gate`).
- Pre-sweep: `GATE=1 ./sweep_r9700_avg.sh` — ffn_gemm avg s2 via run_plan + enrich (~3 min).
- idle_energy_J / dispatch_energy_J populated on measure rows (after baselines in full sweep).
- spot-check one row: window_energy_J ≈ mean_power_W × 30 s (~7 kJ on ffn_gemm avg); energy_per_op sane.
Only then are R9700 numbers analysis-grade. Commit manually (Lincoln).
