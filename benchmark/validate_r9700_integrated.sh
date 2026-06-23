#!/usr/bin/env bash
set -u
cd "$HOME/ryzen_benchmarks/pytorch-attention/benchmark" || exit 1
RUN_ID="ffn_gemm_r9700_s0_validate"
POWER_CSV="results/gpu_power/${RUN_ID}.csv"
RUNS_CSV="results/runs_r9700_validate.csv"
ENRICHED="results/runs_r9700_validate_enriched.csv"
mkdir -p results/gpu_power

python3 gpu_power.py --out "$POWER_CSV" --interval 0.1 --device-id 0 &
SP=$!
cleanup(){ kill -0 "$SP" 2>/dev/null && { kill "$SP" 2>/dev/null; wait "$SP" 2>/dev/null; }; }
trap cleanup EXIT
sleep 1

python3 harness.py --engine r9700 --operator ffn_gemm --mode measure \
  --shape-index 0 --duration 30 --warmup 5 --sync-every 64 --repeats 1 \
  --device-id 0 --run-id "$RUN_ID" --outfile "$RUNS_CSV"
RC=$?

sleep 1; cleanup; trap - EXIT
[ "$RC" -ne 0 ] && { echo "FATAL: harness exited $RC"; exit "$RC"; }

python3 parse_energy.py --power-backend amdsmi --runs "$RUNS_CSV" \
  --gpu-power-dir results/gpu_power --outfile "$ENRICHED"
echo "==================== ENRICHED ROW ===================="
cat "$ENRICHED"
