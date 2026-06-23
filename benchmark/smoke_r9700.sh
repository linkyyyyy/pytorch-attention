#!/usr/bin/env bash
# smoke_r9700.sh — Phase 2 step-7 plumbing validation (host-feed, single op).
# Run from a clean standalone terminal with Cursor CLOSED.
set -u

BENCH_DIR="$HOME/ryzen_benchmarks/pytorch-attention/benchmark"
cd "$BENCH_DIR" || { echo "FATAL: cannot cd to $BENCH_DIR"; exit 1; }

RUN_ID="smoke_ffn_gemm_r9700_s0"
POWER_CSV="results/gpu_power/${RUN_ID}.csv"
RUNS_CSV="results/runs_smoke.csv"
ENRICHED_CSV="results/runs_smoke_enriched.csv"

mkdir -p results/gpu_power

python3 gpu_power.py --out "$POWER_CSV" --interval 0.1 --device-id 0 &
SP=$!

cleanup() {
  if kill -0 "$SP" 2>/dev/null; then
    kill "$SP" 2>/dev/null
    wait "$SP" 2>/dev/null
  fi
}
trap cleanup EXIT

sleep 1   # settle + a little idle baseline

python3 harness.py \
  --operator ffn_gemm \
  --engine r9700 \
  --mode measure \
  --shape-index 0 \
  --duration 5 \
  --warmup 2 \
  --repeats 1 \
  --device-id 0 \
  --run-id "$RUN_ID" \
  --outfile "$RUNS_CSV"
HARNESS_RC=$?

sleep 1
cleanup
trap - EXIT

if [ "$HARNESS_RC" -ne 0 ]; then
  echo "FATAL: harness exited $HARNESS_RC — skipping enrich"
  exit "$HARNESS_RC"
fi

python3 parse_energy.py \
  --power-backend amdsmi \
  --runs "$RUNS_CSV" \
  --gpu-power-dir results/gpu_power \
  --outfile "$ENRICHED_CSV"

echo "==================== ENRICHED ROW ===================="
cat "$ENRICHED_CSV"
echo "==================== POWER TAIL ======================"
tail -n 5 "$POWER_CSV"
