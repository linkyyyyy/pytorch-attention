#!/usr/bin/env bash
# smoke_pathb.sh — Path B E=P·t gate: resident-loop window + amd-smi sampler.
# Clean standalone terminal, Cursor CLOSED (this captures power).
set -u
cd "$HOME/ryzen_benchmarks/pytorch-attention/benchmark" || exit 1

RUN_ID="mgx_ffn_gemm_r9700_s0"
POWER_CSV="results/gpu_power/${RUN_ID}.csv"
WINDOW_LOG="results/${RUN_ID}_window.log"
mkdir -p results/gpu_power

python3 gpu_power.py --out "$POWER_CSV" --interval 0.1 --device-id 0 &
SP=$!
cleanup() { kill -0 "$SP" 2>/dev/null && { kill "$SP" 2>/dev/null; wait "$SP" 2>/dev/null; }; }
trap cleanup EXIT

sleep 1   # idle baseline before the window

python3 migraphx_measure_one.py \
  --graph onnx_graphs/ffn_gemm.onnx \
  --run-id "$RUN_ID" \
  --duration 30 --warmup 5 --sync-every 64 \
  | tee "$WINDOW_LOG"
RC=$?

sleep 1
cleanup
trap - EXIT

echo "==================== WINDOW MARKERS ===================="
grep -E "WINDOW_OPEN|WINDOW_CLOSE" "$WINDOW_LOG"
echo "==================== POWER WINDOW SLICE ================="
# rough in-window power readout: mean power_w over the timed window
python3 - "$POWER_CSV" "$WINDOW_LOG" << 'INNER_EOF'
import csv, sys, re
power_csv, wlog = sys.argv[1], sys.argv[2]
text = open(wlog).read()
t0 = float(re.search(r"t_start=([\d.]+)", text).group(1))
t1 = float(re.search(r"t_end=([\d.]+)", text).group(1))
pw, busy = [], []
with open(power_csv) as f:
    for row in csv.DictReader(f):
        try: ep = float(row["sample_epoch"])
        except (KeyError, ValueError): continue
        if t0 <= ep <= t1:
            if row.get("power_w"): pw.append(float(row["power_w"]))
            if row.get("gfx_busy_pct"): busy.append(float(row["gfx_busy_pct"]))
n = len(pw)
if n > 0:
    print(f"in-window samples: {n}")
    print(f"power_w mean: {sum(pw)/n:.1f} W")
    print(f"gfx_busy_pct: {sum(busy)/n:.1f}  (max {max(busy):.0f})")
else:
    print("No samples found in window!")
INNER_EOF
exit "$RC"
