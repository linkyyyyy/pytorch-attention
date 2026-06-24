#!/usr/bin/env bash
# sweep_r9700_avg.sh — avg-corner R9700 session driver (MIGraphX-direct harness + external gpu_power).
#
# harness.py NEVER spawns gpu_power.py; this script (and run_plan.py) bracket each harness call.
# RUN=1 invocation: use a clean standalone terminal with Cursor/editor CLOSED (package noise).
# parse_energy averages ALL idle rows (open, mid*, close) into one idle floor; distinct run-ids
# preserve per-idle power CSVs for the thermal-drift curve review in the SUMMARY.
# RUN=1   — full avg-corner sweep (~75–90 min, Cursor CLOSED)
# GATE=1  — 3-min avg-corner smoke: ffn_gemm s2 via run_plan + enrich + gate check
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || { echo "FATAL: cannot cd to $SCRIPT_DIR"; exit 1; }

SESSION="$(date +%Y%m%d_%H%M%S)_r9700"
OUT="results/runs_${SESSION}.csv"
GP="results/gpu_power"
PLAN="results/plan_r9700_avg.txt"
NCHUNKS="${NCHUNKS:-3}"
RUN="${RUN:-0}"
GATE="${GATE:-0}"

PYTHON="$(which python3)"

# ---------------------------------------------------------------------------
# gpu_power sampler lifecycle — orphan samplers corrupt CSV (concurrent writes).
# ---------------------------------------------------------------------------
kill_all_gpu_power_samplers() {
  if pgrep -f 'gpu_power\.py' >/dev/null 2>&1; then
    echo "[sampler] killing stale gpu_power.py process(es)"
    pkill -TERM -f 'gpu_power\.py' 2>/dev/null || true
    sleep 0.5
    pkill -KILL -f 'gpu_power\.py' 2>/dev/null || true
    sleep 0.2
  fi
}

stop_sampler_pgid() {
  local sp="$1"
  local pgid
  pgid=$(ps -o pgid= -p "$sp" 2>/dev/null | tr -d ' ')
  if [ -z "$pgid" ]; then
    return 0
  fi
  if kill -0 "$sp" 2>/dev/null; then
    kill -TERM -- "-${pgid}" 2>/dev/null || kill -TERM "$sp" 2>/dev/null || true
  fi
  local i=0
  while kill -0 "$sp" 2>/dev/null && [ "$i" -lt 50 ]; do
    sleep 0.1
    i=$((i + 1))
  done
  if kill -0 "$sp" 2>/dev/null; then
    kill -KILL -- "-${pgid}" 2>/dev/null || kill -KILL "$sp" 2>/dev/null || true
  fi
  wait "$sp" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# onnx path for one plan row (matches operators._onnx_path)
# ---------------------------------------------------------------------------
onnx_path_for_row() {
  local op="$1" idx="$2"
  if [ "$idx" = "0" ]; then
    echo "onnx_graphs/${op}.onnx"
  else
    echo "onnx_graphs/${op}_s${idx}.onnx"
  fi
}

# ---------------------------------------------------------------------------
# bracket_harness MODE RUN_ID [extra harness args...]
# Mirrors validate_r9700_integrated.sh / smoke_r9700.sh gpu_power start/stop pattern.
# Baseline/measure failures do not abort the session script.
# ---------------------------------------------------------------------------
bracket_harness() {
  local mode="$1" run_id="$2"
  shift 2
  local power_csv="${GP}/${run_id}.csv"
  echo "[bracket] mode=${mode} run_id=${run_id} power=${power_csv}"

  setsid python3 gpu_power.py --out "$power_csv" --interval 0.1 --device-id 0 &
  local sp=$!

  _bracket_cleanup() {
    stop_sampler_pgid "$sp"
  }
  trap _bracket_cleanup RETURN

  sleep 1

  python3 harness.py \
    --engine r9700 \
    --mode "$mode" \
    --run-id "$run_id" \
    --outfile "$OUT" \
    --device-id 0 \
    "$@"
  local rc=$?

  sleep 1
  _bracket_cleanup
  trap - RETURN

  if [ "$rc" -ne 0 ]; then
    echo "[WARN] bracket_harness mode=${mode} run_id=${run_id} harness exited $rc"
  fi
  return 0
}

# ---------------------------------------------------------------------------
# Post-enrich validation: Branch-B method gate + energy/busy sanity
# Pass: window_energy_method starts with "trapz_power_w" (gfx1201 Branch B).
# Fail: counter_delta_uj / counter_delta* (dead accumulator on gfx1201).
# ---------------------------------------------------------------------------
validate_enriched_gate() {
  local enriched_csv="$1"
  local label="${2:-gate}"

  echo ""
  echo "==================== ENRICHED GATE (${label}) ===================="

  if [ ! -f "$enriched_csv" ]; then
    echo "[GATE FAIL] enriched CSV missing: ${enriched_csv}"
    return 1
  fi

  python3 - "$enriched_csv" << 'GATE_EOF'
import csv
import sys

path = sys.argv[1]
SKIP = {"idle", "dispatch_baseline", "dispatch"}
rows = [r for r in csv.DictReader(open(path, newline="", encoding="utf-8"))
        if r.get("operator", "") not in SKIP]

if not rows:
    print("[GATE FAIL] no measure rows in enriched CSV")
    sys.exit(1)

def method_ok(method: str) -> tuple[bool, str]:
    m = (method or "").strip()
    if not m:
        return False, "empty window_energy_method"
    if m.startswith("trapz_power_w"):
        return True, ""
    if m == "counter_delta_uj" or m.startswith("counter_delta"):
        return False, f"dead counter_delta path on gfx1201 ({m!r})"
    return False, f"unexpected window_energy_method ({m!r}); want trapz_power_w:*"

failures = 0
for r in rows:
    rid = r.get("run_id", "?")
    method = r.get("window_energy_method", "")
    ok, reason = method_ok(method)
    try:
        ej = float(r.get("window_energy_J") or "nan")
    except ValueError:
        ej = float("nan")
    try:
        busy = float(r.get("gfx_busy_mean_pct") or "nan")
    except ValueError:
        busy = float("nan")

    print(f"  {rid}: method={method!r} window_energy_J={ej:.1f} gfx_busy_mean_pct={busy:.1f}")

    if not ok:
        print(f"    [GATE FAIL] {reason}")
        failures += 1
    elif ej < 4000 or ej > 12000:
        print(f"    [GATE WARN] window_energy_J={ej:.1f} outside ~7 kJ band (4000–12000 J)")
    if busy < 90:
        print(f"    [GATE WARN] gfx_busy_mean_pct={busy:.1f} < 90 (placement?)")

if failures:
    print(f"[GATE FAIL] {failures} measure row(s) failed method-string gate")
    sys.exit(1)

print("[GATE OK] all measure rows use trapz_power_w (Branch B)")
sys.exit(0)
GATE_EOF
}

# ---------------------------------------------------------------------------
# GATE=1 — single-op avg-corner smoke (ffn_gemm shape_index 2) via run_plan + enrich
# ---------------------------------------------------------------------------
run_avg_corner_gate() {
  local gate_plan="results/plan_gate_ffn_gemm_s2.txt"
  local gate_out="results/runs_gate.csv"
  local gate_enriched="results/runs_gate_enriched.csv"
  local gate_row

  gate_row=$(grep -E '^ffn_gemm\|2\|' "$PLAN" | head -n 1 || true)
  if [ -z "$gate_row" ]; then
    echo "FATAL: no ffn_gemm shape_index=2 row in ${PLAN}"
    return 1
  fi

  {
    head -n 1 "$PLAN"
    echo "$gate_row"
  } > "$gate_plan"

  echo "==================== AVG-CORNER GATE (GATE=1) ===================="
  echo "plan_row: ${gate_row}"
  echo "run_id base (via run_plan): r9700_avg_ffn_gemm_s2"
  echo "out: ${gate_out}  enriched: ${gate_enriched}"
  echo "(~3 min: warmup 5s + 30s window × 1 repeat + sampler bracket)"
  echo ""

  kill_all_gpu_power_samplers
  rm -f "$gate_out" "$gate_enriched" "${GP}/r9700_avg_ffn_gemm_s2.csv"

  python3 run_plan.py \
    --plan "$gate_plan" \
    --engines r9700 \
    --power-backend amdsmi \
    --gpu-power-dir "$GP" \
    --outfile "$gate_out" \
    --python "$PYTHON" \
    --device-id 0 \
    --duration 30 \
    --warmup 5 \
    --repeats 1 \
    --cooldown 0
  local plan_rc=$?
  if [ "$plan_rc" -ne 0 ]; then
    echo "[GATE FAIL] run_plan.py exited ${plan_rc}"
    return 1
  fi

  python3 parse_energy.py \
    --power-backend amdsmi \
    --runs "$gate_out" \
    --gpu-power-dir "$GP" \
    --outfile "$gate_enriched"

  validate_enriched_gate "$gate_enriched" "ffn_gemm_s2_avg"
}

# ---------------------------------------------------------------------------
# Step 1 — plan + dirs
# ---------------------------------------------------------------------------
mkdir -p "$GP"

python3 operators.py --sweep-plan --corner avg --plan-out "$PLAN"

# ---------------------------------------------------------------------------
# Step 2 — print plan, resolved rows, count; assert onnx graphs exist
# ---------------------------------------------------------------------------
echo "==================== SWEEP PLAN ===================="
cat "$PLAN"
echo "==================== RESOLVED ROWS ================="
{
  read -r _header
  while IFS='|' read -r op idx block_id shape_class tier fusion_member input_shape skip_engines; do
    [ -z "$op" ] && continue
    echo "operator=${op} shape_index=${idx} shape_class=${shape_class} block_id=${block_id}"
  done < "$PLAN"
} | tee /tmp/sweep_r9700_rows.txt

PLAN_DATA_ROWS=$(($(wc -l < "$PLAN") - 1))
echo "[plan] data_rows=${PLAN_DATA_ROWS} nchunks=${NCHUNKS}"

MISSING=()
while IFS='|' read -r op idx block_id shape_class tier fusion_member input_shape skip_engines; do
  [ -z "$op" ] && continue
  gpath="$(onnx_path_for_row "$op" "$idx")"
  if [ ! -f "$gpath" ]; then
    MISSING+=("$gpath (operator=$op shape_index=$idx shape_class=$shape_class)")
  fi
done < <(tail -n +2 "$PLAN")

if [ "${#MISSING[@]}" -gt 0 ]; then
  echo "FATAL: missing avg-corner ONNX graph(s). Run: python3 operators.py --all-shapes"
  printf '  %s\n' "${MISSING[@]}"
  exit 1
fi

# ---------------------------------------------------------------------------
# Step 3 — eyeball gate (RUN != 1 → print sequence, exit 0)
# ---------------------------------------------------------------------------
chunk_size() {
  local total="$1" n="$2"
  echo $(( (total + n - 1) / n ))
}

CSIZE=$(chunk_size "$PLAN_DATA_ROWS" "$NCHUNKS")

echo "==================== WOULD-RUN SEQUENCE (RUN=${RUN}) ===================="
echo "SESSION=${SESSION}"
echo "OUT=${OUT}  GP=${GP}  PLAN=${PLAN}"
echo "1. bracket_harness idle     idle_r9700_open           --duration 30 --repeats 1"
echo "2. bracket_harness dispatch dispatch_r9700_base      --duration 30 --repeats 1"
for i in $(seq 1 "$NCHUNKS"); do
  start=$(( (i - 1) * CSIZE + 1 ))
  end=$(( i * CSIZE ))
  [ "$end" -gt "$PLAN_DATA_ROWS" ] && end="$PLAN_DATA_ROWS"
  echo "3.${i} run_plan.py --plan results/plan_chunk_${i}.txt  (plan rows ${start}-${end} of ${PLAN_DATA_ROWS})"
  if [ "$i" -lt "$NCHUNKS" ]; then
    echo "   bracket_harness idle idle_r9700_mid${i} --duration 30 --repeats 1"
  fi
done
echo "4. bracket_harness idle idle_r9700_close --duration 30 --repeats 1"
echo "5. parse_energy.py → results/runs_${SESSION}_enriched.csv"
echo ""
echo "To execute: RUN=1 ./sweep_r9700_avg.sh"
echo "Avg gate:   GATE=1 ./sweep_r9700_avg.sh   (~3 min, Cursor CLOSED)"
echo "Optional:    NCHUNKS=3 RUN=1 ./sweep_r9700_avg.sh"

if [ "$GATE" = "1" ]; then
  run_avg_corner_gate
  exit $?
fi

if [ "$RUN" != "1" ]; then
  echo "[gate] RUN!=1 — eyeball only, exiting 0"
  exit 0
fi

kill_all_gpu_power_samplers

# ---------------------------------------------------------------------------
# Step 5 — opening baselines
# run-id must NOT end in _r<digits>: harness _strip_repeat_suffix would eat it as a repeat suffix
# ---------------------------------------------------------------------------
bracket_harness idle     idle_r9700_open           --duration 30 --repeats 1
bracket_harness dispatch dispatch_r9700_base       --duration 30 --repeats 1

# ---------------------------------------------------------------------------
# Step 6 — split plan into chunk files (header + contiguous data rows)
# ---------------------------------------------------------------------------
PLAN_HEADER=$(head -n 1 "$PLAN")
mapfile -t PLAN_ROWS < <(tail -n +2 "$PLAN")

for i in $(seq 1 "$NCHUNKS"); do
  chunk_file="results/plan_chunk_${i}.txt"
  start=$(( (i - 1) * CSIZE ))
  end=$(( start + CSIZE ))
  [ "$end" -gt "${#PLAN_ROWS[@]}" ] && end="${#PLAN_ROWS[@]}"
  {
    echo "$PLAN_HEADER"
    if [ "$start" -lt "$end" ]; then
      printf '%s\n' "${PLAN_ROWS[@]:$start:$((end - start))}"
    fi
  } > "$chunk_file"
  echo "[chunk] wrote ${chunk_file} rows=$((end - start))"
done

# ---------------------------------------------------------------------------
# Step 7 — measure chunks (continue-on-error; no set -e)
# ---------------------------------------------------------------------------
declare -a CHUNK_RCS=()

for i in $(seq 1 "$NCHUNKS"); do
  echo ""
  echo "==================== CHUNK ${i}/${NCHUNKS} ===================="
  python3 run_plan.py \
    --plan "results/plan_chunk_${i}.txt" \
    --engines r9700 \
    --power-backend amdsmi \
    --gpu-power-dir "$GP" \
    --outfile "$OUT" \
    --python "$PYTHON" \
    --device-id 0 \
    --duration 30 \
    --warmup 5 \
    --repeats 5 \
    --cooldown 5 \
    2>&1 | tee "results/chunk_${i}.log"
  CHUNK_RCS[$i]=${PIPESTATUS[0]}

  if [ "$i" -lt "$NCHUNKS" ]; then
    bracket_harness idle "idle_r9700_mid${i}" --duration 30 --repeats 1
  fi
done

# ---------------------------------------------------------------------------
# Step 8 — closing idle bookend
# ---------------------------------------------------------------------------
bracket_harness idle idle_r9700_close --duration 30 --repeats 1

# ---------------------------------------------------------------------------
# Step 9 — SUMMARY (always)
# ---------------------------------------------------------------------------
echo ""
echo "==================== SESSION SUMMARY ===================="
echo "SESSION=${SESSION}  OUT=${OUT}"

for i in $(seq 1 "$NCHUNKS"); do
  rc="${CHUNK_RCS[$i]:-?}"
  done_line=$(grep -E '^\[run_plan\] done ' "results/chunk_${i}.log" 2>/dev/null | tail -n 1 || true)
  echo "chunk ${i}: exit_code=${rc}  ${done_line:-[no run_plan done line]}"
done

echo ""
echo "--- failed / warned ops (from chunk logs) ---"
grep -hE '\[WARN\].*returned exit code|\[WARN\].*sampler produced' results/chunk_*.log 2>/dev/null || echo "(none)"

echo ""
echo "--- idle drift curve (mean power_w W from gpu_power CSV) ---"
mean_power_w() {
  local f="$1"
  if [ ! -f "$f" ]; then
    echo "${f}: MISSING"
    return
  fi
  awk -F, 'NR==1{for(i=1;i<=NF;i++)if($i=="power_w")c=i; next}
    c && $c!="" && $c+0==$c {s+=$c; n++}
    END{if(n>0) printf "%s: mean power_w=%.1f W (n=%d)\n", FILENAME, s/n, n; else print FILENAME": no numeric power_w samples"}' "$f"
}
mean_power_w "${GP}/idle_r9700_open.csv"
for i in $(seq 1 $((NCHUNKS - 1))); do
  mean_power_w "${GP}/idle_r9700_mid${i}.csv"
done
mean_power_w "${GP}/idle_r9700_close.csv"

# ---------------------------------------------------------------------------
# Step 10 — enrich
# ---------------------------------------------------------------------------
ENRICHED="results/runs_${SESSION}_enriched.csv"
echo ""
echo "==================== ENRICH ===================="
python3 parse_energy.py \
  --power-backend amdsmi \
  --runs "$OUT" \
  --gpu-power-dir "$GP" \
  --outfile "$ENRICHED"

echo ""
echo "REMINDER: Branch-B parse_energy fix (e_end > e_start; gpu_power skips energy_uj when"
echo "accumulator==0) must be committed on RadeonR9700 for window_energy_method=trapz_power_w:*."
echo "Without it, enrich may select counter_delta→0 J and under-report window energy."
echo "Enriched: ${ENRICHED}"

validate_enriched_gate "$ENRICHED" "full_sweep" || exit 1
