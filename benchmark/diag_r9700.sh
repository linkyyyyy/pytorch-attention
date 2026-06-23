#!/usr/bin/env bash
set -u
cd "$HOME/ryzen_benchmarks/pytorch-attention/benchmark" || exit 1

echo "===== (1) is migraphx-driver available? ====="
which migraphx-driver 2>/dev/null || ls /opt/rocm/bin 2>/dev/null | grep -i migraphx || echo "migraphx-driver NOT on PATH"

echo
echo "===== (2) CPU EP anchor: does ffn_gemm run on CPU on this box? ====="
python3 harness.py --operator ffn_gemm --engine cpu --mode measure --shape-index 0 \
  --duration 3 --warmup 1 --repeats 1 --outfile /tmp/cpu_anchor.csv 2>&1 | tail -n 15

echo
echo "===== (3) MIGraphX eval trace: which instruction throws the variant error? ====="
MIGRAPHX_TRACE_EVAL=1 python3 harness.py --operator ffn_gemm --engine r9700 --mode measure \
  --shape-index 0 --duration 3 --warmup 1 --repeats 1 --outfile /tmp/mgx_trace.csv 2>&1 | tail -n 50
