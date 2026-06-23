#!/usr/bin/env bash
set -u
cd "$HOME/ryzen_benchmarks/pytorch-attention/benchmark" || exit 1
G=onnx_graphs/ffn_gemm.onnx

echo "===== (A) migraphx-driver run: compile + execute on GPU, NO onnxruntime ====="
migraphx-driver run "$G" 2>&1 | tail -n 40

echo
echo "===== (B) migraphx-driver perf: compile + timed GPU run ====="
migraphx-driver perf "$G" 2>&1 | tail -n 25
