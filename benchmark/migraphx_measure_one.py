#!/usr/bin/env python3
"""
migraphx_measure_one.py — Path B power-bracketed single-operator window.
Resident inputs (offload_copy=False, one H2D), async-enqueue + gpu_sync draining
so the wall-clock window contains real GPU work. Emits window epochs to stdout for
parse_energy.py to slice the gpu_power trace. NOT the sweep — one op, gate only.
"""
import argparse
import time
import numpy as np
import migraphx

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="onnx_graphs/ffn_gemm.onnx")
    ap.add_argument("--run-id", default="mgx_ffn_gemm_r9700_s0")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--warmup", type=float, default=5.0)
    ap.add_argument("--sync-every", type=int, default=64,
                    help="enqueue depth before draining with gpu_sync (window granularity)")
    args = ap.parse_args()

    prog = migraphx.parse_onnx(args.graph)
    prog.compile(migraphx.get_target("gpu"), offload_copy=False)

    pshapes = prog.get_parameter_shapes()
    gpu_inputs = {
        name: migraphx.to_gpu(migraphx.argument(
            np.random.rand(*pshapes[name].lens()).astype(np.float32)))
        for name in pshapes
    }

    def drained_batch(depth: int) -> None:
        for _ in range(depth):
            prog.run(gpu_inputs)
        migraphx.gpu_sync()   # block until the enqueued batch has actually executed

    # warmup (kept out of the window)
    print(f"[WARMUP_START] run_id={args.run_id} warmup_s={args.warmup} sync_every={args.sync_every}")
    w_end = time.perf_counter() + args.warmup
    while time.perf_counter() < w_end:
        drained_batch(args.sync_every)
    migraphx.gpu_sync()
    print(f"[WARMUP_END] run_id={args.run_id}")

    # timed window — epochs printed for parse_energy alignment
    t_start = time.time()
    print(f"[WINDOW_OPEN] run_id={args.run_id} t_start={t_start:.6f} engine=r9700 graph={args.graph}")
    iters = 0
    perf0 = time.perf_counter()
    deadline = perf0 + args.duration
    while time.perf_counter() < deadline:
        drained_batch(args.sync_every)
        iters += args.sync_every
    migraphx.gpu_sync()
    wall = time.perf_counter() - perf0
    t_end = time.time()
    print(f"[WINDOW_CLOSE] run_id={args.run_id} t_end={t_end:.6f} "
          f"iterations={iters} wall_time_s={wall:.6f} mean_latency_ms={wall*1000.0/iters:.6f}")

if __name__ == "__main__":
    main()
