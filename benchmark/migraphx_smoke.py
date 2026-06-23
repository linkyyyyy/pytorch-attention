#!/usr/bin/env python3
"""
migraphx_smoke.py — Path B mechanism validation on ONE operator.
Confirms the MIGraphX Python API can parse+compile our ONNX graph, hold inputs
resident on the GPU (offload_copy=False), and loop the kernel; and determines
whether run() is synchronous or async. NOT a measurement, NOT the sweep.
Run from the benchmark dir.
"""
import time
import numpy as np
import migraphx

GRAPH = "onnx_graphs/ffn_gemm.onnx"
N = 2000

print("=== API surface (so we can correct any version-specific name) ===")
print("migraphx version:", getattr(migraphx, "__version__", "n/a"))
print("module:", [a for a in dir(migraphx) if not a.startswith("_")])

prog = migraphx.parse_onnx(GRAPH)
print("program attrs:", [a for a in dir(prog) if not a.startswith("_")])

# offload_copy=False -> inputs/outputs stay resident; we manage to_gpu/from_gpu.
prog.compile(migraphx.get_target("gpu"), offload_copy=False)
print("[ok] compiled to gpu (offload_copy=False -> resident)")

pshapes = prog.get_parameter_shapes()
print("=== parameters ===")
for name in pshapes:
    print(f"  {name}: lens={pshapes[name].lens()}")

# upload inputs once (single H2D)
gpu_inputs = {}
for name in pshapes:
    arr = np.random.rand(*pshapes[name].lens()).astype(np.float32)
    gpu_inputs[name] = migraphx.to_gpu(migraphx.argument(arr))
print("[ok] inputs uploaded to GPU (resident)")

out = prog.run(gpu_inputs)
print("[ok] single run -> type", type(out), "len", len(out) if hasattr(out, "__len__") else "?")

# --- sync vs async probe ---
t0 = time.perf_counter()
for _ in range(N):
    prog.run(gpu_inputs)
t_noread = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(N):
    r = prog.run(gpu_inputs)
    _ = np.array(migraphx.from_gpu(r[0]))   # forces completion (D2H)
t_read = time.perf_counter() - t0

print("=== sync/async probe (driver perf measured ~117 us/iter for this kernel) ===")
print(f"  no-readback : {t_noread*1000:8.2f} ms total | {t_noread*1e6/N:7.2f} us/iter")
print(f"  with-readback:{t_read*1000:8.2f} ms total | {t_read*1e6/N:7.2f} us/iter")
print("  no-readback << 117us => run() is async-enqueue (sync once at window end)")
print("  no-readback ~ 117us  => run() blocks per call")
