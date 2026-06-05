"""
gemm_plumbing_test.py
=====================================================================
WEEK 1 PIPELINE PLUMBING TEST  (CPU Execution Provider only)

PURPOSE
    Prove that data can flow through every joint of the toolchain:
        PyTorch  ->  ONNX export  ->  ONNX Runtime load  ->  inference  ->  timing
    This is NOT a benchmark. The workload (a single matrix multiply) is
    deliberately trivial so that if ANY stage breaks, the cause is
    unambiguous. No quantization, no DirectML/Vitis EPs, no power capture.
    Those come later, once this passes.

HOW TO RUN
    1. Activate your Ryzen AI conda environment, e.g.:
           conda activate ryzen-ai-1.6.0
    2. From a clean standalone terminal (cmd or PowerShell):
           python gemm_plumbing_test.py
    3. Read the [STAGE] / [OK] / [FAIL] markers in the output.

WHAT SUCCESS LOOKS LIKE
    Every stage prints [OK], the numerical check prints "MATCH", and you
    get a mean latency number at the end with a final "PIPELINE OK" banner.
    See the notes at the bottom of this file for details.
=====================================================================
"""

import sys
import time

import numpy as np

# ---------------------------------------------------------------------
# Stage 0: environment report (helps debugging if something fails later)
# ---------------------------------------------------------------------
print("=" * 65)
print("[STAGE 0] Environment report")
print("=" * 65)
print(f"Python        : {sys.version.split()[0]}")

try:
    import torch
    print(f"PyTorch       : {torch.__version__}")
except ImportError:
    print("[FAIL] PyTorch is not installed in this environment.")
    sys.exit(1)

try:
    import onnxruntime as ort
    print(f"ONNX Runtime  : {ort.__version__}")
    print(f"Available EPs : {ort.get_available_providers()}")
except ImportError:
    print("[FAIL] onnxruntime is not installed in this environment.")
    sys.exit(1)

print()

# ---------------------------------------------------------------------
# Config: the trivial GEMM workload.
# A single matrix multiply C = A @ B. Big enough to be a real GEMM,
# small enough to run instantly. Fixed seed => reproducible.
# ---------------------------------------------------------------------
M, K, N = 512, 512, 512          # C[M,N] = A[M,K] @ B[K,N]
OPSET = 17                        # a stable, widely-supported ONNX opset
ONNX_PATH = "gemm_test.onnx"
WARMUP_ITERS = 10                 # discarded: first runs include one-time graph opt
TIMED_ITERS = 100                 # averaged for the reported latency

torch.manual_seed(0)
np.random.seed(0)


# ---------------------------------------------------------------------
# Stage 1: define a trivial model in PyTorch.
# nn.Linear IS a GEMM (y = x @ W^T + b), the workhorse op of every
# transformer. We disable bias to keep it a pure matmul.
# ---------------------------------------------------------------------
print("[STAGE 1] Define trivial GEMM model in PyTorch ...")
try:
    class TinyGEMM(torch.nn.Module):
        def __init__(self, k, n):
            super().__init__()
            self.fc = torch.nn.Linear(k, n, bias=False)

        def forward(self, x):
            return self.fc(x)

    model = TinyGEMM(K, N).eval()
    dummy_input = torch.randn(M, K)            # shape [512, 512]

    # reference output from PyTorch itself (the "ground truth" math)
    with torch.no_grad():
        torch_output = model(dummy_input).numpy()
    print(f"   [OK] model built; PyTorch output shape {torch_output.shape}")
except Exception as e:
    print(f"   [FAIL] could not build/run the PyTorch model: {e}")
    sys.exit(1)
print()


# ---------------------------------------------------------------------
# Stage 2: export to ONNX.
# First real joint. Watch for opset / dynamic-axis / unmapped-op issues.
# ---------------------------------------------------------------------
print("[STAGE 2] Export model to ONNX ...")
try:
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        opset_version=OPSET,
        # dynamic batch dim so the graph isn't hard-locked to 512 rows.
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"   [OK] exported to '{ONNX_PATH}' (opset {OPSET})")
except Exception as e:
    print(f"   [FAIL] ONNX export failed: {e}")
    sys.exit(1)
print()


# ---------------------------------------------------------------------
# Stage 3: load the ONNX file in ONNX Runtime on the CPU EP.
# A model can export 'successfully' yet fail to load. This separates
# those two failure modes. We pin CPUExecutionProvider explicitly.
# ---------------------------------------------------------------------
print("[STAGE 3] Load ONNX model in ONNX Runtime (CPU EP) ...")
try:
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    print(f"   [OK] session created; using {sess.get_providers()}")
except Exception as e:
    print(f"   [FAIL] ORT could not load the model: {e}")
    sys.exit(1)
print()


# ---------------------------------------------------------------------
# Stage 4: run inference + validate the math round-tripped correctly.
# If ORT output != PyTorch output, the export silently corrupted the
# math -- exactly the kind of thing that would quietly ruin a benchmark.
# ---------------------------------------------------------------------
print("[STAGE 4] Run inference on CPU EP and validate numerics ...")
try:
    ort_input = {"input": dummy_input.numpy().astype(np.float32)}
    ort_output = sess.run(["output"], ort_input)[0]
    print(f"   [OK] ORT output shape {ort_output.shape}")

    if np.allclose(torch_output, ort_output, rtol=1e-3, atol=1e-5):
        max_diff = float(np.max(np.abs(torch_output - ort_output)))
        print(f"   [OK] numerical check: MATCH  (max abs diff {max_diff:.2e})")
    else:
        max_diff = float(np.max(np.abs(torch_output - ort_output)))
        print(f"   [FAIL] numerical MISMATCH  (max abs diff {max_diff:.2e})")
        sys.exit(1)
except Exception as e:
    print(f"   [FAIL] inference failed: {e}")
    sys.exit(1)
print()


# ---------------------------------------------------------------------
# Stage 5: minimal timing harness (wall-clock only; NOT power).
# Warmup runs are discarded because the first inference includes one-time
# graph optimization we don't want polluting the numbers.
# ---------------------------------------------------------------------
print("[STAGE 5] Timing harness (wall-clock only) ...")
try:
    for _ in range(WARMUP_ITERS):          # warmup: discarded
        sess.run(["output"], ort_input)

    start = time.perf_counter()
    for _ in range(TIMED_ITERS):
        sess.run(["output"], ort_input)
    elapsed = time.perf_counter() - start

    mean_ms = (elapsed / TIMED_ITERS) * 1000.0
    print(f"   [OK] {TIMED_ITERS} iters in {elapsed:.4f}s")
    print(f"   [OK] mean latency: {mean_ms:.4f} ms / inference")
except Exception as e:
    print(f"   [FAIL] timing harness failed: {e}")
    sys.exit(1)
print()

print("=" * 65)
print("  PIPELINE OK  --  PyTorch -> ONNX -> ORT -> inference -> timing")
print("  CPU EP validated end to end. Ready to add iGPU / NPU EPs next.")
print("=" * 65)

# =====================================================================
# WHAT SUCCESS LOOKS LIKE
# ---------------------------------------------------------------------
#  * Stage 0 prints versions AND a provider list that includes (at least)
#    'CPUExecutionProvider'. Ideally you also see 'DmlExecutionProvider'
#    and 'VitisAIExecutionProvider' -- confirming Day 3's engines are
#    already visible. (We don't USE them here, just nice to see them.)
#  * Stages 1-5 each print [OK].
#  * Stage 4 prints "MATCH" with a tiny max abs diff (e.g. ~1e-6). This is
#    the most important line: it proves the export didn't corrupt the math.
#  * Stage 5 prints a mean latency (some small number of ms). The exact
#    value does not matter -- this is plumbing, not a benchmark.
#  * You see the final "PIPELINE OK" banner.
#
# IF SOMETHING FAILS
#  * The [FAIL] line tells you the exact stage. Because the workload is one
#    trivial op, the failure is about the TOOLCHAIN (versions, export,
#    loading), never about model complexity. That's the whole point.
# =====================================================================
