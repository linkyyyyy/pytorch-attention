"""
gemm_validate.py - Standalone NPU-path validation for ffn_gemm.

Proves the full chain on the registry GEMM (same graph the harness uses):
    FP32 static (opset 20) -> Quark XINT8 -> VitisAI EP -> partition + correctness.

This is NOT the measured-energy run. For energy, use harness.py under uProf in a
clean standalone terminal (editor closed).

Run from benchmark/:
    conda activate ryzen-ai-1.6.0
    python -m npu.gemm_validate
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

from npu.path import DEFAULT_VAIP_CACHE, ensure_xint8_model, make_npu_session
from operators import build_operator_graph

OPERATOR = "ffn_gemm"
SHAPE_INDEX = 0
CACHE_KEY = "ffn_gemm_s0"


def main() -> None:
    fp32_path, meta, _ = build_operator_graph(OPERATOR, SHAPE_INDEX)
    feeds = meta["feeds"]
    output_names = meta["output_names"]
    print(f"[graph] {fp32_path}  operator={OPERATOR}  shape_index={SHAPE_INDEX}")

    int8_path = ensure_xint8_model(fp32_path, feeds)
    sess, partition = make_npu_session(
        int8_path, CACHE_KEY, cache_dir=DEFAULT_VAIP_CACHE
    )

    x = {
        name: np.random.randn(*arr.shape).astype(np.float32)
        for name, arr in feeds.items()
    }
    out = sess.run(output_names, x)[0]
    print(f"[run] out shape={out.shape}  all_finite={np.isfinite(out).all()}")

    cpu = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    ref = cpu.run(output_names, x)[0]
    rel = np.linalg.norm(out - ref) / (np.linalg.norm(ref) + 1e-9)
    print(f"[run] relative L2 vs FP32 CPU = {rel:.4f}  (expect ~0.01-0.10 for INT8)")

    if partition:
        print(f"\n[validate] {partition.summary()}")
    print(
        "\nIf nodes landed on the NPU and output is finite, the path is LIVE.\n"
        "Next: measured window via harness.py --engine npu (clean terminal, editor closed)."
    )


if __name__ == "__main__":
    main()
