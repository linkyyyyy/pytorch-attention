"""
dml_ortvalue_smoke_test.py — probe OrtValue.ortvalue_from_numpy(..., 'dml', 0).

Run from ryzen-ai-1.6.0 before iGPU measurements:
    conda activate ryzen-ai-1.6.0
    cd benchmark
    python dml_ortvalue_smoke_test.py
"""

from __future__ import annotations

import sys

import numpy as np


def main() -> int:
    print("=" * 65)
    print("[SMOKE] DML OrtValue device string probe")
    print("=" * 65)

    try:
        import onnxruntime as ort
    except ImportError:
        print("[FAIL] onnxruntime not installed")
        return 1

    print(f"ORT version : {ort.__version__}")
    print(f"Providers   : {ort.get_available_providers()}")

    if "DmlExecutionProvider" not in ort.get_available_providers():
        print("[FAIL] DmlExecutionProvider not available")
        return 1

    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    try:
        value = ort.OrtValue.ortvalue_from_numpy(arr, "dml", 0)
        print(f"[OK] ortvalue_from_numpy(arr, 'dml', 0) succeeded")
        print(f"     OrtValue device_type={value.device_name()}")
    except Exception as exc:
        print(f"[FAIL] ortvalue_from_numpy(arr, 'dml', 0) raised: {exc}")
        print("[INFO] harness.py will fall back to sess.run(feeds) for iGPU")
        return 1

    from harness import probe_dml_ortvalue

    if probe_dml_ortvalue(0):
        print("[OK] harness.probe_dml_ortvalue(0) -> True (IOBinding+dml path)")
    else:
        print("[FAIL] harness probe unexpectedly returned False after direct OK")
        return 1

    print("=" * 65)
    print("  DML ORTVALUE SMOKE OK")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
