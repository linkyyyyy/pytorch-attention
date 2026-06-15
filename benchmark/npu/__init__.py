"""
NPU branch tooling (VitisAI EP / Quark XINT8) for the operator energy harness.

Run validation:  cd benchmark && python -m npu.gemm_validate
"""

from npu.path import (
    DEFAULT_VAIP_CACHE,
    QDQ_BOUNDARY_NOTE,
    VaiPartition,
    ensure_xint8_model,
    make_npu_session,
    quantize_op_to_xint8,
    resolve_vaip_config,
    xint8_path_for,
)

__all__ = [
    "DEFAULT_VAIP_CACHE",
    "QDQ_BOUNDARY_NOTE",
    "VaiPartition",
    "ensure_xint8_model",
    "make_npu_session",
    "quantize_op_to_xint8",
    "resolve_vaip_config",
    "xint8_path_for",
]
