"""
path.py - NPU-branch helpers for harness.py (VitisAI EP / XINT8).

Quantization and the one-time VAI compile MUST happen outside the measured window.
The measured window reuses the same _run_repeat / _duration_loop path as CPU/iGPU.

Env: ryzen-ai-1.6.0 (native Windows), HX 370 tower.
"""

from __future__ import annotations

import contextlib
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
DEFAULT_VAIP_CACHE = BENCHMARK_DIR / ".vaip_cache"

# Ships in ...\voe-*-win_amd64\ ; override with BENCHMARK_VAIP_CONFIG if needed.
VAIP_CONFIG = "vaip_config.json"

QDQ_BOUNDARY_NOTE = (
    "QDQ_boundary_on_CPU: graph-boundary QuantizeLinear/DequantizeLinear fall back to "
    "VITIS_EP_CPU; isolated microbenchmark energy includes that small CPU quant/dequant tax."
)

VAI_OPS_RE = re.compile(
    r"\[Vitis AI EP\]\s+No\. of Operators\s*:\s*NPU\s+(\d+)\s+VITIS_EP_CPU\s+(\d+)"
)
VAI_SUBG_RE = re.compile(
    r"\[Vitis AI EP\]\s+No\. of Subgraphs\s*:\s*NPU\s+(\d+)\s+Actually running on NPU\s+(\d+)"
)


@dataclass(frozen=True)
class VaiPartition:
    ops_npu: int
    ops_vitis_ep_cpu: int
    subgraphs_npu: int
    subgraphs_running_on_npu: int

    def summary(self) -> str:
        return (
            f"VAI_PARTITION: ops_NPU={self.ops_npu} ops_VITIS_EP_CPU={self.ops_vitis_ep_cpu} "
            f"subgraphs_NPU={self.subgraphs_npu} running_NPU={self.subgraphs_running_on_npu}; "
            f"{QDQ_BOUNDARY_NOTE}"
        )

    def to_dict(self) -> dict[str, int | str]:
        return {
            "ops_npu": self.ops_npu,
            "ops_vitis_ep_cpu": self.ops_vitis_ep_cpu,
            "subgraphs_npu": self.subgraphs_npu,
            "subgraphs_running_on_npu": self.subgraphs_running_on_npu,
            "qdq_boundary_note": QDQ_BOUNDARY_NOTE,
        }


def xint8_path_for(fp32_path: Path) -> Path:
    return fp32_path.with_name(f"{fp32_path.stem}_xint8.onnx")


def resolve_vaip_config() -> str:
    import os

    override = os.environ.get("BENCHMARK_VAIP_CONFIG", "").strip()
    if override and Path(override).is_file():
        return override

    candidate = Path(VAIP_CONFIG)
    if candidate.is_file():
        return str(candidate.resolve())

    import site

    for base in site.getsitepackages():
        for cfg in Path(base).glob("voe-*/vaip_config.json"):
            return str(cfg.resolve())
    return VAIP_CONFIG


def parse_vai_partition_log(log: str) -> VaiPartition | None:
    ops_m = VAI_OPS_RE.search(log)
    sub_m = VAI_SUBG_RE.search(log)
    if not ops_m or not sub_m:
        return None
    return VaiPartition(
        ops_npu=int(ops_m.group(1)),
        ops_vitis_ep_cpu=int(ops_m.group(2)),
        subgraphs_npu=int(sub_m.group(1)),
        subgraphs_running_on_npu=int(sub_m.group(2)),
    )


def quantize_op_to_xint8(
    fp32_path: Path,
    int8_path: Path,
    feeds_template: dict[str, np.ndarray],
    n_calib: int = 16,
) -> None:
    """Offline prep: FP32 static graph -> XINT8. Never call inside the measured window."""
    from quark.onnx import ModelQuantizer
    from quark.onnx.quantization.config import Config, get_default_config

    class CalibReader(CalibrationDataReader):
        # Distribution-matched synthetic calibration — operator energy unit, not task accuracy.
        def __init__(self) -> None:
            self.data = [
                {
                    name: np.random.randn(*arr.shape).astype(np.float32)
                    for name, arr in feeds_template.items()
                }
                for _ in range(n_calib)
            ]
            self._it = iter(self.data)

        def get_next(self) -> dict[str, np.ndarray] | None:
            return next(self._it, None)

        def rewind(self) -> None:
            self._it = iter(self.data)

    int8_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = Config(global_quant_config=get_default_config("XINT8"))
    ModelQuantizer(cfg).quantize_model(str(fp32_path), str(int8_path), CalibReader())
    print(f"[quark] {int8_path.name}  (XINT8, synthetic calib n={n_calib})")


def ensure_xint8_model(
    fp32_path: Path,
    feeds_template: dict[str, np.ndarray],
    *,
    force: bool = False,
) -> Path:
    int8_path = xint8_path_for(fp32_path)
    if force or not int8_path.exists() or int8_path.stat().st_mtime < fp32_path.stat().st_mtime:
        quantize_op_to_xint8(fp32_path, int8_path, feeds_template)
    else:
        print(f"[quark] reuse {int8_path.name}")
    return int8_path


def make_npu_session(
    int8_path: Path,
    cache_key: str,
    *,
    cache_dir: Path = DEFAULT_VAIP_CACHE,
    intra_op_num_threads: int = 12,
) -> tuple[Any, VaiPartition | None]:
    """
    Session creation triggers the one-time NPU compile — call before the measured window.
    VAI EP run() is synchronous; no DirectML synchronize_outputs() hack.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = intra_op_num_threads

    provider_options = [{
        "config_file": resolve_vaip_config(),
        "cacheDir": str(cache_dir),
        "cacheKey": cache_key,
    }]

    log_buf = io.StringIO()
    with contextlib.redirect_stdout(log_buf), contextlib.redirect_stderr(log_buf):
        sess = ort.InferenceSession(
            str(int8_path),
            sess_options=so,
            providers=["VitisAIExecutionProvider"],
            provider_options=provider_options,
        )

    log = log_buf.getvalue()
    partition = parse_vai_partition_log(log)
    if partition:
        print(f"[vai] {partition.summary()}")
    else:
        print("[vai] partition summary not found in compile log")
        if log.strip():
            print(log.strip()[-2000:])
    print("[vai] providers:", sess.get_providers())
    return sess, partition
