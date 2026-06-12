"""
harness.py — ONE shared measurement loop for operator energy benchmarking.

Python does NOT measure energy. uProf does. This script drives a marked power window
and logs alignment markers for post-hoc energy integration.

Headline formula (analysis, post-uProf):
    energy_per_op = (window_energy - dispatch_energy) / iterations
Idle baseline captures the static floor separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from npu import VaiPartition, ensure_xint8_model, make_npu_session
from operators import (
    DISPATCH_BASELINE_PATH,
    OPSET,
    build_dispatch_baseline_graph,
    build_operator_graph,
    fusion_member_csv,
    get_entry,
    measure_context_from_profile,
    warn_skip_out_of_range_shape_index,
)

BENCHMARK_DIR = Path(__file__).parent
RESULTS_DIR = BENCHMARK_DIR / "results"
PARTITIONS_DIR = RESULTS_DIR / "partitions"
DEFAULT_OUTFILE = RESULTS_DIR / "runs.csv"
METADATA_PATH = RESULTS_DIR / "metadata.json"
VAIP_CACHE_DIR = BENCHMARK_DIR / ".vaip_cache"

INTRA_OP_NUM_THREADS = 12
GRAPH_OPTIMIZATION_LEVEL_NAME = "ORT_ENABLE_ALL"
COOLDOWN_S = 5.0
IDLE_SLEEP_S = 0.001  # window body: tight sleep loop for full warmup/window duration

# None = not probed yet; set by probe_dml_ortvalue() on first iGPU session.
_DML_ORTVALUE_AVAILABLE: bool | None = None

CSV_HEADER = [
    "run_id",
    "operator",
    "cluster",
    "tier",
    "block_id",
    "shape_class",
    "fusion_member",
    "engine",
    "device_id",
    "shape_index",
    "input_shape",
    "dtype",
    "opset",
    "intra_op_num_threads",
    "graph_optimization_level",
    "igpu_vgm_mb",
    "repeat_idx",
    "warmup_s",
    "window_s",
    "iterations_completed",
    "wall_time_s",
    "t_start_epoch",
    "t_end_epoch",
    "mean_latency_ms",
    "idle_power_w",
    "active_power_w",
    "window_energy_J",
    "idle_energy_J",
    "energy_per_op_J",
    "dispatch_energy_J",
    "notes",
]

# Illustrative rows for schema documentation (not written to disk automatically).
CSV_EXAMPLE_ROWS: list[dict[str, Any]] = [
    {
        "run_id": "attn_score_matmul_vit_avg_cpu_r0",
        "operator": "attn_score_matmul",
        "cluster": "B1",
        "tier": "isolated",
        "block_id": "vit_avg",
        "shape_class": "avg",
        "fusion_member": "True",
        "engine": "cpu",
        "device_id": 0,
        "shape_index": 0,
        "input_shape": "12x197x64@12x64x197",
        "dtype": "float32",
        "opset": 20,
        "intra_op_num_threads": 12,
        "graph_optimization_level": "ORT_ENABLE_ALL",
        "igpu_vgm_mb": "512",
        "repeat_idx": 0,
        "warmup_s": 5.0,
        "window_s": 30.0,
        "iterations_completed": 27980,
        "wall_time_s": "30.000000",
        "t_start_epoch": "1780925790.173200",
        "t_end_epoch": "1780925820.173200",
        "mean_latency_ms": "1.072195",
        "idle_power_w": "",
        "active_power_w": "",
        "window_energy_J": "",
        "idle_energy_J": "",
        "energy_per_op_J": "",
        "dispatch_energy_J": "",
        "notes": "threads=12; graph_opt=ORT_ENABLE_ALL; vgm_mb=512; EP_OK: intended=CPUExecutionProvider",
    },
    {
        "run_id": "fused_block_vit_avg_igpu_r0",
        "operator": "fused_block",
        "cluster": "B1",
        "tier": "fused_block",
        "block_id": "vit_avg",
        "shape_class": "avg",
        "fusion_member": "True",
        "engine": "igpu",
        "device_id": 0,
        "shape_index": 0,
        "input_shape": "fused_qkv_attn_proj",
        "dtype": "float32",
        "opset": 20,
        "intra_op_num_threads": 12,
        "graph_optimization_level": "ORT_ENABLE_ALL",
        "igpu_vgm_mb": "512",
        "repeat_idx": 0,
        "warmup_s": 5.0,
        "window_s": 30.0,
        "iterations_completed": 41800,
        "wall_time_s": "30.000000",
        "t_start_epoch": "1780925790.173200",
        "t_end_epoch": "1780925820.173200",
        "mean_latency_ms": "0.716000",
        "idle_power_w": "",
        "active_power_w": "",
        "window_energy_J": "",
        "idle_energy_J": "",
        "energy_per_op_J": "",
        "dispatch_energy_J": "",
        "notes": "threads=12; graph_opt=ORT_ENABLE_ALL; vgm_mb=512; EP_OK: intended=DmlExecutionProvider",
    },
    {
        "run_id": "dispatch_baseline_igpu_r0",
        "operator": "dispatch_baseline",
        "cluster": "baseline",
        "tier": "n/a",
        "block_id": "n/a",
        "shape_class": "n/a",
        "fusion_member": "n/a",
        "engine": "igpu",
        "device_id": 0,
        "shape_index": -1,
        "input_shape": "1",
        "dtype": "float32",
        "opset": 20,
        "intra_op_num_threads": 12,
        "graph_optimization_level": "ORT_ENABLE_ALL",
        "igpu_vgm_mb": "512",
        "repeat_idx": 0,
        "warmup_s": 5.0,
        "window_s": 30.0,
        "iterations_completed": 394000,
        "wall_time_s": "30.000000",
        "t_start_epoch": "1780925790.173200",
        "t_end_epoch": "1780925820.173200",
        "mean_latency_ms": "0.076000",
        "idle_power_w": "",
        "active_power_w": "",
        "window_energy_J": "",
        "idle_energy_J": "",
        "energy_per_op_J": "",
        "dispatch_energy_J": "",
        "notes": "threads=12; graph_opt=ORT_ENABLE_ALL; vgm_mb=512",
    },
    {
        "run_id": "idle_cpu_r0",
        "operator": "idle",
        "cluster": "baseline",
        "tier": "n/a",
        "block_id": "n/a",
        "shape_class": "n/a",
        "fusion_member": "n/a",
        "engine": "cpu",
        "device_id": 0,
        "shape_index": -1,
        "input_shape": "n/a",
        "dtype": "n/a",
        "opset": "n/a",
        "intra_op_num_threads": 12,
        "graph_optimization_level": "ORT_ENABLE_ALL",
        "igpu_vgm_mb": "512",
        "repeat_idx": 0,
        "warmup_s": 5.0,
        "window_s": 30.0,
        "iterations_completed": 3000,
        "wall_time_s": "30.000000",
        "t_start_epoch": "1780925790.173200",
        "t_end_epoch": "1780925820.173200",
        "mean_latency_ms": "10.000000",
        "idle_power_w": "",
        "active_power_w": "",
        "window_energy_J": "",
        "idle_energy_J": "",
        "energy_per_op_J": "",
        "dispatch_energy_J": "",
        "notes": "threads=12; graph_opt=ORT_ENABLE_ALL; vgm_mb=512",
    },
]

CSV_SCHEMA_DOC: dict[str, Any] = {
    "columns": CSV_HEADER,
    "block_context_columns": {
        "tier": {
            "description": "Whether the run measures a single isolated operator or a fused attention block.",
            "allowed_measure": ["isolated", "fused_block"],
            "allowed_baseline": ["n/a"],
        },
        "block_id": {
            "description": "Attention-block config identifier (e.g. vit_avg, pvt_stage1, efficientformer_small).",
            "allowed_measure": "free-form string; empty when not block-scoped",
            "allowed_baseline": ["n/a"],
        },
        "shape_class": {
            "description": "DSE corner label for (N, D) attention-block shape.",
            "allowed_measure": ["small", "avg", "large", ""],
            "allowed_baseline": ["n/a"],
        },
        "fusion_member": {
            "description": "Whether row counts toward Tier-1-vs-Tier-2 fusion gap (attention-core ops).",
            "allowed_measure": ["True", "False"],
            "allowed_baseline": ["n/a"],
        },
    },
    "example_rows": CSV_EXAMPLE_ROWS,
}


def _igpu_vgm_mb() -> str:
    return os.environ.get("BENCHMARK_IGPU_VGM_MB", "512")


def _strip_repeat_suffix(run_id_base: str) -> str:
    """Guard against double _rN suffix when caller passes a full run_id as --run-id."""
    m = re.match(r"^(.+)_r(\d+)$", run_id_base.strip())
    return m.group(1) if m else run_id_base.strip()


def resolve_run_id(run_id_base: str, repeat_idx: int) -> str:
    """Harness owns repeat suffixing: base + exactly one _r{repeat_idx}."""
    return f"{_strip_repeat_suffix(run_id_base)}_r{repeat_idx}"


def _npu_cache_key(operator: str, shape_index: int, mode: str) -> str:
    if mode == "dispatch":
        return "dispatch_baseline"
    return f"{operator}_s{shape_index}"


def _write_partition_sidecar(run_id: str, partition: VaiPartition) -> None:
    PARTITIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = PARTITIONS_DIR / f"{run_id}.json"
    payload = {"run_id": run_id, **partition.to_dict()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[vai] partition sidecar -> {path}")


def _ensure_dispatch_baseline() -> Path:
    if not DISPATCH_BASELINE_PATH.exists():
        build_dispatch_baseline_graph(DISPATCH_BASELINE_PATH, OPSET)
    return DISPATCH_BASELINE_PATH


def _read_graph_opset(onnx_path: Path) -> int:
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    return int(model.opset_import[0].version)


def _ensure_csv_schema_in_metadata(metadata: dict[str, Any]) -> None:
    metadata["csv_schema"] = CSV_SCHEMA_DOC


def _session_metadata_fields() -> dict[str, Any]:
    try:
        import onnxruntime as ort
        ort_version = ort.__version__
        providers = ort.get_available_providers()
    except ImportError:
        ort_version = "unknown"
        providers = []
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ort_version": ort_version,
        "available_providers": providers,
        "opset": OPSET,
        "intra_op_num_threads": INTRA_OP_NUM_THREADS,
        "graph_optimization_level": GRAPH_OPTIMIZATION_LEVEL_NAME,
        "igpu_vgm_mb": _igpu_vgm_mb(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "ryzen-ai-1.6.0"),
        "uprof_version": "",  # fill manually after session
        "gpu_driver_version": "",  # fill manually after session
        "notes": (
            "energy_per_op = (window_energy - dispatch_energy) / iterations; "
            "idle baseline for static floor"
        ),
    }


def _write_metadata_once() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {}
    if METADATA_PATH.exists():
        try:
            metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}

    needs_write = False
    if "csv_schema" not in metadata:
        _ensure_csv_schema_in_metadata(metadata)
        needs_write = True

    # Template metadata.json (empty created_utc) → fill session header on first harness run.
    if not metadata.get("created_utc"):
        metadata.update(_session_metadata_fields())
        needs_write = True

    if needs_write:
        METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"[metadata] wrote {METADATA_PATH}")


def _make_session_options(engine: str, enable_profiling: bool) -> Any:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = INTRA_OP_NUM_THREADS
    if enable_profiling:
        so.enable_profiling = True
    if engine == "igpu":
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return so


def _create_session(onnx_path: Path, engine: str, device_id: int, enable_profiling: bool) -> Any:
    import onnxruntime as ort

    so = _make_session_options(engine, enable_profiling)
    if engine == "cpu":
        providers = ["CPUExecutionProvider"]
    elif engine == "igpu":
        providers = [("DmlExecutionProvider", {"device_id": device_id})]
    else:
        raise ValueError(f"Unsupported engine: {engine}")
    return ort.InferenceSession(str(onnx_path), so, providers=providers)


def _check_ep_placement(sess: Any, engine: str) -> str:
    notes: list[str] = []
    prof_file: str | None = None
    try:
        prof_file = sess.end_profiling()
        if not prof_file:
            return ""
        with open(prof_file, encoding="utf-8") as f:
            data = json.load(f)
        expected = "CPUExecutionProvider" if engine == "cpu" else "DmlExecutionProvider"
        fallbacks: list[str] = []
        for item in data:
            args = item.get("args", {})
            provider = args.get("provider", "") if isinstance(args, dict) else ""
            if provider and provider != expected:
                fallbacks.append(f"EP_FALLBACK: node '{item.get('name')}' on {provider}")
        if fallbacks:
            notes.extend(fallbacks)
        else:
            notes.append(f"EP_OK: intended={expected}")
    except Exception as exc:
        notes.append(f"EP_CHECK_ERROR: {exc}")
    finally:
        if prof_file:
            try:
                os.remove(prof_file)
            except OSError:
                pass
    return "; ".join(notes)


def probe_dml_ortvalue(device_id: int = 0) -> bool:
    """
    Return whether this ORT build accepts OrtValue.ortvalue_from_numpy(..., 'dml', device_id).
    Result is cached for the process lifetime.
    """
    global _DML_ORTVALUE_AVAILABLE
    if _DML_ORTVALUE_AVAILABLE is not None:
        return _DML_ORTVALUE_AVAILABLE
    try:
        import onnxruntime as ort

        probe = np.array([1.0], dtype=np.float32)
        ort.OrtValue.ortvalue_from_numpy(probe, "dml", device_id)
        _DML_ORTVALUE_AVAILABLE = True
        print(f"[DML_IO] OrtValue device string 'dml' OK (device_id={device_id})")
    except Exception as exc:
        _DML_ORTVALUE_AVAILABLE = False
        print(
            f"[DML_IO] OrtValue 'dml' unavailable; falling back to sess.run(feeds): {exc}"
        )
    return _DML_ORTVALUE_AVAILABLE


def igpu_use_iobinding(device_id: int) -> bool:
    return probe_dml_ortvalue(device_id)


def _create_io_binding(
    sess: Any,
    feeds: dict[str, np.ndarray],
    output_names: list[str],
    engine: str,
    device_id: int,
) -> Any | None:
    """Bind inputs/outputs once for IOBinding path. Returns None if iGPU numpy-feed fallback."""
    if engine == "igpu" and not igpu_use_iobinding(device_id):
        return None

    io_binding = sess.io_binding()
    for name, arr in feeds.items():
        contiguous = np.ascontiguousarray(arr)
        if engine == "igpu":
            import onnxruntime as ort

            ort_value = ort.OrtValue.ortvalue_from_numpy(contiguous, "dml", device_id)
            io_binding.bind_ortvalue_input(name, ort_value)
        else:
            io_binding.bind_cpu_input(name, contiguous)
    for name in output_names:
        if engine == "igpu":
            io_binding.bind_output(name, "dml", device_id)
        else:
            io_binding.bind_output(name, "cpu")
    return io_binding


def _run_inference_iteration(
    sess: Any,
    *,
    use_iobinding: bool,
    io_binding: Any | None,
    feeds: dict[str, np.ndarray] | None,
    output_names: list[str] | None,
    engine: str = "cpu",
) -> None:
    if use_iobinding:
        assert io_binding is not None
        sess.run_with_iobinding(io_binding)
        if engine == "igpu":
            io_binding.synchronize_outputs()  # DirectML/iGPU only — not NPU
    else:
        assert feeds is not None and output_names is not None
        sess.run(output_names, feeds)  # VAI EP: synchronous, blocks until NPU finishes


def _run_profile_check(
    onnx_path: Path,
    engine: str,
    device_id: int,
    feeds: dict[str, np.ndarray],
    output_names: list[str],
) -> str:
    """Single profiled iteration before warmup; profiling never enters the timed window."""
    sess_profile = _create_session(onnx_path, engine, device_id, enable_profiling=True)
    print(f"[PROFILE_SESSION] providers={sess_profile.get_providers()}")
    use_iobinding = engine != "igpu" or igpu_use_iobinding(device_id)
    io_binding = _create_io_binding(sess_profile, feeds, output_names, engine, device_id)
    _run_inference_iteration(
        sess_profile,
        use_iobinding=use_iobinding,
        io_binding=io_binding,
        feeds=feeds,
        output_names=output_names,
        engine=engine,
    )
    return _check_ep_placement(sess_profile, engine)


def _duration_loop(
    execute_fn: Callable[[], None],
    duration_s: float,
) -> tuple[int, float]:
    iterations = 0
    start = time.perf_counter()
    deadline = start + duration_s
    while time.perf_counter() < deadline:
        execute_fn()
        iterations += 1
    elapsed = time.perf_counter() - start
    return iterations, elapsed


def _append_csv_row(outfile: Path, row: dict[str, Any]) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    write_header = not outfile.exists() or outfile.stat().st_size == 0
    with outfile.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_HEADER})


def _run_repeat(
    *,
    run_id: str,
    mode: str,
    operator: str,
    cluster: str,
    tier: str,
    block_id: str,
    shape_class: str,
    fusion_member: str,
    engine: str,
    device_id: int,
    shape_index: int,
    input_shape: str,
    dtype: str,
    warmup_s: float,
    window_s: float,
    repeat_idx: int,
    outfile: Path,
    sess: Any | None,
    io_binding: Any | None,
    feeds: dict[str, np.ndarray] | None,
    output_names: list[str] | None,
    use_iobinding: bool,
    ep_note: str,
    csv_opset: int | str,
) -> None:
    def execute() -> None:
        if mode == "idle":
            time.sleep(IDLE_SLEEP_S)  # full window = many sleeps via _duration_loop (no ORT)
        else:
            assert sess is not None
            _run_inference_iteration(
                sess,
                use_iobinding=use_iobinding,
                io_binding=io_binding,
                feeds=feeds,
                output_names=output_names,
                engine=engine,
            )

    print(
        f"[WARMUP_START] run_id={run_id} mode={mode} engine={engine} "
        f"warmup_s={warmup_s} threads={INTRA_OP_NUM_THREADS} "
        f"graph_opt={GRAPH_OPTIMIZATION_LEVEL_NAME}"
    )
    _duration_loop(execute, warmup_s)
    print(f"[WARMUP_END] run_id={run_id}")

    t_start = time.time()
    print(
        f"[WINDOW_OPEN] run_id={run_id} t_start={t_start:.6f} "
        f"mode={mode} engine={engine} operator={operator}"
    )

    iterations, wall_time_s = _duration_loop(execute, window_s)

    t_end = time.time()
    print(
        f"[WINDOW_CLOSE] run_id={run_id} t_end={t_end:.6f} "
        f"iterations={iterations} wall_time_s={wall_time_s:.6f}"
    )

    if mode == "idle":
        mean_latency_ms: float | str = ""
    else:
        mean_latency_ms = (wall_time_s / iterations * 1000.0) if iterations else float("nan")

    notes_parts = [
        f"threads={INTRA_OP_NUM_THREADS}",
        f"graph_opt={GRAPH_OPTIMIZATION_LEVEL_NAME}",
        f"vgm_mb={_igpu_vgm_mb()}",
    ]
    if ep_note:
        notes_parts.append(ep_note)

    _append_csv_row(
        outfile,
        {
            "run_id": run_id,
            "operator": operator,
            "cluster": cluster,
            "tier": tier,
            "block_id": block_id,
            "shape_class": shape_class,
            "fusion_member": fusion_member,
            "engine": engine,
            "device_id": device_id,
            "shape_index": shape_index,
            "input_shape": input_shape,
            "dtype": dtype,
            "opset": csv_opset,
            "intra_op_num_threads": INTRA_OP_NUM_THREADS,
            "graph_optimization_level": GRAPH_OPTIMIZATION_LEVEL_NAME,
            "igpu_vgm_mb": _igpu_vgm_mb(),
            "repeat_idx": repeat_idx,
            "warmup_s": warmup_s,
            "window_s": window_s,
            "iterations_completed": iterations,
            "wall_time_s": f"{wall_time_s:.6f}",
            "t_start_epoch": f"{t_start:.6f}",
            "t_end_epoch": f"{t_end:.6f}",
            "mean_latency_ms": (
                "" if mean_latency_ms == "" else f"{float(mean_latency_ms):.6f}"
            ),
            "idle_power_w": "",
            "active_power_w": "",
            "window_energy_J": "",
            "idle_energy_J": "",
            "energy_per_op_J": "",
            "dispatch_energy_J": "",
            "notes": "; ".join(notes_parts),
        },
    )


def _prep_npu_quantize_only(args: argparse.Namespace) -> None:
    """Offline XINT8 quantize for one graph; no session, no measured window."""
    if args.mode == "measure":
        if not warn_skip_out_of_range_shape_index(args.operator, args.shape_index):
            return
        onnx_path, meta, _ = build_operator_graph(args.operator, args.shape_index)
        label = args.operator
    elif args.mode == "dispatch":
        onnx_path = _ensure_dispatch_baseline()
        meta = {"feeds": {"input": np.array([0.5], dtype=np.float32)}}
        label = "dispatch_baseline"
    else:
        raise SystemExit("--prep-npu-quantize requires --mode measure or dispatch")

    int8_path = ensure_xint8_model(onnx_path, meta["feeds"], force=args.force_requantize)
    print(f"[prep] {label} -> {int8_path}")


def run_harness(args: argparse.Namespace) -> None:
    _write_metadata_once()

    if args.prep_npu_quantize:
        _prep_npu_quantize_only(args)
        return

    outfile = Path(args.outfile)

    csv_opset: int | str = OPSET

    if args.mode == "measure":
        if not warn_skip_out_of_range_shape_index(args.operator, args.shape_index):
            return
        onnx_path, meta, entry = build_operator_graph(args.operator, args.shape_index)
        profile = entry.shape_profiles[args.shape_index]
        operator = entry.name
        cluster = entry.cluster
        input_shape = profile["input_shape"]
        dtype = entry.dtype
        csv_opset = _read_graph_opset(onnx_path)
        print(f"[graph] operator={operator} opset={csv_opset} (read from {onnx_path.name})")
    elif args.mode == "dispatch":
        onnx_path = _ensure_dispatch_baseline()
        meta = {"feeds": {"input": np.array([0.5], dtype=np.float32)}, "output_names": ["output"]}
        operator = "dispatch_baseline"
        cluster = "baseline"
        input_shape = "1"
        dtype = "float32"
        csv_opset = _read_graph_opset(onnx_path)
    else:  # idle
        onnx_path = None
        meta = None
        operator = "idle"
        cluster = "baseline"
        input_shape = "n/a"
        dtype = "n/a"
        csv_opset = "n/a"

    if args.engine == "npu" and args.mode in ("measure", "dispatch"):
        dtype = "xint8"

    if args.mode == "measure":
        tier, block_id, shape_class = measure_context_from_profile(
            profile,
            tier=args.tier,
            block_id=args.block_id,
            shape_class=args.shape_class,
        )
        fusion_member = fusion_member_csv(profile, entry)
    else:
        tier = "n/a"
        block_id = "n/a"
        shape_class = "n/a"
        fusion_member = "n/a"

    base_run_id = args.run_id or (
        f"{operator}_{args.engine}"
        if args.mode != "idle"
        else f"idle_{args.engine}"
    )

    for repeat_idx in range(args.repeats):
        run_id = resolve_run_id(base_run_id, repeat_idx)
        sess = None
        io_binding = None
        ep_note = ""
        feeds = meta["feeds"] if meta else None
        output_names = meta["output_names"] if meta else None

        if args.mode != "idle":
            if args.engine == "npu":
                assert onnx_path is not None and feeds is not None
                int8_path = ensure_xint8_model(onnx_path, feeds)
                cache_key = _npu_cache_key(
                    operator,
                    args.shape_index if args.mode == "measure" else -1,
                    args.mode,
                )
                sess, partition = make_npu_session(
                    int8_path,
                    cache_key,
                    cache_dir=VAIP_CACHE_DIR,
                    intra_op_num_threads=INTRA_OP_NUM_THREADS,
                )
                if repeat_idx == 0 and partition is not None:
                    ep_note = partition.summary()
                    _write_partition_sidecar(run_id, partition)
                    print(f"[EP_CHECK] run_id={run_id} {ep_note}")
                elif repeat_idx == 0:
                    ep_note = "VAI_PARTITION: not captured in compile log"
                    print(f"[EP_CHECK] run_id={run_id} {ep_note}")
                use_iobinding = False
                io_binding = None
                print(
                    f"[MEASURE_SESSION] run_id={run_id} int8={int8_path.name} "
                    f"providers={sess.get_providers()}"
                )
            else:
                if repeat_idx == 0:
                    ep_note = _run_profile_check(
                        onnx_path,
                        args.engine,
                        args.device_id,
                        feeds,
                        output_names,
                    )
                    print(f"[EP_CHECK] run_id={run_id} {ep_note}")

                sess = _create_session(
                    onnx_path,
                    args.engine,
                    args.device_id,
                    enable_profiling=False,
                )
                print(f"[MEASURE_SESSION] run_id={run_id} providers={sess.get_providers()}")
                use_iobinding = args.engine != "igpu" or igpu_use_iobinding(args.device_id)
                io_binding = _create_io_binding(
                    sess, feeds, output_names, args.engine, args.device_id
                )
                if args.engine == "igpu":
                    path = "iobinding+dml" if use_iobinding else "sess.run(feeds)"
                    print(f"[DML_IO] run_id={run_id} inference path={path}")
        else:
            use_iobinding = False

        _run_repeat(
            run_id=run_id,
            mode=args.mode,
            operator=operator,
            cluster=cluster,
            tier=tier,
            block_id=block_id,
            shape_class=shape_class,
            fusion_member=fusion_member,
            engine=args.engine,
            device_id=args.device_id,
            shape_index=args.shape_index if args.mode == "measure" else -1,
            input_shape=input_shape,
            dtype=dtype,
            warmup_s=args.warmup,
            window_s=args.duration,
            repeat_idx=repeat_idx,
            outfile=outfile,
            sess=sess,
            io_binding=io_binding,
            feeds=feeds,
            output_names=output_names,
            use_iobinding=use_iobinding if args.mode != "idle" else False,
            ep_note=ep_note,
            csv_opset=csv_opset,
        )

        if repeat_idx < args.repeats - 1:
            print(f"[COOLDOWN] sleeping {COOLDOWN_S}s before next repeat")
            time.sleep(COOLDOWN_S)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Operator energy measurement harness")
    parser.add_argument("--operator", default="", help="Registry operator name (measure mode)")
    parser.add_argument("--engine", required=True, choices=["cpu", "igpu", "npu"])
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--warmup", type=float, default=5.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--mode", choices=["measure", "idle", "dispatch"], default="measure")
    parser.add_argument("--shape-index", type=int, default=0)
    parser.add_argument(
        "--tier",
        choices=["isolated", "fused_block"],
        default="isolated",
        help="isolated=single operator; fused_block=full attention block graph",
    )
    parser.add_argument(
        "--block-id",
        default="",
        help="Attention-block config id (e.g. vit_avg, pvt_stage1); empty when not block-scoped",
    )
    parser.add_argument(
        "--shape-class",
        choices=["", "small", "avg", "large"],
        default="",
        help="DSE corner: small / avg / large (from attention_block_dimensions.md)",
    )
    parser.add_argument("--outfile", default=str(DEFAULT_OUTFILE))
    parser.add_argument("--run-id", default="", help="Run id prefix (set by run_sweep.bat)")
    parser.add_argument(
        "--prep-npu-quantize",
        action="store_true",
        help="Offline Quark XINT8 quantize only (no session, no measured window)",
    )
    parser.add_argument(
        "--force-requantize",
        action="store_true",
        help="Re-run Quark even if _xint8.onnx exists",
    )
    args = parser.parse_args(argv)

    if args.prep_npu_quantize and args.engine != "npu":
        parser.error("--prep-npu-quantize requires --engine npu")

    if args.mode == "measure":
        if not args.operator:
            parser.error("--operator is required for --mode measure")
        get_entry(args.operator)
    return args


def main() -> None:
    print("=" * 72)
    print("[harness] Operator energy window driver (energy measured by uProf, not here)")
    print(f"[harness] opset={OPSET} threads={INTRA_OP_NUM_THREADS} opt={GRAPH_OPTIMIZATION_LEVEL_NAME}")
    print(
        "[harness] analysis formula: energy_per_op = (window_energy - dispatch_energy) / iterations"
    )
    print("=" * 72)
    args = parse_args()
    run_harness(args)


if __name__ == "__main__":
    main()
