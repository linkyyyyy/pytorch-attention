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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from operators import (
    DISPATCH_BASELINE_PATH,
    OPSET,
    build_dispatch_baseline_graph,
    build_operator_graph,
    get_entry,
)

BENCHMARK_DIR = Path(__file__).parent
RESULTS_DIR = BENCHMARK_DIR / "results"
DEFAULT_OUTFILE = RESULTS_DIR / "runs.csv"
METADATA_PATH = RESULTS_DIR / "metadata.json"

INTRA_OP_NUM_THREADS = 12
GRAPH_OPTIMIZATION_LEVEL_NAME = "ORT_ENABLE_ALL"
COOLDOWN_S = 5.0

# None = not probed yet; set by probe_dml_ortvalue() on first iGPU session.
_DML_ORTVALUE_AVAILABLE: bool | None = None

CSV_HEADER = [
    "run_id",
    "operator",
    "cluster",
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
    "mean_latency_ms",
    "idle_power_w",
    "active_power_w",
    "window_energy_J",
    "idle_energy_J",
    "energy_per_op_J",
    "dispatch_energy_J",
    "notes",
]


def _igpu_vgm_mb() -> str:
    return os.environ.get("BENCHMARK_IGPU_VGM_MB", "512")


def _ensure_dispatch_baseline() -> Path:
    if not DISPATCH_BASELINE_PATH.exists():
        build_dispatch_baseline_graph(DISPATCH_BASELINE_PATH, OPSET)
    return DISPATCH_BASELINE_PATH


def _read_graph_opset(onnx_path: Path) -> int:
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    return int(model.opset_import[0].version)


def _write_metadata_once() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if METADATA_PATH.exists():
        return
    try:
        import onnxruntime as ort
        ort_version = ort.__version__
        providers = ort.get_available_providers()
    except ImportError:
        ort_version = "unknown"
        providers = []

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ort_version": ort_version,
        "available_providers": providers,
        "opset": OPSET,
        "intra_op_num_threads": INTRA_OP_NUM_THREADS,
        "graph_optimization_level": GRAPH_OPTIMIZATION_LEVEL_NAME,
        "igpu_vgm_mb": _igpu_vgm_mb(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "uprof_version": "",  # fill manually after session
        "gpu_driver_version": "",  # fill manually after session
        "notes": (
            "energy_per_op = (window_energy - dispatch_energy) / iterations; "
            "idle baseline for static floor"
        ),
    }
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
) -> None:
    if use_iobinding:
        assert io_binding is not None
        sess.run_with_iobinding(io_binding)
        io_binding.synchronize_outputs()  # 1 iteration = 1 completed GPU execution
    else:
        assert feeds is not None and output_names is not None
        sess.run(output_names, feeds)


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
            time.sleep(0.001)
        else:
            assert sess is not None
            _run_inference_iteration(
                sess,
                use_iobinding=use_iobinding,
                io_binding=io_binding,
                feeds=feeds,
                output_names=output_names,
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
            "mean_latency_ms": f"{mean_latency_ms:.6f}",
            "idle_power_w": "",
            "active_power_w": "",
            "window_energy_J": "",
            "idle_energy_J": "",
            "energy_per_op_J": "",
            "dispatch_energy_J": "",
            "notes": "; ".join(notes_parts),
        },
    )


def run_harness(args: argparse.Namespace) -> None:
    _write_metadata_once()
    outfile = Path(args.outfile)

    csv_opset: int | str = OPSET

    if args.mode == "measure":
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

    base_run_id = args.run_id or f"{operator}_{args.engine}"

    for repeat_idx in range(args.repeats):
        run_id = f"{base_run_id}_r{repeat_idx}"
        sess = None
        io_binding = None
        ep_note = ""
        feeds = meta["feeds"] if meta else None
        output_names = meta["output_names"] if meta else None

        if args.mode != "idle":
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
    parser.add_argument("--engine", required=True, choices=["cpu", "igpu"])
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--warmup", type=float, default=5.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--mode", choices=["measure", "idle", "dispatch"], default="measure")
    parser.add_argument("--shape-index", type=int, default=0)
    parser.add_argument("--outfile", default=str(DEFAULT_OUTFILE))
    parser.add_argument("--run-id", default="", help="Run id prefix (set by run_sweep.bat)")
    args = parser.parse_args(argv)

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
