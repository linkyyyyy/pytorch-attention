"""
run_plan.py — Execute sweep-plan measure rows via uProf-wrapped harness.

Reads pipe-delimited plan CSV (from operators.py --sweep-plan) with Python csv
so empty block_id / shape_class fields do not shift tokens.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

BENCHMARK_DIR = Path(__file__).parent
DEFAULT_PY = Path(r"C:\ProgramData\miniconda3\envs\ryzen-ai-1.6.0\python.exe")
DEFAULT_UPROF = "AMDuProfCLI.exe"

PLAN_FIELDS = (
    "operator",
    "shape_index",
    "block_id",
    "shape_class",
    "tier",
    "fusion_member",
    "input_shape",
    "skip_engines",
)


def load_plan(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="|"))
    if not rows:
        raise ValueError(f"No plan rows in {path}")
    missing = set(PLAN_FIELDS) - set(rows[0].keys())
    if missing:
        raise ValueError(f"Plan missing columns {sorted(missing)}: {path}")
    return rows


def parse_skip_engines(cell: str) -> frozenset[str]:
    return frozenset(e.strip() for e in cell.split(",") if e.strip())


def harness_argv(
    *,
    python: Path,
    operator: str,
    engine: str,
    shape_index: str,
    tier: str,
    block_id: str,
    shape_class: str,
    duration: float,
    warmup: float,
    repeats: int,
    device_id: int,
    run_id: str,
    outfile: Path,
) -> list[str]:
    """Harness child argv (no uProf wrapper). Omits empty block-id / shape-class flags."""
    argv = [
        str(python),
        str(BENCHMARK_DIR / "harness.py"),
        "--operator",
        operator,
        "--engine",
        engine,
        "--duration",
        str(duration),
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
        "--device-id",
        str(device_id),
        "--mode",
        "measure",
        "--shape-index",
        shape_index,
        "--tier",
        tier,
        "--run-id",
        run_id,
        "--outfile",
        str(outfile),
    ]
    if block_id:
        argv.extend(["--block-id", block_id])
    if shape_class:
        argv.extend(["--shape-class", shape_class])
    return argv


def uprof_command_line(
    *,
    uprof_cli: str,
    uprof_out: Path,
    harness_cmd: list[str],
) -> list[str]:
    return [
        uprof_cli,
        "timechart",
        "--event",
        "power",
        "--interval",
        "100",
        "-o",
        str(uprof_out),
        *harness_cmd,
    ]


def format_command(cmd: list[str]) -> str:
    return subprocess.list2cmdline(cmd)


def execute_job(
    harness_cmd: list[str],
    run_id: str,
    args: argparse.Namespace,
) -> int:
    """Run one measure job with uProf wrap (default) or parallel amd-smi sampler."""
    if args.power_backend == "uprof":
        uprof_dir = Path(args.uprof_dir)
        full_cmd = uprof_command_line(
            uprof_cli=args.uprof_cli,
            uprof_out=uprof_dir / run_id,
            harness_cmd=harness_cmd,
        )
        print(f"[RUN] {run_id}")
        print(format_command(full_cmd))
        if args.dry_run:
            return 0
        uprof_dir.mkdir(parents=True, exist_ok=True)
        return subprocess.run(full_cmd, cwd=BENCHMARK_DIR).returncode

    out_csv = Path(args.gpu_power_dir) / f"{run_id}.csv"
    sampler_cmd = [
        str(args.python),
        str(BENCHMARK_DIR / "gpu_power.py"),
        "--out",
        str(out_csv),
        "--interval",
        str(args.sampler_interval),
        "--device-id",
        str(args.device_id),
    ]
    print(f"[RUN] {run_id}")
    print("  sampler:", format_command(sampler_cmd))
    print("  harness:", format_command(harness_cmd))
    if args.dry_run:
        return 0

    Path(args.gpu_power_dir).mkdir(parents=True, exist_ok=True)
    sampler = subprocess.Popen(sampler_cmd, cwd=BENCHMARK_DIR)
    time.sleep(args.sampler_lead)
    rc = subprocess.run(harness_cmd, cwd=BENCHMARK_DIR).returncode
    time.sleep(args.sampler_lead)
    sampler.terminate()
    try:
        sampler.wait(timeout=10)
    except subprocess.TimeoutExpired:
        sampler.kill()
        sampler.wait()

    if (
        (not out_csv.exists())
        or out_csv.stat().st_size == 0
        or sampler.returncode not in (0, None, -15)
    ):
        print(
            f"[WARN] {run_id}: sampler produced no/empty trace or exited abnormally "
            f"(rc={sampler.returncode}) — energy will be unrecoverable for this job"
        )
    return rc


def iter_measure_jobs(
    plan_rows: list[dict[str, str]],
    engines: list[str],
) -> list[tuple[dict[str, str], str, str]]:
    """(plan_row, engine, run_id) for each planned measure job."""
    jobs: list[tuple[dict[str, str], str, str]] = []
    for row in plan_rows:
        op = row["operator"]
        idx = row["shape_index"]
        skip = parse_skip_engines(row.get("skip_engines", ""))
        for engine in engines:
            run_id = f"{op}_{engine}_s{idx}"
            jobs.append((row, engine, run_id))
    return jobs


def run_plan(args: argparse.Namespace) -> int:
    plan_path = Path(args.plan)
    outfile = Path(args.outfile)
    python = Path(args.python)
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]

    plan_rows = load_plan(plan_path)
    jobs = iter_measure_jobs(plan_rows, engines)

    failures = 0
    run_n = 0
    skip_n = 0

    print(f"[run_plan] plan={plan_path} rows={len(plan_rows)} jobs={len(jobs)} dry_run={args.dry_run}")

    for row, engine, run_id in jobs:
        op = row["operator"]
        skip = parse_skip_engines(row.get("skip_engines", ""))
        if engine in skip:
            skip_n += 1
            print(
                f"[SKIP] {op} shape_index={row['shape_index']} engine={engine} "
                f"(skip_engines={row.get('skip_engines', '')})"
            )
            continue

        harness_cmd = harness_argv(
            python=python,
            operator=op,
            engine=engine,
            shape_index=row["shape_index"],
            tier=row["tier"],
            block_id=row.get("block_id", ""),
            shape_class=row.get("shape_class", ""),
            duration=args.duration,
            warmup=args.warmup,
            repeats=args.repeats,
            device_id=args.device_id,
            run_id=run_id,
            outfile=outfile,
        )

        run_n += 1
        print()
        rc = execute_job(harness_cmd, run_id, args)
        if not args.dry_run and rc != 0:
            failures += 1
            print(f"[WARN] {run_id} returned exit code {rc}")
        if args.cooldown > 0 and not args.dry_run:
            time.sleep(args.cooldown)

    print()
    print(f"[run_plan] done runs={run_n} skips={skip_n} failures={failures}")
    return 1 if failures else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run sweep-plan measure matrix via uProf+harness")
    p.add_argument("--plan", required=True, help="Pipe-delimited sweep plan CSV")
    p.add_argument(
        "--uprof-dir",
        default=None,
        help="uProf session output directory (required when --power-backend uprof)",
    )
    p.add_argument("--outfile", required=True, help="Harness runs.csv path")
    p.add_argument("--python", default=str(DEFAULT_PY), help="Absolute python.exe path")
    p.add_argument("--uprof-cli", default=DEFAULT_UPROF, help="AMDuProfCLI executable")
    p.add_argument("--engines", default="cpu,igpu,npu", help="Comma-separated engines")
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--warmup", type=float, default=5.0)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--cooldown", type=float, default=5.0, help="Seconds between uProf wraps (0=off)")
    p.add_argument(
        "--power-backend",
        choices=["uprof", "amdsmi"],
        default="uprof",
        help="Power trace backend: uProf (default) or amd-smi sampler for discrete GPU",
    )
    p.add_argument(
        "--gpu-power-dir",
        default=None,
        help="gpu_power.py CSV output directory (required when --power-backend amdsmi)",
    )
    p.add_argument(
        "--sampler-interval",
        type=float,
        default=0.1,
        help="amd-smi sample interval in seconds (default 0.1 = 100 ms)",
    )
    p.add_argument(
        "--sampler-lead",
        type=float,
        default=0.5,
        help="Seconds of sampling before/after harness for window bracketing",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print [SKIP]/[RUN] commands only; do not execute",
    )
    args = p.parse_args(argv)

    if args.power_backend == "amdsmi" and not args.gpu_power_dir:
        p.error("--gpu-power-dir is required when --power-backend amdsmi")
    if args.power_backend == "uprof" and not args.uprof_dir:
        p.error("--uprof-dir is required when --power-backend uprof")

    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_plan(args)


if __name__ == "__main__":
    sys.exit(main())
