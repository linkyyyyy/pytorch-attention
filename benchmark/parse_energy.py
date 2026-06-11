"""
parse_energy.py — Join harness timing data with uProf power traces.

Deterministic, reproducible post-processing (no AI/LLM). Reads runs.csv epoch window
bounds and integrates socket0-package-power from uProf timechart CSVs.

Window energy stays RAW (WINDOW_ENERGY_IS_RAW=True in analysis.py).
parse_energy joins uProf integrals only; baseline subtraction is in analysis.py.
"""

from __future__ import annotations

import argparse
import csv
import re
import warnings
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

BENCHMARK_DIR = Path(__file__).parent
DEFAULT_RUNS = BENCHMARK_DIR / "results" / "runs.csv"
DEFAULT_OUT = BENCHMARK_DIR / "results" / "runs_enriched.csv"
DEFAULT_UPROF_DIR = BENCHMARK_DIR / "results" / "uprof"
PLOT_DIR = BENCHMARK_DIR / "results" / "plots"

TZ = ZoneInfo("Europe/Athens")
MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

PROFILE_START_RE = re.compile(
    r"Profile Start Time:,(\w+)-(\d+)-(\d+)_(\d+)-(\d+)-(\d+)",
)
RECORDS_HEADER = "RecordId,Timestamp,socket0-package-power"
WINDOW_OPEN_RE = re.compile(r"\[WINDOW_OPEN\].*t_start=([\d.]+)")
WINDOW_CLOSE_RE = re.compile(r"\[WINDOW_CLOSE\].*t_end=([\d.]+)")

DISPATCH_OPERATORS = frozenset({"dispatch_baseline", "dispatch"})
IDLE_OPERATOR = "idle"


@dataclass(frozen=True)
class PowerSample:
    epoch: float
    watts: float


def parse_profile_start_date(lines: Iterable[str]) -> date:
    for line in lines:
        m = PROFILE_START_RE.search(line)
        if m:
            mon, day, year, _, _, _ = m.groups()
            return date(int(year), MONTH_MAP[mon], int(day))
    raise ValueError("Profile Start Time not found in uProf CSV preamble")


def parse_uprof_timestamp(ts: str) -> tuple[int, int, int, int]:
    """Parse HH:MM:SS:ms (colon before milliseconds)."""
    parts = ts.strip().split(":")
    if len(parts) != 4:
        raise ValueError(f"Expected HH:MM:SS:ms, got {ts!r}")
    return int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])


def seconds_of_day(h: int, m: int, s: int, ms: int) -> float:
    return h * 3600 + m * 60 + s + ms / 1000.0


def to_epoch(session_day: date, sod: float, day_offset: int) -> float:
    base = datetime(
        session_day.year, session_day.month, session_day.day,
        tzinfo=TZ,
    )
    return base.timestamp() + day_offset * 86400 + sod


def parse_uprof_csv(path: Path) -> list[PowerSample]:
    """
    Parse uProf timechart CSV → (epoch_seconds, power_watts) samples.
    Uses Profile Start Time for session date; handles midnight rollover.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    session_day = parse_profile_start_date(lines)

    records_start = None
    for i, line in enumerate(lines):
        if line.startswith(RECORDS_HEADER):
            records_start = i + 1
            break
    if records_start is None:
        raise ValueError(f"PROFILE RECORDS header not found in {path}")

    samples: list[PowerSample] = []
    prev_sod: float | None = None
    day_offset = 0

    for line in lines[records_start:]:
        if not line.strip():
            continue
        row = next(csv.reader([line]))
        if len(row) < 3:
            continue
        try:
            ts = row[1].strip()
            power = float(row[2].strip())
        except (ValueError, IndexError):
            continue

        h, m, s, ms = parse_uprof_timestamp(ts)
        sod = seconds_of_day(h, m, s, ms)
        if prev_sod is not None and sod < prev_sod - 0.5:
            day_offset += 1
        prev_sod = sod
        epoch = to_epoch(session_day, sod, day_offset)
        samples.append(PowerSample(epoch, power))

    if not samples:
        raise ValueError(f"No power samples parsed from {path}")
    return samples


def _interp_power(samples: list[PowerSample], t: float) -> float:
    if t <= samples[0].epoch:
        return samples[0].watts
    if t >= samples[-1].epoch:
        return samples[-1].watts
    for i in range(len(samples) - 1):
        t0, p0 = samples[i].epoch, samples[i].watts
        t1, p1 = samples[i + 1].epoch, samples[i + 1].watts
        if t0 <= t <= t1:
            if t1 == t0:
                return p0
            frac = (t - t0) / (t1 - t0)
            return p0 + frac * (p1 - p0)
    return samples[-1].watts


def integrate_power(
    samples: list[PowerSample],
    t_start: float,
    t_end: float,
) -> float:
    """
    Trapezoidal integration of power (W) over [t_start, t_end] → joules.
    Interpolates power at window edges.
    """
    if t_end <= t_start:
        return 0.0
    if not samples:
        return 0.0

    p_start = _interp_power(samples, t_start)
    p_end = _interp_power(samples, t_end)

    interior = [(s.epoch, s.watts) for s in samples if t_start < s.epoch < t_end]
    timeline = [(t_start, p_start)] + interior + [(t_end, p_end)]
    timeline.sort(key=lambda x: x[0])

    energy = 0.0
    for i in range(len(timeline) - 1):
        t0, p0 = timeline[i]
        t1, p1 = timeline[i + 1]
        dt = t1 - t0
        if dt > 0:
            energy += 0.5 * (p0 + p1) * dt
    return energy


def energy_per_op(
    window_energy_j: float,
    dispatch_energy_j: float,
    iterations: int,
) -> float:
    if iterations <= 0:
        return float("nan")
    return (window_energy_j - dispatch_energy_j) / iterations


def parse_window_from_log(log_path: Path) -> tuple[float, float]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    m_open = WINDOW_OPEN_RE.search(text)
    m_close = WINDOW_CLOSE_RE.search(text)
    if not m_open or not m_close:
        raise ValueError(f"Could not find WINDOW_OPEN/CLOSE in {log_path}")
    return float(m_open.group(1)), float(m_close.group(1))


def _uprof_match_stems(run_id: str) -> list[str]:
    """Keys for matching uProf -o folder names and harness --run-id in CSV preamble."""
    stems: list[str] = [run_id]
    if run_id.endswith("._r0"):
        stems.append(run_id[:-4])  # dispatch_baseline_npu_r0._r0 -> ...npu_r0.
        stems.append(run_id.replace("._r0", ""))  # -> ...npu_r0
    m = re.match(r"^(.*)_r\d+$", run_id)
    if m:
        stems.append(m.group(1))  # ffn_gemm_npu_smoke_r0 -> ffn_gemm_npu_smoke
    # de-dupe, longest first (more specific match first)
    seen: set[str] = set()
    out: list[str] = []
    for s in sorted(stems, key=len, reverse=True):
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def find_uprof_csv(uprof_dir: Path, run_id: str) -> Path | None:
    for stem in _uprof_match_stems(run_id):
        direct = uprof_dir / f"{stem}.csv"
        if direct.exists():
            return direct
        nested = uprof_dir / stem / "timechart.csv"
        if nested.is_file():
            return nested
        # uProf -o paths may omit trailing punctuation from a mistyped --run-id
        stem_rstrip = stem.rstrip(".")
        if stem_rstrip != stem:
            nested2 = uprof_dir / stem_rstrip / "timechart.csv"
            if nested2.is_file():
                return nested2
    for p in uprof_dir.rglob("timechart.csv"):
        text = p.read_text(encoding="utf-8", errors="replace")
        for stem in _uprof_match_stems(run_id):
            if stem in text:
                return p
    return None


def _samples_in_window_count(samples: list[PowerSample], t_start: float, t_end: float) -> int:
    return sum(1 for s in samples if t_start <= s.epoch <= t_end)


def save_plot(
    samples: list[PowerSample],
    t_start: float,
    t_end: float,
    run_id: str,
    out_dir: Path,
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("--plot requires matplotlib") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    times = [s.epoch for s in samples]
    powers = [s.watts for s in samples]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, powers, linewidth=0.8, label="socket0-package-power (W)")
    ax.axvline(t_start, color="green", linestyle="--", linewidth=1.2, label="t_start")
    ax.axvline(t_end, color="red", linestyle="--", linewidth=1.2, label="t_end")
    ax.set_xlabel("epoch (s)")
    ax.set_ylabel("package power (W)")
    ax.set_title(f"{run_id} — uProf window alignment")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = out_dir / f"{run_id}_power.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _run_base(run_id: str) -> str:
    m = re.match(r"^(.*)_r\d+$", run_id.strip())
    return m.group(1) if m else run_id.strip()


def _window_energy_for_row(
    row: dict[str, str],
    uprof_dir: Path,
) -> tuple[float, float, float, list[PowerSample]] | None:
    """Integrate uProf window for one harness row; None if join fails."""
    run_id = row["run_id"]
    t_start_s = row.get("t_start_epoch", "").strip()
    t_end_s = row.get("t_end_epoch", "").strip()
    if not t_start_s or not t_end_s:
        log_path = uprof_dir / f"{run_id}.log"
        if log_path.exists():
            t_start, t_end = parse_window_from_log(log_path)
        else:
            return None
    else:
        t_start, t_end = float(t_start_s), float(t_end_s)

    uprof_path = find_uprof_csv(uprof_dir, run_id)
    if not uprof_path:
        return None

    samples = parse_uprof_csv(uprof_path)
    window_j = integrate_power(samples, t_start, t_end)
    return t_start, t_end, window_j, samples


def load_idle_mean_j(
    rows: list[dict[str, str]],
    uprof_dir: Path,
) -> float | None:
    """Mean idle window energy (J) over all idle captures; engine tag is metadata only."""
    values: list[float] = []
    for row in rows:
        if row.get("operator") != IDLE_OPERATOR:
            continue
        joined = _window_energy_for_row(row, uprof_dir)
        if joined:
            values.append(joined[2])
    if not values:
        return None
    return sum(values) / len(values)


def load_dispatch_mean_j_by_engine(
    rows: list[dict[str, str]],
    uprof_dir: Path,
) -> dict[str, float]:
    """Mean dispatch window energy (J) per engine over all dispatch captures."""
    by_engine: dict[str, list[float]] = {}
    for row in rows:
        if row.get("operator") not in DISPATCH_OPERATORS:
            continue
        joined = _window_energy_for_row(row, uprof_dir)
        if not joined:
            continue
        by_engine.setdefault(row["engine"], []).append(joined[2])
    return {engine: sum(vals) / len(vals) for engine, vals in by_engine.items()}


def _dedup_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Dedup exact run_id (keep latest) then baseline rows per (engine, run_base)."""
    by_run_id: dict[str, dict[str, str]] = {}
    for row in rows:
        rid = row["run_id"]
        if rid in by_run_id:
            warnings.warn(f"Duplicate run_id {rid!r}; keeping latest row.", stacklevel=2)
        by_run_id[rid] = row
    deduped = list(by_run_id.values())

    kept: list[dict[str, str]] = []
    seen_baseline: set[tuple[str, str, str]] = set()
    for row in deduped:
        op = row.get("operator", "")
        if op in DISPATCH_OPERATORS:
            key = ("dispatch", row["engine"], _run_base(row["run_id"]))
        elif op == IDLE_OPERATOR:
            key = ("idle", row["engine"], _run_base(row["run_id"]))
        else:
            kept.append(row)
            continue
        if key in seen_baseline:
            warnings.warn(
                f"Duplicate baseline row for engine={row['engine']!r} "
                f"run_base={_run_base(row['run_id'])!r} mode={key[0]}; keeping latest.",
                stacklevel=2,
            )
            kept = [r for r in kept if not (
                (r.get("operator") in DISPATCH_OPERATORS and key[0] == "dispatch"
                 and r["engine"] == row["engine"]
                 and _run_base(r["run_id"]) == key[2])
                or (r.get("operator") == IDLE_OPERATOR and key[0] == "idle"
                    and r["engine"] == row["engine"]
                    and _run_base(r["run_id"]) == key[2])
            )]
        seen_baseline.add(key)
        kept.append(row)
    return kept


def enrich_runs(
    runs_path: Path,
    uprof_dir: Path,
    out_path: Path,
    *,
    plot: bool = False,
) -> list[dict[str, str]]:
    with runs_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {runs_path}")

    fieldnames = list(rows[0].keys())
    for col in ("window_energy_J", "idle_energy_J", "dispatch_energy_J", "energy_per_op_J"):
        if col not in fieldnames:
            fieldnames.append(col)

    idle_mean_j = load_idle_mean_j(rows, uprof_dir)
    dispatch_mean_j = load_dispatch_mean_j_by_engine(rows, uprof_dir)
    if idle_mean_j is None:
        print("[WARN] no idle captures; idle_energy_J left empty — headline will be unsubtracted")
    else:
        print(f"[idle_mean_J] {idle_mean_j:.2f} J (mean over all idle captures)")
    warned_dispatch_engines: set[str] = set()
    enriched: list[dict[str, str]] = []

    for row in rows:
        run_id = row["run_id"]
        op = row.get("operator", "")
        engine = row["engine"]

        joined = _window_energy_for_row(row, uprof_dir)
        if not joined:
            print(f"[SKIP] {run_id}: uProf join failed (epochs or CSV missing)")
            row.setdefault("window_energy_J", "")
            row.setdefault("idle_energy_J", "")
            row.setdefault("dispatch_energy_J", "")
            row.setdefault("energy_per_op_J", "")
            enriched.append(row)
            continue

        t_start, t_end, window_j, samples = joined
        n_in_window = _samples_in_window_count(samples, t_start, t_end)

        row["window_energy_J"] = f"{window_j:.6f}"
        row["energy_per_op_J"] = ""

        if op == IDLE_OPERATOR:
            row["idle_energy_J"] = ""
            row["dispatch_energy_J"] = ""
            print(
                f"[{run_id}] idle floor t_start={t_start:.6f} t_end={t_end:.6f} "
                f"samples_in_window={n_in_window} window_energy_J={window_j:.2f}"
            )
        elif op in DISPATCH_OPERATORS:
            row["idle_energy_J"] = ""
            row["dispatch_energy_J"] = ""
            print(
                f"[{run_id}] dispatch baseline t_start={t_start:.6f} t_end={t_end:.6f} "
                f"samples_in_window={n_in_window} window_energy_J={window_j:.2f}"
            )
        else:
            row["idle_energy_J"] = f"{idle_mean_j:.6f}" if idle_mean_j is not None else ""
            dispatch_j = dispatch_mean_j.get(engine)
            row["dispatch_energy_J"] = f"{dispatch_j:.6f}" if dispatch_j is not None else ""
            if dispatch_j is None and engine not in warned_dispatch_engines:
                print(
                    f"[WARN] no dispatch captures for engine={engine!r}; "
                    "dispatch_energy_J left empty — secondary metric will be unsubtracted"
                )
                warned_dispatch_engines.add(engine)
            print(
                f"[{run_id}] t_start={t_start:.6f} t_end={t_end:.6f} "
                f"samples_in_window={n_in_window} window_energy_J={window_j:.2f} "
                f"idle_energy_J={idle_mean_j or 'n/a'} "
                f"dispatch_energy_J={dispatch_j if dispatch_j is not None else 'n/a'}"
            )

        if plot:
            png = save_plot(samples, t_start, t_end, run_id, PLOT_DIR)
            print(f"  plot -> {png}")

        enriched.append(row)

    enriched = _dedup_rows(enriched)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(enriched)

    print(f"[OK] wrote {out_path} ({len(enriched)} rows)")
    return enriched


def run_toy_test(
    uprof_csv: Path,
    t_start: float,
    t_end: float,
    iterations: int,
    dispatch_j: float,
    *,
    plot: bool,
) -> None:
    print("=" * 72)
    print("[TEST] toy_igpu verification")
    print(f"  uProf CSV: {uprof_csv}")
    print(f"  t_start={t_start:.6f} t_end={t_end:.6f} duration={t_end - t_start:.3f}s")
    print(f"  iterations={iterations}")
    print("=" * 72)

    samples = parse_uprof_csv(uprof_csv)
    window_j = integrate_power(samples, t_start, t_end)
    n_in_window = _samples_in_window_count(samples, t_start, t_end)
    e_per_op = energy_per_op(window_j, dispatch_j, iterations)

    in_window_powers = [s.watts for s in samples if t_start <= s.epoch <= t_end]
    avg_w = sum(in_window_powers) / len(in_window_powers) if in_window_powers else float("nan")

    print(f"samples_in_window={n_in_window}")
    print(f"mean_package_power_in_window={avg_w:.2f} W")
    print(f"window_energy_J={window_j:.2f}")
    print(f"dispatch_energy_J={dispatch_j:.2f}")
    print(f"energy_per_op_J={e_per_op:.6e}")
    print(
        f"sanity: ~{avg_w:.0f} W x {t_end - t_start:.1f} s "
        f"~ {avg_w * (t_end - t_start):.0f} J"
    )

    if plot:
        png = save_plot(samples, t_start, t_end, "toy_igpu", PLOT_DIR)
        print(f"plot -> {png}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Integrate uProf power with harness windows")
    p.add_argument("--runs", type=Path, default=DEFAULT_RUNS, help="Input runs.csv")
    p.add_argument("--uprof-dir", type=Path, default=DEFAULT_UPROF_DIR, help="uProf CSV directory")
    p.add_argument("--outfile", type=Path, default=DEFAULT_OUT, help="Enriched output CSV")
    p.add_argument("--plot", action="store_true", help="Save per-run power PNG with window markers")

    p.add_argument("--test-toy", action="store_true", help="Run toy_igpu verification case")
    p.add_argument(
        "--uprof-csv",
        type=Path,
        default=BENCHMARK_DIR / "uprof_toy" / "AMDuProf-python-Timechart_Jun-08-2026_16-36-25" / "timechart.csv",
    )
    p.add_argument("--t-start", type=float, default=1780925790.173200)
    p.add_argument("--t-end", type=float, default=1780925800.172677)
    p.add_argument("--iterations", type=int, default=13560)
    p.add_argument("--dispatch-j", type=float, default=0.0, help="Dispatch baseline J for toy test")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.test_toy:
        run_toy_test(
            args.uprof_csv,
            args.t_start,
            args.t_end,
            args.iterations,
            args.dispatch_j,
            plot=args.plot,
        )
        return
    enrich_runs(args.runs, args.uprof_dir, args.outfile, plot=args.plot)


if __name__ == "__main__":
    main()
