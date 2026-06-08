"""
parse_energy.py — Join harness timing data with uProf power traces.

Deterministic, reproducible post-processing (no AI/LLM). Reads runs.csv epoch window
bounds and integrates socket0-package-power from uProf timechart CSVs.

Headline formula:
    energy_per_op_J = (window_energy_J - dispatch_energy_J) / iterations_completed
"""

from __future__ import annotations

import argparse
import csv
import re
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


def find_uprof_csv(uprof_dir: Path, run_id: str) -> Path | None:
    direct = uprof_dir / f"{run_id}.csv"
    if direct.exists():
        return direct
    for p in uprof_dir.rglob("timechart.csv"):
        if run_id in p.read_text(encoding="utf-8", errors="replace"):
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


def load_dispatch_energies(
    rows: list[dict[str, str]],
    uprof_dir: Path,
) -> dict[tuple[str, int], float]:
    """Per (engine, repeat_idx) dispatch window energy in joules."""
    out: dict[tuple[str, int], float] = {}
    for row in rows:
        op = row.get("operator", "")
        if op not in ("dispatch_baseline", "dispatch"):
            continue
        engine = row["engine"]
        repeat = int(row.get("repeat_idx", 0))
        uprof_path = find_uprof_csv(uprof_dir, row["run_id"])
        if not uprof_path:
            continue
        samples = parse_uprof_csv(uprof_path)
        t_start = float(row.get("t_start_epoch") or 0)
        t_end = float(row.get("t_end_epoch") or 0)
        if t_start and t_end:
            out[(engine, repeat)] = integrate_power(samples, t_start, t_end)
    return out


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
    for col in ("window_energy_J", "dispatch_energy_J", "energy_per_op_J"):
        if col not in fieldnames:
            fieldnames.append(col)

    dispatch_map = load_dispatch_energies(rows, uprof_dir)
    enriched: list[dict[str, str]] = []

    for row in rows:
        run_id = row["run_id"]
        op = row.get("operator", "")
        engine = row["engine"]
        repeat = int(row.get("repeat_idx", 0))
        iterations = int(row.get("iterations_completed", 0) or 0)

        if op in ("idle",):
            row["window_energy_J"] = ""
            row["dispatch_energy_J"] = ""
            row["energy_per_op_J"] = ""
            enriched.append(row)
            continue

        t_start_s = row.get("t_start_epoch", "").strip()
        t_end_s = row.get("t_end_epoch", "").strip()
        if not t_start_s or not t_end_s:
            log_path = uprof_dir / f"{run_id}.log"
            if log_path.exists():
                t_start, t_end = parse_window_from_log(log_path)
            else:
                print(f"[SKIP] {run_id}: missing t_start_epoch/t_end_epoch and no log")
                enriched.append(row)
                continue
        else:
            t_start, t_end = float(t_start_s), float(t_end_s)

        uprof_path = find_uprof_csv(uprof_dir, run_id)
        if not uprof_path:
            print(f"[SKIP] {run_id}: uProf CSV not found under {uprof_dir}")
            enriched.append(row)
            continue

        samples = parse_uprof_csv(uprof_path)
        window_j = integrate_power(samples, t_start, t_end)
        n_in_window = _samples_in_window_count(samples, t_start, t_end)

        dispatch_j = 0.0
        if op not in ("dispatch_baseline", "dispatch"):
            dispatch_j = dispatch_map.get((engine, repeat), 0.0)

        e_per_op = energy_per_op(window_j, dispatch_j, iterations)

        print(
            f"[{run_id}] t_start={t_start:.6f} t_end={t_end:.6f} "
            f"samples_in_window={n_in_window} window_energy_J={window_j:.2f} "
            f"dispatch_energy_J={dispatch_j:.2f} energy_per_op_J={e_per_op:.6e}"
        )

        row["window_energy_J"] = f"{window_j:.6f}"
        row["dispatch_energy_J"] = f"{dispatch_j:.6f}" if dispatch_j else ""
        row["energy_per_op_J"] = f"{e_per_op:.6e}" if iterations else ""

        if plot:
            png = save_plot(samples, t_start, t_end, run_id, PLOT_DIR)
            print(f"  plot -> {png}")

        enriched.append(row)

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
