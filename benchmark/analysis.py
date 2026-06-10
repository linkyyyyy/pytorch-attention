"""
analysis.py — Deterministic fusion-gap analysis (no AI/LLM).

Consumes runs.csv (post-uProf enrichment) and reports:
  (a) per-operator energy_per_op_J ± std
  (b) Tier-1 vs Tier-2 fusion gap per (engine, block_id)

Pipeline stages are separate functions so each step is inspectable.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BENCHMARK_DIR = Path(__file__).parent
DEFAULT_RUNS = BENCHMARK_DIR / "results" / "runs.csv"
DEFAULT_OUT = BENCHMARK_DIR / "results" / "analysis_out.csv"
DEFAULT_SYNTHETIC = BENCHMARK_DIR / "results" / "runs_synthetic.csv"

DISPATCH_OPERATORS = frozenset({"dispatch_baseline", "dispatch"})
SKIP_OPERATORS = frozenset({"idle"})

# Confirm against one real uProf-joined runs.csv row before trusting production results.
# True  → window_energy_J is gross (dispatch overhead still inside); subtract dispatch here.
# False → window_energy_J is already net (harness/parse_energy removed dispatch); do NOT subtract again.
WINDOW_ENERGY_IS_RAW = True

# shape_index disambiguates q_proj / k_proj / v_proj (same operator name, different profiles).
AGG_GROUP_COLS = [
    "operator",
    "engine",
    "block_id",
    "shape_class",
    "tier",
    "fusion_member",
    "shape_index",
]


def _parse_bool_series(s: pd.Series) -> pd.Series:
    def _one(v: Any) -> bool | float:
        if pd.isna(v):
            return np.nan
        if isinstance(v, bool):
            return v
        text = str(v).strip().lower()
        if text in ("true", "1", "yes"):
            return True
        if text in ("false", "0", "no"):
            return False
        return np.nan

    return s.map(_one)


def load_runs(path: Path) -> pd.DataFrame:
    print(f"[1/5] load_runs: {path}")
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    # Restore empty cells for numeric coercion.
    df = df.replace({"": np.nan})
    print(f"      rows={len(df)} columns={len(df.columns)}")
    return df


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def dispatch_baseline_per_engine(df: pd.DataFrame) -> pd.Series:
    """
    Mean dispatch overhead (J/op) per engine from dispatch-baseline rows.
    Averages duplicates (e.g. dispatch_baseline_cpu_r0) and warns.
    """
    mask = df["operator"].isin(DISPATCH_OPERATORS)
    base = df.loc[mask].copy()
    if base.empty:
        warnings.warn("No dispatch-baseline rows found; using dispatch_energy_J column only.")
        return pd.Series(dtype=float)

    base["energy_per_op_J"] = (
        base["window_energy_J"] - base["dispatch_energy_J"].fillna(0)
    ) / base["iterations_completed"]

    dupes = base.groupby("engine").size()
    for engine, count in dupes.items():
        if count > 1:
            warnings.warn(
                f"Multiple dispatch-baseline rows for engine={engine!r} (n={count}); "
                "averaging for dedupe."
            )

    return base.groupby("engine")["energy_per_op_J"].mean()


def baseline_subtract(df: pd.DataFrame) -> pd.DataFrame:
    convention = "RAW (subtract dispatch)" if WINDOW_ENERGY_IS_RAW else "NET (no dispatch subtract)"
    print(f"[2/5] baseline_subtract — WINDOW_ENERGY_IS_RAW={WINDOW_ENERGY_IS_RAW} ({convention})")
    numeric_cols = [
        "window_energy_J",
        "dispatch_energy_J",
        "iterations_completed",
        "repeat_idx",
        "shape_index",
    ]
    work = _coerce_numeric(df, numeric_cols)

    if "fusion_member" in work.columns:
        work["fusion_member"] = _parse_bool_series(work["fusion_member"])
    else:
        work["fusion_member"] = np.nan
        warnings.warn("Column fusion_member missing; fusion gap may be incomplete.")

    dispatch_by_engine = dispatch_baseline_per_engine(work)
    if WINDOW_ENERGY_IS_RAW and not dispatch_by_engine.empty:
        print(f"      dispatch baseline J/op by engine:\n{dispatch_by_engine.to_string()}")

    measure = work[~work["operator"].isin(DISPATCH_OPERATORS | SKIP_OPERATORS)].copy()
    measure = measure[measure["iterations_completed"] > 0]

    if WINDOW_ENERGY_IS_RAW:
        # Use row dispatch_energy_J when present; else engine baseline rate × iterations.
        fallback_dispatch_j = (
            measure["engine"].map(dispatch_by_engine) * measure["iterations_completed"]
        )
        dispatch_j = measure["dispatch_energy_J"].fillna(fallback_dispatch_j).fillna(0)

        populated_dispatch = measure["dispatch_energy_J"].notna()
        bad_order = populated_dispatch & (
            measure["dispatch_energy_J"] > measure["window_energy_J"]
        )
        if bad_order.any():
            for run_id in measure.loc[bad_order, "run_id"].astype(str):
                warnings.warn(
                    f"DISPATCH CONVENTION MISMATCH? run_id={run_id}: "
                    "dispatch_energy_J > window_energy_J while WINDOW_ENERGY_IS_RAW=True — "
                    "window may already be net; consider WINDOW_ENERGY_IS_RAW=False.",
                    stacklevel=2,
                )

        measure["energy_per_op_J"] = (
            measure["window_energy_J"] - dispatch_j
        ) / measure["iterations_completed"]
    else:
        measure["energy_per_op_J"] = (
            measure["window_energy_J"] / measure["iterations_completed"]
        )

    n_bad = measure["energy_per_op_J"].isna().sum()
    if n_bad:
        warnings.warn(f"{n_bad} measure rows have NaN energy_per_op_J after baseline subtract.")

    print(f"      measure rows={len(measure)}")
    return measure


def aggregate_repeats(df: pd.DataFrame) -> pd.DataFrame:
    print("[3/5] aggregate_repeats")
    cols = [c for c in AGG_GROUP_COLS if c in df.columns]
    missing = set(AGG_GROUP_COLS) - set(cols)
    if missing:
        raise ValueError(f"Missing grouping columns for aggregation: {sorted(missing)}")

    agg = (
        df.groupby(cols, dropna=False)["energy_per_op_J"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg = agg.rename(columns={"mean": "energy_per_op_J_mean", "std": "energy_per_op_J_std"})
    agg["energy_per_op_J_std"] = agg["energy_per_op_J_std"].fillna(0.0)
    print(f"      aggregated groups={len(agg)}")
    return agg


def fusion_gap_table(agg: pd.DataFrame) -> pd.DataFrame:
    print("[4/5] fusion_gap_table")
    rows: list[dict[str, Any]] = []

    for (engine, block_id), grp in agg.groupby(["engine", "block_id"], dropna=False):
        if pd.isna(block_id) or str(block_id).strip() in ("", "n/a"):
            continue

        isolated = grp[
            (grp["tier"] == "isolated")
            & (grp["fusion_member"] == True)  # noqa: E712
        ]
        fused = grp[grp["tier"] == "fused_block"]

        if isolated.empty or fused.empty:
            continue
        if len(fused) != 1:
            warnings.warn(
                f"engine={engine} block_id={block_id}: expected 1 fused_block row, "
                f"got {len(fused)}; using mean of fused rows."
            )

        isolated_nan_n = int(isolated["energy_per_op_J_mean"].isna().sum())
        incomplete = isolated_nan_n > 0
        if incomplete:
            warnings.warn(
                f"engine={engine} block_id={block_id}: {isolated_nan_n} isolated "
                "fusion_member row(s) have NaN energy_per_op_J_mean — "
                "isolated_sum_J set to NaN (incomplete=True).",
                stacklevel=2,
            )
            isolated_sum = np.nan
            isolated_std = np.nan
        else:
            isolated_sum = isolated["energy_per_op_J_mean"].sum()
            isolated_std = float(
                np.sqrt((isolated["energy_per_op_J_std"] ** 2).sum())
            )

        fused_mean = float(fused["energy_per_op_J_mean"].mean())
        fused_std = float(
            np.sqrt((fused["energy_per_op_J_std"] ** 2).mean())
        )

        if incomplete or pd.isna(isolated_sum):
            gap_j = np.nan
            gap_pct = np.nan
        else:
            gap_j = isolated_sum - fused_mean
            gap_pct = 100.0 * gap_j / isolated_sum if isolated_sum else np.nan

        rows.append(
            {
                "engine": engine,
                "block_id": block_id,
                "shape_class": isolated["shape_class"].iloc[0],
                "isolated_n_ops": len(isolated),
                "isolated_nan_ops": isolated_nan_n,
                "incomplete": incomplete,
                "isolated_sum_J": isolated_sum,
                "isolated_sum_std_J": isolated_std,
                "fused_J": fused_mean,
                "fused_std_J": fused_std,
                "gap_J": gap_j,
                "gap_pct": gap_pct,
            }
        )

    out = pd.DataFrame(rows)
    print(f"      fusion-gap rows={len(out)}")
    return out


def per_operator_table(agg: pd.DataFrame) -> pd.DataFrame:
    """operator × engine × shape_class (and tier) with mean ± std."""
    cols = ["operator", "engine", "shape_class", "tier", "block_id", "fusion_member"]
    present = [c for c in cols if c in agg.columns]
    tail = [c for c in (
        "shape_index",
        "energy_per_op_J_mean",
        "energy_per_op_J_std",
        "count",
    ) if c in agg.columns]
    out = agg[present + tail].copy()
    out = out.sort_values(present).reset_index(drop=True)
    return out


def write_analysis_out(
    per_op: pd.DataFrame,
    fusion: pd.DataFrame,
    path: Path,
) -> None:
    print(f"[5/5] write_analysis_out: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    per_op_out = per_op.copy()
    per_op_out.insert(0, "table", "per_operator")

    fusion_out = fusion.copy()
    fusion_out.insert(0, "table", "fusion_gap")

    combined = pd.concat([per_op_out, fusion_out], ignore_index=True, sort=False)
    combined.to_csv(path, index=False)
    print(f"      wrote {len(combined)} rows ({len(per_op_out)} per_operator + {len(fusion_out)} fusion_gap)")


def print_tables(per_op: pd.DataFrame, fusion: pd.DataFrame) -> None:
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:.6e}")

    print("\n=== (a) Per-operator comparison ===")
    print(per_op.to_string(index=False))

    print("\n=== (b) Fusion gap (isolated core sum vs fused_block) ===")
    if fusion.empty:
        print("(no fusion-gap rows — need isolated fusion_member + fused_block for same block_id)")
    else:
        print(fusion.to_string(index=False))


def run_pipeline(runs_path: Path, out_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_runs(runs_path)
    measured = baseline_subtract(raw)
    agg = aggregate_repeats(measured)
    per_op = per_operator_table(agg)
    fusion = fusion_gap_table(agg)
    print_tables(per_op, fusion)
    write_analysis_out(per_op, fusion, out_path)
    return per_op, fusion


# ---------------------------------------------------------------------------
# Synthetic fixture — hand-checkable fusion gap for sdpa_avg / cpu
# ---------------------------------------------------------------------------
# 7 isolated fusion_member ops (mJ): 4+4+4+3+3+3+1 = 22 mJ → 0.022 J total
# fused attn_block_fused: 16 mJ → 0.016 J
# expected gap: 6 mJ (0.006 J), gap_pct ≈ 27.27%

_SYNTHETIC_HEADER = [
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

_ITERATIONS = 10_000


def _synthetic_row(
    *,
    run_id: str,
    operator: str,
    tier: str,
    block_id: str,
    shape_class: str,
    fusion_member: bool,
    engine: str,
    shape_index: int,
    energy_per_op_j: float,
    repeat_idx: int = 0,
    input_shape: str = "",
) -> dict[str, Any]:
    window_j = energy_per_op_j * _ITERATIONS
    return {
        "run_id": run_id,
        "operator": operator,
        "cluster": "B1",
        "tier": tier,
        "block_id": block_id,
        "shape_class": shape_class,
        "fusion_member": str(fusion_member),
        "engine": engine,
        "device_id": "0",
        "shape_index": str(shape_index),
        "input_shape": input_shape,
        "dtype": "float32",
        "opset": "20",
        "intra_op_num_threads": "12",
        "graph_optimization_level": "ORT_ENABLE_ALL",
        "igpu_vgm_mb": "512",
        "repeat_idx": str(repeat_idx),
        "warmup_s": "5",
        "window_s": "30",
        "iterations_completed": str(_ITERATIONS),
        "wall_time_s": "30.0",
        "t_start_epoch": "1780925790.0",
        "t_end_epoch": "1780925820.0",
        "mean_latency_ms": "1.0",
        "idle_power_w": "",
        "active_power_w": "",
        "window_energy_J": f"{window_j:.6f}",
        "idle_energy_J": "",
        "energy_per_op_J": "",
        "dispatch_energy_J": "0",
        "notes": "synthetic",
    }


def write_runs_synthetic(path: Path) -> Path:
    """
    Build runs_synthetic.csv with known arithmetic for sdpa_avg / cpu.

    Isolated core ops (mJ): q=4, k=4, v=4, score=3, softmax=3, av=3, out=1 → 22 mJ
    Fused block: 16 mJ → gap 6 mJ (27.27%)
    Includes duplicate dispatch_baseline rows to exercise dedupe + warn.
    """
    print(f"[synthetic] write_runs_synthetic: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    mj_to_j = 1e-3
    isolated_ops = [
        ("qkv_proj_gemm", 0, 4.0, "q_proj"),
        ("qkv_proj_gemm", 1, 4.0, "k_proj"),
        ("qkv_proj_gemm", 2, 4.0, "v_proj"),
        ("attn_score_matmul", 3, 3.0, "attn_qkt"),
        ("softmax", 4, 3.0, "attn_softmax"),
        ("attn_value_matmul", 5, 3.0, "attn_av"),
        ("out_proj_gemm", 6, 1.0, "out_proj"),
    ]

    rows: list[dict[str, Any]] = []
    for op, shape_idx, mj, label in isolated_ops:
        rows.append(
            _synthetic_row(
                run_id=f"{op}_{label}_cpu_r0",
                operator=op,
                tier="isolated",
                block_id="sdpa_avg",
                shape_class="avg",
                fusion_member=True,
                engine="cpu",
                shape_index=shape_idx,
                energy_per_op_j=mj * mj_to_j,
                input_shape=label,
            )
        )

    # Second repeat on q_proj to exercise mean/std aggregation (same energy → std=0).
    rows.append(
        _synthetic_row(
            run_id="qkv_proj_gemm_q_proj_cpu_r1",
            operator="qkv_proj_gemm",
            tier="isolated",
            block_id="sdpa_avg",
            shape_class="avg",
            fusion_member=True,
            engine="cpu",
            shape_index=0,
            energy_per_op_j=4.0 * mj_to_j,
            repeat_idx=1,
            input_shape="q_proj",
        )
    )

    rows.append(
        _synthetic_row(
            run_id="attn_block_fused_sdpa_avg_cpu_r0",
            operator="attn_block_fused",
            tier="fused_block",
            block_id="sdpa_avg",
            shape_class="avg",
            fusion_member=True,
            engine="cpu",
            shape_index=1,
            energy_per_op_j=16.0 * mj_to_j,
            input_shape="197x768",
        )
    )

    # Context op — must NOT enter fusion sum.
    rows.append(
        _synthetic_row(
            run_id="layer_norm_sdpa_avg_cpu_r0",
            operator="layer_norm",
            tier="isolated",
            block_id="sdpa_avg",
            shape_class="avg",
            fusion_member=False,
            engine="cpu",
            shape_index=0,
            energy_per_op_j=50.0 * mj_to_j,
            input_shape="197x768",
        )
    )

    # Duplicate dispatch baselines (dedupe test).
    for rid in ("dispatch_baseline_cpu_r0", "dispatch_baseline_cpu_r0_dup"):
        rows.append(
            _synthetic_row(
                run_id=rid,
                operator="dispatch_baseline",
                tier="n/a",
                block_id="n/a",
                shape_class="n/a",
                fusion_member=False,
                engine="cpu",
                shape_index=-1,
                energy_per_op_j=0.05 * mj_to_j,
                input_shape="1",
            )
        )

    df = pd.DataFrame(rows, columns=_SYNTHETIC_HEADER)
    df.to_csv(path, index=False)
    print(f"      wrote {len(df)} synthetic rows")
    print("      expected: isolated_sum=0.022 J (22 mJ), fused=0.016 J (16 mJ), gap=0.006 J (~27.27%)")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fusion-gap analysis on runs.csv")
    p.add_argument("--runs", type=Path, default=DEFAULT_RUNS, help="Input runs.csv")
    p.add_argument("--outfile", type=Path, default=DEFAULT_OUT, help="Output analysis_out.csv")
    p.add_argument(
        "--write-synthetic",
        type=Path,
        nargs="?",
        const=DEFAULT_SYNTHETIC,
        metavar="PATH",
        help="Write runs_synthetic.csv fixture and exit (default path if no PATH)",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="Write synthetic fixture, run pipeline on it, print results",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.write_synthetic is not None:
        write_runs_synthetic(args.write_synthetic)
        if not args.demo:
            return 0

    if args.demo:
        synth_path = args.write_synthetic or DEFAULT_SYNTHETIC
        write_runs_synthetic(synth_path)
        print("\n" + "=" * 72)
        print("DEMO: running pipeline on synthetic fixture")
        print("=" * 72 + "\n")
        run_pipeline(synth_path, args.outfile)
        return 0

    run_pipeline(args.runs, args.outfile)
    return 0


if __name__ == "__main__":
    sys.exit(main())
