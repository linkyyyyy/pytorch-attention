"""One-off R9700 vs Phase-1 HX370 comparison (off-tower)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BENCHMARK = Path(__file__).parent
ENRICHED = BENCHMARK / "results/runs_20260624_133311_r9700_enriched.csv"
ANALYSIS_OUT = BENCHMARK / "results/analysis_out_r9700.csv"
STEP4 = BENCHMARK / "results/STEP4_FOR_SHEET_primary.csv"
OUT_MD = BENCHMARK / "results/R9700_CROSS_ENGINE_REPORT.md"
OUT_CSV = BENCHMARK / "results/r9700_vs_hx370_comparison.csv"

MEMORY_BOUND = {
    "softmax",
    "gelu",
    "layer_norm",
    "group_norm",
    "batch_norm",
    "residual_add",
    "avg_pool_token_mixer",
    "depthwise_conv2d",
    "xcit_cov_matmul",
}
COMPUTE_BOUND = {
    "ffn_gemm",
    "qkv_proj_gemm",
    "out_proj_gemm",
    "attn_score_matmul",
    "attn_value_matmul",
    "patch_embed_conv2d",
    "downsample_conv2d",
    "sra_conv2d",
}


def ensure_step4() -> None:
    if STEP4.exists():
        return
    raw = subprocess.check_output(
        ["git", "show", "uProfAnalysis:benchmark/STEP4_FOR_SHEET_primary.csv"],
        cwd=BENCHMARK.parent,
        text=True,
    )
    lines = [ln for ln in raw.splitlines() if not ln.startswith("#")]
    STEP4.write_text("\n".join(lines) + "\n", encoding="utf-8")


def op_key(operator: str, shape_index: int | float) -> str:
    si = int(shape_index) if pd.notna(shape_index) else 0
    if operator == "ffn_gemm" and si in (2, 3):
        return f"ffn_gemm[s{si}]"
    if operator == "qkv_proj_gemm":
        return f"qkv_proj_gemm[s{si}]"
    if si in (0, 1) and operator in (
        "attn_score_matmul",
        "attn_value_matmul",
        "out_proj_gemm",
        "softmax",
        "gelu",
        "layer_norm",
        "residual_add",
        "attn_block_fused",
    ):
        return f"{operator}[s1]"
    return operator


def main() -> int:
    ensure_step4()

    enr = pd.read_csv(ENRICHED)
    for c in (
        "window_energy_J",
        "idle_energy_J",
        "dispatch_energy_J",
        "iterations_completed",
        "energy_per_op_J",
        "gfx_busy_mean_pct",
    ):
        if c in enr.columns:
            enr[c] = pd.to_numeric(enr[c], errors="coerce")

    # --- idle drift ---
    idle = enr[enr["operator"] == "idle"].copy()
    open_j = float(idle.loc[idle["run_id"].str.contains("open"), "window_energy_J"].iloc[0])
    close_j = float(idle.loc[idle["run_id"].str.contains("close"), "window_energy_J"].iloc[0])
    mid_j = idle.loc[idle["run_id"].str.contains("mid"), "window_energy_J"].astype(float)
    session_idle_mean = float(idle["window_energy_J"].mean())
    drift_pct = 100.0 * (close_j - open_j) / open_j

    disp = enr[enr["operator"].isin({"dispatch_baseline", "dispatch"})]
    dispatch_j = float(disp["window_energy_J"].iloc[0])

    # method tags
    meas = enr[~enr["operator"].isin({"idle", "dispatch_baseline", "dispatch"})]
    method_counts = meas["window_energy_method"].value_counts()

    # --- analysis.py headline grid ---
    ao = pd.read_csv(ANALYSIS_OUT)
    r9700 = ao[ao["table"] == "per_operator"].copy()
    r9700["op_key"] = r9700.apply(
        lambda r: op_key(r["operator"], r.get("shape_index", 0)), axis=1
    )

    # --- phase 1 step4 ---
    s4 = pd.read_csv(STEP4)
    for c in ("cpu_FP32", "igpu_FP32", "npu_INT8"):
        s4[c] = pd.to_numeric(s4[c], errors="coerce")

    rows = []
    for _, r in r9700.iterrows():
        key = r["op_key"]
        base = s4[s4["operator"] == key]
        if base.empty and "[" not in key:
            base = s4[s4["operator"] == r["operator"]]
        cpu = float(base["cpu_FP32"].iloc[0]) if not base.empty else np.nan
        igpu = float(base["igpu_FP32"].iloc[0]) if not base.empty else np.nan
        r97 = float(r["energy_per_op_J_mean"])
        rows.append(
            {
                "op_key": key,
                "operator": r["operator"],
                "shape_index": r.get("shape_index", ""),
                "cluster_hint": (
                    "MEMORY_BOUND"
                    if r["operator"] in MEMORY_BOUND
                    else "COMPUTE_BOUND"
                    if r["operator"] in COMPUTE_BOUND
                    else "MIXED"
                ),
                "r9700_J": r97,
                "cpu_FP32_J": cpu,
                "igpu_FP32_J": igpu,
                "ratio_r9700_cpu": r97 / cpu if cpu and cpu > 0 else np.nan,
                "ratio_r9700_igpu": r97 / igpu if igpu and igpu > 0 else np.nan,
                "bandwidth_hypothesis": (
                    "r9700 wins vs igpu"
                    if r["operator"] in MEMORY_BOUND and igpu and r97 < igpu
                    else "r9700 loses vs igpu"
                    if r["operator"] in MEMORY_BOUND and igpu and r97 >= igpu
                    else "n/a"
                ),
            }
        )

    cmp_df = pd.DataFrame(rows).sort_values("op_key")
    cmp_df.to_csv(OUT_CSV, index=False)

    mem = cmp_df[cmp_df["cluster_hint"] == "MEMORY_BOUND"]
    mem_wins_igpu = int((mem["ratio_r9700_igpu"] < 1.0).sum())
    mem_total = int(mem["ratio_r9700_igpu"].notna().sum())

    dense = cmp_df[cmp_df["cluster_hint"] == "COMPUTE_BOUND"]
    dense_wins_cpu = int((dense["ratio_r9700_cpu"] < 1.0).sum())
    dense_total = int(dense["ratio_r9700_cpu"].notna().sum())

    md = f"""# R9700 cross-engine analysis — session `20260624_133311`

Generated off-tower from `runs_20260624_133311_r9700_enriched.csv` + `analysis.py`.

## Baseline policy check (`analysis.py`)

| Policy | Expected | Observed |
|---|---|---|
| `WINDOW_ENERGY_IS_RAW` | True — subtract idle + dispatch in analysis | True (default) |
| Idle floor | Session mean over all idle rows | **{session_idle_mean:.2f} J** (n={len(idle)}: open + 2×mid + close) |
| Dispatch floor | Per-engine mean from dispatch row(s) | **{dispatch_j:.2f} J** (`dispatch_r9700_base`) |
| Headline | `(window - idle) / iters` | `energy_per_op_J` in analysis_out |
| Secondary | `(window - dispatch) / iters` | `energy_per_op_dispatch` in analysis_out |

`parse_energy.py` pre-fills `idle_energy_J` / `dispatch_energy_J` on measure rows with the same
session means; `analysis.py` recomputes from idle/dispatch rows when present (warns on multiple idle).

## Energy integration method

All {int(method_counts.sum())} measure rows: **`{method_counts.index[0]}`** (Branch B — SMU counter dead on gfx1201).

## Idle drift (in-session)

| Window | Energy (J) |
|---|---:|
| open | {open_j:.2f} |
| mid1 | {float(mid_j.iloc[0]):.2f} |
| mid2 | {float(mid_j.iloc[1]):.2f} |
| close | {close_j:.2f} |
| **open -> close drift** | **{drift_pct:+.1f}%** |

## Bandwidth hypothesis (memory-bound vs iGPU FP32)

Central hypothesis: GDDR6 R9700 should beat shared-LPDDR5x 890M on memory-bound ops.

- Memory-bound ops with both engines: **{mem_wins_igpu}/{mem_total}** where r9700 < igpu (lower J/op = win)
- Compute-bound ops vs cpu FP32: **{dense_wins_cpu}/{dense_total}** where r9700 < cpu

### Memory-bound detail

```
{mem[['op_key','r9700_J','igpu_FP32_J','ratio_r9700_igpu','bandwidth_hypothesis']].to_string(index=False)}
```

### Largest r9700 advantages vs iGPU (memory-bound, ratio < 1)

```
{mem.nsmallest(5,'ratio_r9700_igpu')[['op_key','r9700_J','igpu_FP32_J','ratio_r9700_igpu']].to_string(index=False) if not mem.empty else '(none)'}
```

### Full comparison table

See `{OUT_CSV.name}`.

## Methodology / limitations (paper carry-forward)

1. **Cross-instrument:** HX370 package power (uProf, Windows, ORT) vs R9700 board `socket_power` (amd-smi, Ubuntu, MIGraphX-direct). Within-engine idle-subtracted deltas are clean; absolute cross-engine J/op comparisons carry instrument + runtime caveat.
2. **Runtime asymmetry:** R9700 uses MIGraphX Python API (`offload_copy=False`, resident GPU buffers, `sync_every=64`); HX370 uses single ORT build (CPU/DML/Vitis). Not a matched-runtime comparison.
3. **Branch B integration:** All R9700 windows integrated via trapezoidal `socket_power` (`trapz_power_w:no_counter`) — SMU energy accumulator returned zero on gfx1201.
4. **Idle drift:** In-session open->close idle drift **{drift_pct:+.1f}%** on the discrete rail; headline uses session-mean idle (285 J class), not per-repeat idle pairing.
5. **Precision:** R9700 + cpu/iGPU Phase-1 headline = FP32; NPU Phase-1 = XINT8 (excluded from this FP32 cross-GPU slice).
6. **Dispatch validity:** R9700 dispatch baseline runs on GPU (MIGraphX); unlike NPU CPU-fallback dispatch on HX370, dispatch-subtraction is methodologically valid for r9700 secondary metric.
"""
    OUT_MD.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n[OK] wrote {OUT_MD} and {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
