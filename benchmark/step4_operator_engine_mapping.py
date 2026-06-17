#!/usr/bin/env python3
"""
Step 4 — operator-level deployment engine mapping (uProfAnalysis branch).

Energy sources (--source):
  section3 (default): ANALYSIS_REFERENCE.md §3 committed snapshot.
  primary: results/analysis_out.csv + analysis_out_20260615_145614_cpu_int8.csv.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# --- CONFIG (Chris reviews tie logic) ---
TIE_THRESHOLD = 0.10
TIE_POLICY = "report"
PRECISION_GUARD = 0.25
SIGN_MARGIN = 0.10  # sign-divergence guard (mirrors deployment tie rule)
GROUP_KEY = ("operator", "shape_class")

BENCHMARK_DIR = Path(__file__).parent
ANALYSIS_REF = BENCHMARK_DIR / "ANALYSIS_REFERENCE.md"
RESULTS_DIR = BENCHMARK_DIR / "results"
PRIMARY_ANALYSIS = RESULTS_DIR / "analysis_out.csv"
CONTROL_ANALYSIS = RESULTS_DIR / "analysis_out_20260615_145614_cpu_int8.csv"
SESSION_PRODUCTION = "20260612_143854"
SESSION_CONTROL = "20260615_145614_cpu_int8"
RUNS_PRODUCTION = RESULTS_DIR / f"runs_{SESSION_PRODUCTION}.csv"
RUNS_CONTROL = RESULTS_DIR / f"runs_{SESSION_CONTROL}.csv"

OUT_CSV = RESULTS_DIR / "step4_operator_engine_mapping.csv"
OUT_MD = BENCHMARK_DIR / "STEP4_OPERATOR_ENGINE_MAPPING.md"
OUT_HANDOFF = BENCHMARK_DIR / "STEP4_HANDOFF.md"
OUT_SHEET = BENCHMARK_DIR / "STEP4_FOR_SHEET.csv"
OUT_CSV_PRIMARY = RESULTS_DIR / "step4_operator_engine_mapping_primary.csv"
OUT_MD_PRIMARY = BENCHMARK_DIR / "STEP4_OPERATOR_ENGINE_MAPPING_primary.md"
OUT_SHEET_PRIMARY = BENCHMARK_DIR / "STEP4_FOR_SHEET_primary.csv"
OUT_CSV_PRIMARY_COMMITTED = BENCHMARK_DIR / "step4_operator_engine_mapping_primary.csv"

PROVENANCE_BANNER_SECTION3 = (
    "PROVENANCE: energies from ANALYSIS_REFERENCE.md §3 (committed production snapshot), "
    "NOT primary results/ CSVs. Paper-grade table must be regenerated from tower CSVs "
    "(analysis_out.csv + analysis_out_20260615_145614_cpu_int8.csv) before publication."
)
PROVENANCE_BANNER_PRIMARY = (
    "PROVENANCE: energies from primary tower CSVs "
    "(analysis_out.csv + analysis_out_20260615_145614_cpu_int8.csv) at full precision."
)

SCHEMA_KEYS = ("operator", "shape_index")
ENERGY_COLS = ("cpu_FP32", "igpu_FP32", "npu_INT8", "cpu_INT8")
RATIO_COLS = ("precision", "architecture_ratio", "gap")
SCHEMA_COLS = (
    "operator",
    "shape_index",
    "shape_class",
    "operator_label",
    *ENERGY_COLS,
    *RATIO_COLS,
    "trust_flag",
)
RAW_COLS = tuple(f"raw_{c}" for c in (*ENERGY_COLS, *RATIO_COLS))

COMPUTE_BOUND_DENSE = frozenset(
    {
        "ffn_gemm",
        "qkv_proj_gemm",
        "out_proj_gemm",
        "attn_score_matmul",
        "attn_value_matmul",
        "downsample_conv2d",
        "patch_embed_conv2d",
        "xcit_cov_matmul",
        "sra_conv2d",
        "attn_block_fused",
    }
)
MEMORY_BOUND = frozenset(
    {
        "softmax",
        "gelu",
        "layer_norm",
        "group_norm",
        "batch_norm",
        "residual_add",
        "avg_pool_token_mixer",
        "depthwise_conv2d",
    }
)


def _parse_num(token: str) -> float | None:
    t = token.strip().rstrip("†‡").strip()
    if not t or t in ("—", "-", "EXCLUDED", "N/A"):
        return None
    return float(t)


def _parse_operator_label(label: str) -> tuple[str, int]:
    label = label.strip().rstrip("†‡").strip()
    m = re.match(r"^(.+?)\[s(\d+)\]$", label)
    if m:
        return m.group(1), int(m.group(2))
    return label, 0


def format_operator_label(operator: str, shape_index: int) -> str:
    return f"{operator}[s{shape_index}]" if shape_index else operator


def _shape_index_int(value) -> int:
    return int(value) if pd.notna(value) else 0


def _trust_flag_for(operator: str, op_raw: str = "") -> str:
    trust = ""
    if "†" in op_raw or operator == "avg_pool_token_mixer":
        trust = "LOW-TRUST cpu_int8"
    if "‡" in op_raw or operator == "sra_conv2d":
        trust = "UNTRUSTED npu INT8"
    if operator == "attn_block_fused":
        trust = "N/A npu"
    return trust


def _derive_ratios(
    cpu_fp32: float | None,
    cpu_int8: float | None,
    npu_int8: float | None,
) -> tuple[float | None, float | None, float | None]:
    precision = (
        cpu_fp32 / cpu_int8
        if cpu_fp32 is not None and cpu_int8 is not None and cpu_int8 != 0
        else None
    )
    architecture_ratio = (
        cpu_int8 / npu_int8
        if cpu_int8 is not None and npu_int8 is not None and npu_int8 != 0
        else None
    )
    gap = (
        cpu_fp32 / npu_int8
        if cpu_fp32 is not None and npu_int8 is not None and npu_int8 != 0
        else None
    )
    return precision, architecture_ratio, gap


def _validate_schema(df: pd.DataFrame) -> None:
    missing = (set(SCHEMA_COLS) | set(RAW_COLS)) - set(df.columns)
    if missing:
        raise ValueError(f"build_decomposition_df schema missing columns: {sorted(missing)}")
    if df.duplicated(subset=list(SCHEMA_KEYS)).any():
        raise ValueError("duplicate (operator, shape_index) keys in decomposition table")
    if not (df["shape_class"] == "avg").all():
        raise ValueError("shape_class must be 'avg' for all rows")


def build_decomposition_df(records: list[dict]) -> pd.DataFrame:
    if not records:
        raise ValueError("build_decomposition_df: zero records")
    df = pd.DataFrame(records)
    for col in SCHEMA_COLS:
        if col not in df.columns:
            df[col] = None
    for col in RAW_COLS:
        if col not in df.columns:
            df[col] = ""
    _validate_schema(df)
    return df


def parse_section3(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8")
    m = re.search(r"## 3\. Decomposition table\s*\n(.*?)(?:\n---|\n## 4\.)", text, re.DOTALL)
    if not m:
        raise SystemExit("HALT: §3 Decomposition table not found in ANALYSIS_REFERENCE.md")

    records: list[dict] = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line.startswith("|") or line.startswith("| Operator") or line.startswith("|-"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) != 8:
            continue
        op_raw, cf, ci, ni, ig, pr, ar, gap = parts
        base, shape_index = _parse_operator_label(op_raw)
        records.append(
            {
                "operator": base,
                "shape_index": shape_index,
                "shape_class": "avg",
                "operator_label": format_operator_label(base, shape_index),
                "cpu_FP32": _parse_num(cf),
                "cpu_INT8": _parse_num(ci),
                "npu_INT8": _parse_num(ni),
                "igpu_FP32": _parse_num(ig),
                "precision": _parse_num(pr),
                "architecture_ratio": _parse_num(ar) if ar not in ("EXCLUDED", "N/A") else None,
                "gap": _parse_num(gap),
                "trust_flag": _trust_flag_for(base, op_raw),
                "raw_cpu_FP32": cf,
                "raw_cpu_INT8": ci,
                "raw_npu_INT8": ni,
                "raw_igpu_FP32": ig,
                "raw_precision": pr,
                "raw_architecture_ratio": ar,
                "raw_gap": gap,
            }
        )

    if not records:
        raise SystemExit("HALT: §3 table unparseable (zero data rows)")
    return build_decomposition_df(records)


def load_primary() -> pd.DataFrame:
    if not PRIMARY_ANALYSIS.exists():
        raise SystemExit(f"HALT: primary analysis CSV not found: {PRIMARY_ANALYSIS}")
    if not CONTROL_ANALYSIS.exists():
        raise SystemExit(f"HALT: control analysis CSV not found: {CONTROL_ANALYSIS}")

    orig = pd.read_csv(PRIMARY_ANALYSIS)
    orig = orig[orig["table"] == "per_operator"]
    ctrl = pd.read_csv(CONTROL_ANALYSIS)
    ctrl = ctrl[ctrl["table"] == "per_operator"]

    def _series(df: pd.DataFrame, engine: str) -> pd.Series:
        sub = df[df["engine"] == engine].copy()
        sub["shape_index"] = sub["shape_index"].map(_shape_index_int)
        return sub.set_index(["operator", "shape_index"])["energy_per_op_J_mean"]

    cpu_fp32 = _series(orig, "cpu")
    igpu_fp32 = _series(orig, "igpu")
    npu_int8 = _series(orig, "npu")
    cpu_int8 = _series(ctrl, "cpu_int8")

    keys = sorted(
        set(cpu_fp32.index) | set(igpu_fp32.index) | set(npu_int8.index) | set(cpu_int8.index),
        key=lambda k: (k[0], k[1]),
    )

    records: list[dict] = []
    for operator, shape_index in keys:
        cf = cpu_fp32.get((operator, shape_index))
        ig = igpu_fp32.get((operator, shape_index))
        ni = npu_int8.get((operator, shape_index))
        ci = cpu_int8.get((operator, shape_index))
        cf_f = float(cf) if pd.notna(cf) else None
        ig_f = float(ig) if pd.notna(ig) else None
        ni_f = float(ni) if pd.notna(ni) else None
        ci_f = float(ci) if pd.notna(ci) else None
        precision, architecture_ratio, gap = _derive_ratios(cf_f, ci_f, ni_f)
        records.append(
            {
                "operator": operator,
                "shape_index": shape_index,
                "shape_class": "avg",
                "operator_label": format_operator_label(operator, shape_index),
                "cpu_FP32": cf_f,
                "igpu_FP32": ig_f,
                "npu_INT8": ni_f,
                "cpu_INT8": ci_f,
                "precision": precision,
                "architecture_ratio": architecture_ratio,
                "gap": gap,
                "trust_flag": _trust_flag_for(operator),
                "raw_cpu_FP32": "",
                "raw_cpu_INT8": "",
                "raw_npu_INT8": "",
                "raw_igpu_FP32": "",
                "raw_precision": "",
                "raw_architecture_ratio": "",
                "raw_gap": "",
            }
        )

    if not records:
        raise SystemExit("HALT: primary loader produced zero decomposition rows")
    return build_decomposition_df(records)


def load_decomposition(source: str) -> pd.DataFrame:
    if source == "section3":
        if not ANALYSIS_REF.exists():
            raise SystemExit("HALT: ANALYSIS_REFERENCE.md not found")
        return parse_section3(ANALYSIS_REF)
    if source == "primary":
        return load_primary()
    raise SystemExit(f"HALT: unknown source {source!r}")


def classify_mechanism(operator: str) -> tuple[str, str | None]:
    rule_dense = (
        operator.endswith("_gemm")
        or operator.endswith("_matmul")
        or ("conv2d" in operator and operator != "depthwise_conv2d")
        or operator == "attn_block_fused"
    )
    rule_mem = operator in MEMORY_BOUND

    in_dense = operator in COMPUTE_BOUND_DENSE
    in_mem = operator in MEMORY_BOUND

    if in_dense and in_mem:
        return "UNCLASSIFIED", f"{operator} in both maps"
    if in_dense:
        if rule_mem and not rule_dense:
            return "UNCLASSIFIED", f"map=dense rule=memory for {operator}"
        return "COMPUTE_BOUND_DENSE", None
    if in_mem:
        if rule_dense:
            return "UNCLASSIFIED", f"map=memory rule=dense for {operator}"
        return "MEMORY_BOUND", None
    if rule_dense:
        return "COMPUTE_BOUND_DENSE", None
    if rule_mem:
        return "MEMORY_BOUND", None
    return "UNCLASSIFIED", f"no confident rule/map for {operator}"


def _relative_margin(energies: dict[str, float]) -> float:
    vals = {k: v for k, v in energies.items() if pd.notna(v)}
    if len(vals) < 2:
        return float("nan")
    e_min, e_max = min(vals.values()), max(vals.values())
    return (e_max - e_min) / e_min if e_min > 0 else float("inf")


def _pick_winner(energies: dict[str, float], rel: float) -> tuple[str, bool]:
    vals = {k: v for k, v in energies.items() if pd.notna(v)}
    if not vals:
        return "no candidates", False
    if len(vals) == 1:
        return next(iter(vals)), False
    is_tie = rel <= TIE_THRESHOLD
    if is_tie:
        if TIE_POLICY == "lowest_energy":
            return min(vals, key=vals.get), True
        return f"tie ({'/'.join(sorted(vals.keys()))})", True
    return min(vals, key=vals.get), False


def _naive_winner(row: pd.Series) -> str:
    grid = {"cpu": row["cpu_FP32"], "igpu": row["igpu_FP32"], "npu": row["npu_INT8"]}
    vals = {k: v for k, v in grid.items() if pd.notna(v)}
    if not vals:
        return "n/a"
    e_min = min(vals.values())
    return sorted(k for k, v in vals.items() if v == e_min)[0]


def _sign_fields(row: pd.Series, op: str) -> dict[str, str | float | bool]:
    """gap/arch distances + sign-margin-guarded sign_divergence."""
    na = {
        "gap_dist": "",
        "arch_dist": "",
        "sign_marginal": "",
        "sign_divergence": "N/A",
    }
    if op in ("sra_conv2d", "attn_block_fused", "avg_pool_token_mixer"):
        return na
    cf, ci, ni = row["cpu_FP32"], row["cpu_INT8"], row["npu_INT8"]
    if any(pd.isna(x) for x in (cf, ci, ni)):
        return na

    gap = float(cf) / float(ni)
    arch = float(ci) / float(ni)
    gap_dist = abs(gap - 1.0)
    arch_dist = abs(arch - 1.0)

    headline = "CPU" if gap < 1 else "NPU"
    best_i8 = "CPU" if arch < 1 else "NPU"
    raw_diverge = headline != best_i8

    if raw_diverge:
        # Ratio closest to parity is the noise-sensitive side of the flip.
        deciding_dist = min(gap_dist, arch_dist)
        sign_marginal = deciding_dist < SIGN_MARGIN
        sign_divergence = not sign_marginal
    else:
        deciding_dist = min(gap_dist, arch_dist)
        sign_marginal = False
        sign_divergence = False

    return {
        "gap_dist": round(gap_dist, 4),
        "arch_dist": round(arch_dist, 4),
        "sign_marginal": sign_marginal,
        "sign_divergence": str(sign_divergence),
    }


def _diverges_deployment(winner: str, naive: str) -> str:
    if winner.startswith("EXCLUDED") or winner.startswith("no matched recommendation"):
        return "N/A"
    if winner.startswith("tie ("):
        inner = winner[5:-1]
        members = set(inner.split("/"))
        if naive in members:
            return "False"
        return "True"
    return str(winner != naive)


def process_row(row: pd.Series) -> dict:
    op = row["operator"]
    label = row["operator_label"]
    sc = row["shape_class"]

    bucket, conflict = classify_mechanism(op)
    if bucket == "UNCLASSIFIED":
        raise SystemExit(f"HALT: UNCLASSIFIED {label}: {conflict}")

    excluded: list[str] = []
    notes: list[str] = []
    trust_flag = row["trust_flag"] or ""

    if op == "sra_conv2d":
        trust_flag = "UNTRUSTED npu INT8"
    elif op == "attn_block_fused":
        trust_flag = "N/A npu"
    elif op == "avg_pool_token_mixer":
        trust_flag = "LOW-TRUST cpu_int8 (irrelevant to FP32 routing)"
        notes.append("LOW-TRUST cpu_int8 cell irrelevant to MEMORY_BOUND FP32 recommendation")

    if bucket == "COMPUTE_BOUND_DENSE":
        deployment_precision = "INT8 (matched QDQ)"
        ranking_metric = "matched_INT8_energy"
        candidates = {"cpu": row["cpu_INT8"], "npu": row["npu_INT8"]}
        excluded.append("igpu: not evaluable at INT8")
        arch = row["architecture_ratio"]

        if op == "sra_conv2d":
            candidates = {k: v for k, v in candidates.items() if k != "npu"}
            excluded.append("npu: UNTRUSTED INT8 (Gate 2)")
            winner = "EXCLUDED (npu INT8 untrusted)"
            is_tie = False
            margin_pct = float("nan")
        elif op == "attn_block_fused":
            excluded.append("npu: N/A (VitisAI fused crash)")
            candidates = {k: v for k, v in candidates.items() if k != "npu"}
            winner = "no matched recommendation (npu N/A)"
            is_tie = False
            margin_pct = float("nan")
        else:
            rel = _relative_margin(candidates)
            winner, is_tie = _pick_winner(candidates, rel)
            margin_pct = rel * 100
    else:
        deployment_precision = "FP32"
        ranking_metric = "FP32_energy"
        candidates = {"cpu": row["cpu_FP32"], "igpu": row["igpu_FP32"]}
        excluded.append("npu: not evaluable at FP32")
        arch = None
        rel = _relative_margin(candidates)
        winner, is_tie = _pick_winner(candidates, rel)
        margin_pct = rel * 100

    naive = _naive_winner(row)
    sign = _sign_fields(row, op)
    div_dep = _diverges_deployment(winner, naive)

    return {
        "operator": label,
        "shape_class": sc,
        "mechanism": bucket,
        "deployment_precision": deployment_precision,
        "ranking_metric": ranking_metric,
        "candidate_engines": "; ".join(f"{k}={v:.6g}" for k, v in candidates.items() if pd.notna(v)),
        "excluded_engines_reason": "; ".join(excluded),
        "winner": winner,
        "margin_pct": round(margin_pct, 4) if pd.notna(margin_pct) else "",
        "is_tie": is_tie,
        "architecture_ratio": round(arch, 4) if arch is not None and pd.notna(arch) else "",
        "precision": row["precision"] if pd.notna(row["precision"]) else "",
        "gap": row["gap"] if pd.notna(row["gap"]) else "",
        "gap_dist": sign["gap_dist"],
        "arch_dist": sign["arch_dist"],
        "sign_marginal": sign["sign_marginal"],
        "naive_winner": naive,
        "sign_divergence": sign["sign_divergence"],
        "diverges_deployment": div_dep,
        "trust_flag": trust_flag,
        "notes": "; ".join(notes),
        "mechanism_conflict": conflict or "",
    }


def write_markdown(
    out_df: pd.DataFrame,
    *,
    out_md: Path,
    energy_source: str,
    provenance_banner: str,
) -> None:
    lines = [
        "# Step 4 — Operator Engine Mapping",
        "",
        f"> **{provenance_banner}**",
        "",
        "Operator-level deployment recommendations (avg corner). **No model-level claims.**",
        "",
        f"**Branch:** uProfAnalysis | **Energy source:** {energy_source}",
        f"**Config:** TIE_THRESHOLD={TIE_THRESHOLD}, TIE_POLICY={TIE_POLICY!r}, "
        f"PRECISION_GUARD={PRECISION_GUARD} (advisory only), SIGN_MARGIN={SIGN_MARGIN}",
        "",
        "## Methods / scope limitation",
        "",
        "Complementary engine coverage asymmetry: **COMPUTE_BOUND_DENSE** ranks matched INT8 "
        "(cpu_INT8 vs npu_INT8; iGPU excluded). **MEMORY_BOUND** ranks FP32 (cpu_FP32 vs "
        "igpu_FP32; NPU excluded). CPU is the only engine in both grids. No operator receives "
        "a clean 3-way matched-precision ranking.",
        "",
        "## Mapping table",
        "",
    ]
    display = [
        "operator",
        "shape_class",
        "mechanism",
        "winner",
        "margin_pct",
        "is_tie",
        "architecture_ratio",
        "sign_divergence",
        "diverges_deployment",
        "naive_winner",
        "trust_flag",
    ]
    lines.append("| " + " | ".join(display) + " |")
    lines.append("|" + "|".join(["---"] * len(display)) + "|")
    for _, r in out_df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in display) + " |")

    # sign_divergence audit
    sd_true = out_df[out_df["sign_divergence"] == "True"]["operator"].tolist()

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "### (a) GEMM / dense conv sweep → NPU (architectural)",
            "",
            "Trusted **COMPUTE_BOUND_DENSE** rows with valid npu_INT8 recommend **npu** on "
            "matched-INT8 energy. `precision` < 1 on every dense GEMM/conv (INT8 penalizes CPU); "
            "`architecture_ratio` 3–14× on winners confirms the advantage is **architectural "
            "(XDNA2)**, not a quantization artifact. `sign_divergence=False` on those GEMMs "
            "(headline and best-INT8 agree: NPU).",
            "",
            "### (b) sign_divergence pair: softmax + depthwise_conv2d",
            "",
            "These reproduce the Step-3 decomposition trap — **headline winner ≠ best-INT8 engine** "
            "(`sign_divergence=True`). This is an INT8-vs-FP32 attribution question, **separate** "
            "from the deployment recommendation:",
            "",
            "- **softmax[s1]:** gap=0.216 → headline **CPU** (cpu_FP32 < npu_INT8); arch=5.75 → "
            "best-INT8 **NPU**. Deployment (MEMORY_BOUND, FP32 grid) → **cpu** — same as naive.",
            "- **depthwise_conv2d:** gap=0.515 → headline **CPU**; arch=3.15 → best-INT8 **NPU**. "
            "Deployment → **tie (cpu/igpu)** at 1.8%; `diverges_deployment=False` (naive cpu ∈ tie set).",
            "",
            f"**sign_divergence=True ops (SIGN_MARGIN={SIGN_MARGIN}):** "
            f"{', '.join(sd_true) or '(none)'} — consolidation pair only.",
            "",
            "**batch_norm footnote:** near-parity headline (gap=1.03) within sign margin; "
            "robust best-INT8 engine = CPU (arch=0.81). NOT contested-middle (it has a clear "
            "INT8 winner); excluded from the divergence pair on headline-noise grounds. "
            "`avg_pool_token_mixer` sign fields N/A (LOW-TRUST cpu_int8).",
            "",
            "### (c) Contested middle → 10% rule ties",
            "",
            "Attention matmuls where no engine dominates at matched INT8 (`attn_score_matmul`, "
            "`attn_value_matmul`, `xcit_cov_matmul`) route per mechanism. **10% deployment ties** "
            "at avg corner: **depthwise_conv2d** and **group_norm** (FP32 grid). "
            "TIE_POLICY=`report` for audit. (`batch_norm` is not in this band — see footnote above.)",
            "",
            "For MEMORY_BOUND ops, arch/precision/gap values in the CSV are labelled decomposition "
            "cross-reference (§3) only — not ranking inputs.",
            "",
            "### Trust exclusions",
            "",
            "- **sra_conv2d:** EXCLUDED — npu INT8 UNTRUSTED",
            "- **attn_block_fused:** no matched recommendation — npu N/A",
            "- **avg_pool_token_mixer:** LOW-TRUST cpu_int8 irrelevant to FP32 routing",
            "",
        ]
    )
    out_md.write_text("\n".join(lines), encoding="utf-8")


SHEET_COLS = [
    "operator",
    "shape_class",
    "cpu_FP32",
    "cpu_INT8",
    "npu_INT8",
    "igpu_FP32",
    "precision",
    "architecture_ratio",
    "gap",
    "mechanism",
    "winner",
    "margin_pct",
    "is_tie",
    "sign_divergence",
    "diverges_deployment",
    "naive_winner",
    "trust_flag",
]


def write_sheet(
    decomp: pd.DataFrame,
    out_df: pd.DataFrame,
    *,
    out_path: Path,
    provenance: str,
) -> None:
    """Chris sheet-import: one flat row per operator (energies + routing), not raw runs log."""
    decomp_idx = decomp.set_index(["operator", "shape_index"])
    rows: list[dict] = []
    for _, r in out_df.iterrows():
        op_label = r["operator"]
        base, shape_index = _parse_operator_label(op_label)
        key = (base, shape_index)
        if key not in decomp_idx.index:
            raise ValueError(f"sheet export: missing decomposition row for {op_label}")
        d = decomp_idx.loc[key]
        rows.append(
            {
                "operator": op_label,
                "shape_class": r["shape_class"],
                "cpu_FP32": d["cpu_FP32"],
                "cpu_INT8": d["cpu_INT8"],
                "npu_INT8": d["npu_INT8"],
                "igpu_FP32": d["igpu_FP32"],
                "precision": d["precision"],
                "architecture_ratio": d["architecture_ratio"],
                "gap": d["gap"],
                "mechanism": r["mechanism"],
                "winner": r["winner"],
                "margin_pct": r["margin_pct"],
                "is_tie": r["is_tie"],
                "sign_divergence": r["sign_divergence"],
                "diverges_deployment": r["diverges_deployment"],
                "naive_winner": r["naive_winner"],
                "trust_flag": r["trust_flag"],
            }
        )
    sheet_df = pd.DataFrame(rows)[SHEET_COLS]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        f.write(f"# {provenance}\n")
    sheet_df.to_csv(out_path, mode="a", index=False, float_format="%.17g")


def write_handoff() -> None:
    text = """# Step 4 handoff — paste into next Claude prompt, then delete

**Branch:** `uProfAnalysis` | **Scope:** operator-level only, avg corner, no model claims
**Status:** Step 4 mapping **finalized** (sign-margin guard). Next: paper scaffold / Chris sign-off / tower CSV regen.

---

## Read first (30 s)

| What | Where |
|------|-------|
| Durable project truth | `PROJECT_CONTEXT.md` §6–§7 |
| Energy numbers (committed snapshot) | `ANALYSIS_REFERENCE.md` §3 — **21 ops**, real J/op + ratios |
| Step 4 full table + prose | `STEP4_OPERATOR_ENGINE_MAPPING.md` |
| Step 4 machine output | `results/step4_operator_engine_mapping.csv` |
| Repro script | `step4_operator_engine_mapping.py` |

**PROVENANCE:** This laptop has **no production CSVs** (`results/` gitignored). Step 4 energies parsed **only** from `ANALYSIS_REFERENCE.md` §3 — **not** `results/analysis_out.csv` (demo). Regenerate from tower before paper.

---

## Sessions (tower)

| Session | Engines | Idle |
|---------|---------|------|
| `20260612_143854` | cpu/igpu FP32, npu XINT8 | 216.22 J |
| `20260615_145614_cpu_int8` | cpu INT8 control | 326.49 J |

Decomposition identity `precision × arch = gap` → 1.000 across 20 triplet ops (consistency check only).

---

## Step 4 router (mechanism → metric)

| Bucket | Ops | Rank on | Candidates |
|--------|-----|---------|------------|
| **COMPUTE_BOUND_DENSE** | GEMMs, dense convs, `attn_block_fused`, `sra_conv2d` | matched INT8 | cpu_INT8 vs npu_INT8 (iGPU out) |
| **MEMORY_BOUND** | softmax, gelu, norms, pool, residual, **depthwise_conv2d** | FP32 | cpu_FP32 vs igpu_FP32 (NPU out) |

`PRECISION_GUARD=0.25` advisory only — never routes.
`TIE_THRESHOLD=0.10`, `TIE_POLICY=report`.
`SIGN_MARGIN=0.10` — raw headline≠best-INT8 flips within margin are guarded out.

---

## Winners at a glance (avg corner)

**NPU (matched INT8):** all trusted GEMMs + `downsample_conv2d`, `patch_embed_conv2d` (arch 3–14×, precision < 1 on dense).

**CPU (matched INT8, contested):** `attn_score_matmul`, `attn_value_matmul`, `xcit_cov_matmul`.

**CPU (FP32):** `softmax`, `gelu`, `layer_norm`, `batch_norm`, `residual_add`.

**iGPU (FP32):** `avg_pool_token_mixer`.

**Ties (FP32, ≤10%):** `depthwise_conv2d` tie(cpu/igpu), `group_norm` tie(cpu/igpu).

**Excluded / N/A:** `sra_conv2d×npu` UNTRUSTED; `attn_block_fused×npu` N/A.

---

## Two divergence columns — do not conflate

| Column | Meaning |
|--------|---------|
| **`sign_divergence`** | Step-3 trap (SIGN_MARGIN guarded): headline winner ≠ best-INT8 winner. **True:** `softmax[s1]`, `depthwise_conv2d` **ONLY**. N/A: sra, attn_block_fused, avg_pool. |
| **`diverges_deployment`** | Step-4 trap: deploy winner ≠ naive production-grid argmin. **True:** `batch_norm`, `group_norm`. **False:** softmax, depthwise (tie or match). N/A: EXCLUDED / no matched rec. |

**Marginal flip (guarded out):** `batch_norm` — raw headline≠best-INT8 but gap=1.03 within SIGN_MARGIN; robust best-INT8=CPU (arch=0.81). Not in divergence pair.

**Paper headline pair:** softmax + depthwise — INT8-vs-FP32 attribution trap, separate from FP32 deployment pick.

---

## Trust carry-forwards (always apply)

- `sra_conv2d × npu` → UNTRUSTED
- `attn_block_fused × npu` → N/A
- `avg_pool_token_mixer × cpu_int8` → LOW-TRUST (irrelevant to FP32 routing)

---

## Deferred / not done

- [ ] Step 4 regen from **tower** `results/` CSVs (paper-grade)
- [ ] Chris sign-off on methodology + tie/sign-margin policy
- [ ] Paper scaffold (intro / methods / results)
- [ ] small/large SDPA corners; Track-2 shape fixes; model-level work
- [ ] `uProfBenchmarking`: idle re-run + sra remediation

---

## Do not touch unless asked

`PROJECT_CONTEXT.md`, `ANALYSIS_REFERENCE.md`, harness/pipeline/session files, tower `results/` artifacts.
"""
    OUT_HANDOFF.write_text(text, encoding="utf-8")


def run_routing(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([process_row(row) for _, row in df.iterrows()])


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 4 operator engine mapping")
    parser.add_argument(
        "--source",
        choices=("section3", "primary"),
        default="section3",
        help="Energy source: §3 markdown snapshot (default) or primary tower CSVs",
    )
    args = parser.parse_args()
    source = args.source

    if source == "section3":
        out_csv = OUT_CSV
        out_md = OUT_MD
        out_sheet = OUT_SHEET
        provenance = PROVENANCE_BANNER_SECTION3
        energy_source = "ANALYSIS_REFERENCE.md §3"
        write_handoff_flag = True
        committed_csv_mirror = None
    else:
        out_csv = OUT_CSV_PRIMARY
        out_md = OUT_MD_PRIMARY
        out_sheet = OUT_SHEET_PRIMARY
        provenance = PROVENANCE_BANNER_PRIMARY
        energy_source = "primary tower CSVs (full precision)"
        write_handoff_flag = False
        committed_csv_mirror = OUT_CSV_PRIMARY_COMMITTED

    print("=" * 72)
    print("STEP 0 — ENERGY SOURCE")
    print("=" * 72)

    df = load_decomposition(source)
    cols = [
        "operator",
        "shape_class",
        "cpu_FP32",
        "cpu_INT8",
        "npu_INT8",
        "igpu_FP32",
        "precision",
        "architecture_ratio",
        "gap",
    ]
    print(f'  source = "{energy_source}"')
    print(f"  op count = {len(df)}")
    print(f"  columns = {cols}")
    if source == "section3":
        print("  FORBIDDEN sources not read (results/analysis_out.csv, runs_synthetic.csv, fixtures)")
    print()

    print("=" * 72)
    print("STEP 1 — MECHANISTIC ROUTING")
    print("=" * 72)
    dense_guard_hits = 0
    dense_total = 0
    for _, row in df.iterrows():
        op, label = row["operator"], row["operator_label"]
        bucket, conflict = classify_mechanism(op)
        prec_dev = abs(row["precision"] - 1) if pd.notna(row["precision"]) else float("nan")
        exceeds = pd.notna(prec_dev) and prec_dev > PRECISION_GUARD
        flag = "EXCEEDS" if exceeds else "ok"
        print(
            f"  {label:28} -> {bucket:22} |precision-1|={prec_dev:.3f} {flag}"
            + (f"  CONFLICT: {conflict}" if conflict else "")
        )
        if bucket == "UNCLASSIFIED":
            print(f"HALT: UNCLASSIFIED {label}")
            return 1
        if bucket == "COMPUTE_BOUND_DENSE":
            dense_total += 1
            if exceeds:
                dense_guard_hits += 1
    print(f"\n  COMPUTE_BOUND_DENSE tripping PRECISION_GUARD: {dense_guard_hits}/{dense_total}")
    if dense_guard_hits < dense_total:
        print("  WARNING: not all COMPUTE_BOUND_DENSE ops exceed guard (see attn_value_matmul, xcit_cov_matmul)")

    print("\n" + "=" * 72)
    print("STEPS 2–5 — METRIC, TIES, TRUST, DIVERGENCE")
    print("=" * 72)

    records = [process_row(row) for _, row in df.iterrows()]
    out_df = pd.DataFrame(records)

    out_cols = [
        "operator",
        "shape_class",
        "mechanism",
        "deployment_precision",
        "ranking_metric",
        "candidate_engines",
        "excluded_engines_reason",
        "winner",
        "margin_pct",
        "is_tie",
        "architecture_ratio",
        "precision",
        "gap",
        "gap_dist",
        "arch_dist",
        "sign_marginal",
        "naive_winner",
        "sign_divergence",
        "diverges_deployment",
        "trust_flag",
    ]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        f.write(f"# {provenance}\n")
    out_df[out_cols].to_csv(out_csv, mode="a", index=False)

    write_markdown(
        out_df,
        out_md=out_md,
        energy_source=energy_source,
        provenance_banner=provenance,
    )
    write_sheet(df, out_df, out_path=out_sheet, provenance=provenance)
    if committed_csv_mirror is not None:
        committed_csv_mirror.write_text(out_csv.read_text(encoding="utf-8"), encoding="utf-8")
    if write_handoff_flag:
        write_handoff()

    # sign_divergence expectation audit
    expected_sd = {"softmax[s1]", "depthwise_conv2d"}
    actual_sd = set(out_df.loc[out_df["sign_divergence"] == "True", "operator"])
    print(f"\n  sign_divergence=True: {sorted(actual_sd)}")
    if actual_sd != expected_sd:
        print(f"  NOTE: expected {sorted(expected_sd)}; diff = {sorted(actual_sd ^ expected_sd)}")
        for op in sorted(actual_sd ^ expected_sd):
            r = out_df[out_df["operator"] == op].iloc[0]
            print(f"    {op}: gap={r['gap']} arch={r['architecture_ratio']} sign_divergence={r['sign_divergence']}")

    print(f"\nWrote: {out_csv}")
    print(f"Wrote: {out_md}")
    print(f"Wrote: {out_sheet}")
    if committed_csv_mirror is not None:
        print(f"Wrote: {committed_csv_mirror}")
    if write_handoff_flag:
        print(f"Wrote: {OUT_HANDOFF}")

    # Diff summary
    print("\n=== FILE DIFF SUMMARY ===")
    print("  step4_operator_engine_mapping.py: +SIGN_MARGIN guard; _sign_fields() adds gap_dist/arch_dist/sign_marginal; batch_norm guarded out of sign_divergence.")
    print("  step4_operator_engine_mapping.csv: +gap_dist, arch_dist, sign_marginal cols; batch_norm sign_divergence True->False.")
    print("  STEP4_OPERATOR_ENGINE_MAPPING.md: pair={softmax,depthwise} only; batch_norm footnote; contested-middle prose fixed.")
    print("  STEP4_HANDOFF.md: sign_divergence set aligned; batch_norm to marginal flip line; diverges_deployment table updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
