#!/usr/bin/env python3
"""
F2 — precision/architecture decomposition chart (operator-level, avg corner).

Energy source: benchmark/ANALYSIS_REFERENCE.md §3 ONLY (committed snapshot).
No reads from results/ (demo fixtures) and no reliance on derived columns.

Outputs:
  figures/F2_decomposition.png
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_REF = ROOT / "benchmark" / "ANALYSIS_REFERENCE.md"
OUT_PNG = ROOT / "figures" / "F2_decomposition.png"

# Strict SIGN/TRUST matching: include only compute-bound-dense ops with a trusted
# matched-INT8 triplet (cpu_FP32, cpu_INT8, npu_INT8 all present).
INCLUDE_OPS = {
    "ffn_gemm",
    "qkv_proj_gemm",
    "out_proj_gemm",
    "attn_score_matmul",
    "attn_value_matmul",
    "xcit_cov_matmul",
    "downsample_conv2d",
    "patch_embed_conv2d",
}
EXCLUDE_OPS = {"sra_conv2d", "attn_block_fused"}

# If any identity check deviates by more than this, print a flag.
IDENTITY_MAX_REL_DEV = 0.005  # 0.5%

FOOTNOTE_SCOPE = (
    "SCOPE: Single-corner (avg, N=197, D=768). Coverage asymmetry: no operator receives a clean "
    "3-way matched ranking. Contested-middle ties (TIE_THRESHOLD=0.10) are sensitive to tensor "
    "dimensions; parameter sweeps could move tied winners (future work, per supervisor sign-off)."
)
FOOTNOTE_SOURCE = (
    "SOURCE: ANALYSIS_REFERENCE.md §3 (validated against primary tower CSVs); "
    "energy-layer diff 83/83 numeric PASS + 1 expected-N/A, 2026-06-17."
)
FOOTNOTE_WRAP_WIDTH = 115
XLABEL_PAD = 10


def _parse_num(token: str) -> float | None:
    t = token.strip().rstrip("†‡").strip()
    if not t or t in ("—", "-", "EXCLUDED", "N/A"):
        return None
    return float(t)


def _parse_operator_label(op_raw: str) -> tuple[str, str, int]:
    """
    Returns (operator_base, operator_label_string, shape_index).
    Example: 'ffn_gemm[s2]' -> ('ffn_gemm', 'ffn_gemm[s2]', 2)
    """
    op_raw = op_raw.strip()
    # Remove footnote markers for the parsing regex only.
    op_clean = op_raw.replace("†", "").replace("‡", "").strip()
    m = re.match(r"^(.+?)\[s(\d+)\]$", op_clean)
    if m:
        base = m.group(1).strip()
        si = int(m.group(2))
        return base, f"{base}[s{si}]", si
    # Single-profile ops: no suffix.
    # Examples: 'downsample_conv2d', 'group_norm'
    base = op_clean.strip()
    return base, base, 0


def parse_section3(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    start = text.find("## 3. Decomposition table")
    if start == -1:
        raise RuntimeError("HALT: §3 table not found in ANALYSIS_REFERENCE.md (start marker missing)")

    # End at the next section header (## 4. ...) to avoid accidentally capturing regime buckets.
    end = text.find("## 4. Regime buckets", start)
    if end == -1:
        # Fallback: stop at the horizontal rule between sections.
        end = text.find("\n---", start)
    if end == -1:
        raise RuntimeError("HALT: §3 table not found in ANALYSIS_REFERENCE.md (end marker missing)")

    section = text[start:end]

    out: list[dict] = []
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|") or line.startswith("| Operator") or line.startswith("|-"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) != 8:
            continue
        op_raw, cpu_fp32_s, cpu_i8_s, npu_i8_s, igpu_fp32_s, precision_s, arch_s, gap_s = parts
        base, label, shape_index = _parse_operator_label(op_raw)

        out.append(
            {
                "operator": base,
                "operator_label": label,
                "shape_index": shape_index,
                "cpu_FP32": _parse_num(cpu_fp32_s),
                "cpu_INT8": _parse_num(cpu_i8_s),
                "npu_INT8": _parse_num(npu_i8_s),
                "igpu_FP32": _parse_num(igpu_fp32_s),
                "precision_ratio": _parse_num(precision_s),
                "architecture_ratio": _parse_num(arch_s),
                "original_gap": _parse_num(gap_s),
            }
        )

    if not out:
        raise RuntimeError("HALT: §3 table unparseable (zero data rows).")
    return out


def main() -> None:
    if not ANALYSIS_REF.exists():
        raise SystemExit(f"HALT: missing {ANALYSIS_REF}")

    rows = parse_section3(ANALYSIS_REF)

    # Select only compute-bound-dense ops with a trusted matched-INT8 triplet.
    selected: list[dict] = []
    for r in rows:
        op = r["operator"]
        if op in EXCLUDE_OPS:
            continue
        if op not in INCLUDE_OPS:
            continue
        if r["cpu_FP32"] is None or r["cpu_INT8"] is None or r["npu_INT8"] is None:
            # Matched INT8 triplet is required.
            continue
        selected.append(r)

    # Sort by architecture_ratio DESCENDING.
    # Recompute ratios from raw cells (do not trust snapshot derived columns).
    for r in selected:
        cpu_fp32 = float(r["cpu_FP32"])
        cpu_i8 = float(r["cpu_INT8"])
        npu_i8 = float(r["npu_INT8"])
        precision_ratio = cpu_fp32 / cpu_i8
        architecture_ratio = cpu_i8 / npu_i8
        original_gap = cpu_fp32 / npu_i8
        identity_check = (precision_ratio * architecture_ratio) / original_gap

        r["precision_ratio_recalc"] = precision_ratio
        r["architecture_ratio_recalc"] = architecture_ratio
        r["original_gap_recalc"] = original_gap
        r["identity_check"] = identity_check

    selected.sort(key=lambda rr: rr["architecture_ratio_recalc"], reverse=True)

    print("F2 decomposition chart")
    print('source = "ANALYSIS_REFERENCE.md §3"')
    print(f"op count (filtered matched-INT8 triplets) = {len(selected)}")
    print("columns used = operator, cpu_FP32, cpu_INT8, npu_INT8")
    print()

    print("Per-op ratios + identity check (recomputed from raw cells):")
    print(
        "operator_label | precision_ratio | architecture_ratio | original_gap | identity_check | FLAG"
    )
    print(
        "----------------|------------------|-----------------------|--------------|------------------|------"
    )

    any_flag = False
    for r in selected:
        precision_ratio = r["precision_ratio_recalc"]
        architecture_ratio = r["architecture_ratio_recalc"]
        original_gap = r["original_gap_recalc"]
        identity_check = r["identity_check"]

        rel_dev = abs(identity_check - 1.0) / 1.0
        flag = rel_dev > IDENTITY_MAX_REL_DEV
        if flag:
            any_flag = True
        print(
            f"{r['operator_label']:<18} | {precision_ratio:>16.6g} | {architecture_ratio:>21.6g} | {original_gap:>12.6g} | {identity_check:>16.6g} | {flag}"
        )

    excluded_ops_note = []
    # Explicitly list why those ops were excluded for visibility.
    for op in sorted(EXCLUDE_OPS):
        excluded_ops_note.append(f"{op} excluded (npu match not trusted / NPU cell absent)")

    print()
    if excluded_ops_note:
        print("Excluded op note:")
        for x in excluded_ops_note:
            print(f"- {x}")

    # --- Plot ---
    labels = [r["operator_label"] for r in selected]
    precision_vals = [r["precision_ratio_recalc"] for r in selected]
    arch_vals = [r["architecture_ratio_recalc"] for r in selected]

    y = np.arange(len(selected))
    bar_h = 0.36
    offset = bar_h / 2

    fig, ax = plt.subplots(figsize=(12, 8))

    color_prec = "#4C78A8"
    color_arch = "#F58518"

    ax.barh(y + offset, precision_vals, height=bar_h, color=color_prec, alpha=0.95, label="precision (cpu FP32 / cpu INT8)")
    ax.barh(y - offset, arch_vals, height=bar_h, color=color_arch, alpha=0.95, label="architecture (cpu INT8 / npu INT8)")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # top-to-bottom in descending order

    ax.set_xscale("log")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0)

    ax.set_xlabel("ratio (log scale)", labelpad=XLABEL_PAD)

    # Zone labels: data-x left/right of parity line; axes-fraction y (mid stack, not row-anchored).
    blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(
        0.72,
        0.5,
        "INT8 hurts CPU / CPU cheaper",
        transform=blend,
        ha="right",
        va="center",
        fontsize=9,
    )
    ax.text(
        1.45,
        0.5,
        "NPU cheaper",
        transform=blend,
        ha="left",
        va="center",
        fontsize=9,
    )

    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, axis="x", which="both", alpha=0.25)

    # Tight x-lims around data.
    all_vals = precision_vals + arch_vals
    min_v = min([v for v in all_vals if v > 0])
    max_v = max(all_vals)
    ax.set_xlim(max(min_v / 2, 0.3), max_v * 1.2)

    foot = textwrap.fill(f"{FOOTNOTE_SCOPE}\n\n{FOOTNOTE_SOURCE}", width=FOOTNOTE_WRAP_WIDTH)
    fig.tight_layout()
    fig.text(
        0.01, -0.06, foot,
        transform=fig.transFigure,
        va="top", ha="left",
        fontsize=9, linespacing=1.3, color="#333333",
    )

    out_png_str = str(OUT_PNG)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print()
    print("Suggested caption (paste into paper):")
    print(
        "F2. Precision/architecture decomposition: per-operator grouped bars show precision = cpu_FP32/cpu_INT8 and architecture = cpu_INT8/npu_INT8 (x-axis log; vertical line at 1.0). "
        "Across compute-bound dense ops, precision tends to fall below 1 while architecture rises well above 1 (~3-14x), indicating that the NPU advantage is architectural rather than a quantization artifact."
    )
    print()
    print(f"PNG path: {out_png_str}")

    if any_flag:
        print("NOTE: one or more ops deviated from identity_check by >0.5% (should not happen algebraically; audit parsing).")


if __name__ == "__main__":
    main()

