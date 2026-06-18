#!/usr/bin/env python3
"""
Chris figure suite — off-tower plots from committed STEP4_FOR_SHEET_primary.csv.

Figures A–D: sheet CSV only (no results/).
Figure E: benchmark/figure_e_operator_engine.csv (local tower export).

Outputs under figures/ (matplotlib only).
"""

from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

BENCHMARK_DIR = Path(__file__).resolve().parent
ROOT = BENCHMARK_DIR.parent
FIGURES_DIR = ROOT / "figures"
DEFAULT_SHEET = BENCHMARK_DIR / "STEP4_FOR_SHEET_primary.csv"
DEFAULT_FIGURE_E = BENCHMARK_DIR / "figure_e_operator_engine.csv"

# Colorblind-safe engine palette (shared across all figures)
ENGINE_COLORS = {
    "cpu": "#0072B2",       # blue
    "cpu_light": "#56B4E9", # lighter blue (cpu_FP32)
    "cpu_dark": "#004C73",  # darker blue (cpu_INT8)
    "igpu": "#E69F00",      # orange
    "npu": "#009E73",       # green
}
TIE_COLOR_LEFT = ENGINE_COLORS["cpu"]
TIE_COLOR_RIGHT = ENGINE_COLORS["igpu"]
EXCLUDED_COLOR = "#999999"
NA_COLOR = "#CCCCCC"

FOOTNOTE_SCOPE = (
    "SCOPE: Single-corner (avg, N=197, D=768). Coverage asymmetry: no operator receives a clean "
    "3-way matched ranking. Contested-middle ties (TIE_THRESHOLD=0.10) are sensitive to tensor "
    "dimensions; parameter sweeps could move tied winners (future work, per supervisor sign-off)."
)
FOOTNOTE_SOURCE = (
    "SOURCE: STEP4_FOR_SHEET_primary.csv (full-precision primary tower CSVs); "
    "energy-layer diff 83/83 numeric PASS + 1 expected-N/A, 2026-06-17."
)
FOOTNOTE_SOURCE_E = (
    "SOURCE: figure_e_operator_engine.csv (tower-exported P/t/E decomposition); "
    "energy-layer diff 83/83 numeric PASS + 1 expected-N/A, 2026-06-17."
)

FOOTNOTE_WRAP_WIDTH = 115
CAPTION_WRAP_WIDTH = 100
XLABEL_PAD = 10


def _title_with_caption(main: str, caption: str | None = None, width: int = CAPTION_WRAP_WIDTH) -> str:
    if not caption:
        return main
    return f"{main}\n\n{textwrap.fill(caption.strip(), width=width)}"


def _combined_footnote(source_line: str) -> str:
    return textwrap.fill(f"{FOOTNOTE_SCOPE}\n\n{source_line}", width=FOOTNOTE_WRAP_WIDTH)


def add_bottom_footnotes(fig: plt.Figure, source_line: str = FOOTNOTE_SOURCE) -> None:
    """SCOPE + SOURCE below figure coords; bbox_inches='tight' expands canvas to fit."""
    fig.text(
        0.01, -0.06, _combined_footnote(source_line),
        transform=fig.transFigure,
        va="top", ha="left",
        fontsize=9, linespacing=1.3, color="#333333",
    )


def load_sheet(csv_path: Path | str) -> pd.DataFrame:
    """Load STEP4 sheet; skip # provenance banner; handle \\r\\n and empty cells."""
    df = pd.read_csv(csv_path, comment="#", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})
    for col in ("cpu_FP32", "cpu_INT8", "npu_INT8", "igpu_FP32", "precision", "architecture_ratio", "gap", "margin_pct"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_figure_e(csv_path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, comment="#", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    numeric_cols = (
        "iterations_completed", "window_duration_s", "avg_power_W", "execution_time_ms",
        "energy_gross_J", "idle_energy_per_op_J", "energy_net_J", "energy_net_CV_pct",
        "energy_per_token_J", "mean_latency_ms",
    )
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _parse_bool(val) -> bool:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return False
    s = str(val).strip().lower()
    if s in ("", "n/a", "nan", "none"):
        return False
    return s == "true"


def _has_trust_flag(val) -> bool:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return False
    s = str(val).strip()
    return bool(s) and s.lower() not in ("nan", "none")


def _sort_by_winner_name(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(by=["winner", "operator"], key=lambda s: s.astype(str).str.lower()).reset_index(drop=True)


def _winner_category(winner: str) -> str:
    w = str(winner).strip()
    wl = w.lower()
    if "tie" in wl:
        return "tie"
    if "excluded" in wl:
        return "excluded"
    if "no matched" in wl or "n/a" in wl:
        return "na"
    if w == "cpu":
        return "cpu"
    if w == "npu":
        return "npu"
    if w == "igpu":
        return "igpu"
    return "other"


def _draw_winner_patch(ax, x, y, width, height, winner: str) -> None:
    cat = _winner_category(winner)
    if cat == "tie":
        rect_l = mpatches.FancyBboxPatch(
            (x, y), width / 2, height, boxstyle="round,pad=0.01",
            facecolor=TIE_COLOR_LEFT, edgecolor="black", linewidth=0.6,
        )
        rect_r = mpatches.FancyBboxPatch(
            (x + width / 2, y), width / 2, height, boxstyle="round,pad=0.01",
            facecolor=TIE_COLOR_RIGHT, edgecolor="black", linewidth=0.6,
        )
        ax.add_patch(rect_l)
        ax.add_patch(rect_r)
        ax.hatch = None
    elif cat == "excluded":
        rect = mpatches.FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.01",
            facecolor=EXCLUDED_COLOR, edgecolor="black", linewidth=0.6, hatch="///",
        )
        ax.add_patch(rect)
    elif cat == "na":
        rect = mpatches.FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.01",
            facecolor=NA_COLOR, edgecolor="#666666", linewidth=0.6, linestyle="--",
        )
        ax.add_patch(rect)
    else:
        color = ENGINE_COLORS.get(cat, "#888888")
        rect = mpatches.FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.01",
            facecolor=color, edgecolor="black", linewidth=0.6,
        )
        ax.add_patch(rect)


def plot_routing_map(df: pd.DataFrame, out_path: Path | None = None) -> Path:
    """Figure A — operator -> engine routing map."""
    out_path = out_path or FIGURES_DIR / "routing_map.png"
    dense = _sort_by_winner_name(df[df["mechanism"] == "COMPUTE_BOUND_DENSE"])
    memory = _sort_by_winner_name(df[df["mechanism"] == "MEMORY_BOUND"])
    blocks = [
        ("COMPUTE_BOUND_DENSE", dense),
        ("MEMORY_BOUND", memory),
    ]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    y = len(dense) + len(memory) + 4  # start from top
    y_dir = -1  # grow downward
    row_h = 0.38
    gap = 0.12
    block_gap = 0.55
    labels: list[tuple[float, str]] = []

    for block_name, block_df in blocks:
        y += y_dir * block_gap
        ax.text(0.05, y + row_h * 0.15, block_name, fontsize=10, fontweight="bold", va="bottom")
        y += y_dir * 0.25
        for _, row in block_df.iterrows():
            op = str(row["operator"])
            winner = str(row["winner"])
            labels.append((y, op))
            _draw_winner_patch(ax, 3.2, y - row_h * 0.35, 1.6, row_h * 0.7, winner)
            ax.text(2.9, y, winner, ha="right", va="center", fontsize=7, style="italic")

            badge_x = 5.4
            bx = 0.0
            if _has_trust_flag(row.get("trust_flag")):
                ax.text(badge_x + bx, y, "⚠", fontsize=11, ha="center", va="center", color="#CC6600")
                bx += 0.35
            if _parse_bool(row.get("is_tie")):
                ax.text(badge_x + bx, y, "=", fontsize=12, fontweight="bold", ha="center", va="center")
                bx += 0.35
            if _parse_bool(row.get("sign_divergence")):
                ax.text(badge_x + bx, y, "±", fontsize=12, fontweight="bold", ha="center", va="center", color="#8B008B")
                bx += 0.35
            if _parse_bool(row.get("diverges_deployment")):
                ax.text(badge_x + bx, y, "≠", fontsize=12, fontweight="bold", ha="center", va="center", color="#B22222")
            y += y_dir * (row_h + gap)

    for ypos, op in labels:
        ax.text(0.05, ypos, op, ha="left", va="center", fontsize=8)

    y_min, y_max = min(p for p, _ in labels) - 0.5, max(p for p, _ in labels) + 0.5
    ax.set_ylim(y_min, y_max)

    ax.text(3.2, y_max + 0.35, "Winner", ha="left", fontsize=9, fontweight="bold")
    ax.text(5.4, y_max + 0.35, "Flags", ha="center", fontsize=9, fontweight="bold")

    legend_handles = [
        mpatches.Patch(facecolor=ENGINE_COLORS["cpu"], edgecolor="black", label="cpu"),
        mpatches.Patch(facecolor=ENGINE_COLORS["npu"], edgecolor="black", label="npu"),
        mpatches.Patch(facecolor=ENGINE_COLORS["igpu"], edgecolor="black", label="igpu"),
        mpatches.Patch(facecolor=TIE_COLOR_LEFT, edgecolor="black", label="tie (cpu/igpu)"),
        mpatches.Patch(facecolor=EXCLUDED_COLOR, edgecolor="black", hatch="///", label="EXCLUDED / untrusted"),
        mpatches.Patch(facecolor=NA_COLOR, edgecolor="#666666", linestyle="--", label="no matched recommendation"),
        Line2D([0], [0], marker="$⚠$", color="w", markerfacecolor="#CC6600", markersize=10, label="TRUST flag"),
        Line2D([0], [0], marker="$=$", color="k", markersize=10, label="TIE (is_tie)"),
        Line2D([0], [0], marker="$±$", color="#8B008B", markersize=10, label="SIGN-DIVERGENCE"),
        Line2D([0], [0], marker="$≠$", color="#B22222", markersize=10, label="DEPLOYMENT-DIVERGENCE"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.95, ncol=2)

    ax.set_title("Operator -> engine routing (deployment recommendation)", fontsize=13, fontweight="bold", pad=20)
    ax.axis("off")
    fig.tight_layout()
    add_bottom_footnotes(fig)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return out_path


def plot_arch_ratio_dense(df: pd.DataFrame, out_path: Path | None = None) -> Path:
    """Figure B — architecture_ratio log bars, COMPUTE_BOUND_DENSE only."""
    out_path = out_path or FIGURES_DIR / "arch_ratio_dense.png"
    sub = df[
        (df["mechanism"] == "COMPUTE_BOUND_DENSE") & df["architecture_ratio"].notna()
    ].copy()
    sub = sub.sort_values("architecture_ratio", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(sub))
    bar_h = 0.65

    for i, row in sub.iterrows():
        val = float(row["architecture_ratio"])
        op = str(row["operator"])
        winner = str(row["winner"])
        trust = str(row.get("trust_flag", ""))
        is_untrusted = "UNTRUSTED" in trust.upper() or "EXCLUDED" in winner.upper()

        if is_untrusted:
            ax.barh(i, val, height=bar_h, color=EXCLUDED_COLOR, hatch="///", edgecolor="black", linewidth=0.5)
            ax.text(val * 1.05, i, "EXCLUDED — npu INT8 untrusted", va="center", fontsize=7, color="#444444")
        else:
            cat = _winner_category(winner)
            color = ENGINE_COLORS.get(cat, "#888888")
            ax.barh(i, val, height=bar_h, color=color, edgecolor="black", linewidth=0.5)
        ax.text(val * 0.92 if val >= 1 else val * 1.08, i, f"{val:.3g}", va="center", ha="right" if val >= 1 else "left", fontsize=7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub["operator"], fontsize=8)
    ax.set_xscale("log")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.2, zorder=0)
    ax.text(1.02, len(sub) - 0.5, "1.0 = parity; >1 NPU wins matched-INT8", fontsize=8, va="top")
    ax.set_xlabel(
        "architecture_ratio = cpu_INT8 / npu_INT8 (matched INT8), log scale",
        fontsize=10, labelpad=XLABEL_PAD,
    )
    caption = (
        "Memory-bound operators are omitted from this matched-INT8 architecture_ratio comparison: "
        "they are routed on the FP32 grid due to low arithmetic intensity (INT8 yields little benefit "
        "for memory-bound ops), so for these operators architecture_ratio (cpu_INT8/npu_INT8) is a "
        "decomposition cross-reference, not a deployment ranking input."
    )
    ax.set_title(
        _title_with_caption(
            "Architectural advantage at matched INT8 — compute-bound dense operators",
            caption,
        ),
        fontsize=9, pad=20, fontweight="bold",
    )

    legend_handles = [
        mpatches.Patch(facecolor=ENGINE_COLORS["cpu"], label="cpu winner"),
        mpatches.Patch(facecolor=ENGINE_COLORS["npu"], label="npu winner"),
        mpatches.Patch(facecolor=EXCLUDED_COLOR, hatch="///", label="EXCLUDED (untrusted npu)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)
    fig.tight_layout()
    add_bottom_footnotes(fig)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return out_path


def plot_decomposition_identity(df: pd.DataFrame, out_path: Path | None = None) -> Path:
    """Figure C — precision × architecture_ratio = gap cross-check."""
    out_path = out_path or FIGURES_DIR / "decomposition_identity.png"

    def _include(row) -> bool:
        op = str(row["operator"])
        trust = str(row.get("trust_flag", ""))
        if "attn_block_fused" in op:
            return False
        if "UNTRUSTED" in trust.upper() or "EXCLUDED" in str(row["winner"]).upper():
            return False
        return (
            pd.notna(row["precision"])
            and pd.notna(row["architecture_ratio"])
            and pd.notna(row["gap"])
        )

    sub = df[df.apply(_include, axis=1)].copy()
    sub = sub.sort_values("operator").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(sub))

    for i, row in sub.iterrows():
        prec = float(row["precision"])
        arch = float(row["architecture_ratio"])
        gap = float(row["gap"])
        op = str(row["operator"])
        trust = str(row.get("trust_flag", ""))
        low_trust = "LOW-TRUST" in trust.upper()

        xs = [prec, arch, gap]
        if low_trust:
            ax.plot(xs, [i, i, i], color=EXCLUDED_COLOR, linewidth=2, linestyle="--", zorder=1)
            ax.scatter(xs, [i, i, i], c=[EXCLUDED_COLOR] * 3, s=40, zorder=2, marker="s")
            label_suffix = " [LOW-TRUST]"
        else:
            ax.plot(xs, [i, i, i], color="#555555", linewidth=1.5, zorder=1)
            ax.scatter(
                [prec, arch, gap], [i, i, i],
                c=[ENGINE_COLORS["cpu_light"], ENGINE_COLORS["npu"], "#333333"],
                s=45, zorder=2,
            )
            label_suffix = ""

        product = prec * arch
        rel_err = abs(product - gap) / gap if gap else 0
        ax.text(max(xs) * 1.15, i, f"×≈{gap:.3g} ({rel_err*100:.2f}%)", va="center", fontsize=6.5)
        if _parse_bool(row.get("sign_divergence")):
            ax.annotate(
                "gap<1, arch>1", xy=(arch, i), xytext=(arch * 2.5, i + 0.35),
                fontsize=7, color="#8B008B",
                arrowprops=dict(arrowstyle="->", color="#8B008B", lw=0.8),
            )

        ax.text(0.01, i, op + label_suffix, transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.set_xscale("log")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(1.02, len(sub) - 1, "parity (1.0)", fontsize=8, va="top")
    ax.set_xlabel(
        "log scale: precision (●) → architecture_ratio (●) → gap (●)",
        fontsize=10, labelpad=XLABEL_PAD,
    )
    ax.set_title(
        _title_with_caption(
            "Energy-gap decomposition: precision x architecture (INT8-vs-FP32 attribution)",
            "Attribution cross-check only; deployment routing for memory-bound ops is on the FP32 grid (see routing map).",
        ),
        fontsize=9, pad=20,
    )

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ENGINE_COLORS["cpu_light"], label="precision"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ENGINE_COLORS["npu"], label="architecture_ratio"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#333333", label="gap"),
        mpatches.Patch(facecolor=EXCLUDED_COLOR, hatch="///", label="LOW-TRUST (avg_pool)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)
    fig.tight_layout()
    add_bottom_footnotes(fig)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return out_path


def plot_divergence_spotlight(df: pd.DataFrame, out_path: Path | None = None) -> Path:
    """Figure D — sign-divergence and deployment-divergence panels."""
    out_path = out_path or FIGURES_DIR / "divergence_spotlight.png"

    sign_ops = df[df["sign_divergence"].apply(_parse_bool)].copy()
    deploy_ops = df[df["diverges_deployment"].apply(_parse_bool)].copy()

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15, 7))

    # LEFT — sign divergence
    engines_l = ["cpu_FP32", "cpu_INT8", "npu_INT8"]
    colors_l = [ENGINE_COLORS["cpu_light"], ENGINE_COLORS["cpu_dark"], ENGINE_COLORS["npu"]]
    x = np.arange(len(sign_ops))
    w = 0.25
    for j, (eng, col) in enumerate(zip(engines_l, colors_l)):
        vals = sign_ops[eng].astype(float).values
        ax_l.bar(x + (j - 1) * w, vals, width=w, label=eng.replace("_", " "), color=col, edgecolor="black", linewidth=0.4)

    ax_l.set_yscale("log")
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(sign_ops["operator"], rotation=15, ha="right", fontsize=8)
    ax_l.set_ylabel("Energy (J/op)", fontsize=9)
    ax_l.set_xlabel("Operator", fontsize=9, labelpad=XLABEL_PAD)
    ax_l.set_title("Sign-divergence: FP32 headline winner ≠ best-INT8 engine (attribution)", fontsize=9, pad=20)
    for idx, (_, row) in enumerate(sign_ops.iterrows()):
        ax_l.text(
            idx, ax_l.get_ylim()[1] * 0.55,
            "headline (cpu_FP32 vs npu_INT8) -> CPU\nbest-INT8 (cpu_INT8 vs npu_INT8) -> NPU",
            ha="center", fontsize=6.5, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    if "depthwise_conv2d" in list(sign_ops["operator"]):
        depthwise_idx = list(sign_ops["operator"]).index("depthwise_conv2d")
        ax_l.text(
            depthwise_idx, ax_l.get_ylim()[0] * 3,
            "deployment: tie (cpu/igpu) on FP32 grid",
            ha="center", fontsize=6.5, style="italic", color="#555555",
        )
    ax_l.legend(fontsize=7, loc="upper right")

    # RIGHT — deployment divergence
    engines_r = ["cpu_FP32", "igpu_FP32", "npu_INT8"]
    colors_r = [ENGINE_COLORS["cpu_light"], ENGINE_COLORS["igpu"], ENGINE_COLORS["npu"]]
    x2 = np.arange(len(deploy_ops))
    for j, (eng, col) in enumerate(zip(engines_r, colors_r)):
        vals = deploy_ops[eng].astype(float).values
        ax_r.bar(x2 + (j - 1) * w, vals, width=w, label=eng.replace("_", " "), color=col, edgecolor="black", linewidth=0.4)

    ax_r.set_yscale("log")
    ax_r.set_xticks(x2)
    ax_r.set_xticklabels(deploy_ops["operator"], rotation=15, ha="right", fontsize=8)
    ax_r.set_ylabel("Energy (J/op)", fontsize=9)
    ax_r.set_xlabel("Operator", fontsize=9, labelpad=XLABEL_PAD)
    ax_r.set_title("Deployment-divergence: naive cheapest cell ≠ deployment engine", fontsize=9, pad=20)
    for idx, (_, row) in enumerate(deploy_ops.iterrows()):
        naive = str(row["naive_winner"])
        winner = str(row["winner"])
        ax_r.text(
            idx, ax_r.get_ylim()[1] * 0.4,
            f"naive cheapest cell -> {naive}\ndeployment-correct -> {winner}",
            ha="center", fontsize=7, bbox=dict(boxstyle="round", facecolor="#FFF8DC", alpha=0.9),
        )
    ax_r.legend(fontsize=7, loc="upper right")

    fig.suptitle("Two divergence traps (single-corner, avg)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    add_bottom_footnotes(fig)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return out_path


def _engine_marker_style(engine: str, trust: str) -> dict:
    eng = str(engine)
    trust_s = str(trust) if pd.notna(trust) else ""
    style = {"marker": "o", "s": 50, "edgecolors": "black", "linewidths": 0.4}
    if "N/A npu" in trust_s or eng == "npu_INT8" and trust_s.startswith("N/A"):
        style.update(facecolors=NA_COLOR, hatch="///", marker="s")
        return style
    if "UNTRUSTED" in trust_s.upper():
        style.update(facecolors=EXCLUDED_COLOR, hatch="///", marker="D")
        return style
    if "LOW-TRUST" in trust_s.upper():
        style.update(facecolors=EXCLUDED_COLOR, hatch="..", marker="^")
        return style
    color_map = {
        "cpu_FP32": ENGINE_COLORS["cpu_light"],
        "cpu_INT8": ENGINE_COLORS["cpu_dark"],
        "igpu_FP32": ENGINE_COLORS["igpu"],
        "npu_INT8": ENGINE_COLORS["npu"],
    }
    style["c"] = color_map.get(eng, "#888888")
    return style


def plot_figure_e_power_time(df_e: pd.DataFrame, out_path: Path | None = None) -> Path:
    """Figure E — gross package power vs amortized execution time."""
    out_path = out_path or FIGURES_DIR / "figure_e_power_time_decomposition.png"
    sub = df_e[df_e["execution_time_ms"].notna() & df_e["avg_power_W"].notna()].copy()

    fig, ax = plt.subplots(figsize=(11, 8))

    p_lo = float(sub["avg_power_W"].min()) * 0.8
    p_hi = float(sub["avg_power_W"].max()) * 1.2
    t_min = float(sub["execution_time_ms"].min()) * 0.5
    t_max = float(sub["execution_time_ms"].max()) * 2.0
    t_grid = np.logspace(np.log10(t_min), np.log10(t_max), 200)

    e_levels = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
    for e in e_levels:
        p_contour = e * 1000.0 / t_grid
        visible = (p_contour >= p_lo) & (p_contour <= p_hi)
        if not visible.any():
            continue
        ax.plot(t_grid[visible], p_contour[visible], color="#DDDDDD", linewidth=0.8, zorder=0)
        mid = visible.sum() // 2
        ti = np.where(visible)[0][mid]
        ax.text(t_grid[ti], p_contour[ti], f"E={e}J", fontsize=6, color="#888888")

    ax.set_ylim(p_lo, p_hi)
    ax.set_xlim(t_min, t_max)

    sign_ops = {"softmax[s1]", "depthwise_conv2d"}

    for _, row in sub.iterrows():
        t = float(row["execution_time_ms"])
        p = float(row["avg_power_W"])
        eng = str(row["engine"])
        op = str(row["operator"])
        trust = row.get("trust_flag", "")
        kw = _engine_marker_style(eng, trust)
        fc = kw.pop("c", kw.pop("facecolors", "#888888"))
        ec = kw.pop("edgecolors", "black")
        marker = kw.pop("marker", "o")
        s = kw.pop("s", 50)
        lw = kw.pop("linewidths", 0.4)
        ax.scatter(t, p, facecolors=fc, edgecolors=ec, marker=marker, s=s, linewidths=lw, zorder=3, **kw)
        if op in sign_ops:
            ax.scatter(t, p, s=180, facecolors="none", edgecolors="#8B008B", linewidths=2, zorder=4)
            ax.annotate(
                op, (t, p), xytext=(8, 8), textcoords="offset points", fontsize=7, color="#8B008B",
                arrowprops=dict(arrowstyle="->", color="#8B008B", lw=0.7),
            )

    ax.set_xscale("log")
    ax.set_xlabel(
        "execution_time_ms (amortized throughput-time, log scale)",
        fontsize=10, labelpad=XLABEL_PAD,
    )
    ax.set_ylabel("avg_power_W (gross mean PACKAGE power)", fontsize=10)
    caption = (
        "Caveats: avg_power represents gross mean PACKAGE power (cannot be engine-isolated). "
        "execution_time represents amortized throughput-time over the window, not single-shot latency."
    )
    ax.set_title(
        _title_with_caption(
            "Energy Decomposition: Gross Package Power vs. Amortized Execution Time",
            caption,
        ),
        fontsize=9, pad=20, fontweight="bold",
    )

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ENGINE_COLORS["cpu_light"], label="cpu_FP32", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ENGINE_COLORS["cpu_dark"], label="cpu_INT8", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ENGINE_COLORS["igpu"], label="igpu_FP32", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ENGINE_COLORS["npu"], label="npu_INT8", markersize=8),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=EXCLUDED_COLOR, label="UNTRUSTED npu", markersize=8),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=EXCLUDED_COLOR, label="LOW-TRUST cpu_int8", markersize=8),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=NA_COLOR, label="N/A npu", markersize=8),
        Line2D([0], [0], marker="o", color="#8B008B", markerfacecolor="none", markersize=12, label="sign-divergence highlight"),
        Line2D([0], [0], color="#DDDDDD", linewidth=1, label="iso-E_gross contours"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.95)
    fig.tight_layout()
    add_bottom_footnotes(fig, source_line=FOOTNOTE_SOURCE_E)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Chris figure suite from STEP4 primary sheet.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_SHEET, help="STEP4_FOR_SHEET_primary.csv path")
    parser.add_argument("--figure-e-csv", type=Path, default=DEFAULT_FIGURE_E, help="figure_e_operator_engine.csv path")
    parser.add_argument(
        "--figures", nargs="+", choices=["a", "b", "c", "d", "e", "all"], default=["all"],
        help="Which figures to render",
    )
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sheet = load_sheet(args.csv)
    want = set(args.figures)
    if "all" in want:
        want = {"a", "b", "c", "d", "e"}

    outputs = []
    if "a" in want:
        outputs.append(plot_routing_map(sheet))
    if "b" in want:
        outputs.append(plot_arch_ratio_dense(sheet))
    if "c" in want:
        outputs.append(plot_decomposition_identity(sheet))
    if "d" in want:
        outputs.append(plot_divergence_spotlight(sheet))
    if "e" in want:
        if not args.figure_e_csv.exists():
            raise SystemExit(f"HALT: Figure E CSV not found: {args.figure_e_csv}")
        df_e = load_figure_e(args.figure_e_csv)
        outputs.append(plot_figure_e_power_time(df_e))

    for p in outputs:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
