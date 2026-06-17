#!/usr/bin/env python3
"""
Step 4 provenance diff — read-only validation of ANALYSIS_REFERENCE.md §3 vs primary CSVs.

Does NOT modify ANALYSIS_REFERENCE.md or routing thresholds.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

from step4_operator_engine_mapping import (
    CONTROL_ANALYSIS,
    ENERGY_COLS,
    PRIMARY_ANALYSIS,
    RATIO_COLS,
    RUNS_CONTROL,
    RUNS_PRODUCTION,
    SESSION_CONTROL,
    SESSION_PRODUCTION,
    load_primary,
    parse_section3,
    run_routing,
    ANALYSIS_REF,
    RESULTS_DIR,
)

OUT_DIFF_CSV = RESULTS_DIR / "step4_provenance_diff.csv"

TRUST_DISPOSITIONS = {
    ("sra_conv2d", "npu_INT8"): "expected-EXCLUDED-arch",
    ("attn_block_fused", "npu_INT8"): "expected-ABSENT",
    ("avg_pool_token_mixer", "cpu_INT8"): "LOW-TRUST-present",
}


def detect_printed_precision(token: str) -> int | None:
    t = token.strip().rstrip("†‡").strip()
    if not t or t in ("—", "-", "EXCLUDED", "N/A"):
        return None
    if "e" in t.lower():
        mantissa = t.lower().split("e")[0].lstrip("+-")
        if "." in mantissa:
            return len(mantissa.replace(".", ""))
        return len(mantissa)
    if "." in t:
        frac = t.split(".")[1].rstrip("0")
        return len(frac) if frac else 0
    return 0


def round_to_printed(value: float, token: str) -> float:
    places = detect_printed_precision(token)
    if places is None:
        return value
    return round(value, places)


def _s3_float(token: str) -> float | None:
    t = token.strip().rstrip("†‡").strip()
    if not t or t in ("—", "-", "EXCLUDED", "N/A"):
        return None
    return float(t)


def _rel_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-30)
    return abs(a - b) / denom


def gate0_runs() -> None:
    print("=" * 72)
    print("GATE 0 — SESSION + ROW COUNTS")
    print("=" * 72)
    for path, expected_session, expect_total, expect_baselines in (
        (RUNS_PRODUCTION, SESSION_PRODUCTION, 314, 4),
        (RUNS_CONTROL, SESSION_CONTROL, 107, 2),
    ):
        if not path.exists():
            raise SystemExit(f"HALT: missing runs CSV {path}")
        df = pd.read_csv(path)
        session = path.stem.replace("runs_", "")
        baselines = df[df["operator"].str.contains("idle|dispatch", case=False, na=False, regex=True)]
        print(f"  {path.name}: rows={len(df)} baselines={len(baselines)} session_id={session}")
        if session != expected_session:
            raise SystemExit(
                f"HALT: session id mismatch {session!r} != {expected_session!r}"
            )
        if len(df) != expect_total:
            print(f"  WARNING: expected {expect_total} total rows, got {len(df)}")
    print(f"  primary analysis: {PRIMARY_ANALYSIS.name} exists={PRIMARY_ANALYSIS.exists()}")
    print(f"  control analysis: {CONTROL_ANALYSIS.name} exists={CONTROL_ANALYSIS.exists()}")
    print()


def _energy_disposition(operator: str, column: str) -> str:
    if (operator, column) in TRUST_DISPOSITIONS:
        return TRUST_DISPOSITIONS[(operator, column)]
    if operator == "sra_conv2d" and column == "npu_INT8":
        return "UNTRUSTED-measure-still-diff"
    if operator == "avg_pool_token_mixer" and column == "cpu_INT8":
        return "LOW-TRUST-present"
    return ""


def energy_layer_diff(s3: pd.DataFrame, primary: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    rows: list[dict] = []
    pass_n = fail_n = 0
    s3_idx = s3.set_index(["operator", "shape_index"])
    pri_idx = primary.set_index(["operator", "shape_index"])

    all_keys = sorted(set(s3_idx.index) | set(pri_idx.index), key=lambda k: (k[0], k[1]))
    for key in all_keys:
        operator, shape_index = key
        for col in ENERGY_COLS:
            raw_col = f"raw_{col}"
            disposition = _energy_disposition(operator, col)
            if key not in s3_idx.index:
                rows.append(
                    {
                        "operator": operator,
                        "shape_index": shape_index,
                        "metric": col,
                        "layer": "energy",
                        "s3_value": "",
                        "primary_value": pri_idx.loc[key, col] if key in pri_idx.index else "",
                        "abs_diff": "",
                        "rel_diff": "",
                        "printed_precision": "",
                        "verdict": "FAIL",
                        "disposition": "missing-in-section3",
                    }
                )
                fail_n += 1
                continue
            if key not in pri_idx.index:
                rows.append(
                    {
                        "operator": operator,
                        "shape_index": shape_index,
                        "metric": col,
                        "layer": "energy",
                        "s3_value": s3_idx.loc[key, col],
                        "primary_value": "",
                        "abs_diff": "",
                        "rel_diff": "",
                        "printed_precision": "",
                        "verdict": "FAIL",
                        "disposition": "missing-in-primary",
                    }
                )
                fail_n += 1
                continue

            raw_tok = str(s3_idx.loc[key, raw_col])
            s3_val = s3_idx.loc[key, col]
            pri_val = pri_idx.loc[key, col]

            if operator == "attn_block_fused" and col == "npu_INT8":
                if pd.isna(s3_val) and pd.isna(pri_val):
                    verdict = "PASS"
                    pass_n += 1
                else:
                    verdict = "FAIL"
                    fail_n += 1
                rows.append(
                    {
                        "operator": operator,
                        "shape_index": shape_index,
                        "metric": col,
                        "layer": "energy",
                        "s3_value": s3_val,
                        "primary_value": pri_val,
                        "abs_diff": "",
                        "rel_diff": "",
                        "printed_precision": detect_printed_precision(raw_tok),
                        "verdict": verdict,
                        "disposition": "expected-ABSENT",
                    }
                )
                continue

            if pd.isna(s3_val) and pd.isna(pri_val):
                verdict = "PASS"
                pass_n += 1
                abs_d, rel_d = 0.0, 0.0
            elif pd.isna(s3_val) or pd.isna(pri_val):
                verdict = "FAIL"
                fail_n += 1
                abs_d, rel_d = float("nan"), float("nan")
            else:
                pri_rounded = round_to_printed(float(pri_val), raw_tok)
                s3_parsed = float(s3_val)
                abs_d = abs(pri_rounded - s3_parsed)
                rel_d = _rel_diff(pri_rounded, s3_parsed)
                if pri_rounded == s3_parsed:
                    verdict = "PASS"
                    pass_n += 1
                else:
                    verdict = "FAIL"
                    fail_n += 1
            rows.append(
                {
                    "operator": operator,
                    "shape_index": shape_index,
                    "metric": col,
                    "layer": "energy",
                    "s3_value": s3_val,
                    "primary_value": pri_val,
                    "abs_diff": abs_d if verdict == "FAIL" else 0.0,
                    "rel_diff": rel_d if verdict == "FAIL" else 0.0,
                    "printed_precision": detect_printed_precision(raw_tok),
                    "verdict": verdict,
                    "disposition": disposition,
                }
            )

    return pd.DataFrame(rows), pass_n, fail_n


def ratio_layer_diff(s3: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    s3_idx = s3.set_index(["operator", "shape_index"])
    pri_idx = primary.set_index(["operator", "shape_index"])

    for key in sorted(s3_idx.index, key=lambda k: (k[0], k[1])):
        operator, shape_index = key
        if key not in pri_idx.index:
            continue
        s3_row = s3_idx.loc[key]
        pri_row = pri_idx.loc[key]

        derived = {
            "precision": (
                pri_row["cpu_FP32"] / pri_row["cpu_INT8"]
                if pd.notna(pri_row["cpu_FP32"])
                and pd.notna(pri_row["cpu_INT8"])
                and pri_row["cpu_INT8"] != 0
                else None
            ),
            "architecture_ratio": (
                pri_row["cpu_INT8"] / pri_row["npu_INT8"]
                if pd.notna(pri_row["cpu_INT8"])
                and pd.notna(pri_row["npu_INT8"])
                and pri_row["npu_INT8"] != 0
                else None
            ),
            "gap": (
                pri_row["cpu_FP32"] / pri_row["npu_INT8"]
                if pd.notna(pri_row["cpu_FP32"])
                and pd.notna(pri_row["npu_INT8"])
                and pri_row["npu_INT8"] != 0
                else None
            ),
        }

        for metric in RATIO_COLS:
            raw_col = f"raw_{metric}"
            raw_tok = str(s3_row[raw_col])
            if raw_tok.strip() in ("EXCLUDED", "N/A", "—", "-", ""):
                rows.append(
                    {
                        "operator": operator,
                        "shape_index": shape_index,
                        "metric": metric,
                        "layer": "ratio",
                        "s3_value": raw_tok,
                        "primary_value": derived[metric],
                        "abs_diff": "",
                        "rel_diff": "",
                        "printed_precision": "",
                        "verdict": "SKIP",
                        "disposition": raw_tok.strip() or "N/A",
                    }
                )
                continue
            s3_val = s3_row[metric]
            pri_val = derived[metric]
            pri_rounded = None
            if pd.isna(s3_val) or pri_val is None:
                verdict = "SKIP" if pd.isna(s3_val) and pri_val is None else "INFO"
            else:
                pri_rounded = round_to_printed(float(pri_val), raw_tok)
                verdict = "MATCH" if pri_rounded == float(s3_val) else "MISMATCH"
            abs_d = (
                abs(pri_rounded - float(s3_val))
                if verdict == "MISMATCH" and pri_rounded is not None and pd.notna(s3_val)
                else ""
            )
            rows.append(
                {
                    "operator": operator,
                    "shape_index": shape_index,
                    "metric": metric,
                    "layer": "ratio",
                    "s3_value": s3_val,
                    "primary_value": pri_val,
                    "abs_diff": abs_d,
                    "rel_diff": "",
                    "printed_precision": detect_printed_precision(raw_tok),
                    "verdict": verdict,
                    "disposition": "",
                }
            )
    return pd.DataFrame(rows)


def output_layer_diff(s3: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    out_s3 = run_routing(s3).set_index("operator")
    out_pri = run_routing(primary).set_index("operator")
    cols = ["winner", "sign_divergence", "diverges_deployment", "naive_winner"]
    rows = []
    for op in sorted(set(out_s3.index) | set(out_pri.index)):
        for c in cols:
            v_s3 = out_s3.loc[op, c] if op in out_s3.index else "MISSING"
            v_pri = out_pri.loc[op, c] if op in out_pri.index else "MISSING"
            rows.append(
                {
                    "operator": op,
                    "shape_index": "",
                    "metric": c,
                    "layer": "output",
                    "s3_value": v_s3,
                    "primary_value": v_pri,
                    "abs_diff": "",
                    "rel_diff": "",
                    "printed_precision": "",
                    "verdict": "MATCH" if v_s3 == v_pri else "DIFF",
                    "disposition": "secondary",
                }
            )
    return pd.DataFrame(rows)


def pattern_note(energy_df: pd.DataFrame) -> str:
    fails = energy_df[energy_df["verdict"] == "FAIL"]
    if fails.empty:
        return "All energy cells PASS — safe to promote primary for paper-grade table."
    by_col = fails.groupby("metric").size()
    by_op = fails.groupby("operator").size()
    if len(fails) == 1:
        return "Isolated single-cell FAIL — likely §3 transcription typo."
    if len(by_col) == 1:
        return (
            f"Fails clustered on column {by_col.index[0]} — likely systematic loader/join issue; "
            "do NOT promote."
        )
    if len(by_op) == 1:
        return (
            f"Fails clustered on operator {by_op.index[0]} — check label/key reconciliation; "
            "do NOT promote."
        )
    return "Mixed failure pattern — inspect per-cell; do NOT promote until resolved."


def main() -> int:
    if not ANALYSIS_REF.exists():
        print("HALT: ANALYSIS_REFERENCE.md not found")
        return 1

    gate0_runs()

    print("=" * 72)
    print("LOAD ORACLE + PRIMARY")
    print("=" * 72)
    s3 = parse_section3(ANALYSIS_REF)
    primary = load_primary()
    print(f"  section3 rows: {len(s3)}")
    print(f"  primary rows:  {len(primary)}")
    print()

    print("=" * 72)
    print("ENERGY-LAYER DIFF (GATE)")
    print("=" * 72)
    energy_df, pass_n, fail_n = energy_layer_diff(s3, primary)
    total = pass_n + fail_n
    print(f"  total cells: {total}  PASS: {pass_n}  FAIL: {fail_n}")
    if fail_n:
        print("\n  FAILURES:")
        for _, r in energy_df[energy_df["verdict"] == "FAIL"].iterrows():
            print(
                f"    {r['operator']}[s{r['shape_index']}] {r['metric']}: "
                f"s3={r['s3_value']} primary={r['primary_value']} "
                f"(rounded primary @ prec {r['printed_precision']}) "
                f"abs={r['abs_diff']:.6g} rel={r['rel_diff']:.6g}"
            )
    print(f"\n  PATTERN: {pattern_note(energy_df)}")
    print()

    print("=" * 72)
    print("RATIO DIFF (report-only, not gate)")
    print("=" * 72)
    ratio_df = ratio_layer_diff(s3, primary)
    mism = ratio_df[ratio_df["verdict"] == "MISMATCH"]
    print(f"  ratio rows: {len(ratio_df)}  MISMATCH: {len(mism)}")
    if len(mism):
        for _, r in mism.head(10).iterrows():
            print(
                f"    {r['operator']}[s{r['shape_index']}] {r['metric']}: "
                f"s3={r['s3_value']} derived={r['primary_value']}"
            )
    print()

    print("=" * 72)
    print("OUTPUT DIFF (secondary)")
    print("=" * 72)
    output_df = output_layer_diff(s3, primary)
    diffs = output_df[output_df["verdict"] == "DIFF"]
    print(f"  output comparisons: {len(output_df)}  DIFF: {len(diffs)}")
    if len(diffs):
        for _, r in diffs.iterrows():
            print(f"    {r['operator']} {r['metric']}: s3={r['s3_value']!r} primary={r['primary_value']!r}")
    print()

    combined = pd.concat([energy_df, ratio_df, output_df], ignore_index=True)
    OUT_DIFF_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_DIFF_CSV, index=False)
    print(f"Wrote: {OUT_DIFF_CSV}")

    if fail_n:
        print("\nHALT: energy-layer gate FAILED")
        return 1
    print("\nENERGY-LAYER GATE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
