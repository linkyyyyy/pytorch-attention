"""Build benchmark/figure_e_operator_engine.csv from existing tower captures (analysis only)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BENCHMARK_DIR = Path(__file__).resolve().parent
OUT_PATH = BENCHMARK_DIR / "figure_e_operator_engine.csv"
SHEET_PATH = BENCHMARK_DIR / "STEP4_FOR_SHEET_primary.csv"
ENR_MAIN_PATH = BENCHMARK_DIR / "results" / "runs_enriched.csv"
ENR_INT8_PATH = BENCHMARK_DIR / "results" / "runs_20260615_145614_cpu_int8_enriched.csv"
AO_MAIN_PATH = BENCHMARK_DIR / "results" / "analysis_out.csv"
AO_INT8_PATH = BENCHMARK_DIR / "results" / "analysis_out_20260615_145614_cpu_int8.csv"

ENGINE_COLS = ["cpu_FP32", "cpu_INT8", "npu_INT8", "igpu_FP32"]
SESSIONS = "20260612_143854 (cpu_FP32, igpu_FP32, npu XINT8) + 20260615_145614_cpu_int8 (cpu_INT8 control)"
TOKEN_N = 197


def parse_op_label(label: str) -> tuple[str, int]:
    m = re.match(r"^(.+?)\[s(\d+)\]$", label)
    if m:
        return m.group(1), int(m.group(2))
    return label, 0


def is_measure(df: pd.DataFrame) -> pd.DataFrame:
    mask = ~df["operator"].isin(["idle", "dispatch_baseline", "dispatch"]) & df["operator"].notna()
    out = df.loc[mask].copy()
    out["shape_index"] = out["shape_index"].fillna(0).astype(int)
    return out


def duration_row(r: pd.Series) -> tuple[float, str]:
    if pd.notna(r.get("t_start_epoch")) and pd.notna(r.get("t_end_epoch")):
        return float(r["t_end_epoch"]) - float(r["t_start_epoch"]), "t_end_epoch_minus_t_start_epoch"
    return float(r["wall_time_s"]), "wall_time_s"


def agg_cell(df: pd.DataFrame) -> dict:
    durs: list[float] = []
    fields: list[str] = []
    for _, row in df.iterrows():
        d, f = duration_row(row)
        durs.append(d)
        fields.append(f)
    field = fields[0] if len(set(fields)) == 1 else "mixed"
    it = df["iterations_completed"].astype(float)
    we = df["window_energy_J"].astype(float)
    idle = df["idle_energy_J"].astype(float)
    it_mean = float(it.mean())
    dur_mean = float(np.mean(durs))
    we_mean = float(we.mean())
    idle_mean = float(idle.mean())
    return {
        "iterations_completed": it_mean,
        "window_duration_s": dur_mean,
        "duration_field_used": field,
        "window_energy_J": we_mean,
        "idle_energy_J": idle_mean,
        "energy_gross_J": we_mean / it_mean,
        "idle_energy_per_op_J": float((idle / it).mean()),
        "energy_net_J": float(((we - idle) / it).mean()),
        "mean_latency_ms": float(df["mean_latency_ms"].mean()),
        "dispatch_energy_J": float(df["dispatch_energy_J"].mean()) if "dispatch_energy_J" in df else np.nan,
    }


def ao_engine_col(engine: str) -> str | None:
    if engine == "cpu":
        return "cpu_FP32"
    if engine == "cpu_int8":
        return "cpu_INT8"
    if engine == "igpu":
        return "igpu_FP32"
    if engine == "npu":
        return "npu_INT8"
    return None


def trust_flag_for(base: str, ec: str) -> str:
    if base == "attn_block_fused" and ec == "npu_INT8":
        return "N/A npu (VAI EP batch mismatch)"
    if base == "sra_conv2d" and ec == "npu_INT8":
        return "UNTRUSTED npu INT8 (CPU fallback)"
    if base == "avg_pool_token_mixer" and ec == "cpu_INT8":
        return "LOW-TRUST cpu_int8 (high CV%)"
    return ""


def select_sub(enr_m: pd.DataFrame, enr_i: pd.DataFrame, base: str, si: int, ec: str) -> pd.DataFrame:
    if ec == "cpu_INT8":
        return enr_i[(enr_i["operator"] == base) & (enr_i["shape_index"] == si) & (enr_i["engine"] == "cpu_int8")]
    if ec == "cpu_FP32":
        return enr_m[(enr_m["operator"] == base) & (enr_m["shape_index"] == si) & (enr_m["engine"] == "cpu")]
    if ec == "igpu_FP32":
        return enr_m[(enr_m["operator"] == base) & (enr_m["shape_index"] == si) & (enr_m["engine"] == "igpu")]
    if ec == "npu_INT8":
        return enr_m[(enr_m["operator"] == base) & (enr_m["shape_index"] == si) & (enr_m["engine"] == "npu")]
    return pd.DataFrame()


def require_inputs() -> None:
    for p in (SHEET_PATH, ENR_MAIN_PATH, ENR_INT8_PATH, AO_MAIN_PATH, AO_INT8_PATH):
        if not p.exists():
            raise SystemExit(f"HALT: required input missing: {p}")


def build_rows() -> tuple[pd.DataFrame, dict]:
    require_inputs()
    sheet = pd.read_csv(SHEET_PATH, comment="#")
    enr_m = is_measure(pd.read_csv(ENR_MAIN_PATH))
    enr_i = is_measure(pd.read_csv(ENR_INT8_PATH))

    ao = pd.concat(
        [
            pd.read_csv(AO_MAIN_PATH),
            pd.read_csv(AO_INT8_PATH),
        ],
        ignore_index=True,
    )
    ao = ao[ao["table"] == "per_operator"].copy()
    ao["shape_index"] = ao["shape_index"].fillna(0).astype(int)

    ao_lookup: dict[tuple[str, int, str], float] = {}
    for _, r in ao.iterrows():
        ec = ao_engine_col(str(r["engine"]))
        if ec is None:
            continue
        mean = float(r["energy_per_op_J_mean"])
        std = float(r["energy_per_op_J_std"]) if pd.notna(r["energy_per_op_J_std"]) else np.nan
        cv = (std / mean * 100.0) if pd.notna(std) and mean != 0 else np.nan
        ao_lookup[(str(r["operator"]), int(r["shape_index"]), ec)] = cv

    rows: list[dict] = []
    for _, s in sheet.iterrows():
        op_label = str(s["operator"])
        base, si = parse_op_label(op_label)
        mech = s["mechanism"]
        winner = s["winner"]
        for ec in ENGINE_COLS:
            tf = trust_flag_for(base, ec)
            if base == "attn_block_fused" and ec == "npu_INT8":
                rows.append(
                    {
                        "operator": op_label,
                        "engine": ec,
                        "mechanism": mech,
                        "iterations_completed": "",
                        "window_duration_s": "",
                        "duration_field_used": "",
                        "avg_power_W": "",
                        "execution_time_ms": "",
                        "energy_gross_J": "",
                        "idle_energy_per_op_J": "",
                        "energy_net_J": "",
                        "energy_net_CV_pct": "",
                        "energy_per_token_J": "",
                        "mean_latency_ms": "",
                        "trust_flag": tf,
                        "winner": winner,
                    }
                )
                continue

            sub = select_sub(enr_m, enr_i, base, si, ec)
            if sub.empty:
                raise SystemExit(f"HALT: no enriched rows for {op_label} × {ec}")

            a = agg_cell(sub)
            it = a["iterations_completed"]
            dur = a["window_duration_s"]
            we = a["window_energy_J"]
            eg = a["energy_gross_J"]
            idle_po = a["idle_energy_per_op_J"]
            en = a["energy_net_J"]
            ap = we / dur
            et = dur / it * 1000.0
            ept = en / TOKEN_N
            cv = ao_lookup.get((base, si, ec), np.nan)

            rows.append(
                {
                    "operator": op_label,
                    "engine": ec,
                    "mechanism": mech,
                    "iterations_completed": it,
                    "window_duration_s": dur,
                    "duration_field_used": a["duration_field_used"],
                    "avg_power_W": ap,
                    "execution_time_ms": et,
                    "energy_gross_J": eg,
                    "idle_energy_per_op_J": idle_po,
                    "energy_net_J": en,
                    "energy_net_CV_pct": cv,
                    "energy_per_token_J": ept,
                    "mean_latency_ms": a["mean_latency_ms"],
                    "trust_flag": tf,
                    "winner": winner,
                }
            )

    out = pd.DataFrame(rows)
    gates = run_gates(out, sheet, enr_m, enr_i)
    return out, gates


def run_gates(out: pd.DataFrame, sheet: pd.DataFrame, enr_m: pd.DataFrame, enr_i: pd.DataFrame) -> dict:
    g1_fails: list[tuple] = []
    max_g1 = 0.0
    for _, r in out.iterrows():
        if r["energy_gross_J"] == "" or pd.isna(r["energy_gross_J"]):
            continue
        eg = float(r["energy_gross_J"])
        ap = float(r["avg_power_W"])
        et = float(r["execution_time_ms"])
        rel = abs(ap * (et / 1000.0) - eg) / eg if eg else 0.0
        max_g1 = max(max_g1, rel)
        if rel > 1e-3:
            g1_fails.append((r["operator"], r["engine"], rel))

    g2_fails: list[tuple] = []
    max_g2 = 0.0
    for _, s in sheet.iterrows():
        op_label = str(s["operator"])
        base, _ = parse_op_label(op_label)
        for ec in ENGINE_COLS:
            if base == "attn_block_fused" and ec == "npu_INT8":
                continue
            sheet_val = float(s[ec])
            row = out[(out["operator"] == op_label) & (out["engine"] == ec)]
            if row.empty:
                g2_fails.append((op_label, ec, "missing row"))
                continue
            en = float(row.iloc[0]["energy_net_J"])
            diff = abs(en - sheet_val)
            max_g2 = max(max_g2, diff)
            if diff > 1e-6:
                g2_fails.append((op_label, ec, sheet_val, en, diff))

    g3_flags: list[tuple] = []
    for _, r in out.iterrows():
        if r["energy_gross_J"] == "":
            continue
        eg = float(r["energy_gross_J"])
        en = float(r["energy_net_J"])
        idle_po = float(r["idle_energy_per_op_J"])
        if not (eg > en > 0):
            g3_flags.append((r["operator"], r["engine"], "gross>net>0 violated", eg, en))
        if idle_po >= eg:
            g3_flags.append((r["operator"], r["engine"], "possible idle drift (idle>=gross)", idle_po, eg))

    # G4: confirm energy_net uses idle subtraction, not dispatch
    g4_ok = True
    g4_notes: list[str] = []
    for _, s in sheet.iterrows():
        op_label = str(s["operator"])
        base, si = parse_op_label(op_label)
        for ec in ENGINE_COLS:
            if base == "attn_block_fused" and ec == "npu_INT8":
                continue
            sub = select_sub(enr_m, enr_i, base, si, ec)
            if sub.empty:
                continue
            idle_net = ((sub["window_energy_J"] - sub["idle_energy_J"]) / sub["iterations_completed"]).mean()
            if "dispatch_energy_J" in sub.columns:
                dispatch_net = ((sub["window_energy_J"] - sub["dispatch_energy_J"]) / sub["iterations_completed"]).mean()
                row = out[(out["operator"] == op_label) & (out["engine"] == ec)].iloc[0]
                en = float(row["energy_net_J"])
                if abs(en - dispatch_net) < 1e-9 and abs(en - idle_net) > 1e-6:
                    g4_ok = False
                    g4_notes.append(f"{op_label}×{ec}: energy_net matches dispatch not idle")

    return {
        "g1_pass": len(g1_fails) == 0,
        "g1_max_rel": max_g1,
        "g1_fails": g1_fails,
        "g2_pass": len(g2_fails) == 0,
        "g2_max_abs": max_g2,
        "g2_fails": g2_fails,
        "g3_flags": g3_flags,
        "g4_pass": g4_ok,
        "g4_notes": g4_notes,
    }


def write_csv(out: pd.DataFrame, gates: dict) -> None:
    g1_status = "PASS" if gates["g1_pass"] else "FAIL"
    g2_status = "PASS" if gates["g2_pass"] else "FAIL"
    banner = (
        "# PROVENANCE: sessions="
        f"{SESSIONS}; "
        "avg_power = gross mean PACKAGE power (window_energy_J/window_duration_s), not engine-isolated; "
        "execution_time = amortized throughput-time (window_duration/iterations), not single-shot latency; "
        "energy_net = window-minus-idle headline metric; "
        f"energy_per_token = energy_net/197 (fixed-N=197 normalization); "
        f"G1={g1_status} (max_rel={gates['g1_max_rel']:.3e}); "
        f"G2={g2_status} (max_abs={gates['g2_max_abs']:.3e}); "
        "consistent with energy gate 83/83 numeric PASS + 1 expected-N/A; date 2026-06-17."
    )
    cols = [
        "operator",
        "engine",
        "mechanism",
        "iterations_completed",
        "window_duration_s",
        "duration_field_used",
        "avg_power_W",
        "execution_time_ms",
        "energy_gross_J",
        "idle_energy_per_op_J",
        "energy_net_J",
        "energy_net_CV_pct",
        "energy_per_token_J",
        "mean_latency_ms",
        "trust_flag",
        "winner",
    ]
    lines = [banner, ",".join(cols)]
    for _, r in out[cols].iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if v == "" or (isinstance(v, float) and pd.isna(v)):
                vals.append("")
            elif isinstance(v, (int, np.integer)):
                vals.append(str(v))
            elif isinstance(v, float):
                vals.append(repr(v))
            else:
                vals.append(str(v))
        lines.append(",".join(vals))
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    out, gates = build_rows()
    print(f"rows: {len(out)}")
    print(f"G1: {'PASS' if gates['g1_pass'] else 'FAIL'} max_rel={gates['g1_max_rel']:.6e}")
    if gates["g1_fails"]:
        print("  fails:", gates["g1_fails"][:10])
    print(f"G2: {'PASS' if gates['g2_pass'] else 'FAIL'} max_abs={gates['g2_max_abs']:.6e}")
    if gates["g2_fails"]:
        print("  fails:", gates["g2_fails"][:10])
    print(f"G3: {len(gates['g3_flags'])} flag(s)")
    for f in gates["g3_flags"][:10]:
        print(" ", f)
    print(f"G4: {'PASS' if gates['g4_pass'] else 'FAIL'}")
    for n in gates["g4_notes"]:
        print(" ", n)

    if not gates["g1_pass"] or not gates["g2_pass"]:
        print("HALT: G1 or G2 failed — CSV not written.")
        return 1

    write_csv(out, gates)
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
