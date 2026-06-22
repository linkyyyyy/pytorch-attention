import csv, sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # benchmark/ on path
import parse_energy as pe

GP_HDR = ["sample_epoch","mono_s","power_w","energy_uj","gfx_busy_pct",
          "vram_used_mb","sclk_mhz","device_id"]

def _mk_gp(path, *, power, busy, counter):
    # counter: "ramp" | "none" | "nonmono"; samples 999.5..1030.5 step 0.5
    e0 = 999.5
    rows = []
    for k in range(63):
        ep = 999.5 + 0.5*k
        if counter == "none":
            uj = ""
        else:
            val = int(power*1_000_000*(ep-e0))
            if counter == "nonmono" and 1005.0 <= ep <= 1005.6:
                val -= 100_000_000
            uj = str(val)
        rows.append({"sample_epoch":f"{ep:.6f}","mono_s":f"{k*0.5:.6f}",
                     "power_w":f"{power:.3f}","energy_uj":uj,"gfx_busy_pct":f"{busy:.1f}",
                     "vram_used_mb":"512","sclk_mhz":"2200","device_id":"0"})
    with open(path,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=GP_HDR); w.writeheader(); w.writerows(rows)

def _samples(path):
    return pe.parse_amdsmi_csv(Path(path))

def test_counter_delta_matches_integration(tmp_path):
    p = tmp_path/"op.csv"; _mk_gp(p, power=150, busy=90, counter="ramp")
    j, method = pe.amdsmi_window_energy(_samples(p), 1000.0, 1030.0)
    assert method == "counter_delta_uj"
    assert abs(j - 4500.0) < 1e-6

def test_no_counter_falls_to_integration(tmp_path):
    p = tmp_path/"op.csv"; _mk_gp(p, power=150, busy=90, counter="none")
    j, method = pe.amdsmi_window_energy(_samples(p), 1000.0, 1030.0)
    assert method == "trapz_power_w:no_counter"
    assert abs(j - 4500.0) < 1e-6

def test_wrap_guard_falls_to_integration(tmp_path):
    p = tmp_path/"op.csv"; _mk_gp(p, power=150, busy=90, counter="nonmono")
    j, method = pe.amdsmi_window_energy(_samples(p), 1000.0, 1030.0)
    assert method == "trapz_power_w:counter_nonmonotonic"
    assert abs(j - 4500.0) < 1e-6

def _runs_csv(path):
    hdr = ["run_id","operator","engine","shape_index","t_start_epoch","t_end_epoch","iterations_completed"]
    rows = [("opA_r9700_s0_r0","opA","r9700","0"),("opD_r9700_s0_r0","opD","r9700","0"),
            ("idle_r9700_r0","idle","r9700","-1"),
            ("dispatch_baseline_r9700_r0","dispatch_baseline","r9700","-1")]
    with open(path,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr); w.writeheader()
        for rid,op,eng,si in rows:
            w.writerow({"run_id":rid,"operator":op,"engine":eng,"shape_index":si,
                        "t_start_epoch":"1000.000000","t_end_epoch":"1030.000000",
                        "iterations_completed":"10000"})

def test_enrich_amdsmi_end_to_end(tmp_path, capsys):
    gp = tmp_path/"gpu_power"; gp.mkdir()
    _mk_gp(gp/"opA_r9700_s0.csv", power=150, busy=90, counter="ramp")
    _mk_gp(gp/"opD_r9700_s0.csv", power=150, busy=1,  counter="ramp")   # trips busy gate
    _mk_gp(gp/"idle_r9700.csv", power=20, busy=0, counter="none")
    _mk_gp(gp/"dispatch_baseline_r9700.csv", power=30, busy=4, counter="none")
    runs = tmp_path/"runs.csv"; _runs_csv(runs)
    out = tmp_path/"enriched.csv"
    pe.enrich_runs(runs, gp, out, backend="amdsmi")
    captured = capsys.readouterr().out
    assert "mean gfx_busy=1.0%" in captured              # placement gate fired
    rows = {r["run_id"]: r for r in csv.DictReader(open(out))}
    a = rows["opA_r9700_s0_r0"]
    assert a["window_energy_method"] == "counter_delta_uj"
    assert a["gfx_busy_mean_pct"] == "90.00"
    assert a["idle_energy_J"] == "600.000000"
    assert a["dispatch_energy_J"] == "900.000000"
