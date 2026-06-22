"""
gpu_power.py — Standalone Linux/ROCm power+energy sampler for the r9700 engine.

Runs as its own process bracketing ONE harness invocation (all repeats), writing a
timestamped CSV that parse_energy.py later slices by harness window epochs.

Logs instantaneous power AND the SMU energy accumulator when available so the
counter-delta-vs-integration choice is made downstream.
"""

from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from pathlib import Path
from typing import Any

CSV_HEADER = [
    "sample_epoch",
    "mono_s",
    "power_w",
    "energy_uj",
    "gfx_busy_pct",
    "vram_used_mb",
    "sclk_mhz",
    "device_id",
]

_stop = False


def _handle_signal(signum: int, _frame: Any) -> None:
    global _stop
    _stop = True


def read_sample(handle: Any, device_id: int) -> dict[str, str]:
    """
    THE tower-verification point — all amd-smi reads live here.

    amdsmi Python API names and signatures vary across ROCm releases; this laptop
    checkout cannot exercise them. Tomorrow: fix only this function after running
    `python -c "import amdsmi; help(amdsmi)"` on the tower.

    Each field is wrapped in its own try/except so an unsupported counter yields ""
    instead of killing the sampler loop.
    """
    import amdsmi

    out: dict[str, str] = {
        "power_w": "",
        "energy_uj": "",
        "gfx_busy_pct": "",
        "vram_used_mb": "",
        "sclk_mhz": "",
    }

    # -------------------------------------------------------------------------
    # TODO(tower): confirm function names/signatures against installed ROCm.
    # Candidate APIs (version-dependent — verify on tower):
    #   Power:        amdsmi_get_power_info(handle, sensor_type)
    #                 amdsmi_get_power_info(handle, AmdSmiPowerType.POWER_TYPE_TOTAL_BOARD)
    #   Energy accum: amdsmi_get_energy_count(handle)  -> may be absent on RDNA
    #   GFX busy:     amdsmi_get_gpu_activity(handle)  -> .gfx_activity or similar
    #   VRAM:         amdsmi_get_gpu_memory_usage(handle, mem_type)
    #   SCLK:         amdsmi_get_clk_info(handle, clk_type) for AmdSmiClkType.GFX
    # -------------------------------------------------------------------------

    try:
        # Board / total GPU power (watts).
        power_info = amdsmi.amdsmi_get_power_info(handle, amdsmi.AmdSmiPowerType.POWER_TYPE_TOTAL_BOARD)
        if isinstance(power_info, dict):
            mw = power_info.get("average_socket_power") or power_info.get("power") or power_info.get("current")
            if mw is not None:
                out["power_w"] = str(float(mw) / 1000.0 if float(mw) > 1000 else float(mw))
        elif power_info is not None:
            out["power_w"] = str(float(power_info))
    except Exception:
        try:
            power_info = amdsmi.amdsmi_get_power_info(handle)
            if isinstance(power_info, dict) and "average_socket_power" in power_info:
                out["power_w"] = str(float(power_info["average_socket_power"]) / 1000.0)
        except Exception:
            pass

    try:
        # SMU energy accumulator (microjoules). Expected to fail on many RDNA cards.
        energy = amdsmi.amdsmi_get_energy_count(handle)
        if isinstance(energy, dict):
            uj = energy.get("energy_accumulator") or energy.get("energy") or energy.get("value")
            if uj is not None:
                out["energy_uj"] = str(int(uj))
        elif energy is not None:
            out["energy_uj"] = str(int(energy))
    except Exception:
        out["energy_uj"] = ""

    try:
        activity = amdsmi.amdsmi_get_gpu_activity(handle)
        if isinstance(activity, dict):
            pct = activity.get("gfx_activity") or activity.get("gfx_busy_percent")
            if pct is not None:
                out["gfx_busy_pct"] = str(float(pct))
        elif activity is not None:
            out["gfx_busy_pct"] = str(float(activity))
    except Exception:
        pass

    try:
        mem = amdsmi.amdsmi_get_gpu_memory_usage(handle, amdsmi.AmdSmiMemoryType.VRAM)
        if isinstance(mem, dict):
            used = mem.get("vram_used") or mem.get("used") or mem.get("memory_used")
            if used is not None:
                out["vram_used_mb"] = str(float(used) / (1024 * 1024))
        elif mem is not None:
            out["vram_used_mb"] = str(float(mem) / (1024 * 1024))
    except Exception:
        pass

    try:
        clk = amdsmi.amdsmi_get_clk_info(handle, amdsmi.AmdSmiClkType.GFX)
        if isinstance(clk, dict):
            mhz = clk.get("clk") or clk.get("current") or clk.get("clk_freq")
            if mhz is not None:
                out["sclk_mhz"] = str(float(mhz))
        elif clk is not None:
            out["sclk_mhz"] = str(float(clk))
    except Exception:
        pass

    _ = device_id  # reserved for per-device sensor selection if API requires it
    return out


def init(device_id: int) -> tuple[Any, Any]:
    """
    Initialize amdsmi and return (amdsmi_module, processor_handle).

    API names are version-dependent — verify on tower if import/init fails.
    """
    try:
        import amdsmi
    except ImportError:
        print(
            "[gpu_power] FATAL: cannot import amdsmi — source the ROCm env / amdsmi not importable",
            file=sys.stderr,
        )
        sys.exit(2)

    amdsmi.amdsmi_init()
    handles = amdsmi.amdsmi_get_processor_handles()
    if device_id < 0 or device_id >= len(handles):
        print(
            f"[gpu_power] FATAL: device_id={device_id} out of range "
            f"(found {len(handles)} processor handle(s))",
            file=sys.stderr,
        )
        amdsmi.amdsmi_shut_down()
        sys.exit(2)
    return amdsmi, handles[device_id]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ROCm discrete-GPU power sampler (amd-smi)")
    p.add_argument("--out", type=Path, required=True, help="Output CSV path")
    p.add_argument("--interval", type=float, default=0.1, help="Sample interval in seconds")
    p.add_argument("--device-id", type=int, default=0, help="GPU device index")
    p.add_argument("--flush-every", type=int, default=20, help="Flush CSV every N samples")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(
        f"[gpu_power] device={args.device_id} interval={args.interval}s -> {args.out}",
        flush=True,
    )

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    amdsmi_mod, handle = init(args.device_id)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_samples = 0
    energy_ever_populated = False
    wall_start = time.monotonic()

    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        f.flush()

        next_tick = time.monotonic()
        while not _stop:
            sample_epoch = time.time()
            mono_s = time.monotonic()
            fields = read_sample(handle, args.device_id)
            if fields.get("energy_uj"):
                energy_ever_populated = True

            writer.writerow(
                {
                    "sample_epoch": f"{sample_epoch:.6f}",
                    "mono_s": f"{mono_s:.6f}",
                    "power_w": fields["power_w"],
                    "energy_uj": fields["energy_uj"],
                    "gfx_busy_pct": fields["gfx_busy_pct"],
                    "vram_used_mb": fields["vram_used_mb"],
                    "sclk_mhz": fields["sclk_mhz"],
                    "device_id": str(args.device_id),
                }
            )
            n_samples += 1
            if n_samples % args.flush_every == 0:
                f.flush()

            next_tick += args.interval
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)

        f.flush()

    try:
        amdsmi_mod.amdsmi_shut_down()
    except Exception:
        pass

    wall_duration = time.monotonic() - wall_start
    print(
        f"[gpu_power] shutdown n_samples={n_samples} wall_s={wall_duration:.3f} "
        f"energy_uj_populated={'yes' if energy_ever_populated else 'no'}",
        flush=True,
    )

    if n_samples == 0:
        print("[gpu_power] FATAL: zero samples collected — trace is empty", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
