"""Pure rotation, no pose change - the FULL (alpha x ae) grid, steel feed.

    python sweep_rotation_ae_fixedpose.py                    linear map only (fast)
    python sweep_rotation_ae_fixedpose.py --ap 1 --no-plots  the threshold/reachability screen
    python sweep_rotation_ae_fixedpose.py --simulate --cells 90/60,0/10
    python sweep_rotation_ae_fixedpose.py --resume

`sweep_rotation_fixedpose.py` fixed `ae` at 60% and only swept `alpha`, to keep
the first real coupled check on the pose-vs-rotation decomposition cheap. This
is that sweep's full generalisation: the same (alpha, ae) grid
`sweep_rotation_ae.py` uses (19 x 4 = 76 cells), through the SAME wrist
re-clocking fix (`sweep_rotation_fixedpose.config_for`) so every cell holds the
receptance fixed and only turns the cut direction and the radial engagement.

FZ DEFAULTS TO 0.12 mm/tooth, NOT 0.18 - every other sweep in this project uses
0.18 (a value that was never itself the point of comparison), but for THIS
grid the user asked for something more in tune with steel, so the default
moved. Pass --fz to override.

THE DEPTH IS NOT SET HERE. This script's job is prediction and reachability
only - `--ap` is a single probe depth (1 mm by default, cheap, matches how
`AP_MM` was originally derived for `sweep_rotation_ae.py`) used to read off
`pred_ap_crit_mid_mm` per cell. Which cells get a coupled pass, and at what
depth (chosen relative to that cell's own threshold, capped so nothing runs at
an unrealistic axial depth), is a decision made AFTER looking at this sweep's
output - see the module docstring in `sweep_rotation_fixedpose.py` for why a
uniform ratio-to-threshold blows up in the high-leverage region.
"""

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import main as run_main                                       # noqa: E402
import analysis                                               # noqa: E402
from analysis import save                                     # noqa: E402
from analysis import pulse as pmod                            # noqa: E402
from sweep_rotation_ae import MID, diameter_mm, measure        # noqa: E402
from sweep_rotation_fixedpose import config_for                # noqa: E402

ALPHA_DEG = tuple(float(a) for a in range(0, 181, 10))
AE_FRAC = (0.10, 0.30, 0.60, 0.80)

RPM = 5000.0
TEETH = 4
FZ = 0.12                        # steel - see the module docstring
AP_MM = 1.0                      # cheap probe depth for the threshold screen
SIM_DT = 1.09375e-4
RASTER_MM = 0.01
FEED_PROFILE = "flying"


def cells() -> list:
    return [(alpha, frac) for frac in AE_FRAC for alpha in ALPHA_DEG]


def cell_name(alpha, frac) -> str:
    return f"alpha{alpha:03.0f}_ae{100 * frac:02.0f}_fixedpose"


def parse_cells(spec: str) -> set:
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if part:
            a, pct = part.split("/")
            out.add((float(a), round(float(pct) / 100.0, 6)))
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="rotation_ae_fixedpose")
    p.add_argument("--simulate", action="store_true")
    p.add_argument("--cells", default=None, metavar="ALPHA/AE%,...")
    p.add_argument("--no-twin", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--replot", action="store_true")
    g = p.add_argument_group("operating point (held across the grid)")
    g.add_argument("--rpm", type=float, default=RPM)
    g.add_argument("--teeth", type=int, default=TEETH)
    g.add_argument("--fz", type=float, default=FZ, metavar="MM")
    g.add_argument("--ap", type=float, default=AP_MM, metavar="MM",
                   help=f"axial depth, same for every cell (default {AP_MM:g} mm "
                        "probe - see the module docstring)")
    g.add_argument("--ap-table", default=None, metavar="JSON",
                   help="path to a {'alpha/ae%%': ap_mm} JSON - overrides --ap "
                        "per cell, for a curated depth after reading the probe")
    g.add_argument("--raster", type=float, default=RASTER_MM, metavar="MM")
    g.add_argument("--sim-dt", type=float, default=SIM_DT, metavar="S")
    g.add_argument("--feed-profile", default=FEED_PROFILE, choices=("ramped", "flying"))
    g.add_argument("--ds", type=float, default=2.0, metavar="MM")
    g.add_argument("--part-length", type=float, default=None, metavar="MM")
    g.add_argument("--part-width", type=float, default=None, metavar="MM")
    g = p.add_argument_group("the tap")
    g.add_argument("--pulse-force", type=float, default=20.0, metavar="N")
    g.add_argument("--pulse-ms", type=float, default=5.0, metavar="MS")
    g.add_argument("--pulse-dir", default="auto", metavar="X,Y,Z|auto")
    g = p.add_argument_group("output")
    g.add_argument("--no-cell-plots", action="store_true")
    g.add_argument("--no-plots", action="store_true")
    g.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def ap_for(alpha, frac, a, ap_table: dict) -> float:
    if ap_table is not None:
        key = f"{alpha:g}/{100*frac:g}"
        if key in ap_table:
            return float(ap_table[key])
    return float(a.ap)


def argv_for(alpha, frac, a, runs_dir, ap_table, *, simulate, tapped) -> list:
    name = cell_name(alpha, frac) + ("_pulse" if tapped else "")
    ap = ap_for(alpha, frac, a, ap_table)
    argv = ["--config", str(config_for(alpha)),
            "--name", name, "--out", str(runs_dir),
            "--rpm", repr(float(a.rpm)), "--teeth", str(int(a.teeth)),
            "--fz", repr(float(a.fz)), "--ae", repr(frac * diameter_mm()),
            "--ap", repr(ap),
            "--raster", repr(float(a.raster)), "--sim-dt", repr(float(a.sim_dt)),
            "--feed-profile", str(a.feed_profile), "--ds", repr(float(a.ds)),
            "--linearize-at", repr(MID),
            "--pulse-force", repr(float(a.pulse_force)),
            "--pulse-ms", repr(float(a.pulse_ms)), "--pulse-dir", str(a.pulse_dir)]
    if tapped:
        argv += ["--pulse-at", repr(MID)]
    if a.part_length is not None:
        argv += ["--part-length", repr(float(a.part_length))]
    if a.part_width is not None:
        argv += ["--part-width", repr(float(a.part_width))]
    if not simulate:
        argv += ["--predict-only"]
    if a.no_plots or a.no_cell_plots:
        argv += ["--no-plots"]
    if a.verbose:
        argv += ["--verbose"]
    return argv


def _load_summary(d: Path):
    f = d / "summary.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def _num(row, key):
    try:
        v = float(row.get(key))
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def run_cell(alpha, frac, a, runs_dir, ap_table, *, simulate, twin):
    base_dir = runs_dir / cell_name(alpha, frac)
    pulse_dir = runs_dir / (cell_name(alpha, frac) + "_pulse")

    row = _load_summary(base_dir) if a.resume else None
    need_base = row is None or (simulate and not row.get("sim_valid")
                                and not row.get("sim_diverged"))
    if need_base:
        try:
            row = run_main.main(argv_for(alpha, frac, a, runs_dir, ap_table,
                                         simulate=simulate, tapped=False))
        except Exception as exc:
            traceback.print_exc()
            return ({"run": cell_name(alpha, frac), "alpha_deg": float(alpha),
                     "ae_frac": float(frac), "error": str(exc)}, "FAILED")
    if simulate and twin and row.get("sim_valid") and not row.get("sim_diverged"):
        pulsed = _load_summary(pulse_dir) if a.resume else None
        if pulsed is None or (not pulsed.get("sim_valid")
                              and not pulsed.get("sim_diverged")):
            run_main.main(argv_for(alpha, frac, a, runs_dir, ap_table,
                                   simulate=True, tapped=True))

    row = dict(row)
    row.update({"alpha_deg": float(alpha), "ae_frac": float(frac),
                "ae_pct": 100.0 * float(frac),
                "ae_mm": float(frac) * diameter_mm(), "simulated": bool(simulate),
                "ap_used_mm": ap_for(alpha, frac, a, ap_table)})
    if simulate:
        row.update(measure(row, base_dir, pulse_dir))
    row["lin_state"] = pmod.lin_state(_num(row, "pred_growth_mid_1_s"))
    row["sim_state"] = pmod.sim_state(row) if simulate else "unmeasured"
    return row, ("reused" if not need_base else "ran")


def main(argv=None):
    a = parse_args(argv)
    sweep_dir = Path(analysis.OUT) / a.name
    runs_dir = sweep_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    ap_table = None
    if a.ap_table:
        ap_table = json.loads(Path(a.ap_table).read_text(encoding="utf-8"))

    want_sim = parse_cells(a.cells) if a.cells else None
    plan = cells()

    print(f"sweep    {a.name}: {len(plan)} cells, alpha "
          f"{ALPHA_DEG[0]:g}..{ALPHA_DEG[-1]:g} deg x ae {[100*f for f in AE_FRAC]}%")
    print(f"         {a.rpm:g} rpm, {a.teeth} teeth, fz {a.fz:g} mm/tooth"
          f"{' (per-cell ap table)' if ap_table else f', ap {a.ap:g} mm'}")
    if a.simulate:
        n = len(want_sim) if want_sim else len(plan)
        print(f"         SIMULATING {'the ' + str(n) + ' named cell(s)' if want_sim else 'every cell'}")
    else:
        print("         prediction only (pass --simulate for coupled passes)")
    print()

    rows = []
    for i, (alpha, frac) in enumerate(plan, 1):
        simulate = bool(a.simulate)
        if simulate and want_sim is not None:
            simulate = (alpha, round(frac, 6)) in want_sim
        print(f"[{i:3d}/{len(plan)}] alpha {alpha:5.0f}  ae {100*frac:4.0f}%"
              f"{'   [COUPLED PASS]' if simulate else ''}")
        row, status = run_cell(alpha, frac, a, runs_dir, ap_table,
                               simulate=simulate, twin=not a.no_twin)
        print(f"         {status}")
        rows.append(row)

    save.write_csv(sweep_dir / "sweep.csv", rows)
    save.write_json(sweep_dir / "sweep.json",
                    {"name": a.name, "n_cells": len(plan), "alpha_deg": list(ALPHA_DEG),
                     "ae_frac": list(AE_FRAC), "rpm": float(a.rpm), "teeth": int(a.teeth),
                     "fz_mm": float(a.fz), "ap_mm": float(a.ap),
                     "ap_table": bool(ap_table), "sim_dt": float(a.sim_dt),
                     "feed_profile": str(a.feed_profile), "linearize_at": MID,
                     "simulated": bool(a.simulate), "twin": not a.no_twin,
                     "fixed_pose": True})

    n_ok = sum(1 for r in rows if "error" not in r)
    n_fail = len(rows) - n_ok
    print()
    print(f"out      {sweep_dir}")
    print(f"         sweep.csv            {len(rows)} rows  ({n_fail} failed)")
    return rows


if __name__ == "__main__":
    main()
