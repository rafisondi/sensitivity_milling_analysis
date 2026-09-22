"""`sweep_rotation_ae_fixedpose.py`'s grid, run as precut steady-state passes.

    python steadystate_precut/sweep_rotation_ae.py                      linear map only
    python steadystate_precut/sweep_rotation_ae.py --simulate --rpm 5000 --ap 10 \\
        --fz 0.12 --part-length 150 --resume
    python steadystate_precut/sweep_rotation_ae.py --simulate --rpm 5000,6500,8000 \\
        --ap 10 --fz 0.12 --part-length 150 --resume

The same cells - alpha 0..180 deg in 10 deg steps x ae 10/30/60/80 % of D - and
the same poses: every alpha uses `sweep_rotation_fixedpose.config_for(alpha)`,
which turns the workpiece and re-clocks the wrist so the arm sits in one pose
at every orientation. What changes is the pass: instead of lead-in, entry, edge,
exit and lead-out, each cell opens mid-cut on a stock already milled up to
`--s0`, preloaded and compensated, and cuts `--length` mm of steady edge
(`steadystate_precut.precut`). The tapped twin run gives `sim_sigma_1_s`
against the eigenvalue at the tap node, exactly as in the fixed-pose sweep, so
the two `sweep.csv` files line up cell for cell.

DEFAULT GEOMETRY. `--s0 15 --length 120 --tap-at 0.5` puts the tap at s = 75 mm,
the middle of a 150 mm edge - where the full-pass sweep taps too - and ends the
cut at s = 135 mm, clear of the far corner for every ae in the grid (the widest
disc reach is R = 8 mm at ae 80 %, so the steady span ends at 142 mm). That
needs `--part-length 150`; on the stock 100 mm part lower both.

`--rpm` takes a list; each speed is its own sweep directory,
`out/<name>_rpm<N>/`, with `runs/<cell>/`, `sweep.csv` and `sweep.json` like
the fixed-pose sweep writes.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import analysis                                                 # noqa: E402
from analysis import config as acfg                             # noqa: E402
from analysis import growth, save                               # noqa: E402
from analysis import pulse as pmod                              # noqa: E402
from analysis.pulse import Pulse                                # noqa: E402
from steadystate_precut import pipeline, precut                 # noqa: E402
from sweep_rotation_fixedpose import config_for                 # noqa: E402

ALPHA_DEG = tuple(float(a) for a in range(0, 181, 10))
AE_FRAC = (0.10, 0.30, 0.60, 0.80)

RPM = 5000.0
TEETH = 4
FZ = 0.12
AP_MM = 1.0
SIM_DT = 1.09375e-4
RASTER_MM = 0.01
S0_MM = 15.0
LENGTH_MM = 120.0
TAP_AT = 0.5


def cells() -> list:
    return [(alpha, frac) for frac in AE_FRAC for alpha in ALPHA_DEG]


def cell_name(alpha, frac) -> str:
    return f"alpha{alpha:03.0f}_ae{100 * frac:02.0f}_steady"


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
    p.add_argument("--name", default="steady_rotation_ae_fixedpose",
                   help="sweep directory prefix; each rpm writes out/<name>_rpm<N>/")
    p.add_argument("--simulate", action="store_true")
    p.add_argument("--cells", default=None, metavar="ALPHA/AE%,...",
                   help="simulate only these cells (the rest stay prediction-only)")
    p.add_argument("--no-twin", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="reuse every finished pass on disk")

    g = p.add_argument_group("the precut job")
    g.add_argument("--s0", type=float, default=S0_MM, metavar="MM")
    g.add_argument("--length", type=float, default=LENGTH_MM, metavar="MM")
    g.add_argument("--tap-at", type=float, default=TAP_AT, metavar="FRAC",
                   help="tap point and linearisation pose, fraction of the "
                        "steady path (default 0.5)")

    g = p.add_argument_group("operating point (held across the grid)")
    g.add_argument("--rpm", default=str(RPM), metavar="RPM,...")
    g.add_argument("--teeth", type=int, default=TEETH)
    g.add_argument("--fz", type=float, default=FZ, metavar="MM")
    g.add_argument("--ap", type=float, default=AP_MM, metavar="MM")
    g.add_argument("--ap-table", default=None, metavar="JSON",
                   help="{'alpha/ae%%': ap_mm} JSON, overrides --ap per cell")
    g.add_argument("--raster", type=float, default=RASTER_MM, metavar="MM")
    g.add_argument("--sim-dt", type=float, default=SIM_DT, metavar="S")
    g.add_argument("--ds", type=float, default=2.0, metavar="MM")
    g.add_argument("--part-length", type=float, default=None, metavar="MM")
    g.add_argument("--part-width", type=float, default=None, metavar="MM")

    g = p.add_argument_group("the tap")
    g.add_argument("--pulse-force", type=float, default=20.0, metavar="N")
    g.add_argument("--pulse-ms", type=float, default=5.0, metavar="MS")
    g.add_argument("--pulse-dir", default="auto", metavar="X,Y,Z|auto")

    g = p.add_argument_group("output")
    g.add_argument("--no-cell-plots", action="store_true")
    g.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def ap_for(alpha, frac, a, ap_table) -> float:
    if ap_table is not None:
        key = f"{alpha:g}/{100 * frac:g}"
        if key in ap_table:
            return float(ap_table[key])
    return float(a.ap)


def job_for(alpha, frac, rpm, a, ap_table):
    cfg = acfg.load_base(config_for(alpha), toolpath=None)
    D = float(cfg.mill.diameter_mm)
    cfg = acfg.apply_operating_point(
        cfg, ap_mm=ap_for(alpha, frac, a, ap_table), ae_mm=frac * D, rpm=rpm,
        n_teeth=a.teeth, fz_mm=a.fz, sim_dt=a.sim_dt, raster_mm=a.raster,
        part_length_mm=a.part_length, part_width_mm=a.part_width)
    acfg.check_compliant(cfg)
    return precut.build_job(cfg, a.s0, a.length, ds_mm=a.ds), D


def _tap(a) -> Pulse:
    d = ((1.0, 1.0, 1.0) if str(a.pulse_dir).lower() == "auto"
         else tuple(float(v) for v in str(a.pulse_dir).split(",")))
    return Pulse(at_frac=a.tap_at, force_N=a.pulse_force,
                 duration_s=1e-3 * a.pulse_ms, dir_w=d)


def _load(d: Path):
    f = d / "summary.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def _finished(row) -> bool:
    return bool(row) and (bool(row.get("sim_valid")) or bool(row.get("sim_diverged")))


def _predict_row(job, a):
    """The linear side only, at the tap node - for cells not simulated."""
    from analysis import plant, stability
    setup = precut.prepare(job, verbose=a.verbose)
    rec, info = plant.receptance_at_fraction(setup, a.tap_at, verbose=a.verbose)
    stab = stability.predict(setup, rec, verbose=a.verbose)
    ap_crit = stability.critical_depth(stab)
    return {**acfg.operating_point_row(job.cfg), **job.row(), **info,
            **stability.prediction_row(stab, ap_crit),
            **stability.mid_node_row(stab, ap_crit, a.tap_at)}


def run_cell(alpha, frac, rpm, a, runs_dir, ap_table, *, simulate, twin):
    name = cell_name(alpha, frac)
    base_dir, pulse_dir = runs_dir / name, runs_dir / f"{name}_pulse"
    job, D = job_for(alpha, frac, rpm, a, ap_table)
    status = "predicted"

    if not simulate:
        row = _predict_row(job, a)
        row["lin_state"] = pmod.lin_state(row.get("pred_growth_mid_1_s"))
        row["sim_state"] = "unmeasured"
    else:
        kw = dict(out_root=runs_dir, tap=_tap(a), tap_at=a.tap_at,
                  linearize_at=a.tap_at, no_plots=a.no_cell_plots,
                  verbose=a.verbose, auto_dir=str(a.pulse_dir).lower() == "auto")
        base = _load(base_dir) if a.resume else None
        status = "reused"
        if not _finished(base):
            base = pipeline.run(job, name=name, tapped=False, **kw)
            status = "ran"
        pulsed = None
        if twin and base.get("sim_valid") and not base.get("sim_diverged"):
            pulsed = _load(pulse_dir) if a.resume else None
            if not _finished(pulsed):
                pulsed = pipeline.run(job, name=f"{name}_pulse", tapped=True, **kw)
                status = "ran"
        row = pipeline.twin_row(base, base_dir, pulsed, pulse_dir)
        g = growth.score(base_dir, ae_mm=frac * D)
        row.update({k: g.get(k) for k in ("plateau_verdict", "plateau_g_1_s",
                                          "plateau_floor_1_s", "peak_pct_ae",
                                          "ring_g_1_s", "ring_r2") if k in g})
        row["sim_state"] = pmod.sim_state(row)
        save.write_json(base_dir / "summary.json", row)
        print(pipeline.describe_tap(row), flush=True)

    row.update({"alpha_deg": float(alpha), "ae_frac": float(frac),
                "ae_pct": 100.0 * float(frac), "ae_mm": float(frac) * D,
                "simulated": bool(simulate),
                "ap_used_mm": ap_for(alpha, frac, a, ap_table)})
    return row, status


def sweep_one_rpm(rpm, a, ap_table, want_sim):
    sweep_dir = Path(analysis.OUT) / f"{a.name}_rpm{rpm:g}"
    runs_dir = sweep_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    plan = cells()

    print(f"\nsweep    {sweep_dir.name}: {len(plan)} cells, alpha "
          f"{ALPHA_DEG[0]:g}..{ALPHA_DEG[-1]:g} deg x ae {[100 * f for f in AE_FRAC]}%")
    print(f"         {rpm:g} rpm, {a.teeth} teeth, fz {a.fz:g} mm/tooth, "
          f"{'per-cell ap table' if ap_table else f'ap {a.ap:g} mm'} | precut to "
          f"s0 {a.s0:g} mm, {a.length:g} mm of steady cut, tap at "
          f"{100 * a.tap_at:g}%")
    print()

    rows = []
    for i, (alpha, frac) in enumerate(plan, 1):
        simulate = bool(a.simulate) and (want_sim is None
                                         or (alpha, round(frac, 6)) in want_sim)
        print(f"[{i:3d}/{len(plan)}] alpha {alpha:5.0f}  ae {100 * frac:4.0f}%"
              f"{'   [STEADY COUPLED PASS]' if simulate else ''}", flush=True)
        try:
            row, status = run_cell(alpha, frac, rpm, a, runs_dir, ap_table,
                                   simulate=simulate, twin=not a.no_twin)
        except Exception as exc:
            traceback.print_exc()
            row, status = ({"run": cell_name(alpha, frac), "alpha_deg": float(alpha),
                            "ae_frac": float(frac), "error": str(exc)}, "FAILED")
        print(f"         {status}", flush=True)
        rows.append(row)
        # rewritten after every cell, so a long sweep can be read while it runs
        save.write_csv(sweep_dir / "sweep.csv", rows)

    save.write_json(sweep_dir / "sweep.json", {
        "name": sweep_dir.name, "n_cells": len(plan), "alpha_deg": list(ALPHA_DEG),
        "ae_frac": list(AE_FRAC), "rpm": float(rpm), "teeth": int(a.teeth),
        "fz_mm": float(a.fz), "ap_mm": float(a.ap), "ap_table": bool(ap_table),
        "sim_dt": float(a.sim_dt), "feed_profile": "flying",
        "linearize_at": float(a.tap_at), "tap_at": float(a.tap_at),
        "simulated": bool(a.simulate), "twin": not a.no_twin, "fixed_pose": True,
        "steady_state": True, "s0_mm": float(a.s0), "length_mm": float(a.length),
        "part_length_mm": a.part_length})
    n_fail = sum(1 for r in rows if "error" in r)
    print(f"\nout      {sweep_dir}\n         sweep.csv  {len(rows)} rows "
          f"({n_fail} failed)")
    return rows


def main(argv=None):
    a = parse_args(argv)
    ap_table = (json.loads(Path(a.ap_table).read_text(encoding="utf-8"))
                if a.ap_table else None)
    want_sim = parse_cells(a.cells) if a.cells else None
    return {rpm: sweep_one_rpm(rpm, a, ap_table, want_sim)
            for rpm in (float(v) for v in str(a.rpm).split(",") if v.strip())}


if __name__ == "__main__":
    main()
