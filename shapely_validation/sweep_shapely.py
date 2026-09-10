"""The tooth-passing grid, shapely engine - a costed-down cousin of `sweep.py`.

    python sweep_shapely.py                          the linear side, no sim
    python sweep_shapely.py --simulate                every cell (long - see COST)
    python sweep_shapely.py --simulate --cells 10000/8,10000/4
    python sweep_shapely.py --simulate --resume

THE GRID, AND WHY IT IS SMALLER THAN `sweep.py`'S

`sweep.py` runs `GRID_RPM = (1000, 2000, 3333, 5000, 7500, 10000)` at fixed chip
load, `600x15` (later `500x60`) mm of stock, dexel `dt = 1.09375e-4` s. The
shapely engine costs roughly 60x more per step than dexel on the source repo's
machine (`MillingBenchmarkStability` README) - measured on THIS machine at
`ap = 1 mm` (one axial slice) it is closer to 6-8x, but that is still enough
that the full six-speed grid at 500 mm would run for many hours per slow cell.
So this sweep is scoped down on purpose, all three cuts made together rather
than separately:

    RPM_LIST    (1000, 3000, 5000, 10000)    four speeds, not six
    part length  200 mm, not 500/600         steady state starts ~80 mm in
                                             (measured on the dexel v5 sweep),
                                             so 200 mm leaves ~120 mm settled
    SIM_DT       1.05e-4 s                   close to sweep.py's own odd-valued
                                             step, kept for the same reason -
                                             see that module's docstring

GRID ONLY, NO `rows` DESIGN. `sweep.py`'s `rows` design exists to null-test the
model at a fixed feed while `N` moves; that question does not need re-asking
with a second engine, so this sweep is `grid` (fixed chip load) only - the same
16 (rpm, teeth) cells `compare_error.py` already has dexel numbers for in
`out/tooth_passing_v5`, so every cell here has something to be checked against.

COST. Per-cell time scales as (part length + 50 mm of leads) / feed / SIM_DT *
per-step cost, and `feed = fz * rpm * teeth / 60` at fixed `fz` - so it is
dominated by the SLOW, FEW-TOOTH corner. Measured on this machine at
ap = 1 mm (~8 ms/step in real cutting):

    1000 rpm, 1 tooth    ~envisioned worst cell, ~100 min
    10000 rpm, 8 teeth   fastest cell, ~1-2 min

`--cells` restricts the SIMULATION to named (rpm, teeth) pairs so the grid can
be run in stages rather than as one multi-hour block; the linear side (no
`--simulate`) always covers the full 16 regardless, since it costs seconds.
"""

import argparse
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO))

import main_shapely as run_main                               # noqa: E402
import analysis                                               # noqa: E402
from analysis import save                                     # noqa: E402

#: The rpm axis - four speeds, see the module docstring for why not six.
RPM_LIST = (1000.0, 3000.0, 5000.0, 10000.0)

#: The tooth axis, same as `sweep.py`'s.
TEETH = (1, 2, 4, 8)

#: Chip load held fixed - the same operating point `sweep.py`'s `grid` design
#: and the base config both use.
FZ = 0.18

#: Stock, cut down from v5's 500x60 - see the module docstring.
PART_LENGTH_MM = 200.0
PART_WIDTH_MM = 60.0

SIM_DT = 1.05e-4
FEED_PROFILE = "flying"


def cells() -> list:
    return [(rpm, n) for rpm in RPM_LIST for n in TEETH]


def cell_name(rpm: float, teeth: int) -> str:
    return f"shapely_grid_fz{FZ:g}_rpm{rpm:g}_z{teeth}".replace(".", "p")


def parse_cells(spec: str) -> set:
    """`"10000/8,10000/4"` -> {(10000.0, 8), (10000.0, 4)}."""
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        rpm, n = part.split("/")
        out.add((float(rpm), int(n)))
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="tooth_passing_shapely",
                   help="sweep directory under out/ (default: tooth_passing_shapely)")
    p.add_argument("--simulate", action="store_true",
                   help="run the coupled pass at each cell; without this the "
                        "sweep is the linear side only, seconds not hours")
    p.add_argument("--cells", default=None, metavar="RPM/N,...",
                   help="only these cells, e.g. 10000/8,10000/4 - applies to the "
                        "SIMULATION; the linear side always covers all 16")
    p.add_argument("--resume", action="store_true",
                   help="skip any cell that already has a summary.json")
    p.add_argument("--ap", type=float, default=None, metavar="MM",
                   help="axial depth [mm] (default: the config's own)")
    p.add_argument("--ae", type=float, default=None, metavar="MM")
    p.add_argument("--sim-dt", type=float, default=SIM_DT, metavar="S")
    p.add_argument("--part-length", type=float, default=PART_LENGTH_MM, metavar="MM")
    p.add_argument("--part-width", type=float, default=PART_WIDTH_MM, metavar="MM")
    p.add_argument("--feed-profile", default=FEED_PROFILE,
                   choices=("ramped", "flying"))
    p.add_argument("--ds", type=float, default=2.0, metavar="MM")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def argv_for(rpm, teeth, a, sweep_runs: Path, simulate: bool) -> list:
    argv = ["--name", cell_name(rpm, teeth), "--out", str(sweep_runs),
            "--rpm", repr(float(rpm)), "--teeth", str(int(teeth)),
            "--fz", repr(float(FZ)),
            "--sim-dt", repr(float(a.sim_dt)),
            "--part-length", repr(float(a.part_length)),
            "--part-width", repr(float(a.part_width)),
            "--ds", repr(float(a.ds)), "--feed-profile", str(a.feed_profile)]
    if a.ap is not None:
        argv += ["--ap", repr(float(a.ap))]
    if a.ae is not None:
        argv += ["--ae", repr(float(a.ae))]
    if not simulate:
        argv += ["--predict-only"]
    if a.no_plots:
        argv += ["--no-plots"]
    if a.verbose:
        argv += ["--verbose"]
    return argv


def main(argv=None):
    a = parse_args(argv)
    sweep_dir = Path(analysis.OUT) / a.name
    runs_dir = sweep_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    want_sim = parse_cells(a.cells) if a.cells else None
    plan = cells()

    print(f"sweep    {a.name}: {len(plan)} cells (shapely engine), "
          f"rpm {RPM_LIST} x teeth {TEETH}")
    print(f"         fz {FZ:g} mm/tooth fixed, part {a.part_length:g}x"
          f"{a.part_width:g} mm, dt {a.sim_dt:g} s, {a.feed_profile} feed")
    if a.simulate:
        n = len(want_sim) if want_sim else len(plan)
        print(f"         SIMULATING {'the ' + str(n) + ' named cell(s)' if want_sim else 'every cell - see the module docstring for the cost'}")
    else:
        print("         prediction only (pass --simulate for coupled passes)")
    print()

    rows, skipped = [], []
    for i, (rpm, n) in enumerate(plan, 1):
        name = cell_name(rpm, n)
        d = runs_dir / name

        simulate = bool(a.simulate)
        if simulate and want_sim is not None:
            simulate = (float(rpm), int(n)) in want_sim

        reusable = False
        if a.resume and (d / "summary.json").exists():
            import json
            cached = json.loads((d / "summary.json").read_text(encoding="utf-8"))
            reusable = bool(cached.get("sim_valid")) or not simulate
        if reusable:
            row = cached
            row["design"] = "grid"
            row.setdefault("simulated", bool(row.get("sim_valid", False)))
            rows.append(row)
            print(f"[{i:2d}/{len(plan)}] {name}: reused")
            continue

        fz = FZ
        feed = fz * rpm * n / 60.0
        print(f"[{i:2d}/{len(plan)}] {name}   fz {fz:.4f} mm/tooth, feed "
              f"{feed:.1f} mm/s{'   [COUPLED PASS - shapely]' if simulate else ''}")
        try:
            row = run_main.main(argv_for(rpm, n, a, runs_dir, simulate))
        except Exception as exc:                      # a cell is data, not a stop
            traceback.print_exc()
            print(f"         ! {name} failed: {exc}")
            rows.append({"run": name, "design": "grid", "n_teeth": int(n),
                        "spindle_rpm": float(rpm), "error": str(exc)})
            continue
        row["design"] = "grid"
        row["simulated"] = simulate
        rows.append(row)

    save.write_csv(sweep_dir / "sweep.csv", rows)
    save.write_json(sweep_dir / "sweep.json",
                    {"name": a.name, "design": "grid", "n_cells": len(plan),
                     "teeth": list(TEETH), "rpm_list": list(RPM_LIST),
                     "fz_mm": FZ, "sim_dt": float(a.sim_dt),
                     "part_length_mm": float(a.part_length),
                     "part_width_mm": float(a.part_width),
                     "feed_profile": str(a.feed_profile),
                     "engine": "shapely", "simulated": bool(a.simulate),
                     "skipped_simulation": skipped})

    print()
    print(f"out      {sweep_dir}")
    print(f"         sweep.csv            {len(rows)} rows")
    return rows


if __name__ == "__main__":
    main()
