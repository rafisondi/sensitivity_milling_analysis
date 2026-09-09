"""Does tooth passing matter to a model that averages it away?

    python sweep.py                          the linear grid, plots, no simulation
    python sweep.py --simulate               every cell, coupled  (long)
    python sweep.py --simulate --cells 1000/1,3333/1,3333/4,10000/8
    python sweep.py --resume                 skip cells already on disk
    python sweep.py --design rows            only the fixed-feed rows

THE QUESTION

`F0`, `K_cut` and `C_cut` see the operating point only through two groups:

    N * fz = 60 feed / rpm      ->  F0 and K_cut
    rpm                         ->  C_cut          (the 1/Omega in A_cut_0)

So the linear model has TWO effective parameters where the engine has three, and
the extra one is precisely the tooth-passing frequency `rpm N / 60`. Everything
the ripple does is averaged away before the model sees it. This sweep asks
whether that average is free.

THE TWO DESIGNS, AND WHY BOTH

`rows` - fixed FEED, one rpm, tooth count varied. Inside a row the model's
    output is bit-identical: same `F0`, same `K_cut`, same `C_cut`, because only
    `N` moved and the model sees `N` only through `N fz`. It is a NULL TEST, so
    anything the engine does differently across a row is tooth passing and
    nothing else. The cost is that `fz` swings as `1/N` within the row, which
    caps how wide a row can be before the chip is unphysical at one end and
    under the raster at the other.

`grid` - fixed CHIP LOAD over the whole rpm x teeth plane. The model does move
    here (`F0` scales with `N`), so it is a scaling test rather than a null test,
    but it covers the plane and every cell is physically the same cut per tooth.

RESOLUTION: A FIXED `dt`, NOT A FIXED NUMBER OF STEPS PER TOOTH

The engagement arc is fixed in ANGLE by `ae` - 68 deg at ae = 5 mm on a 16 mm
cutter, whatever the tooth count. So holding the steps per TOOTH fixed does not
hold the cut equally resolved; it makes the steps through the cut scale with `N`
(3.8 at one tooth, 30 at eight), which confounds numerical resolution with the
very axis this sweep varies. At 20 steps per tooth that showed up as a 29-point
swing in the one-tooth force error, purely from where the `dt` clamp landed.

A fixed `dt` removes `N` from that: steps through the cut become a function of
the spindle speed alone, 113 at 1000 rpm down to 11 at 10000 rpm. The price is
that the cells no longer cost the same - the pass duration goes as `1/(rpm N)`,
so `1000/z1` is 520k steps against 6k for `10000/z8` - and that one cell,
`10000 rpm` with eight teeth, sits at 7.5 steps per TOOTH, just under the 8 where
`MillConfig.summary` warns the tooth harmonics alias. Everything else clears it.

`--steps-per-tooth` is still there to run the old rule deliberately.

WHAT IS RECORDED AND WHAT IS NOT

Every run writes its full summary; this script collects them into one CSV. The
prediction side is reduced to scalars as usual. The measurement side is NOT
reduced to a stability verdict here: under a tooth-passing sweep the band-pass
window `analysis.report` fits its envelope in (3.5-57.9 Hz on this arm) collides
with the tooth-passing frequency itself in the slow, few-tooth corner, and a
forced sinusoid inside the pass band flattens the envelope. `sweep_validity.png`
plots where each cell sits against that and the engine's two resolution limits,
so the collision is visible as an operating-point fact. Deciding what to do
about it is for after the passes exist.
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import main as run_main                                       # noqa: E402
import analysis                                               # noqa: E402
from analysis import save                                     # noqa: E402

#: The rpm axis of the fixed-chip-load grid.
GRID_RPM = (1000.0, 2000.0, 3333.0, 5000.0, 7500.0, 10000.0)

#: The tooth axis, shared by both designs. `analysis.plots` pins a colour to
#: each of these, so a partial sweep keeps the same identity as a full one.
TEETH = (1, 2, 4, 8)

#: Chip load the grid holds fixed [mm/tooth] - the base job's own value, so the
#: 3333 rpm / 4 tooth cell reproduces the committed baseline cut.
GRID_FZ = 0.18

#: The fixed-feed rows: the null test. Only these rpms are wide enough to carry
#: all four tooth counts - at 1000 and 2000 rpm the one- and two-tooth cells
#: would need 1.2-2.4 mm of chip per tooth on an 8 mm-radius cutter.
ROW_RPM = (3333.0, 5000.0, 7500.0)
ROW_FEED = 40.0

#: Integration step [s], held fixed across the sweep. See the module docstring
#: for why this rather than a fixed number of steps per tooth.
SIM_DT = 1.0e-4

#: Dexel raster [mm]. At the grid's `fz` this is 18 chip pixels in every cell,
#: comfortably clear of the 2 px floor where the binary raster loses the chip.
RASTER_MM = 0.01

#: The feed profile every cell runs. "flying" holds `v_max` from the first sample,
#: so the deviation carries the cut and nothing else - the smoothstep ramps of
#: "ramped" are through air, but they deflect this arm by up to 168 um on their
#: own, which swamps the cut on the fast cells. See `analysis.feedplan`.
FEED_PROFILE = "flying"

#: Cells whose chip falls below this many raster pixels are not SIMULATED - the
#: binary raster loses the chip and the engine's force stops meaning anything.
#: The linear side is unaffected (it never sees the raster) and still runs.
MIN_CHIP_PX = 2.0

#: Nor are cells whose chip is a large fraction of the cutter radius, where the
#: mechanistic force model is out of its range whatever the raster does.
MAX_FZ_FRAC_OF_RADIUS = 0.10


def cells(design: str) -> list:
    """The (design, rpm, teeth, fz, feed) cells of the requested sweep."""
    out = []
    if design in ("grid", "both"):
        for rpm in GRID_RPM:
            for n in TEETH:
                out.append({"design": "grid", "rpm": rpm, "teeth": n,
                            "fz": GRID_FZ, "feed": None})
    if design in ("rows", "both"):
        for rpm in ROW_RPM:
            for n in TEETH:
                out.append({"design": "rows", "rpm": rpm, "teeth": n,
                            "fz": None, "feed": ROW_FEED})
    return out


def cell_name(c) -> str:
    if c["design"] == "grid":
        tag = f"fz{c['fz']:g}".replace(".", "p")
    else:
        tag = f"feed{c['feed']:g}".replace(".", "p")
    return f"{c['design']}_{tag}_rpm{c['rpm']:g}_z{c['teeth']}".replace(".", "p")


def cell_fz(c) -> float:
    """Chip load [mm/tooth] this cell will actually run at."""
    if c["fz"] is not None:
        return float(c["fz"])
    return 60.0 * float(c["feed"]) / (c["rpm"] * c["teeth"])


def simulable(c, raster_mm: float, radius_mm: float = 8.0):
    """(ok, why not) - whether a coupled pass at this cell would mean anything."""
    fz = cell_fz(c)
    if fz / raster_mm < MIN_CHIP_PX:
        return False, (f"chip {fz:.4f} mm = {fz / raster_mm:.1f} px, under "
                       f"{MIN_CHIP_PX:g}")
    if fz > MAX_FZ_FRAC_OF_RADIUS * radius_mm:
        return False, (f"chip {fz:.3f} mm is over "
                       f"{100 * MAX_FZ_FRAC_OF_RADIUS:g}% of the tool radius")
    return True, ""


def parse_cells(spec: str) -> set:
    """`"1000/1,3333/4"` -> {(1000.0, 1), (3333.0, 4)}."""
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        rpm, teeth = part.split("/")
        out.add((float(rpm), int(teeth)))
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="tooth_passing",
                   help="sweep directory under out/ (default: tooth_passing)")
    p.add_argument("--design", default="both", choices=("grid", "rows", "both"),
                   help="fixed chip load over the plane, fixed feed rows, or both")
    p.add_argument("--simulate", action="store_true",
                   help="run the coupled pass at each cell as well as the "
                        "prediction; without this the sweep is the linear side "
                        "only, which is seconds rather than hours")
    p.add_argument("--cells", default=None, metavar="RPM/N,...",
                   help="only these cells, e.g. 1000/1,3333/4 - applies to the "
                        "SIMULATION; the linear side always covers the design")
    p.add_argument("--resume", action="store_true",
                   help="skip any cell that already has a summary.json")
    p.add_argument("--ap", type=float, default=None, metavar="MM",
                   help="axial depth [mm] (default: the config's 1.0)")
    p.add_argument("--ae", type=float, default=None, metavar="MM",
                   help="radial engagement [mm]")
    p.add_argument("--raster", type=float, default=RASTER_MM, metavar="MM",
                   help="dexel raster [mm]; also sets the chip-pixel guard")
    p.add_argument("--sim-dt", type=float, default=SIM_DT, metavar="S",
                   help="integration step [s], the same in every cell")
    p.add_argument("--steps-per-tooth", type=float, default=None, metavar="N",
                   help="set the step from the TOOTH PERIOD instead of fixing it, "
                        "dt = 60/(rpm N spt). Overrides --sim-dt; this makes the "
                        "steps through the cut scale with the tooth count, so it "
                        "is the wrong rule for a tooth-passing sweep - see the "
                        "module docstring")
    p.add_argument("--feed-profile", default=FEED_PROFILE,
                   choices=("ramped", "flying"),
                   help="'flying' (default) opens at full feed, so the deviation "
                        "carries only the cut; 'ramped' adds the feed ramps and "
                        "the arm's tracking error through them")
    p.add_argument("--ds", type=float, default=2.0, metavar="MM",
                   help="arc-length spacing of the stability nodes [mm]")
    p.add_argument("--no-cell-plots", action="store_true",
                   help="skip the per-run figures; the sweep figures are still "
                        "drawn")
    p.add_argument("--no-plots", action="store_true",
                   help="skip every figure")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def argv_for(c, a, sweep_runs: Path, simulate: bool) -> list:
    """The `main.py` command line for one cell."""
    argv = ["--name", cell_name(c), "--out", str(sweep_runs),
            "--rpm", repr(float(c["rpm"])), "--teeth", str(int(c["teeth"])),
            "--raster", repr(float(a.raster)), "--ds", repr(float(a.ds)),
            "--feed-profile", str(a.feed_profile)]
    if a.steps_per_tooth is not None:
        argv += ["--steps-per-tooth", repr(float(a.steps_per_tooth))]
    else:
        argv += ["--sim-dt", repr(float(a.sim_dt))]
    if c["fz"] is not None:
        argv += ["--fz", repr(float(c["fz"]))]
    else:
        argv += ["--feed", repr(float(c["feed"]))]
    if a.ap is not None:
        argv += ["--ap", repr(float(a.ap))]
    if a.ae is not None:
        argv += ["--ae", repr(float(a.ae))]
    if not simulate:
        argv += ["--predict-only"]
    if a.no_plots or a.no_cell_plots:
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
    plan = cells(a.design)

    print(f"sweep    {a.name}: {len(plan)} cells, design {a.design!r}")
    step_txt = (f"{a.steps_per_tooth:g} steps/tooth"
                if a.steps_per_tooth is not None else f"dt {a.sim_dt:g} s")
    print(f"         teeth {TEETH}, {step_txt}, raster {a.raster:g} mm, "
          f"{a.feed_profile} feed")
    if a.simulate:
        n = len(want_sim) if want_sim else len(plan)
        print(f"         SIMULATING {'the ' + str(n) + ' named cell(s)' if want_sim else 'every cell'}")
    else:
        print("         prediction only (pass --simulate for coupled passes)")
    print()

    rows, skipped = [], []
    for i, c in enumerate(plan, 1):
        name = cell_name(c)
        d = runs_dir / name
        fz = cell_fz(c)

        simulate = bool(a.simulate)
        if simulate and want_sim is not None:
            simulate = (float(c["rpm"]), int(c["teeth"])) in want_sim
        if simulate:
            ok, why = simulable(c, a.raster)
            if not ok:
                print(f"[{i:2d}/{len(plan)}] {name}: prediction only - {why}")
                skipped.append({"cell": name, "reason": why})
                simulate = False

        # A resumed cell is only reusable if it already holds what is being asked
        # for. A prediction-only run leaves a summary.json like any other, so
        # resuming on existence alone would silently skip every cell the caller
        # has just asked to SIMULATE.
        reusable = False
        if a.resume and (d / "summary.json").exists():
            import json
            cached = json.loads((d / "summary.json").read_text(encoding="utf-8"))
            reusable = bool(cached.get("sim_valid")) or not simulate
        if reusable:
            row = cached
            # `design` and `simulated` belong to the SWEEP, not to the run, so
            # they are not in the run's own summary and have to be re-attached
            # here - without them a resumed sweep groups into nothing and draws
            # no figures.
            row["design"] = c["design"]
            row.setdefault("simulated", bool(row.get("sim_valid", False)))
            rows.append(row)
            print(f"[{i:2d}/{len(plan)}] {name}: reused")
            continue

        tpf = c["rpm"] * c["teeth"] / 60.0
        print(f"[{i:2d}/{len(plan)}] {name}   fz {fz:.4f} mm/tooth, "
              f"TPF {tpf:.0f} Hz{'   [COUPLED PASS]' if simulate else ''}")
        try:
            row = run_main.main(argv_for(c, a, runs_dir, simulate))
        except Exception as exc:                      # a cell is data, not a stop
            traceback.print_exc()
            print(f"         ! {name} failed: {exc}")
            rows.append({"run": name, "design": c["design"],
                         "n_teeth": int(c["teeth"]),
                         "spindle_rpm": float(c["rpm"]), "error": str(exc)})
            continue
        row["design"] = c["design"]
        row["simulated"] = simulate
        rows.append(row)

    save.write_csv(sweep_dir / "sweep.csv", rows)
    save.write_json(sweep_dir / "sweep.json",
                    {"name": a.name, "design": a.design, "n_cells": len(plan),
                     "teeth": list(TEETH), "grid_rpm": list(GRID_RPM),
                     "grid_fz_mm": GRID_FZ, "row_rpm": list(ROW_RPM),
                     "row_feed_mm_s": ROW_FEED,
                     "sim_dt": (None if a.steps_per_tooth is not None
                                else float(a.sim_dt)),
                     "steps_per_tooth": (None if a.steps_per_tooth is None
                                         else float(a.steps_per_tooth)),
                     "feed_profile": str(a.feed_profile),
                     "raster_mm": float(a.raster), "simulated": bool(a.simulate),
                     "skipped_simulation": skipped})

    written = {}
    if not a.no_plots:
        from analysis import plots
        for design in sorted({r.get("design") for r in rows if r.get("design")}):
            sub = [r for r in rows if r.get("design") == design
                   and r.get("n_teeth") is not None]
            for k, v in plots.sweep_figures(sweep_dir / design, sub).items():
                written[f"{design}/{k}"] = v
            # The two sides on one axis, for whatever has actually been run.
            for k, v in plots.overlay_figures(sweep_dir / design, sub).items():
                written[f"{design}/{k}"] = v

    print()
    print(f"out      {sweep_dir}")
    print(f"         sweep.csv            {len(rows)} rows")
    for k, v in written.items():
        print(f"         {k:<28} {Path(v).relative_to(sweep_dir)}")
    return rows


if __name__ == "__main__":
    main()
