"""Does radial engagement matter to a model that sees it only as a mean force?

    python sweep_ae.py                       the linear side, plots, no simulation
    python sweep_ae.py --simulate            every cell, coupled  (long)
    python sweep_ae.py --simulate --cells 2,5,10
    python sweep_ae.py --resume              skip cells already on disk

THE QUESTION

`F0`, `K_cut` and `C_cut` are built from the engagement geometry at each node
(`stabsim.engagement`, called once per node by `analysis.stability.predict`), so
unlike the tooth-passing axis in `sweep.py`, `ae` is NOT averaged away before the
linear model sees it - the entry/exit angles it is built from move with `ae`
directly. What is still only approximate is the LINEARISATION itself: `F0 + K_cut
x + C_cut xdot` is a one-term truncation of a force that is a genuinely nonlinear
(and, past half immersion, discontinuous-in-angle) function of the chip
thickness. A wider engagement arc does not change what the model is told, but it
does change how much of the cut that truncation has to cover in one node - a
small ae has the tool engaged over a narrow arc where the force is close to
linear in the chip; a large one sweeps through more of the chip-thickness curve
per revolution. This sweep asks whether that shows up as a bigger prediction
error.

ONE AXIS, NOT TWO DESIGNS

`sweep.py` needs a null test and a scaling test because tooth count and rpm both
reach the model only through `N fz` and `rpm` - `ae` has no such confound. It
does not appear in `N fz` at all, so holding `fz` (and therefore `F0`, `K_cut`)
fixed while `ae` alone moves is already a clean isolation: every cell cuts the
same chip per tooth, and whatever changes in the comparison is attributable to
the engagement angle and nothing else. One design is enough.

WHAT IS HELD FIXED

`rpm`, `n_teeth` and `fz` sit at the committed baseline (3333 rpm, 4 teeth,
0.18 mm/tooth) throughout, so `ae = 5 mm` reproduces that baseline exactly and
every other cell is a pure engagement-angle change against it. `sim_dt` is the
same odd, non-revolution-dividing step `sweep.py` derived for this operating
point (see its module docstring) - reused rather than re-derived, because
holding rpm and teeth fixed here means the aliasing argument that produced it
is unaffected by which `ae` is run.

RANGE

`ae` runs from a light 1 mm up to `diameter_mm` (full slot, `tool_offset_mm`
runs 8 mm down to -8 mm) - cells asking for more than the diameter are skipped
before they reach the engine (`simulable`), since no physical single pass
engages more than that. `ap` stays at the base config's 1.0 mm throughout: a
larger `ae` already raises the mean radial force on its own, and stacking that
with the deeper cut the README already found to deflect this arm past where the
cut survives would confound the two.

THIS SCRIPT DOES NOT TOUCH `sweep.py`, `main.py` OR ANY OTHER EXISTING FILE - it
only calls `main.main()` the same way `sweep.py` does, into its own `out/`
subdirectory.

THE STALE-TOOLPATH TRAP, AND WHY THIS SCRIPT BUSTS THE CACHE ITSELF

`analysis.toolpath.ensure` keys its saved path purely on FEED (`name_for` stamps
`_v<speed>`, nothing else), but the waypoints it plans come from
`part.contour_waypoints(mill.tool_offset_mm, ...)` - and `tool_offset_mm = R -
ae`. Since this sweep holds `fz` (and therefore feed) fixed and moves `ae`
alone, every cell after the first asks `ensure` for the SAME file name and gets
the FIRST cell's geometry back, silently, regardless of `--ae`. Run without the
workaround below, this sweep reproduces one frozen offset eleven times and every
downstream number - prediction AND measurement, since the coupled pass also
replays that same cached path - comes out identical across the whole axis. That
is not a finding about the physics; it was checked here by running it once and
seeing a bit-identical `pred_growth_max_trim_1_s` across all eleven cells, which
is themselves-checkable, not a coincidence a real ae sweep would produce.

`_bust_toolpath_cache` deletes the one cache file this sweep's fixed feed maps
to before every cell, forcing `ensure` to rebuild it from THIS cell's `ae`. It
touches only the cache entry this sweep's own feed writes to, and only ever
deletes a file `ensure` can deterministically regenerate - nothing here is
unrecoverable. The same trap is latent in `sweep.py` itself for any future
`ap`/`Ktc`/`Krc` sweep run at one fixed feed; it just never trips there because
none of that sweep's parameters feed into `tool_offset_mm`.
"""

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import main as run_main                                       # noqa: E402
import analysis                                               # noqa: E402
from analysis import config as acfg                           # noqa: E402
from analysis import save, toolpath as _toolpath              # noqa: E402

#: The engagement axis [mm], on the base config's 16 mm cutter: 6-100% immersion.
#: 5 mm is the committed baseline and is kept in the list on purpose.
AE_MM = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0)

#: Held fixed at the base config's own operating point - see the module docstring.
RPM = 3333.0
TEETH = 4
FZ = 0.18

#: Same odd, non-revolution-dividing step `sweep.py` uses at this rpm/tooth pair.
SIM_DT = 1.09375e-4

RASTER_MM = 0.01
FEED_PROFILE = "flying"

#: A cell is not SIMULATED past this fraction of the cutter diameter - beyond it
#: there is no physical single pass, only a redefinition of what "one pass" means.
MAX_AE_FRAC_OF_DIAMETER = 1.0


def cell_name(ae: float) -> str:
    return f"ae{ae:g}_rpm{RPM:g}_z{TEETH}".replace(".", "p")


def simulable(ae: float, diameter_mm: float):
    """(ok, why not) - whether a coupled pass at this ae would mean anything."""
    if ae <= 0.0:
        return False, "ae must be positive"
    if ae > MAX_AE_FRAC_OF_DIAMETER * diameter_mm:
        return False, (f"ae {ae:g} mm exceeds the cutter diameter "
                       f"{diameter_mm:g} mm - not a single pass")
    return True, ""


def _bust_toolpath_cache(ae: float, feed_profile: str):
    """Delete the cached toolpath THIS cell's feed maps to, so it rebuilds at
    THIS cell's `ae` instead of replaying whatever offset wrote it first. See
    the module docstring - `ensure`'s cache key does not include `ae`."""
    cfg = acfg.load_base()
    cfg = acfg.apply_operating_point(cfg, ae_mm=ae, rpm=RPM, n_teeth=TEETH,
                                     fz_mm=FZ)
    name = _toolpath.name_for(cfg, profile=feed_profile)
    f = acfg.TOOLPATH_DIR / f"{name}.npz"
    if f.exists():
        f.unlink()


def parse_cells(spec: str) -> set:
    """`"2,5,10"` -> {2.0, 5.0, 10.0}."""
    return {float(v.strip()) for v in spec.split(",") if v.strip()}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="radial_engagement",
                   help="sweep directory under out/ (default: radial_engagement)")
    p.add_argument("--simulate", action="store_true",
                   help="run the coupled pass at each cell as well as the "
                        "prediction; without this the sweep is the linear side "
                        "only, which is seconds rather than hours")
    p.add_argument("--cells", default=None, metavar="AE,...",
                   help="only these ae values [mm], e.g. 2,5,10 - applies to the "
                        "SIMULATION; the linear side always covers the full axis")
    p.add_argument("--resume", action="store_true",
                   help="skip any cell that already has a summary.json")
    p.add_argument("--ap", type=float, default=None, metavar="MM",
                   help="axial depth [mm] (default: the config's 1.0)")
    p.add_argument("--raster", type=float, default=RASTER_MM, metavar="MM",
                   help="dexel raster [mm]")
    p.add_argument("--sim-dt", type=float, default=SIM_DT, metavar="S",
                   help="integration step [s], the same in every cell")
    p.add_argument("--feed-profile", default=FEED_PROFILE,
                   choices=("ramped", "flying"),
                   help="'flying' (default) opens at full feed, so the deviation "
                        "carries only the cut; 'ramped' adds the feed ramps and "
                        "the arm's tracking error through them")
    p.add_argument("--ds", type=float, default=2.0, metavar="MM",
                   help="arc-length spacing of the stability nodes [mm]")
    p.add_argument("--no-cell-plots", action="store_true",
                   help="skip the per-run figures; the sweep figure is still drawn")
    p.add_argument("--no-plots", action="store_true",
                   help="skip every figure")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def argv_for(ae: float, a, sweep_runs: Path, simulate: bool) -> list:
    """The `main.py` command line for one cell."""
    argv = ["--name", cell_name(ae), "--out", str(sweep_runs),
            "--rpm", repr(RPM), "--teeth", str(TEETH), "--fz", repr(FZ),
            "--ae", repr(float(ae)),
            "--raster", repr(float(a.raster)), "--ds", repr(float(a.ds)),
            "--sim-dt", repr(float(a.sim_dt)),
            "--feed-profile", str(a.feed_profile)]
    if a.ap is not None:
        argv += ["--ap", repr(float(a.ap))]
    if not simulate:
        argv += ["--predict-only"]
    if a.no_plots or a.no_cell_plots:
        argv += ["--no-plots"]
    if a.verbose:
        argv += ["--verbose"]
    return argv


def figure_ae(path: Path, rows: list):
    """ae vs growth rate (both sides) and vs critical depth - one PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in rows if r.get("ae_mm") is not None]
    ok.sort(key=lambda r: r["ae_mm"])
    ae = [r["ae_mm"] for r in ok]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(ae, [r.get("pred_growth_max_trim_1_s") for r in ok],
             "o-", label="predicted (linear)", color="tab:blue")
    sim_ae = [r["ae_mm"] for r in ok if r.get("sim_valid")]
    sim_g = [r.get("sim_growth_1_s") for r in ok if r.get("sim_valid")]
    if sim_ae:
        ax1.plot(sim_ae, sim_g, "s-", label="measured (coupled)", color="tab:red")
    ax1.axhline(0.0, color="black", lw=0.8)
    ax1.axvline(5.0, color="grey", lw=0.8, ls=":", label="committed baseline")
    ax1.set_xlabel("radial engagement  ae  [mm]")
    ax1.set_ylabel("growth rate  [1/s]")
    ax1.set_title("growth rate vs engagement")
    ax1.legend()

    ax2.plot(ae, [r.get("pred_ap_crit_trim_mm") for r in ok], "o-",
             color="tab:blue")
    ax2.axvline(5.0, color="grey", lw=0.8, ls=":")
    ax2.set_xlabel("radial engagement  ae  [mm]")
    ax2.set_ylabel("predicted critical depth  [mm]")
    ax2.set_title("ap_crit vs engagement")

    fig.suptitle(f"radial engagement sweep  ({RPM:g} rpm, {TEETH} teeth, "
                f"fz {FZ:g} mm/tooth)")
    fig.tight_layout()
    out = path / "ae_growth.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv=None):
    a = parse_args(argv)
    sweep_dir = Path(analysis.OUT) / a.name
    runs_dir = sweep_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    want_sim = parse_cells(a.cells) if a.cells else None

    print(f"sweep    {a.name}: {len(AE_MM)} cells, ae {AE_MM[0]:g}..{AE_MM[-1]:g} mm")
    print(f"         {RPM:g} rpm, {TEETH} teeth, fz {FZ:g} mm/tooth fixed, "
          f"dt {a.sim_dt:g} s, {a.feed_profile} feed")
    if a.simulate:
        n = len(want_sim) if want_sim else len(AE_MM)
        print(f"         SIMULATING {'the ' + str(n) + ' named cell(s)' if want_sim else 'every cell'}")
    else:
        print("         prediction only (pass --simulate for coupled passes)")
    print()

    rows, skipped = [], []
    for i, ae in enumerate(AE_MM, 1):
        name = cell_name(ae)
        d = runs_dir / name

        simulate = bool(a.simulate)
        if simulate and want_sim is not None:
            simulate = any(abs(ae - v) < 1e-9 for v in want_sim)
        if simulate:
            ok, why = simulable(ae, diameter_mm=16.0)
            if not ok:
                print(f"[{i:2d}/{len(AE_MM)}] {name}: prediction only - {why}")
                skipped.append({"cell": name, "reason": why})
                simulate = False

        reusable = False
        if a.resume and (d / "summary.json").exists():
            import json
            cached = json.loads((d / "summary.json").read_text(encoding="utf-8"))
            reusable = bool(cached.get("sim_valid")) or not simulate
        if reusable:
            row = cached
            row.setdefault("simulated", bool(row.get("sim_valid", False)))
            rows.append(row)
            print(f"[{i:2d}/{len(AE_MM)}] {name}: reused")
            continue

        print(f"[{i:2d}/{len(AE_MM)}] {name}   ae {ae:g} mm"
              f"{'   [COUPLED PASS]' if simulate else ''}")
        _bust_toolpath_cache(ae, a.feed_profile)
        try:
            row = run_main.main(argv_for(ae, a, runs_dir, simulate))
        except Exception as exc:                      # a cell is data, not a stop
            traceback.print_exc()
            print(f"         ! {name} failed: {exc}")
            rows.append({"run": name, "ae_mm": float(ae), "n_teeth": TEETH,
                        "spindle_rpm": RPM, "error": str(exc)})
            continue
        row["simulated"] = simulate
        rows.append(row)

    save.write_csv(sweep_dir / "sweep.csv", rows)
    save.write_json(sweep_dir / "sweep.json",
                    {"name": a.name, "n_cells": len(AE_MM), "ae_mm": list(AE_MM),
                     "rpm": RPM, "teeth": TEETH, "fz_mm": FZ,
                     "sim_dt": float(a.sim_dt), "feed_profile": str(a.feed_profile),
                     "raster_mm": float(a.raster), "simulated": bool(a.simulate),
                     "skipped_simulation": skipped})

    written = {}
    if not a.no_plots and any(r.get("ae_mm") is not None for r in rows):
        written["ae_growth.png"] = figure_ae(sweep_dir, rows)

    print()
    print(f"out      {sweep_dir}")
    print(f"         sweep.csv            {len(rows)} rows")
    for k, v in written.items():
        print(f"         {k:<20} {Path(v).relative_to(sweep_dir)}")
    return rows


if __name__ == "__main__":
    main()
