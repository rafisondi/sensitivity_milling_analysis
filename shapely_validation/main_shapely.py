"""One coupled robot milling pass, shapely engine - the dexel `main.py` with one
line changed: `sim_coupled_shapely.simulate` in place of `analysis.sim_coupled.simulate`.

    python main_shapely.py --rpm 3333 --teeth 4 --fz 0.18 --part-length 200

Steps 1-5 and 7 (job, plant, prediction, feedforward, everything to disk) are
`analysis.*`, imported unchanged - only step 6, the coupled pass itself, differs.
See `sim_coupled_shapely.py` for what that one line actually changes and what it
does not support (`--steady-state`, which this raises on rather than silently
ignoring).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))    # this dir: shapely_engine, sim_coupled_shapely
sys.path.insert(0, str(REPO))                                # the workspace: analysis, fastsim, robotsim...

import analysis                                              # noqa: E402
from analysis import config as acfg                          # noqa: E402
from analysis import (feedforward, forces, plant, report,     # noqa: E402
                      save, stability, toolpath)

import sim_coupled_shapely as sim_coupled                    # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None,
                   help="base RunConfig JSON (default: configs/base.json)")
    p.add_argument("--name", default=None,
                   help="output directory under out/ (default: from the settings)")
    p.add_argument("--out", default=None, help="output root (default: out/)")

    g = p.add_argument_group("operating point")
    g.add_argument("--ap", type=float, default=None, metavar="MM",
                   help="axial depth = part height [mm]")
    g.add_argument("--ae", type=float, default=None, metavar="MM",
                   help="radial engagement [mm]")
    g.add_argument("--rpm", type=float, default=None, help="spindle speed")
    g.add_argument("--feed", type=float, default=None, metavar="MM_S",
                   help="max feed [mm/s]; regenerates the toolpath")
    g.add_argument("--teeth", type=int, default=None, metavar="N",
                   help="flutes on the cutter")
    g.add_argument("--feed-profile", default="flying",
                   choices=("ramped", "flying"),
                   help="'flying' (default) opens at full feed and stays there")
    g.add_argument("--fz", type=float, default=None, metavar="MM",
                   help="chip load [mm/tooth]; sets the feed from the rpm and "
                        "the tooth count. Mutually exclusive with --feed")
    g.add_argument("--steps-per-tooth", type=float, default=None, metavar="N",
                   help="set sim_dt from the TOOTH PERIOD instead of --sim-dt")
    g.add_argument("--ktc", type=float, default=None, help="Ktc [N/mm^2]")
    g.add_argument("--krc", type=float, default=None, help="Krc [N/mm^2]")
    g.add_argument("--sim-dt", type=float, default=None, help="integration step [s]")
    g.add_argument("--raster", type=float, default=None, metavar="MM",
                   help="unused by the shapely engine; kept so the same argv "
                        "sweep.py builds for the dexel side still parses here")
    g.add_argument("--robot-model", default=None,
                   help="which arm (default: joints123)")
    g.add_argument("--part-length", type=float, default=None, metavar="MM",
                   help="stock length [mm] - how much cut there is to measure")
    g.add_argument("--part-width", type=float, default=None, metavar="MM",
                   help="stock width [mm] - cheap to cut down, unlike length")

    g = p.add_argument_group("the model")
    g.add_argument("--ds", type=float, default=2.0, metavar="MM",
                   help="arc-length spacing of the stability nodes [mm]")
    g.add_argument("--coupling", default="both",
                   choices=("both", "damping", "stiffness", "none"),
                   help="which cut terms the prediction closes")

    g = p.add_argument_group("the DC compensation")
    g.add_argument("--no-compensate", action="store_true")
    g.add_argument("--comp-gain", type=float, default=1.0)
    g.add_argument("--comp-axes", default="xy")

    g = p.add_argument_group("running")
    g.add_argument("--predict-only", action="store_true",
                   help="the linear side only - no time-domain pass")
    g.add_argument("--steady-state", action="store_true",
                   help="NOT SUPPORTED by the shapely engine - raises")
    g.add_argument("--csv-decimate", type=int, default=10, metavar="N")
    g.add_argument("--no-plots", action="store_true")
    g.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def default_name(cfg, compensated) -> str:
    m = cfg.milling()
    return (f"shapely_ap{cfg.part.height_mm:g}_ae{m.radial_engagement_mm:g}"
            f"_rpm{m.spindle_rpm:g}_z{m.n_teeth:g}_f{m.feed_mm_s:g}"
            f"_{'comp' if compensated else 'plain'}").replace(".", "p")


def main(argv=None):
    a = parse_args(argv)
    if a.steady_state:
        raise SystemExit("--steady-state is not supported by the shapely engine "
                         "- see sim_coupled_shapely.py")
    lines = []

    def say(text=""):
        print(text, flush=True)
        lines.append(str(text))

    # ── 1. the job ───────────────────────────────────────────────────────────
    cfg = acfg.load_base(a.config)
    cfg = acfg.apply_operating_point(
        cfg, ap_mm=a.ap, ae_mm=a.ae, rpm=a.rpm, feed_mm_s=a.feed,
        n_teeth=a.teeth, fz_mm=a.fz, steps_per_tooth=a.steps_per_tooth,
        Ktc=a.ktc, Krc=a.krc, sim_dt=a.sim_dt, raster_mm=a.raster,
        robot_model=a.robot_model,
        part_length_mm=a.part_length, part_width_mm=a.part_width)
    acfg.check_compliant(cfg)

    compensated = not a.no_compensate
    name = a.name or default_name(cfg, compensated)
    out_root = Path(a.out) if a.out else analysis.OUT
    d = save.run_dir(out_root, name)

    cfg = toolpath.ensure(cfg, profile=a.feed_profile, verbose=True)

    setup = stability.prepare(cfg, ds_mm=a.ds, verbose=a.verbose)
    say(setup.summary())

    # ── 2. the linear replacement for the arm ────────────────────────────────
    receptance = plant.receptance_from_robot(cfg)
    say(plant.compliance_report(receptance, setup.scene))

    # ── 3-4. the prediction, node by node along the trajectory ───────────────
    stab = stability.predict(setup, receptance, coupling=a.coupling,
                             verbose=a.verbose)
    ap_crit = stability.critical_depth(stab)
    row = {"run": name, "compensated": compensated, "engine": "shapely",
           **acfg.operating_point_row(cfg),
           **stability.prediction_row(stab, ap_crit)}
    wT = np.asarray(stab.omega_T(), float)
    row["omega_T_max"] = float(np.nanmax(wT)) if np.isfinite(wT).any() else np.nan
    row["tooth_trunc_frac"] = 0.5 * row["omega_T_max"] ** 2

    # ── 5. the mean-force feedforward ────────────────────────────────────────
    ff = feedforward.build(setup, receptance, axes=a.comp_axes,
                           gain=a.comp_gain if compensated else 0.0,
                           verbose=a.verbose)
    say("comp     TCP DC compensation: tau_ff = -J^T F0(s), carried on the motors")
    say(ff.summary(ae_mm=setup.mill.radial_engagement_mm))
    row.update(ff.row())

    # ── 6. the truth (shapely) ───────────────────────────────────────────────
    run, force_rows = None, []
    if a.predict_only:
        say("\nsim      skipped (--predict-only)")
    else:
        say("")
        run = sim_coupled.simulate(cfg, ff if compensated else None,
                                   verbose=a.verbose)
        row.update(report.chatter_metrics(run, setup.mill,
                                          modes_hz=receptance.modes_hz))
        row.update(forces.force_error(run, stab, setup.mill))
        row.update(report.agreement(row))
        force_rows = forces.table(run, stab, setup.mill)
        say("")
        say(forces.summary(row))

    # ── 7. everything to disk ────────────────────────────────────────────────
    cfg.save(d / "config.json")
    receptance.in_workpiece(setup.scene).save(d / "plant.npz")
    written = {"config": d / "config.json", "plant": d / "plant.npz"}
    written.update(save.save_stability(d, stab, stability.table(stab, ap_crit)))
    written.update(save.save_forces(d, ff, force_rows))
    if run is not None:
        written.update(save.save_run(d, run, decimate=a.csv_decimate))

    row["feed_profile"] = a.feed_profile
    row["csv_decimate"] = int(a.csv_decimate)
    row["coupling"] = a.coupling
    row["ds_mm"] = float(a.ds)
    save.write_json(d / "summary.json", row)

    if not a.no_plots:
        from analysis import plots
        for k, v in plots.run_figures(d, stab, ff, run, setup, receptance).items():
            written[k] = v
    say("")
    say(f"out      {d}")
    for k, v in written.items():
        say(f"         {k:<18} {Path(v).relative_to(d)}")
    (d / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return row


if __name__ == "__main__":
    main()
