"""Open the pass mid-cut on a precut stock, and look at stability with no entry or exit.

    python steadystate_precut/demo.py                       s0 = 40 mm, 50 mm of cut
    python steadystate_precut/demo.py --s0 30 --length 40 --ap 2 --rpm 5000
    python steadystate_precut/demo.py --tap                 + the tapped twin pass
    python steadystate_precut/demo.py --s0 20,40 --rpm 2000,3333,5000 --tap
    python steadystate_precut/demo.py --predict-only        the job and the linear side

WHAT IS SET UP, IN ORDER (see `steadystate_precut.precut`)

    1  the stock is milled up to s0 along the edge (s0 from the edge's start
       corner), everything the tool disc swept on its way there already gone
    2  the job is placed so the TCP at the config's start pose sits on s0 at
       full depth - the robot's start pose, and the plant's linearisation, are
       where the steady cut starts
    3  the toolpath runs straight along the edge from s0 for --length mm at full
       feed from the first sample, and stops while still fully engaged; it is
       refused if the disc would reach the far corner
    4  the engine carves to a few revolutions short of s0 and mills the rest
       rigidly, so the surface ahead of the tool and its chip history are those
       of a running cut at t = 0
    5  the arm opens on its tracking equilibrium at feed velocity, springs preset
       for the mean force F0, motors carrying -J^T F0: preloaded and compensated

and then runs the same prediction / coupled pass / report `main.py` does, one
`out/<name>_s<s0>_rpm<rpm>/` directory per cell, plus `figures/precut.png`.

`--tap` is the stability measurement proper: a preloaded steady cut has no
transient to fit, so the pass is run again with a short tap on the TCP and the
two are subtracted (`analysis.pulse`), giving `sim_sigma_1_s` against the
eigenvalue at the tap node. With `--tap` the plant is linearised at the tap
point unless `--linearize-at` says otherwise.

`--s0` and `--rpm` take comma-separated lists and run as a grid, like
`move_wp_constantRobPose/demo.py`.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis import config as acfg                            # noqa: E402
from analysis.pulse import Pulse                                # noqa: E402
from steadystate_precut import pipeline, precut                 # noqa: E402

GEN_CONFIG_DIR = REPO / "configs" / "_steadystate_precut"


def parse_float_list(spec) -> list:
    """`"2000,3000"` -> [2000.0, 3000.0]; `None` -> `[None]` (config default)."""
    if spec is None:
        return [None]
    return [float(v) for v in str(spec).split(",") if v.strip()]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(REPO / "configs" / "base.json"))

    g = p.add_argument_group("the precut job")
    g.add_argument("--s0", default="40", metavar="MM,...",
                   help="how far along the edge the stock is already milled, from "
                        "its start corner [mm]; the run starts there (default 40)")
    g.add_argument("--length", type=float, default=50.0, metavar="MM",
                   help="steady cut simulated from s0 [mm] (default 50)")
    g.add_argument("--edge", type=int, default=None,
                   help="edge to cut (default: the config's path.start_edge)")
    g.add_argument("--ds", type=float, default=2.0, metavar="MM",
                   help="stability node spacing [mm]; the prediction's stock is "
                        "precut to s0 - ds")

    g = p.add_argument_group("operating point")
    g.add_argument("--rpm", default=None, metavar="RPM,...",
                   help="one spindle speed, or a comma-separated list")
    g.add_argument("--ae", type=float, default=None, metavar="MM")
    g.add_argument("--ap", type=float, default=None, metavar="MM")
    g.add_argument("--feed", type=float, default=None, metavar="MM_S")
    g.add_argument("--fz", type=float, default=None, metavar="MM",
                   help="chip load; sets the feed from rpm and teeth instead")
    g.add_argument("--teeth", type=int, default=None)
    g.add_argument("--steps-per-tooth", type=float, default=None)
    g.add_argument("--raster", type=float, default=None, metavar="MM")
    g.add_argument("--part-length", type=float, default=None, metavar="MM",
                   help="a longer part allows a longer --length")
    g.add_argument("--part-width", type=float, default=None, metavar="MM")

    g = p.add_argument_group("the model")
    g.add_argument("--plant", default="reduced", choices=("reduced", "full"))
    g.add_argument("--coupling", default="both",
                   choices=("both", "damping", "stiffness", "none"))
    g.add_argument("--linearize-at", type=float, default=None, metavar="FRAC",
                   help="linearise at FRAC of the steady path instead of the "
                        "start pose (s0); with --tap it defaults to --tap-at")
    g.add_argument("--comp-axes", default="xy")

    g = p.add_argument_group("the tap (twin run)")
    g.add_argument("--tap", action="store_true",
                   help="run the tapped twin as well and fit the tap's decay rate")
    g.add_argument("--tap-at", type=float, default=0.3, metavar="FRAC",
                   help="where along the steady path (default 0.3 - early, so "
                        "the ring-down has the rest of the path)")
    g.add_argument("--tap-force", type=float, default=20.0, metavar="N")
    g.add_argument("--tap-ms", type=float, default=5.0, metavar="MS")
    g.add_argument("--tap-dir", default="auto", metavar="X,Y,Z|auto")

    g = p.add_argument_group("running")
    g.add_argument("--predict-only", action="store_true",
                   help="set up the job and print the linear side; no pass")
    g.add_argument("--name", default="steady_precut",
                   help="prefix for the out/<name>_s<s0>_rpm<rpm>/ directories")
    g.add_argument("--out", default=None, help="output root (default: out/)")
    g.add_argument("--save", action="store_true",
                   help="also write each job's RunConfig to configs/_steadystate_precut/")
    g.add_argument("--csv-decimate", type=int, default=10)
    g.add_argument("--no-plots", action="store_true")
    g.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def build(a, s0, rpm):
    cfg = acfg.load_base(a.config, toolpath=None)
    cfg = acfg.apply_operating_point(
        cfg, ap_mm=a.ap, ae_mm=a.ae, rpm=rpm, feed_mm_s=a.feed,
        n_teeth=a.teeth, fz_mm=a.fz, steps_per_tooth=a.steps_per_tooth,
        part_length_mm=a.part_length, part_width_mm=a.part_width,
        raster_mm=a.raster)
    acfg.check_compliant(cfg)
    return precut.build_job(cfg, s0, a.length, edge_index=a.edge, ds_mm=a.ds)


def _tap(a) -> Pulse:
    d = ((1.0, 1.0, 1.0) if str(a.tap_dir).lower() == "auto"
         else tuple(float(v) for v in str(a.tap_dir).split(",")))
    if len(d) != 3:
        raise SystemExit(f"--tap-dir needs three components, got {a.tap_dir!r}")
    return Pulse(at_frac=a.tap_at, force_N=a.tap_force,
                 duration_s=1e-3 * a.tap_ms, dir_w=d)


def _predict_only(job, a):
    """The job and its linear verdict, without the time-domain pass."""
    from analysis import plant, stability
    import main as run_main

    setup = precut.prepare(job, verbose=a.verbose)
    print(setup.summary())
    print(job.summary())
    rec = plant.receptance_from_robot(job.cfg)
    print(plant.compliance_report(rec, setup.scene))
    stab = stability.predict(setup, rec, coupling=a.coupling, verbose=a.verbose)
    ap_crit = stability.critical_depth(stab)
    row = {**acfg.operating_point_row(job.cfg), **job.row(),
           **stability.prediction_row(stab, ap_crit)}
    print(run_main._prediction_block(row))
    eng = np.asarray(stab.engaged, bool)
    g = np.asarray(stab.growth_rate, float)[eng]
    print(f"         steady check: growth over the {int(eng.sum())} engaged "
          f"nodes spans {g.min():+.2f} .. {g.max():+.2f} 1/s "
          f"(a flat line means no entry/exit is left in the path)")
    return row


def main(argv=None):
    a = parse_args(argv)
    out_root = Path(a.out) if a.out else REPO / "out"
    tap = _tap(a)
    lin_at = a.linearize_at
    if a.tap and lin_at is None:
        lin_at = a.tap_at

    rows = []
    for rpm in parse_float_list(a.rpm):
        for s0 in parse_float_list(a.s0):
            job = build(a, s0, rpm)
            tag = f"_s{s0:g}" + ("" if rpm is None else f"_rpm{rpm:g}")
            name = f"{a.name}{tag}".replace(".", "p")
            print(f"\n===== {name} =====")
            if a.save:
                job.cfg.save(GEN_CONFIG_DIR / f"{name}.json")
            if a.predict_only:
                rows.append((name, _predict_only(job, a)))
                continue
            kw = dict(name=name, out_root=out_root, plant_kind=a.plant,
                      coupling=a.coupling, linearize_at=lin_at,
                      comp_axes=a.comp_axes, csv_decimate=a.csv_decimate,
                      no_plots=a.no_plots, verbose=a.verbose,
                      auto_dir=str(a.tap_dir).lower() == "auto")
            row = (pipeline.run_twin(job, tap=tap, tap_at=a.tap_at, **kw)
                   if a.tap else pipeline.run(job, **kw))
            rows.append((name, row))
    summarise(rows)
    return rows


def summarise(rows):
    if not rows:
        return
    print("\n[steady-state cells]")
    print(f"  {'cell':<34}{'pred trim 1/s':>14}{'sim env 1/s':>13}{'R2':>6}"
          f"{'tap sim 1/s':>13}{'tap lin 1/s':>13}   verdict")
    for name, r in rows:
        f = lambda k: float(r.get(k, np.nan)) if r.get(k) is not None else np.nan
        verdict = (r.get("sim_state") or
                   ("DIVERGED" if r.get("sim_diverged") else
                    "UNSTABLE" if r.get("sim_unstable") else
                    "stable" if r.get("sim_valid") else "n/a"))
        print(f"  {name:<34}{f('pred_growth_max_trim_1_s'):>14.2f}"
              f"{f('sim_growth_1_s'):>13.2f}{f('sim_growth_r2'):>6.2f}"
              f"{f('sim_sigma_1_s'):>13.2f}{f('pred_growth_mid_1_s'):>13.2f}"
              f"   {verdict}")


if __name__ == "__main__":
    main()
