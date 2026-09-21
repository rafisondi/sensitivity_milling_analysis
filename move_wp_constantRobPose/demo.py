"""Place the workpiece on the current start pose, then move or turn it.

    python move_wp_constantRobPose/demo.py
    python move_wp_constantRobPose/demo.py --rpm 5000 --ae 8 --ap 2 --edge 0
    python move_wp_constantRobPose/demo.py --dy 15 --attack 45
    python move_wp_constantRobPose/demo.py --attack 30,60,90 --save
    python move_wp_constantRobPose/demo.py --dy 15 --attack 30,90,150 --simulate
    python move_wp_constantRobPose/demo.py --ap 10 --attack 0,10,...,180 \\
        --rpm 2000,3000,5000,7000,10000 --save --simulate

`--rpm` (like `--attack`) takes a comma-separated list, so a rpm x angle GRID
runs as one command: every rpm in the list gets its own full place + attack
sweep, each cell tagged `..._rpm<N>` in its slug and run directory so nothing
collides. `--ap` / `--ae` stay single-valued — sweep those by running the
command again, or add a list for them here the same way if that's wanted next.

Builds `configs/base.json` (the pose this project already uses), sets the
operating point, then:

    1  places the workpiece so the TCP sits on the middle of the edge about
       to be cut, at full ae/ap                       place_at_edge_midpoint
    2  optionally slides it by (--dx, --dy, --dz) mm along its own axes,
       robot pose untouched                                    move_workpiece
    3  optionally turns it by each --attack angle about the point under the
       tool, robot pose untouched                                attack_angle

and PROVES the "robot pose untouched" claim rather than just asserting it: it
solves `cfg.start.theta(...)` before and after each step and prints the
largest joint move, which should read ~0 deg (IK-residual noise only) for
every step here — that is the whole point of doing the move/turn on the
WORKPIECE side of the placement instead of on the robot's start pose.

`--save` writes one RunConfig JSON per step to
`configs/_move_wp_constantRobPose/`, ready for `python main.py --config ...`.

`--simulate` goes further and actually RUNS each step through `main.main` —
the full coupled dexel pass, not just the linear prediction (`--predict-only`
is never passed) — so what comes out is the same `out/<name>/` report
(figures, timeseries, summary.json) a normal `python main.py` run produces,
one per step. This is the "truth" side; it is slower (a real raster cut per
step) than just placing/moving/turning the job, which is instantaneous.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis import config as acfg                           # noqa: E402
from runconfig import RunConfig                                # noqa: E402
from robotsim import kinematics                                # noqa: E402
from move_wp_constantRobPose.workpiece_pose import (            # noqa: E402
    attack_angle, contact_point_w_mm, edge_pivot_mm, move_workpiece,
    place_at_edge_midpoint,
)

GEN_CONFIG_DIR = REPO / "configs" / "_move_wp_constantRobPose"


def theta_deg(cfg) -> np.ndarray:
    robot = kinematics.load_robot(cfg.scene)
    return np.rad2deg(cfg.start.theta(robot, cfg.scene))


def report_step(label: str, cfg, theta0_deg) -> np.ndarray:
    theta1_deg = theta_deg(cfg)
    move = float(np.max(np.abs(theta1_deg - theta0_deg))) if theta0_deg is not None else 0.0
    print(f"[{label}]")
    print("  " + cfg.scene.summary().replace("\n", "\n  "))
    print(f"  start joints    {np.round(theta1_deg, 4)} deg "
          f"| max move vs previous step: {move:.2e} deg")
    if cfg.scene.placed_by_hand:
        # Two different points, easy to conflate: `edge pivot` is a property
        # of the PART (fixed to the feature being cut — what attack_angle
        # turns about); `anchor point` is wherever the TCP's anchor pose
        # happens to sit right now (drifts with every move_workpiece call).
        # They coincide only right after place_at_edge_midpoint.
        print(f"  edge pivot      {np.round(edge_pivot_mm(cfg), 3)} mm  "
              f"(workpiece frame — fixed to the part; what attack_angle turns about)")
        print(f"  anchor point    {np.round(contact_point_w_mm(cfg), 3)} mm  "
              f"(workpiece frame — wherever the TCP's anchor pose is right now)")
    return theta1_deg


def simulate_step(slug: str, label: str, cfg, a) -> dict:
    """Save `cfg` and run it through `main.main` end to end. Returns its row.

    Imported lazily — `main.py` pulls in the whole analysis stack (and, for
    `--engine shapely`, an optional package), which only the `--simulate`
    path needs.
    """
    import main as run_main

    GEN_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg.save(GEN_CONFIG_DIR / f"{slug}.json")

    run_name = f"{a.name}_{slug}"
    out_root = a.out or str(REPO / "out")
    argv = ["--config", str(cfg_path), "--name", run_name, "--out", out_root]
    if a.engine != "dexel":
        argv += ["--engine", a.engine]
    if a.no_plots:
        argv += ["--no-plots"]
    if a.verbose:
        argv += ["--verbose"]

    print(f"\n===== simulating: {label} -> out/{run_name}/ =====")
    row = run_main.main(argv)
    return row


def summarise(rows: list):
    if not rows:
        return
    print("\n[simulated steps]")
    print(f"  {'step':<34}{'pred growth 1/s':>17}{'sim growth 1/s':>16}   verdict")
    for label, row in rows:
        pred = row.get("pred_growth_max_trim_1_s", float("nan"))
        sim = row.get("sim_growth_1_s", float("nan"))
        verdict = ("DIVERGED" if row.get("sim_diverged") else
                   "UNSTABLE" if row.get("sim_unstable") else
                   "stable" if row.get("sim_valid") else "n/a")
        print(f"  {label:<34}{pred:>17.2f}{sim:>16.2f}   {verdict}")


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
    p.add_argument("--edge", type=int, default=None,
                   help="edge to cut (default: the config's own path.start_edge)")
    p.add_argument("--rpm", default=None, metavar="RPM,...",
                   help="one spindle speed, or a comma-separated list to sweep "
                        "(default: the config's own rpm)")
    p.add_argument("--ae", type=float, default=None, metavar="MM")
    p.add_argument("--ap", type=float, default=None, metavar="MM")
    p.add_argument("--dx", type=float, default=0.0, metavar="MM", help="move along workpiece +X")
    p.add_argument("--dy", type=float, default=0.0, metavar="MM", help="move along workpiece +Y")
    p.add_argument("--dz", type=float, default=0.0, metavar="MM", help="move along workpiece +Z")
    p.add_argument("--attack", default=None, metavar="DEG,...",
                   help="one or more attack angles [deg] about the midpoint of "
                        "the edge being cut")
    p.add_argument("--save", action="store_true",
                   help="write a RunConfig JSON per step to configs/_move_wp_constantRobPose/")
    g = p.add_argument_group("simulate the whole pass (slow: a real coupled cut per step)")
    g.add_argument("--simulate", action="store_true",
                   help="run every step (placed, moved, each attack angle) through "
                        "main.main — the full coupled pass, not just the prediction")
    g.add_argument("--name", default="move_wp_constantRobPose",
                   help="prefix for the out/<name>_<step>/ run directories")
    g.add_argument("--out", default=None, help="output root (default: out/)")
    g.add_argument("--engine", default="dexel", choices=("dexel", "shapely"))
    g.add_argument("--no-plots", action="store_true")
    g.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def run_operating_point(rpm, a) -> tuple:
    """Place + optionally move + attack-sweep at one rpm. Returns (cfg, sim_rows)."""
    tag = "" if rpm is None else f"_rpm{rpm:g}"
    rpm_label = "" if rpm is None else f" | rpm {rpm:g}"

    cfg = RunConfig.load(a.config)
    cfg = acfg.apply_operating_point(cfg, ap_mm=a.ap, ae_mm=a.ae, rpm=rpm)

    print(f"\noperating point   ap {cfg.part.height_mm:g} mm | "
          f"ae {cfg.mill.radial_engagement_mm:g} mm | rpm {cfg.mill.spindle_rpm:g}\n")

    sim_rows = []
    theta0 = report_step(f"start pose (pose-derived placement, unchanged){rpm_label}",
                         cfg, None)

    cfg = place_at_edge_midpoint(cfg, edge_index=a.edge)
    theta1 = report_step(f"placed: TCP at the edge midpoint{rpm_label}", cfg, theta0)
    if a.save:
        GEN_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        cfg.save(GEN_CONFIG_DIR / f"01_placed{tag}.json")

    angles = ([float(v) for v in a.attack.split(",") if v.strip()]
             if a.attack else [])
    # Angle 0 (if swept) is the same cell as "placed" — skip the duplicate
    # coupled pass rather than running it twice.
    if a.simulate and 0.0 not in angles:
        sim_rows.append((f"placed{rpm_label}", simulate_step(
            f"01_placed{tag}", f"placed{rpm_label}", cfg, a)))

    if a.dx or a.dy or a.dz:
        cfg = move_workpiece(cfg, d_w_mm=(a.dx, a.dy, a.dz))
        theta1 = report_step(
            f"moved: d_w = ({a.dx:g}, {a.dy:g}, {a.dz:g}) mm{rpm_label}", cfg, theta1)
        if a.save:
            cfg.save(GEN_CONFIG_DIR / f"02_moved{tag}.json")
        if a.simulate:
            sim_rows.append((f"moved{rpm_label}", simulate_step(
                f"02_moved{tag}", f"moved{rpm_label}", cfg, a)))

    # Every angle pivots about the SAME point — the edge midpoint, fixed to
    # the part — so the angles are independent turns from one baseline, not
    # a chained walk.
    for angle in angles:
        cfg_a = attack_angle(cfg, angle)
        label = f"attack angle {angle:g} deg (about the edge midpoint){rpm_label}"
        report_step(label, cfg_a, theta1)
        if a.save:
            cfg_a.save(GEN_CONFIG_DIR / f"03_attack{angle:03.0f}{tag}.json")
        if a.simulate:
            slug = f"03_attack{angle:03.0f}{tag}"
            sim_rows.append((f"attack {angle:g} deg{rpm_label}",
                             simulate_step(slug, label, cfg_a, a)))

    return cfg, sim_rows


def main(argv=None):
    a = parse_args(argv)
    rpm_list = parse_float_list(a.rpm)

    cfg, sim_rows = None, []
    for rpm in rpm_list:
        cfg, rows = run_operating_point(rpm, a)
        sim_rows += rows

    print("\nAll 'max move vs previous step' figures above should read ~1e-6 deg "
          "or smaller (IK-residual noise) — the robot never actually moves; only "
          "the workpiece does.")
    summarise(sim_rows)
    return cfg


if __name__ == "__main__":
    main()
