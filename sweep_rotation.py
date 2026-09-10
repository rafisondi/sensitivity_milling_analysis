"""Does the direction the cut is presented to the arm matter?

    python sweep_rotation.py                     the linear side, plots, no sim
    python sweep_rotation.py --simulate           every cell, coupled  (long)
    python sweep_rotation.py --simulate --cells 3000/90,5000/0
    python sweep_rotation.py --resume             skip cells already on disk

THE GRID

`alpha` runs the full half-turn, 0 to 180 deg in 15 deg steps - past 180 the
workpiece is back to a mirror of the arc already swept, so nothing past it adds
a new arm/feed-direction pairing. It is crossed with three spindle speeds, 3000,
5000 and 10000 rpm, teeth held at 4 throughout: not because rpm and alpha are
expected to interact strongly, but because `C_cut` alone (the process-damping
term) scales as `1/Omega`, so whatever directional sensitivity the plant's
anisotropy produces is not obviously the same size at every speed.

THE QUESTION

Every other sweep in this project moves a CUTTING parameter and leaves the job
sitting exactly where it always has: the same edge, the same corner of the
workpiece, the same arm configuration. But the plant - the M/D/K this whole
comparison is linearised around - is a frozen-pose receptance of a robot whose
joint compliance is sharply anisotropic (README: `G(0)` off-diagonal terms are a
third of the diagonal ones, and the three joint stiffnesses this arm is built
from differ by nearly an order of magnitude). Two cuts identical in every process
number - same ap, ae, rpm, feed, chip load - can still see a different arm if the
workpiece is presented to it turned by some angle, because the FEED direction
rotates against the joints' own compliant axes while nothing about the cut
itself changes. This sweep asks whether that turn is free.

WHAT "ROTATE THE WORKPIECE" MEANS HERE

`fastsim.geometry.Workpiece` is a closed outline extruded along +Z_w, and the
edge being milled lives in the Z_w = 0 plane. `robotsim.Scene.R_cut` is the
attitude the whole cut is run at in the BASE frame - `scene.T_iw` maps
WORKPIECE -> BASE, so post-multiplying it by a rotation about +Z_w turns the
workpiece (and, since the toolpath is planned IN the workpiece frame and only
mapped to the base frame afterwards through the same `T_iw`, the nominal
trajectory with it) about its own extrusion axis, leaving the milling plane and
every process number untouched. Composed with the level default
(`robotsim.scene.default_r_iw`, rpy (0, 0, -90)), that is just

    R_cut(alpha) = default_r_iw() @ Rz(alpha)   ==  rpy (0, 0, -90 + alpha)

because both are pure yaw. `alpha = 0` reproduces the committed baseline
exactly. The base config's `scene.placed_by_hand` is False, so the workpiece
ORIGIN keeps following the start pose as it always did - the anchor point (the
lead-in start, `path.place_by_fk = True`) does not move, only the direction the
rest of the path swings away from it, and the TCP orientation the IK has to
reach at every node with it. That last part is the point: `R_i_tcp = R_cut @
R_w_tcp` rotates with the workpiece, so the wrist clocking - and with it, which
joint configuration the arm sits in along the path - moves with `alpha` too.

WHY THIS IS SEPARATE FROM EVERY OTHER SWEEP HERE, AND WHY THAT IS SAFE

No other script in this project touches `scene`, so there is nothing to
collide with. `alpha` does not reach `MillConfig` at all - it is a pure
`scene.R_cut` edit - so `analysis.toolpath.ensure`'s cache (keyed on feed only,
see the warning in `sweep_ae.py`) is untouched: the SAVED PATH is authored in
the workpiece frame and is the same file at every `alpha`, only the 4x4 that
carries it into the base frame differs, and that transform is read live off
`cfg.scene` on every call rather than cached. This script writes one small JSON
per cell to `configs/_sweep_rotation/` (new, nothing else reads that
directory) and calls `main.main(["--config", ...])` the same way `sweep.py`
calls it with `--rpm` etc. - `main.py`, `runconfig.py` and every existing
config are untouched.

REACHABILITY IS THE REAL LIMIT HERE, NOT RESOLUTION

Teeth and `ae` are held at the baseline throughout, so the CUT is identical at
every column of the grid; `rpm` moves across `RPM_LIST` at fixed `fz`, the same
invariant `sweep.py` holds for its own rpm axis, so `SIM_DT` stays fixed rather
than re-derived (4 teeth clears the steps-per-tooth floor comfortably even at
10000 rpm, unlike the 8-tooth corner that needed the odd-dt argument there).
What CAN fail is the arm: swinging a ~150 mm path around a fixed anchor by a
large `alpha` can
walk the far end toward a joint limit, a wrist singularity or simply out of
reach, and the IK residual `StartPose.describe` reports (or an outright
solver exception) is the signal, not a resolution parameter. Cells fail
independently here (as in `sweep.py`) rather than aborting the sweep; check
`summary.txt` in a failed cell's run directory for the residual before trusting
its neighbours.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import main as run_main                                       # noqa: E402
import analysis                                               # noqa: E402
from analysis import save                                     # noqa: E402

REPO = Path(__file__).resolve().parent
BASE_CONFIG = REPO / "configs" / "base.json"
GEN_CONFIG_DIR = REPO / "configs" / "_sweep_rotation"

#: The rotation axis [deg], about the workpiece's own extrusion axis Z_w.
#: 0 reproduces the committed baseline exactly (rpy yaw -90).
ALPHA_DEG = tuple(float(a) for a in range(0, 181, 15))

#: The spindle-speed axis [rpm], crossed with ALPHA_DEG.
RPM_LIST = (3000.0, 5000.0, 10000.0)

#: Held fixed at the base config's own operating point - only the direction the
#: job is presented to the arm (and, on the second axis, how fast it turns)
#: moves. `fz`, not `feed`, is what is held fixed across RPM_LIST - see
#: `analysis.config.apply_operating_point` on why that is the invariant to
#: hold when rpm moves.
TEETH = 4
FZ = 0.18
AE = 5.0

SIM_DT = 1.09375e-4
RASTER_MM = 0.01
FEED_PROFILE = "flying"


def cells() -> list:
    """The (rpm, alpha) grid, rpm-major so a partial run fills one speed first."""
    return [(rpm, alpha) for rpm in RPM_LIST for alpha in ALPHA_DEG]


def cell_name(rpm: float, alpha: float) -> str:
    return f"alpha{alpha:03.0f}_rpm{rpm:g}_z{TEETH}"


def config_for(alpha: float) -> Path:
    """Write `configs/_sweep_rotation/alpha<±N>.json`: the base config with
    `scene.R_cut_rpy_deg`'s YAW turned by `alpha` about the workpiece's own Z.

    Everything else - including `R_iw_rpy_deg`/`origin_i_m`, which are only a
    RECORD of a past placement and are ignored on load while
    `placed_by_hand` is False - is copied verbatim from `configs/base.json`.
    """
    d = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    sc = d["scene"]
    if sc.get("placed_by_hand"):
        raise RuntimeError(
            "configs/base.json is placed_by_hand=True - this sweep only knows "
            "how to turn a workpiece that is still following the start pose")
    rpy = list(sc.get("R_cut_rpy_deg") or [0.0, 0.0, -90.0])
    rpy[2] = float(rpy[2]) + float(alpha)
    sc["R_cut_rpy_deg"] = rpy

    GEN_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    out = GEN_CONFIG_DIR / f"alpha{alpha:03.0f}.json"
    out.write_text(json.dumps(d, indent=2), encoding="utf-8")
    return out


def parse_cells(spec: str) -> set:
    """`"3000/90,5000/0"` -> {(3000.0, 90.0), (5000.0, 0.0)}."""
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        rpm, alpha = part.split("/")
        out.add((float(rpm), float(alpha)))
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="rotation",
                   help="sweep directory under out/ (default: rotation)")
    p.add_argument("--simulate", action="store_true",
                   help="run the coupled pass at each cell as well as the "
                        "prediction; without this the sweep is the linear side "
                        "only")
    p.add_argument("--cells", default=None, metavar="RPM/ALPHA,...",
                   help="only these (rpm, alpha) cells, e.g. 3000/90,5000/0 - "
                        "applies to the SIMULATION; the linear side always "
                        "covers the full grid")
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
                   help="'flying' (default) opens at full feed; 'ramped' adds "
                        "the feed ramps and the arm's tracking error through "
                        "them")
    p.add_argument("--ds", type=float, default=2.0, metavar="MM",
                   help="arc-length spacing of the stability nodes [mm]")
    p.add_argument("--no-cell-plots", action="store_true",
                   help="skip the per-run figures; the sweep figure is still "
                        "drawn")
    p.add_argument("--no-plots", action="store_true",
                   help="skip every figure")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def argv_for(rpm: float, alpha: float, cfg_path: Path, a, sweep_runs: Path,
            simulate: bool) -> list:
    """The `main.py` command line for one cell."""
    argv = ["--config", str(cfg_path),
            "--name", cell_name(rpm, alpha), "--out", str(sweep_runs),
            "--rpm", repr(float(rpm)), "--teeth", str(TEETH), "--fz", repr(FZ),
            "--ae", repr(AE),
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


def figure_alpha(path: Path, rows: list):
    """alpha vs growth rate (both sides) and vs critical depth, one line per
    rpm - two panels, one PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in rows if r.get("alpha_deg") is not None
          and r.get("spindle_rpm") is not None]
    rpms = sorted({float(r["spindle_rpm"]) for r in ok})
    cmap = plt.get_cmap("viridis")
    colours = {rpm: cmap(i / max(1, len(rpms) - 1)) for i, rpm in enumerate(rpms)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for rpm in rpms:
        sub = sorted((r for r in ok if float(r["spindle_rpm"]) == rpm),
                    key=lambda r: r["alpha_deg"])
        alpha = [r["alpha_deg"] for r in sub]
        c = colours[rpm]
        ax1.plot(alpha, [r.get("pred_growth_max_trim_1_s") for r in sub],
                 "o-", color=c, label=f"{rpm:g} rpm  predicted")
        sim = [(r["alpha_deg"], r["sim_growth_1_s"]) for r in sub if r.get("sim_valid")]
        if sim:
            sa, sg = zip(*sim)
            ax1.plot(sa, sg, "s--", color=c, mfc="none",
                     label=f"{rpm:g} rpm  measured")
        ax2.plot(alpha, [r.get("pred_ap_crit_trim_mm") for r in sub], "o-",
                 color=c, label=f"{rpm:g} rpm")

    ax1.axhline(0.0, color="black", lw=0.8)
    ax1.axvline(0.0, color="grey", lw=0.8, ls=":")
    ax1.set_xlabel("workpiece rotation  alpha  [deg]")
    ax1.set_ylabel("growth rate  [1/s]")
    ax1.set_title("growth rate vs cut direction")
    ax1.legend(fontsize=7.5)

    ax2.axvline(0.0, color="grey", lw=0.8, ls=":")
    ax2.set_xlabel("workpiece rotation  alpha  [deg]")
    ax2.set_ylabel("predicted critical depth  [mm]")
    ax2.set_title("ap_crit vs cut direction")
    ax2.legend(fontsize=7.5)

    fig.suptitle(f"workpiece-rotation sweep  ({TEETH} teeth, fz {FZ:g} mm/tooth, "
                f"ae {AE:g} mm)")
    fig.tight_layout()
    out = path / "rotation_growth.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv=None):
    a = parse_args(argv)
    sweep_dir = Path(analysis.OUT) / a.name
    runs_dir = sweep_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    want_sim = parse_cells(a.cells) if a.cells else None
    plan = cells()

    print(f"sweep    {a.name}: {len(plan)} cells, alpha "
          f"{ALPHA_DEG[0]:g}..{ALPHA_DEG[-1]:g} deg x rpm {RPM_LIST}")
    print(f"         {TEETH} teeth, fz {FZ:g} mm/tooth, ae {AE:g} mm fixed, "
          f"dt {a.sim_dt:g} s, {a.feed_profile} feed")
    if a.simulate:
        n = len(want_sim) if want_sim else len(plan)
        print(f"         SIMULATING {'the ' + str(n) + ' named cell(s)' if want_sim else 'every cell'}")
    else:
        print("         prediction only (pass --simulate for coupled passes)")
    print()

    rows, skipped = [], []
    for i, (rpm, alpha) in enumerate(plan, 1):
        name = cell_name(rpm, alpha)
        d = runs_dir / name

        simulate = bool(a.simulate)
        if simulate and want_sim is not None:
            simulate = any(abs(rpm - r) < 1e-9 and abs(alpha - v) < 1e-9
                           for r, v in want_sim)

        reusable = False
        if a.resume and (d / "summary.json").exists():
            cached = json.loads((d / "summary.json").read_text(encoding="utf-8"))
            reusable = bool(cached.get("sim_valid")) or not simulate
        if reusable:
            row = cached
            row.setdefault("alpha_deg", float(alpha))
            row.setdefault("simulated", bool(row.get("sim_valid", False)))
            rows.append(row)
            print(f"[{i:2d}/{len(plan)}] {name}: reused")
            continue

        cfg_path = config_for(alpha)
        print(f"[{i:2d}/{len(plan)}] {name}   {rpm:g} rpm, alpha {alpha:.0f} deg"
              f"{'   [COUPLED PASS]' if simulate else ''}")
        try:
            row = run_main.main(argv_for(rpm, alpha, cfg_path, a, runs_dir, simulate))
        except Exception as exc:                      # a cell is data, not a stop
            traceback.print_exc()
            print(f"         ! {name} failed: {exc}")
            rows.append({"run": name, "alpha_deg": float(alpha), "n_teeth": TEETH,
                        "spindle_rpm": float(rpm), "error": str(exc)})
            continue
        row["alpha_deg"] = float(alpha)
        row["simulated"] = simulate
        rows.append(row)

    save.write_csv(sweep_dir / "sweep.csv", rows)
    save.write_json(sweep_dir / "sweep.json",
                    {"name": a.name, "n_cells": len(plan),
                     "alpha_deg": list(ALPHA_DEG), "rpm_list": list(RPM_LIST),
                     "teeth": TEETH, "fz_mm": FZ, "ae_mm": AE,
                     "sim_dt": float(a.sim_dt), "feed_profile": str(a.feed_profile),
                     "raster_mm": float(a.raster), "simulated": bool(a.simulate),
                     "skipped_simulation": skipped})

    written = {}
    if not a.no_plots and any(r.get("alpha_deg") is not None for r in rows):
        written["rotation_growth.png"] = figure_alpha(sweep_dir, rows)

    print()
    print(f"out      {sweep_dir}")
    print(f"         sweep.csv            {len(rows)} rows")
    for k, v in written.items():
        print(f"         {k:<20} {Path(v).relative_to(sweep_dir)}")
    return rows


if __name__ == "__main__":
    main()
