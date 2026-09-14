"""Pure rotation, no pose change - a real coupled check on the decomposition.

    python sweep_rotation_fixedpose.py                     linear map only (fast)
    python sweep_rotation_fixedpose.py --simulate           + coupled twin runs
    python sweep_rotation_fixedpose.py --simulate --cells 90,130
    python sweep_rotation_fixedpose.py --resume             reuse what is on disk
    python sweep_rotation_fixedpose.py --replot             figures from sweep.csv only

WHY THIS EXISTS

`sweep_rotation_ae.py` rotates the workpiece by `alpha` and lets the wrist
clock with it - `scene.R_i_tcp = R_cut(alpha) @ R_w_tcp` - so every cell in
that grid changes TWO things at once: which direction the cut sits relative to
the compliance (the effect `alpha` is meant to probe), and which joint
configuration - hence which receptance - the arm is actually in (a side effect
of `place_by_fk` chasing a wrist orientation that rotates with the workpiece).
The analytical decomposition (freeze the receptance, rotate only `K_cut`/
`C_cut` through `R_iw(alpha)`) says the pure-rotation piece alone still moves
`leverage` a lot (0.575 -> 2.573 across the half-turn) while `trunc_frac` stays
exactly flat - i.e. most of what `sweep_rotation_ae.py` measures as "alpha
matters" is really "pose matters", but not all of it. This sweep is the real
coupled check on that: hold the TCP orientation the path is planned to fixed
at every alpha, so the arm sits in the same pose (compliance, receptance,
everything the linear model sees) throughout, while the workpiece - and with
it the feed direction the compliance actually experiences - still turns.

THE FIX, EXACTLY

`scene.R_i_tcp = R_iw @ R_w_tcp` and `R_iw` is pinned to `R_cut` whenever the
scene gives one (`robotsim/scene.py`), so `config_for(alpha)` here turns the
workpiece exactly as `sweep_rotation.config_for` does - `R_cut(alpha) =
R_cut(0) @ Rz(alpha)` - and then solves `R_w_tcp(alpha) = R_cut(alpha)^T @
R_cut(0) @ R_w_tcp(0)`, the wrist re-clocking that exactly cancels it:

    R_i_tcp(alpha) = R_cut(alpha) @ R_w_tcp(alpha) == R_cut(0) @ R_w_tcp(0)

for every alpha (verified numerically to ~1e-16). Because `Rz(alpha)` is a
rotation about the workpiece's own (vertical) Z axis and so is the tool's
spindle axis, this compensation is itself a rotation about the tool's own Z -
it re-clocks the wrist about the spindle, not tilt it - so it changes no
process number and no milling geometry, only which of the (otherwise
physically equivalent) wrist configurations the arm sits in. `anchor_pose()`
already ignores `R_w_tcp` and places the job from FK of the fixed IK seed
regardless of alpha (`StartPose.anchor_pose`, `rpy_deg=None` falls through to
the seed branch when the scene is not yet placed) - so the START was already
alpha-invariant; what this script adds is making the REST OF THE PATH
alpha-invariant too, which `sweep_rotation.py`'s plain `config_for` does not.

WHAT STILL DRIFTS, AND WHY THAT IS FINE

The arm still moves along the path - lead-in, then the edge - so the pose at
the tap (halfway mark) is not bit-identical to the pose at the anchor even
here. That drift is the ordinary, ALPHA-INDEPENDENT drift every other sweep in
this project already carries (same part, same feed, same lead-in at every
alpha); it is not the thing under test. Matching `rotation_ae_rpm5000`'s own
operating point (rpm, ap, fz, ae, part length) exactly, cell for cell, keeps
that residual drift identical to the already-analysed baseline too, so any
remaining difference between this sweep and that one is the wrist-clocking
effect alone, not a change in cost or resolution.

THE GRID

One radial engagement, `ae = 60% D` - the ring the alpha090_ae60 cell that
started this whole investigation sits on - crossed with the same alpha axis
`sweep_rotation_ae.py` uses, 0..180 deg in 10 deg steps. rpm, teeth, fz, ap and
part length default to `rotation_ae_rpm5000`'s own operating point so the two
sweeps are directly comparable, cell for cell.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))

import main as run_main                                       # noqa: E402
import analysis                                               # noqa: E402
from analysis import save                                     # noqa: E402
from sweep_rotation_ae import (                                # noqa: E402
    AP_MM, FZ, MID, RASTER_MM, SIM_DT, STATE_STYLE, diameter_mm, measure,
)
from analysis import pulse as pmod                            # noqa: E402

REPO = Path(__file__).resolve().parent
BASE_CONFIG = REPO / "configs" / "base.json"
GEN_CONFIG_DIR = REPO / "configs" / "_sweep_rotation_fixedpose"

ALPHA_DEG = tuple(float(a) for a in range(0, 181, 10))
AE_FRAC = 0.60

#: Matches `rotation_ae_rpm5000` exactly - see the module docstring.
RPM = 5000.0
TEETH = 4


def _rpy_to_R(rpy_deg) -> np.ndarray:
    return Rotation.from_euler("xyz", np.asarray(rpy_deg, float), degrees=True).as_matrix()


def _R_to_rpy(R: np.ndarray) -> list:
    return Rotation.from_matrix(R).as_euler("xyz", degrees=True).tolist()


def config_for(alpha: float) -> Path:
    """`configs/_sweep_rotation_fixedpose/alpha<N>.json`: the base config with
    the workpiece turned by `alpha` (as `sweep_rotation.config_for` does) AND
    the wrist re-clocked to exactly cancel the resulting change in
    `scene.R_i_tcp` - see the module docstring for the derivation.
    """
    d = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    sc = d["scene"]
    if sc.get("placed_by_hand"):
        raise RuntimeError(
            "configs/base.json is placed_by_hand=True - this sweep only knows "
            "how to turn a workpiece that is still following the start pose")

    R_cut0 = _rpy_to_R(sc.get("R_cut_rpy_deg") or [0.0, 0.0, -90.0])
    R_wtcp0 = _rpy_to_R(sc.get("R_w_tcp_rpy_deg") or [180.0, 0.0, 180.0])
    R_i_tcp_ref = R_cut0 @ R_wtcp0

    R_cut_alpha = R_cut0 @ Rotation.from_euler("z", alpha, degrees=True).as_matrix()
    R_wtcp_alpha = R_cut_alpha.T @ R_i_tcp_ref

    sc["R_cut_rpy_deg"] = _R_to_rpy(R_cut_alpha)
    sc["R_w_tcp_rpy_deg"] = _R_to_rpy(R_wtcp_alpha)

    GEN_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    out = GEN_CONFIG_DIR / f"alpha{alpha:03.0f}.json"
    out.write_text(json.dumps(d, indent=2), encoding="utf-8")
    return out


def cell_name(alpha: float) -> str:
    return f"alpha{alpha:03.0f}_ae{100 * AE_FRAC:02.0f}_fixedpose"


def parse_cells(spec: str) -> set:
    """`"90,130"` -> {90.0, 130.0}."""
    return {float(a.strip()) for a in spec.split(",") if a.strip()}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="rotation_fixedpose")
    p.add_argument("--simulate", action="store_true")
    p.add_argument("--cells", default=None, metavar="ALPHA,...",
                   help="simulate only these alphas, e.g. 90,130 - the linear "
                        "map always covers the whole grid")
    p.add_argument("--no-twin", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--replot", action="store_true")
    g = p.add_argument_group("operating point (held across the grid)")
    g.add_argument("--rpm", type=float, default=RPM)
    g.add_argument("--teeth", type=int, default=TEETH)
    g.add_argument("--fz", type=float, default=FZ, metavar="MM")
    g.add_argument("--ae-pct", type=float, default=100.0 * AE_FRAC, metavar="PCT")
    g.add_argument("--ap", type=float, default=AP_MM, metavar="MM")
    g.add_argument("--raster", type=float, default=RASTER_MM, metavar="MM")
    g.add_argument("--sim-dt", type=float, default=SIM_DT, metavar="S")
    g.add_argument("--feed-profile", default="flying", choices=("ramped", "flying"))
    g.add_argument("--ds", type=float, default=2.0, metavar="MM")
    g.add_argument("--part-length", type=float, default=None, metavar="MM",
                   help="stock length (default: config's own 100 mm, matching "
                        "rotation_ae_rpm5000 - see the module docstring)")
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


def argv_for(alpha, a, runs_dir, *, simulate, tapped) -> list:
    name = cell_name(alpha) + ("_pulse" if tapped else "")
    ae_mm = a.ae_pct / 100.0 * diameter_mm()
    argv = ["--config", str(config_for(alpha)),
            "--name", name, "--out", str(runs_dir),
            "--rpm", repr(float(a.rpm)), "--teeth", str(int(a.teeth)),
            "--fz", repr(float(a.fz)), "--ae", repr(ae_mm),
            "--raster", repr(float(a.raster)), "--sim-dt", repr(float(a.sim_dt)),
            "--feed-profile", str(a.feed_profile), "--ds", repr(float(a.ds)),
            "--linearize-at", repr(MID),
            "--pulse-force", repr(float(a.pulse_force)),
            "--pulse-ms", repr(float(a.pulse_ms)), "--pulse-dir", str(a.pulse_dir)]
    if tapped:
        argv += ["--pulse-at", repr(MID)]
    if a.ap is not None:
        argv += ["--ap", repr(float(a.ap))]
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


def run_cell(alpha, a, runs_dir, *, simulate, twin):
    base_dir = runs_dir / cell_name(alpha)
    pulse_dir = runs_dir / (cell_name(alpha) + "_pulse")

    row = _load_summary(base_dir) if a.resume else None
    need_base = row is None or (simulate and not row.get("sim_valid")
                                and not row.get("sim_diverged"))
    if need_base:
        row = run_main.main(argv_for(alpha, a, runs_dir, simulate=simulate, tapped=False))
    if simulate and twin:
        pulsed = _load_summary(pulse_dir) if a.resume else None
        if pulsed is None or (not pulsed.get("sim_valid")
                              and not pulsed.get("sim_diverged")):
            run_main.main(argv_for(alpha, a, runs_dir, simulate=True, tapped=True))

    row = dict(row)
    row.update({"alpha_deg": float(alpha), "ae_pct": float(a.ae_pct),
                "ae_mm": a.ae_pct / 100.0 * diameter_mm(), "simulated": bool(simulate)})
    if simulate:
        row.update(measure(row, base_dir, pulse_dir))
    row["lin_state"] = pmod.lin_state(_num(row, "pred_growth_mid_1_s"))
    row["sim_state"] = pmod.sim_state(row) if simulate else "unmeasured"
    return row, ("reused" if not need_base else "ran")


def figure(path: Path, rows: list):
    from analysis import plots
    plt = plots._mpl(backend="Agg")

    ok = sorted((r for r in rows if np.isfinite(_num(r, "alpha_deg"))),
                key=lambda r: r["alpha_deg"])
    alpha = np.array([_num(r, "alpha_deg") for r in ok])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    plots._clean(ax1)
    plots._clean(ax2)

    lin = np.array([_num(r, "pred_growth_mid_1_s") for r in ok])
    ax1.plot(alpha, lin, "o-", color=plots.INK, lw=1.6, label="predicted (sigma_lin)")
    sim_rows = [r for r in ok if r.get("sim_state") not in (None, "", "unmeasured")]
    if sim_rows:
        sa = [_num(r, "alpha_deg") for r in sim_rows]
        sg = [_num(r, "sim_sigma_1_s") for r in sim_rows]
        ax1.plot(sa, sg, "s--", color="#c62828", mfc="none", label="measured (sigma_sim)")
    ax1.axhline(0.0, color=plots.INK, lw=0.8)
    ax1.set_xlabel("workpiece rotation  alpha  [deg]")
    ax1.set_ylabel("growth rate  [1/s]")
    ax1.set_title("fixed pose: growth rate vs cut direction", loc="left")
    ax1.legend(fontsize=8)

    trunc = np.array([_num(r, "tooth_trunc_frac") for r in ok])
    ax2.plot(alpha, trunc, "o-", color=plots.INK_2, lw=1.6)
    ax2.set_xlabel("workpiece rotation  alpha  [deg]")
    ax2.set_ylabel("tooth_trunc_frac")
    ax2.set_title("delay-truncation term - should be FLAT here", loc="left")

    fig.suptitle(f"pure rotation, fixed pose  (ae {100*AE_FRAC:.0f}% D, "
                f"{RPM:g} rpm, {TEETH} teeth)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return plots._save(fig, path / "fixedpose_growth_vs_alpha.png")


def main(argv=None):
    a = parse_args(argv)
    sweep_dir = Path(analysis.OUT) / a.name
    runs_dir = sweep_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    if a.replot:
        import csv
        rows = list(csv.DictReader(open(sweep_dir / "sweep.csv", encoding="utf-8")))
        out = figure(sweep_dir, rows)
        print(f"figure   {out}")
        return rows

    want_sim = parse_cells(a.cells) if a.cells else None
    plan = list(ALPHA_DEG)

    print(f"sweep    {a.name}: {len(plan)} cells, ae {a.ae_pct:.0f}% D fixed, "
          f"alpha {plan[0]:g}..{plan[-1]:g} deg")
    print(f"         {a.rpm:g} rpm, {a.teeth} teeth, fz {a.fz:g} mm/tooth, "
          f"ap {a.ap:g} mm - matching rotation_ae_rpm5000's own operating point")
    if a.simulate:
        n = len(want_sim) if want_sim else len(plan)
        print(f"         SIMULATING {'the ' + str(n) + ' named cell(s)' if want_sim else 'every cell'} "
              f"(twin runs{', --no-twin' if a.no_twin else ''})")
    else:
        print("         prediction only (pass --simulate for coupled passes)")
    print()

    rows = []
    for i, alpha in enumerate(plan, 1):
        simulate = bool(a.simulate)
        if simulate and want_sim is not None:
            simulate = any(abs(alpha - v) < 1e-9 for v in want_sim)
        print(f"[{i:2d}/{len(plan)}] alpha {alpha:5.0f} deg"
              f"{'   [COUPLED PASS]' if simulate else ''}")
        try:
            row, status = run_cell(alpha, a, runs_dir, simulate=simulate,
                                   twin=not a.no_twin)
            print(f"         {status}")
        except Exception as exc:                      # a cell is data, not a stop
            traceback.print_exc()
            print(f"         ! alpha{alpha:.0f} failed: {exc}")
            row = {"run": cell_name(alpha), "alpha_deg": float(alpha), "error": str(exc)}
        rows.append(row)

    save.write_csv(sweep_dir / "sweep.csv", rows)
    save.write_json(sweep_dir / "sweep.json",
                    {"name": a.name, "n_cells": len(plan), "alpha_deg": list(ALPHA_DEG),
                     "ae_pct": float(a.ae_pct), "rpm": float(a.rpm), "teeth": int(a.teeth),
                     "fz_mm": float(a.fz), "ap_mm": float(a.ap), "sim_dt": float(a.sim_dt),
                     "raster_mm": float(a.raster), "feed_profile": str(a.feed_profile),
                     "linearize_at": MID, "pulse_at": MID,
                     "pulse_force_N": float(a.pulse_force), "pulse_ms": float(a.pulse_ms),
                     "pulse_dir_w": str(a.pulse_dir),
                     "part_length_mm": (None if a.part_length is None else float(a.part_length)),
                     "part_width_mm": (None if a.part_width is None else float(a.part_width)),
                     "simulated": bool(a.simulate), "twin": not a.no_twin,
                     "fixed_pose": True})

    written = {}
    if not a.no_plots:
        written["fixedpose_growth_vs_alpha.png"] = figure(sweep_dir, rows)

    print()
    print(f"out      {sweep_dir}")
    print(f"         sweep.csv            {len(rows)} rows")
    for k, v in written.items():
        print(f"         {k:<28} {Path(v).relative_to(sweep_dir)}")
    return rows


if __name__ == "__main__":
    main()
