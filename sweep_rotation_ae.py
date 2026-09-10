"""Stable or unstable, by cut direction and radial engagement - model against simulation.

    python sweep_rotation_ae.py                          linear map only (fast)
    python sweep_rotation_ae.py --simulate               + coupled passes, twin runs
    python sweep_rotation_ae.py --simulate --cells 90/60,0/10
    python sweep_rotation_ae.py --resume                 reuse what is on disk
    python sweep_rotation_ae.py --replot                 figures from sweep.csv only

THE GRID

    alpha   0 .. 180 deg in 10 deg steps  - the workpiece AND its planned path turned
                                            about the workpiece's own Z (see
                                            `sweep_rotation.py` for what that means
                                            and why 180 is enough)
    ae      10, 30, 60, 80 % of the tool diameter

76 cells. Above 50 % the tool centre runs INSIDE the part edge (`tool_offset_mm =
R - ae < 0`); the path is the inward offset of the outline and the cut enters
through the part's end face.

WHAT EACH CELL PRODUCES

    linear       the arm linearised at the commanded pose HALF WAY along the
                 driven path, the cut closed at the node there:
                     sigma_lin = max Re(eig A_cl)          -> red if > 0
    simulated    the coupled pass, run twice - once plain, once with a 5 ms tap
                 on the TCP at the same halfway mark - and subtracted:
                     sigma_sim = decay rate of the tap's response
                                                           -> red if it grows

Both are the decay rate of a small perturbation about ONE operating point, at ONE
pose, so they compare directly. `analysis.pulse` sets out why that is the growth
rate to compare, and why the plateau envelope the older metrics fitted is not.
A third number, `sigma_lin_fit`, is the same tap through the linear closed loop
fitted with the same estimator: where it matches `sigma_lin`, the fit window is
adequate and any gap to `sigma_sim` is the model; where it does not, the window is
the limit.

WHAT IS FIXED, AND WHAT IS CORRECTED

Spindle speed, tooth count, chip load and depth are held (CLI flags). The DC
compensation is on (`tau_ff = -J^T F0(s)`), the feed is flying, the step is
unlocked from the revolution (1.09375e-4 s), and the flexible-joint RK4 sees the
command at each stage's own time. The toolpath cache now carries the tool offset,
so the four `ae` values no longer replay one path.

THE DEPTH

At the stock `ap = 1 mm` every cell is stable at the halfway mark. The default
`ap = 8 mm` was read off that map (`AP_MM`); to re-derive it, run the linear map
at `--ap 1` and read `pred_ap_crit_mid_mm` in `sweep.csv` - the depth at which each
cell's halfway node goes unstable. `ap_crit` is only the SMALLEST unstable depth
(growth need not be monotone in `ap`), so the linear map at the chosen depth, not
`ap > ap_crit`, is the prediction the polar plot shows.

COST

Twin runs double the simulation: at 3333 rpm, 4 teeth, fz 0.18 the pass is about
3.9 s, ~40 s of wall time per run, so roughly 80 s per cell and ~100 min for the
full 76. `--no-twin` halves it and falls back to the plateau growth test.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import main as run_main                                       # noqa: E402
import analysis                                               # noqa: E402
from analysis import config as acfg                           # noqa: E402
from analysis import growth, pulse as pmod, save              # noqa: E402
from sweep_rotation import config_for                         # noqa: E402

ALPHA_DEG = tuple(float(a) for a in range(0, 181, 10))
AE_FRAC = (0.10, 0.30, 0.60, 0.80)

RPM = 3333.0
TEETH = 4

#: Axial depth [mm], chosen from the linear map at ap = 1 mm (out/rotation_ae_scan).
#: The halfway-node critical depth runs from 1.55 mm (alpha 130, ae 10%) to beyond
#: the 120 mm search cap, and the count of predicted-unstable cells is flat from
#: 6 to 15 mm: every threshold found lies below 5.2 mm or above 20.6 mm. 8 mm puts
#: the tightest unstable cell (alpha 40, ae 10%, 5.16 mm) 55% past its threshold and
#: stays 2.6x under the next one, so no cell sits on the boundary. Taken from the
#: 10% and 30% rings only - the 60/80% rings were still running when it was fixed.
AP_MM = 8.0
FZ = 0.18
SIM_DT = 1.09375e-4
RASTER_MM = 0.01
FEED_PROFILE = "flying"

#: The halfway mark of the driven path: where the arm is linearised, where the
#: prediction is read, and where the coupled pass is tapped.
MID = 0.5

STATE_STYLE = {
    "stable":     {"color": "#2e7d32", "marker": "o", "filled": True,
                   "label": "stable"},
    "unstable":   {"color": "#c62828", "marker": "X", "filled": True,
                   "label": "unstable"},
    "marginal":   {"color": "#2e7d32", "marker": "o", "filled": False,
                   "label": "no growth resolved"},
    "unmeasured": {"color": "#9e9e9e", "marker": "s", "filled": True,
                   "label": "not measured"},
}


def diameter_mm() -> float:
    return float(acfg.load_base().mill.diameter_mm)


def cells() -> list:
    """(alpha, ae_frac), ae-major, so a partial run fills one ring first."""
    return [(alpha, frac) for frac in AE_FRAC for alpha in ALPHA_DEG]


def cell_name(alpha, frac) -> str:
    return f"alpha{alpha:03.0f}_ae{100 * frac:02.0f}"


def parse_cells(spec: str) -> set:
    """`"90/60,0/10"` (alpha deg / ae % of D) -> {(90.0, 0.60), (0.0, 0.10)}."""
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
    p.add_argument("--name", default="rotation_ae")
    p.add_argument("--simulate", action="store_true",
                   help="run the coupled passes; without it, the linear map only")
    p.add_argument("--cells", default=None, metavar="ALPHA/AE%,...",
                   help="simulate only these, e.g. 90/60,0/10 - the linear map "
                        "always covers the whole grid")
    p.add_argument("--no-twin", action="store_true",
                   help="skip the tapped run (half the cost); the simulated "
                        "verdict falls back to the plateau growth test")
    p.add_argument("--resume", action="store_true",
                   help="reuse runs already on disk")
    p.add_argument("--replot", action="store_true",
                   help="only redraw the figures from an existing sweep.csv")
    g = p.add_argument_group("operating point (held across the grid)")
    g.add_argument("--rpm", type=float, default=RPM)
    g.add_argument("--teeth", type=int, default=TEETH)
    g.add_argument("--fz", type=float, default=FZ, metavar="MM")
    g.add_argument("--ap", type=float, default=AP_MM, metavar="MM",
                   help=f"axial depth (default {AP_MM:g} mm, chosen from the "
                        "ap = 1 mm linear map - see AP_MM)")
    g.add_argument("--raster", type=float, default=RASTER_MM, metavar="MM")
    g.add_argument("--sim-dt", type=float, default=SIM_DT, metavar="S")
    g.add_argument("--feed-profile", default=FEED_PROFILE,
                   choices=("ramped", "flying"))
    g.add_argument("--ds", type=float, default=2.0, metavar="MM")
    g = p.add_argument_group("the tap")
    g.add_argument("--pulse-force", type=float, default=20.0, metavar="N")
    g.add_argument("--pulse-ms", type=float, default=5.0, metavar="MS")
    g.add_argument("--pulse-dir", default="auto", metavar="X,Y,Z|auto",
                   help="'auto' taps along the least-damped mode's input "
                        "direction at the halfway node - see main.py")
    g = p.add_argument_group("output")
    g.add_argument("--no-cell-plots", action="store_true")
    g.add_argument("--no-plots", action="store_true")
    g.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def argv_for(alpha, frac, a, runs_dir, *, simulate, tapped) -> list:
    """The `main.py` command line for one cell (plain or tapped)."""
    name = cell_name(alpha, frac) + ("_pulse" if tapped else "")
    argv = ["--config", str(config_for(alpha)),
            "--name", name, "--out", str(runs_dir),
            "--rpm", repr(float(a.rpm)), "--teeth", str(int(a.teeth)),
            "--fz", repr(float(a.fz)), "--ae", repr(frac * diameter_mm()),
            "--raster", repr(float(a.raster)), "--sim-dt", repr(float(a.sim_dt)),
            "--feed-profile", str(a.feed_profile), "--ds", repr(float(a.ds)),
            "--linearize-at", repr(MID),
            "--pulse-force", repr(float(a.pulse_force)),
            "--pulse-ms", repr(float(a.pulse_ms)), "--pulse-dir", str(a.pulse_dir)]
    if tapped:
        argv += ["--pulse-at", repr(MID)]
    if a.ap is not None:
        argv += ["--ap", repr(float(a.ap))]
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


def measure(row, base_dir: Path, pulse_dir: Path) -> dict:
    """Everything the simulated verdict needs, from what is on disk."""
    out = {}
    g = growth.score(base_dir, ae_mm=_num(row, "ae_mm") or 5.0)
    out.update({k: g.get(k) for k in ("plateau_verdict", "plateau_g_1_s",
                                      "plateau_floor_1_s", "peak_pct_ae",
                                      "ring_g_1_s", "ring_r2") if k in g})
    pulsed = _load_summary(pulse_dir)
    if pulsed is not None:
        out["pulse_sim_diverged"] = bool(pulsed.get("sim_diverged"))
        needed = ("pulse_t_s", "pulse_fit_t0_s", "pulse_fit_t1_s",
                  "pulse_band_lo_hz", "pulse_band_hi_hz", "pulse_f_lowest_hz")
        if all(np.isfinite(_num(row, k)) for k in needed):
            out.update(pmod.twin_rate(
                base_dir, pulse_dir, t_pulse=_num(row, "pulse_t_s"),
                duration_s=1e-3 * _num(row, "pulse_ms"),
                fit_t0=_num(row, "pulse_fit_t0_s"),
                fit_t1=_num(row, "pulse_fit_t1_s"),
                band=(_num(row, "pulse_band_lo_hz"), _num(row, "pulse_band_hi_hz")),
                f_lowest_hz=_num(row, "pulse_f_lowest_hz")))
    return out


def run_cell(alpha, frac, a, runs_dir, *, simulate, twin):
    """(row, status) for one cell - the linear prediction, and if asked the
    plain and tapped coupled passes."""
    base_dir = runs_dir / cell_name(alpha, frac)
    pulse_dir = runs_dir / (cell_name(alpha, frac) + "_pulse")

    row = _load_summary(base_dir) if a.resume else None
    need_base = row is None or (simulate and not row.get("sim_valid")
                                and not row.get("sim_diverged"))
    if need_base:
        row = run_main.main(argv_for(alpha, frac, a, runs_dir,
                                     simulate=simulate, tapped=False))
    if simulate and twin:
        pulsed = _load_summary(pulse_dir) if a.resume else None
        if pulsed is None or (not pulsed.get("sim_valid")
                              and not pulsed.get("sim_diverged")):
            run_main.main(argv_for(alpha, frac, a, runs_dir,
                                   simulate=True, tapped=True))

    row = dict(row)
    row.update({"alpha_deg": float(alpha), "ae_frac": float(frac),
                "ae_pct": 100.0 * float(frac),
                "ae_mm": float(frac) * diameter_mm(), "simulated": bool(simulate)})
    if simulate:
        row.update(measure(row, base_dir, pulse_dir))
    row["lin_state"] = pmod.lin_state(_num(row, "pred_growth_mid_1_s"))
    row["sim_state"] = pmod.sim_state(row) if simulate else "unmeasured"
    return row, ("reused" if not need_base else "ran")


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _polar_panel(ax, rows, state_key, title):
    from analysis import plots
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    rings = [100.0 * f for f in AE_FRAC]
    ax.set_rlim(0, 100)
    ax.set_rticks(rings)
    ax.set_yticklabels([f"{r:g}%D" for r in rings], fontsize=7, color=plots.INK_2)
    ax.set_thetagrids(range(0, 181, 30), labels=[f"{d}°" for d in range(0, 181, 30)],
                      fontsize=8)
    ax.grid(True, color=plots.GRID, lw=0.8)
    for state, st in STATE_STYLE.items():
        pts = [(np.radians(_num(r, "alpha_deg")), _num(r, "ae_pct"))
               for r in rows if r.get(state_key) == state]
        if not pts:
            continue
        th, rr = zip(*pts)
        ax.scatter(th, rr, s=70, marker=st["marker"], linewidths=1.6,
                   facecolors=st["color"] if st["filled"] else "none",
                   edgecolors=st["color"], zorder=3)
    ax.set_title(title, color=plots.INK, fontsize=10, pad=14)


def figure_polar(sweep_dir: Path, rows, a):
    from analysis import plots
    plt = plots._mpl()
    import matplotlib.lines as mlines

    fig = plt.figure(figsize=(13, 5.6))
    ax1 = fig.add_subplot(1, 2, 1, projection="polar")
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    _polar_panel(ax1, rows, "lin_state",
                 "linear model  -  sign of max Re(eig) at the halfway mark")
    _polar_panel(ax2, rows, "sim_state",
                 "coupled simulation  -  does the tapped response grow?")

    both = [r for r in rows if r.get("sim_state") in ("stable", "unstable", "marginal")
            and r.get("lin_state") in ("stable", "unstable")]
    agree = sum((r["lin_state"] == "unstable") == (r["sim_state"] == "unstable")
                for r in both)
    handles = [mlines.Line2D([], [], ls="", marker=st["marker"], ms=9,
                             markerfacecolor=st["color"] if st["filled"] else "none",
                             markeredgecolor=st["color"], label=st["label"])
               for st in STATE_STYLE.values()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9)
    ap = a.ap if a.ap is not None else float(acfg.load_base().part.height_mm)
    fig.suptitle(f"angle = workpiece rotation, radius = radial engagement   |   "
                 f"{a.rpm:g} rpm, {a.teeth} teeth, fz {a.fz:g} mm, ap {ap:g} mm"
                 + (f"   |   agree on {agree} of {len(both)} simulated cells"
                    if both else ""),
                 fontsize=10, color=plots.INK)
    fig.subplots_adjust(bottom=0.14, top=0.85, wspace=0.25)
    out = sweep_dir / "figures" / "polar_stability.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_parity(sweep_dir: Path, rows):
    """sigma_sim vs sigma_lin, and the estimator check sigma_lin_fit vs sigma_lin."""
    from analysis import plots
    plt = plots._mpl()
    colours = dict(zip([100.0 * f for f in AE_FRAC],
                       (plots.C1, plots.C2, plots.C3, plots.C4)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, ykey, ylab, title in (
            (ax1, "sim_sigma_1_s", "sigma_sim  - tap decay, coupled pass  [1/s]",
             "the model against the simulation"),
            (ax2, "pred_pulse_fit_1_s", "sigma_lin_fit  - same tap, linear loop  [1/s]",
             "the estimator against the eigenvalue it should recover")):
        plots._clean(ax, grid="both")
        allv = []
        for pct, c in colours.items():
            sub = [r for r in rows if abs(_num(r, "ae_pct") - pct) < 1e-6]
            x = np.array([_num(r, "pred_growth_mid_1_s") for r in sub])
            y = np.array([_num(r, ykey) for r in sub])
            ok = np.isfinite(x) & np.isfinite(y)
            if not ok.any():
                continue
            weak = np.array([_num(r, "sim_sigma_r2") < 0.5 if ykey == "sim_sigma_1_s"
                             else _num(r, "pred_pulse_fit_r2") < 0.5 for r in sub])
            ax.scatter(x[ok & ~weak], y[ok & ~weak], s=36, color=c,
                       label=f"ae {pct:g}% D", zorder=3)
            ax.scatter(x[ok & weak], y[ok & weak], s=36, facecolors="none",
                       edgecolors=c, zorder=3)
            allv += list(x[ok]) + list(y[ok])
        if allv:
            lo, hi = min(allv), max(allv)
            pad = 0.08 * (hi - lo + 1e-9)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=plots.MUTED,
                    ls=(0, (4, 3)), lw=1.0, label="agreement")
        ax.axhline(0, color=plots.MUTED, lw=0.8)
        ax.axvline(0, color=plots.MUTED, lw=0.8)
        ax.set_xlabel("sigma_lin  - max Re(eig) at the halfway mark  [1/s]")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", color=plots.INK)
        ax.legend(loc="best")
    fig.text(0.01, -0.02, "hollow = fit R2 < 0.5.  Upper-right quadrant: both say "
             "unstable; lower-left: both stable; the off-diagonal quadrants are "
             "the disagreements.", fontsize=8, color=plots.INK_2)
    fig.tight_layout()
    out = sweep_dir / "figures" / "growth_parity.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def read_csv(path: Path) -> list:
    import csv
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main(argv=None):
    a = parse_args(argv)
    sweep_dir = Path(analysis.OUT) / a.name
    runs_dir = sweep_dir / "runs"

    if a.replot:
        rows = read_csv(sweep_dir / "sweep.csv")
        print(figure_polar(sweep_dir, rows, a))
        if any(np.isfinite(_num(r, "sim_sigma_1_s")) or
               np.isfinite(_num(r, "pred_pulse_fit_1_s")) for r in rows):
            print(figure_parity(sweep_dir, rows))
        return rows

    runs_dir.mkdir(parents=True, exist_ok=True)
    want_sim = parse_cells(a.cells) if a.cells else None
    plan = cells()
    D = diameter_mm()
    print(f"sweep    {a.name}: {len(plan)} cells, alpha {ALPHA_DEG[0]:g}..{ALPHA_DEG[-1]:g}"
          f" deg x ae {[f'{100 * f:g}%' for f in AE_FRAC]} of D = {D:g} mm")
    print(f"         {a.rpm:g} rpm, {a.teeth} teeth, fz {a.fz:g}, dt {a.sim_dt:g} s, "
          f"{a.feed_profile}, linearised + tapped at {100 * MID:g}% of the path")
    print("         " + ("SIMULATING" + (" (twin runs)" if not a.no_twin else "")
                        if a.simulate else "linear map only (pass --simulate)"))
    print()

    rows = []
    for i, (alpha, frac) in enumerate(plan, 1):
        name = cell_name(alpha, frac)
        simulate = bool(a.simulate) and (
            want_sim is None or any(abs(alpha - x) < 1e-9 and abs(frac - f) < 1e-6
                                    for x, f in want_sim))
        print(f"[{i:2d}/{len(plan)}] {name}   alpha {alpha:3.0f} deg, ae "
              f"{100 * frac:g}% ({frac * D:g} mm)"
              f"{'   [COUPLED' + (' + TAP]' if not a.no_twin else ']') if simulate else ''}")
        try:
            row, status = run_cell(alpha, frac, a, runs_dir, simulate=simulate,
                                   twin=not a.no_twin)
        except Exception as exc:                       # a cell is data, not a stop
            traceback.print_exc()
            print(f"         ! {name} failed: {exc}")
            rows.append({"run": name, "alpha_deg": alpha, "ae_frac": frac,
                         "ae_pct": 100 * frac, "error": str(exc),
                         "lin_state": "unmeasured", "sim_state": "unmeasured"})
            continue
        print(f"         {status}: sigma_lin {_num(row, 'pred_growth_mid_1_s'):+.2f} 1/s "
              f"-> {row['lin_state']}"
              + (f" | sigma_sim {_num(row, 'sim_sigma_1_s'):+.2f} "
                 f"(floor {_num(row, 'sim_sigma_floor_1_s'):.2f}) -> {row['sim_state']}"
                 if simulate else ""))
        rows.append(row)

    save.write_csv(sweep_dir / "sweep.csv", rows)
    save.write_json(sweep_dir / "sweep.json", {
        "name": a.name, "alpha_deg": list(ALPHA_DEG), "ae_frac": list(AE_FRAC),
        "diameter_mm": D, "rpm": a.rpm, "teeth": a.teeth, "fz_mm": a.fz,
        "ap_mm": a.ap, "sim_dt": a.sim_dt, "raster_mm": a.raster,
        "feed_profile": a.feed_profile, "linearize_at": MID, "pulse_at": MID,
        "pulse_force_N": a.pulse_force, "pulse_ms": a.pulse_ms,
        "pulse_dir_w": a.pulse_dir, "simulated": bool(a.simulate),
        "twin": not a.no_twin})

    if not a.no_plots:
        print()
        print(f"figure   {figure_polar(sweep_dir, rows, a)}")
        if any(np.isfinite(_num(r, "sim_sigma_1_s")) or
               np.isfinite(_num(r, "pred_pulse_fit_1_s")) for r in rows):
            print(f"figure   {figure_parity(sweep_dir, rows)}")
    print(f"out      {sweep_dir / 'sweep.csv'}  ({len(rows)} rows)")
    return rows


if __name__ == "__main__":
    main()
