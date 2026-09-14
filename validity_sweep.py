"""Where does the linearisation stop being trustworthy - swept over any two of
fz, ap, rpm, with the third held fixed.

    python validity_sweep.py --run <dir>                        fz x ap, own rpm
    python validity_sweep.py --run <dir> --rpm 5000,10000 --ap 1.35
    python validity_sweep.py --run <dir> --fz 0.18 --rpm 3000,10000
    python validity_sweep.py --run <dir> --leverage-max 2.0 --trunc-max 0.05

TWO FACTORS, ONE JUSTIFICATION

Chasing why `alpha090_ae60` (rotation_ae_rpm5000) predicted confidently stable
(-7.48 1/s) while the coupled pass diverged, then checking the SAME cell at
10000 rpm where the coupled pass was fine, is what this reduces to. A third
candidate - the quasi-static deflection `G(0) F0` as a fraction of `ae` - was
tried and DROPPED: it doesn't correlate with the measured deviation at all
(r = -0.20 against the residual DC, -0.04 against the AC ripple, across the 56
simulated rpm=5000 cells), because the pipeline actively CANCELS the DC
component it estimates (`tau_ff = -J^T F0(s)`, `analysis.feedforward`) - the
metric was measuring a channel the run itself compensates away. Its earlier
apparent separation was very likely a confound through `ae` (which also drives
LEVERAGE below), not independent evidence.

What is left is a standard first-order error-propagation argument:

    error in the verdict  ~  (sensitivity of the verdict to the approximate
                              term)  x  (how wrong that term actually is)
                          =  LEVERAGE                x  TRUNC_FRAC

  LEVERAGE      growth_mid / open_loop_growth - how much of the closed-loop
                "confidence" the CUT is manufacturing versus what the bare
                structure provides on its own. This is the SENSITIVITY half -
                it says nothing about whether `C_cut` is itself accurate, only
                how much an inaccuracy in it would matter. Responds to `ap`
                (scales `K_cut` AND `C_cut` together) and to `rpm` (`C_cut` ~
                1/rpm - see `stabsim.cut.A_cut_0`). Does NOT respond to `fz`:
                confirmed at the `A_cut_0`/`C_cut` function signatures, which
                take no feed argument at all.

  TRUNC_FRAC    (omega T)^2 / 2, T = 60/(rpm n_teeth) - the fraction of the
                cut force the regenerative-delay TRUNCATION itself throws
                away (`x(t-T) ~ x(t) - T xd(t)`), per
                `stabsim.stability.StabilityAlongPath.omega_T`. This is the
                ACCURACY half - a real Taylor-remainder bound, not a
                heuristic. Responds to `rpm` ONLY (no `fz`, no `ap` - it is
                purely a statement about time, not amplitude or geometry).

Calibrated against a small, noisy sample - 42 simulated cells at 5000 rpm plus
one paired point at 10000 rpm - the PRODUCT separated agree/disagree cells
better than either factor alone (2.88x median ratio, against leverage's 2.16x
and trunc_frac's 1.29x) and tracked the 5000->10000 rpm change far better than
leverage alone did (5.2x against a real 11.9x change in peak deviation, versus
leverage's 1.3x). Still only a first-order estimate of a genuinely nonlinear
failure - treat it as screening, not a verdict.

WHAT THIS SCRIPT DOES

Rebuilds K_cut/C_cut ONCE at the run's own (ae, phi_en, phi_ex, pose) - the
expensive parts (IK, receptance identification) - then sweeps any two of (fz,
ap, rpm) CHEAPLY over that fixed geometry: each is a closed-form rescale of
the same K8/C8 (see the two responses above), so each grid point costs one
eigenproblem, not a simulation or a new IK solve. `fz` does not move either
factor and is kept purely so the reported operating point and `growth`/`F0`
context stay meaningful - it plays no role in the safety gate.

Pass exactly two of --fz/--ap/--rpm as a range ("lo,hi") and the third as a
single value to hold fixed - defaults to the run's own operating point.

WHAT THIS DOES NOT DO. It does not re-measure the engagement geometry at a
swept ap, and the coupled dexel sim it is checked against is a truth this
script never re-runs. It is a screening tool over the cut linearisation's own
fragility axes, not a replacement for actually running the pass.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare_linear as cl                                   # noqa: E402
from runconfig import RunConfig                                # noqa: E402
from analysis import toolpath, plant as plant_mod, stability, plots  # noqa: E402
from stabsim import cut, stability as stabsim_stability        # noqa: E402

AXES = ("fz", "ap", "rpm")


def _find_mid_node(stab):
    s = np.asarray(stab.s_mm, float)
    return int(np.argmin(np.abs(s - 0.5 * s.max())))


def build_reference(run_dir, ds_mm=2.0, verbose=False):
    """Everything the sweep needs, computed ONCE at this run's own geometry."""
    cfg_d = json.loads((Path(run_dir) / "config.json").read_text())
    cfg = cl._repin(RunConfig.from_dict(cfg_d))
    cfg = toolpath.ensure(cfg, profile="flying", verbose=verbose)
    setup = stability.prepare(cfg, ds_mm=ds_mm, verbose=verbose)
    recep = plant_mod.receptance_from_robot(cfg)
    stab = stability.predict(setup, recep, verbose=verbose)

    i = _find_mid_node(stab)
    mill = setup.mill
    r = stab.receptance
    g_open, hz_open = stabsim_stability.dominant_mode(r)

    return {
        "run": str(run_dir), "mill": mill, "receptance": r,
        "K8": stab.K_p[i], "C8": stab.C_p[i],
        "g_open": g_open, "hz_open": hz_open,
        "modes_hz": np.asarray(r.modes_hz, float),
        "fz0": float(mill.feed_per_tooth_mm()), "ap0": float(mill.height_mm),
        "rpm0": float(mill.spindle_rpm), "n_teeth": float(mill.n_teeth),
        "ae_mm": float(stab.ae_mm[i]),
    }


def evaluate(ref, fz, ap, rpm) -> dict:
    """(leverage, trunc_frac, growth) at this (fz, ap, rpm). `fz` only feeds
    `growth` through `K_cut` for context - it plays no role in either factor
    the safety gate actually uses."""
    kappa_ap = ap / ref["ap0"]
    kappa_K = (fz / ref["fz0"]) * kappa_ap                 # K_cut ~ fz * ap
    kappa_C = kappa_ap * (ref["rpm0"] / rpm)               # C_cut ~ ap / rpm

    K = ref["K8"] * kappa_K
    C = ref["C8"] * kappa_C
    growth = stabsim_stability.growth_rate(ref["receptance"], K, C)
    leverage = growth / ref["g_open"] if ref["g_open"] != 0 else np.nan

    T = 60.0 / (rpm * ref["n_teeth"])
    omega_T_max = float(2.0 * np.pi * ref["modes_hz"].max() * T)
    trunc_frac = 0.5 * omega_T_max ** 2

    return {"growth": float(growth), "leverage": float(leverage),
            "trunc_frac": trunc_frac, "risk": float(leverage) * trunc_frac,
            "fz": float(fz), "ap": float(ap), "rpm": float(rpm)}


def _parse_axis(spec: str, default: float) -> tuple:
    """"lo,hi" -> (range, True);  "v" or None -> (single value, False)."""
    if spec is None:
        return (default, default), False
    if "," in spec:
        lo, hi = (float(v) for v in spec.split(","))
        return (lo, hi), True
    return (float(spec), float(spec)), False


def sweep(ref, specs: dict, n=40):
    """specs: {'fz': str|None, 'ap': str|None, 'rpm': str|None} - exactly two
    must be ranges ("lo,hi"); the third a single value or None (-> reference).
    Returns (grid, x_name, x_vals, y_name, y_vals, fixed_name, fixed_val)."""
    parsed = {ax: _parse_axis(specs.get(ax), ref[f"{ax}0"]) for ax in AXES}
    swept = [ax for ax in AXES if parsed[ax][1]]
    if len(swept) != 2:
        raise ValueError(f"give exactly two of --fz/--ap/--rpm as a range "
                         f"(lo,hi) and the third as a single value - got "
                         f"{len(swept)} ranges: {swept}")
    fixed = [ax for ax in AXES if ax not in swept][0]
    x_name, y_name = swept
    x_vals = np.geomspace(*parsed[x_name][0], n)
    y_vals = np.geomspace(*parsed[y_name][0], n)
    fixed_val = parsed[fixed][0][0]

    grid = []
    for yv in y_vals:
        row = []
        for xv in x_vals:
            kwargs = {x_name: xv, y_name: yv, fixed: fixed_val}
            row.append(evaluate(ref, **kwargs))
        grid.append(row)
    return grid, x_name, x_vals, y_name, y_vals, fixed, fixed_val


def best_pair(grid, leverage_max, trunc_max):
    """The safe cell maximising fz*ap*rpm - proportional to volumetric
    material-removal rate, and the right objective regardless of which two
    axes happen to be swept (the third sits at its fixed value in every
    cell, so it still contributes correctly)."""
    best = None
    for row in grid:
        for c in row:
            if c["leverage"] <= leverage_max and c["trunc_frac"] <= trunc_max:
                score = c["fz"] * c["ap"] * c["rpm"]
                if best is None or score > best["_score"]:
                    best = {**c, "_score": score}
    return best


def figure(path, grid, x_name, x_vals, y_name, y_vals, fixed_name, fixed_val,
          ref, leverage_max, trunc_max, best):
    plt = plots._mpl(backend="Agg" if path else None)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    def arr(key):
        return np.array([[c[key] for c in row] for row in grid])

    lev, trunc = arr("leverage"), arr("trunc_frac")
    extent = (x_vals[0], x_vals[-1], y_vals[0], y_vals[-1])

    panels = (
        (axes[0], lev, "leverage", lev <= leverage_max, "leverage"),
        (axes[1], trunc, "delay-truncation error  (omega T)^2/2",
         trunc <= trunc_max, "trunc_frac"),
    )
    for ax, data, title, safe_mask, cbar_label in panels:
        im = ax.imshow(data, origin="lower", extent=extent, aspect="auto",
                       cmap="magma_r")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.contour(x_vals, y_vals, safe_mask.astype(float), levels=[0.5],
                  colors=[plots.C3], linewidths=2.0)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title(title, loc="left", color=plots.INK, fontsize=10)
        fig.colorbar(im, ax=ax, label=cbar_label)
        ref_xy = (ref[f"{x_name}0"], ref[f"{y_name}0"])
        ax.scatter(*ref_xy, marker="x", s=80, color=plots.INK, zorder=5,
                  label="run's own point")
        if best:
            ax.scatter([best[x_name]], [best[y_name]], marker="*", s=200,
                      color=plots.C3, zorder=5, edgecolor=plots.INK,
                      label="best safe pair")
        ax.legend(fontsize=7.5, loc="upper left")

    fig.suptitle(f"{ref['run']}  -  {x_name} x {y_name}  ({fixed_name} fixed "
                f"at {fixed_val:g})  -  red contour = safe boundary",
                x=0.01, ha="left", fontsize=10, color=plots.INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if path:
        return plots._save(fig, path)
    plt.show()
    return None


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--fz", default="0.18", metavar="LO,HI | VALUE",
                   help="held fixed by default - fz moves neither safety "
                        "factor, see the module docstring")
    p.add_argument("--ap", default="0.2,8.0", metavar="LO,HI | VALUE")
    p.add_argument("--rpm", default="1000,20000", metavar="LO,HI | VALUE")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--leverage-max", type=float, default=2.0)
    p.add_argument("--trunc-max", type=float, default=0.05,
                   help="max (omega T)^2/2 - 0.042 was fine at 10000 rpm on "
                        "alpha090_ae60, 0.167 was not at 5000 rpm")
    p.add_argument("--ds", type=float, default=2.0, metavar="MM")
    p.add_argument("--save-dir", default=None)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    ref = build_reference(a.run, ds_mm=a.ds, verbose=a.verbose)
    print(f"run      {ref['run']}")
    print(f"         open-loop growth {ref['g_open']:.3f} 1/s at {ref['hz_open']:.1f} Hz "
         f"(rpm-independent)")
    print(f"         own operating point: fz {ref['fz0']:.4f} mm, ap {ref['ap0']:.2f} mm, "
         f"rpm {ref['rpm0']:g}, ae {ref['ae_mm']:.2f} mm")
    own = evaluate(ref, ref["fz0"], ref["ap0"], ref["rpm0"])
    print(f"         at that point: growth {own['growth']:+.3f} 1/s, "
         f"leverage {own['leverage']:.2f}, trunc_frac {own['trunc_frac']:.3f}, "
         f"risk (product) {own['risk']:.3f}")

    grid, x_name, x_vals, y_name, y_vals, fixed_name, fixed_val = sweep(
        ref, {"fz": a.fz, "ap": a.ap, "rpm": a.rpm}, n=a.n)
    best = best_pair(grid, a.leverage_max, a.trunc_max)

    print(f"\nswept    {x_name} x {y_name}, {fixed_name} fixed at {fixed_val:g}")
    print(f"thresholds  leverage <= {a.leverage_max:g}, trunc_frac <= {a.trunc_max:g}")
    if best:
        print("best safe pair:")
        print(f"  fz = {best['fz']:.4f} mm/tooth, ap = {best['ap']:.3f} mm, "
             f"rpm = {best['rpm']:.0f}")
        print(f"  -> growth {best['growth']:+.3f} 1/s, leverage {best['leverage']:.2f}, "
             f"trunc_frac {best['trunc_frac']:.3f}, risk {best['risk']:.3f}")
    else:
        print("  no point in the swept range satisfies both thresholds - "
             "widen the ranges or relax a threshold")

    if not a.no_plot:
        save_dir = Path(a.save_dir) if a.save_dir else Path(a.run) / "validity"
        out = figure(save_dir / f"validity_{x_name}_{y_name}.png", grid,
                    x_name, x_vals, y_name, y_vals, fixed_name, fixed_val,
                    ref, a.leverage_max, a.trunc_max, best)
        print(f"\nfigure   {out}")
    return grid, best


if __name__ == "__main__":
    main()
