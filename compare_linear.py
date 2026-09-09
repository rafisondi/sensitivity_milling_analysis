"""Integrate the linear cut model along a finished pass, and overlay the two.

    python compare_linear.py --rpm 5000 --teeth 4 --fz 0.18
    python compare_linear.py --rpm 3333 --teeth 4 --feed 40
    python compare_linear.py --run out/tooth_passing/runs/grid_fz0p18_rpm5000_z4

WHAT THIS IS, AND HOW IT DIFFERS FROM `analysis.stability`

`stability_along_path` freezes the cut at each node and reports `max Re(eig)` -
one eigenvalue per node, no time in it. This script instead SCHEDULES the same
three matrices along the path and INTEGRATES them, so the linear model produces a
deflection TRAJECTORY that can be laid on top of the coupled run sample by
sample:

    z     = [dx, dxd]                       the plant's own state, 6 of them
    zdot  = A z + B f
    f(t)  = drive(t) + K_cut(s(t)) dx + C_cut(s(t)) dxd

That is a much sharper test of the linearisation than the eigenvalue: the
eigenvalue only has to get a decay rate right, while this has to reproduce the
entry ramp, the plateau offset and the exit, in all three axes at once.

WHAT DRIVES IT, AND WHY THAT IS NOT SIMPLY `F0`

The run being compared against carried its mean force on the MOTORS
(`analysis.feedforward`), so the joint springs never saw the part of `F0` the
feedforward took. Driving the linear model with the whole of `F0` would compare
a compensated simulation against an uncompensated prediction, and the offset
between them would be the compensation rather than the model.

So the default drive is what the run actually left on the structure,

    drive(t) = F0(t) - carried(t)

read straight out of that run's own `forces/feedforward.npz`. With the stock
`--comp-axes xy` the in-plane part of `F0` is fully carried and the axial part is
not, so the drive comes out as very nearly `[0, 0, F0_z]`. That is not a
degenerate case, it is the point: the linear model then predicts almost NO
in-plane deflection, and whatever in-plane deflection the run actually shows is
the error in `F0` itself, converted to micrometres by the arm's compliance.

`--drive uncompensated` feeds it the whole of `F0` instead, which is the richer
signal and the right comparison against a `--no-compensate` run.

FRAMES. Everything here is the WORKPIECE frame: the receptance is rotated into
it before the cut matrices are built, so `K_cut`/`C_cut` come out on the same
axes as `deflection_w_um` and `force_w` in the run's npz, and no rotation is
applied to anything afterwards.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis                                                # noqa: E402
from analysis import plots, stability                          # noqa: E402
from analysis import plant as plant_mod                        # noqa: E402
from runconfig import RunConfig                                # noqa: E402

DRIVES = ("as-run", "uncompensated")


# ─────────────────────────────────────────────────────────────────────────────
# Finding the run
# ─────────────────────────────────────────────────────────────────────────────

def _cell_of(cfg_json) -> tuple:
    """(rpm, teeth, fz) of a saved config."""
    mill, path = cfg_json["mill"], cfg_json["path"]
    rpm = float(mill["spindle_rpm"])
    n = int(mill["n_teeth"])
    feed = float(path["speed_mm_s"])
    return rpm, n, feed / (rpm / 60.0 * n), feed


def find_run(out_root, rpm, teeth, fz=None, feed=None, tol=1e-3):
    """The run directory matching this operating point, or raise with the options."""
    found = []
    for cfg_file in sorted(Path(out_root).glob("**/config.json")):
        d = cfg_file.parent
        if not (d / "raw" / "coupled.npz").exists():
            continue                                  # prediction only, nothing to compare
        try:
            got = _cell_of(json.loads(cfg_file.read_text(encoding="utf-8")))
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
        found.append((d, got))

    hits = [d for d, (r, n, z, v) in found
            if abs(r - rpm) < 1.0 and n == int(teeth)
            and (fz is None or abs(z - fz) < tol)
            and (feed is None or abs(v - feed) < 1e-2)]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        near = sorted({f"{r:g} rpm / {n} teeth / fz {z:.4f} / feed {v:g}"
                       for _, (r, n, z, v) in found})
        want = (f"{rpm:g} rpm / {teeth} teeth"
                + (f" / fz {fz:g}" if fz else "")
                + (f" / feed {feed:g}" if feed else ""))
        raise SystemExit(
            f"no simulated run at {want}.\nRuns with a coupled pass under "
            f"{out_root}:\n  " + "\n  ".join(near))
    raise SystemExit(
        "that operating point matches more than one run - pass --run:\n  "
        + "\n  ".join(str(h) for h in hits))


# ─────────────────────────────────────────────────────────────────────────────
# The linear pass
# ─────────────────────────────────────────────────────────────────────────────

def _repin(cfg):
    """Point a saved config's `io` at THIS workspace.

    `config.json` records absolute directories, so a run produced anywhere else -
    another machine, a container, a moved checkout - carries paths that do not
    exist here and `cfg.build()` fails looking for its own toolpath. Only the
    directories are rewritten; the operating point, the toolpath NAME and the
    scene are the record of what was run and are left exactly as they were.
    """
    from dataclasses import replace

    from analysis import config as acfg
    return replace(cfg, io=replace(
        cfg.io, out_dir=str(analysis.OUT),
        workpiece_dir=str(acfg.WORKPIECE_DIR),
        toolpath_dir=str(acfg.TOOLPATH_DIR)))


def schedule(d: Path, *, ds_mm=2.0, drive="as-run", verbose=False):
    """Rebuild this run's linear model and put every term on its time grid.

    Returns a dict with the simulation's own histories and the scheduled cut
    matrices, all on the SAME `t`, all in the workpiece frame.

    The cut matrices are recomputed rather than read back, because
    `StabilityAlongPath.save_npz` keeps the verdict and the geometry but not
    `K_p`/`C_p` - and recomputing them from the saved config is exact, since the
    prediction is a pure function of it.
    """
    cfg = _repin(RunConfig.load(str(d / "config.json")))
    with np.load(d / "raw" / "coupled.npz") as z:
        t = np.asarray(z["t"], float)
        force_w = np.asarray(z["force_w"], float)
        dev_um = np.asarray(z["deflection_w_um"], float)
    with np.load(d / "forces" / "feedforward.npz") as z:
        ff_t = np.asarray(z["t"], float)
        ff_s = np.asarray(z["s_mm"], float)
        F0_ff = np.asarray(z["F0_w"], float)
        carried = np.asarray(z["carried_w"], float)

    setup = stability.prepare(cfg, ds_mm=ds_mm, verbose=verbose)
    # In the WORKPIECE frame, so `K_p`/`C_p` come back on the axes the run's
    # deflection and force are already written on.
    recep = plant_mod.receptance_from_robot(cfg).in_workpiece(setup.scene)
    stab = stability.predict(setup, recep, verbose=verbose)

    # s(t) on the simulation grid, then every scheduled term against it.
    s_t = np.interp(t, ff_t, ff_s)
    node_s = np.asarray(stab.s_mm, float)

    def on_t(node_values):
        """(n_node, ...) sampled at the nodes -> (n_t, ...) on the sim grid."""
        v = np.asarray(node_values, float)
        flat = v.reshape(len(node_s), -1)
        out = np.column_stack([np.interp(s_t, node_s, flat[:, j])
                               for j in range(flat.shape[1])])
        return out.reshape((len(t),) + v.shape[1:])

    K_t = on_t(stab.K_p if stab.K_p is not None else np.zeros((len(node_s), 3, 3)))
    C_t = on_t(stab.C_p if stab.C_p is not None else np.zeros((len(node_s), 3, 3)))
    F0_t = on_t(stab.F0_w)

    if drive == "uncompensated":
        drive_t = F0_t
    else:
        # Exactly what this run's motors left for the joint springs to carry.
        drive_t = np.column_stack(
            [np.interp(t, ff_t, (F0_ff - carried)[:, j]) for j in range(3)])

    # THE QUASI-STATIC EXPECTATION, for reference against the integrated one.
    # `G(0) drive` is where a steady load of the current drive would hold the
    # tool: the whole prediction you get from the compliance alone, with no
    # dynamics, no `K_cut` and no `C_cut` in it. On the `from_mdk` plant this is
    # exactly `K^-1`, so it is not an extrapolation - it is the DC limit the
    # integrated response should settle onto wherever the drive is flat, and the
    # gap between the two curves is precisely what the cut terms and the plant's
    # inertia are contributing.
    G0 = recep.dc_gain                                     # (3, 3) [m/N]
    qs_dev_um = (drive_t @ G0.T) * 1e6
    qs_F0 = F0_t

    return {"G0": G0, "qs_dev_um": qs_dev_um, "qs_F0": qs_F0,
            "cfg": cfg, "setup": setup, "receptance": recep, "stab": stab,
            "t": t, "s_mm": s_t, "force_w": force_w, "dev_um": dev_um,
            "K_t": K_t, "C_t": C_t, "F0_t": F0_t, "drive_t": drive_t,
            "mill": setup.mill, "run_dir": d, "drive": drive}


def integrate(sch) -> dict:
    """RK4 the scheduled linear model along the pass.

    THE DEFLECTION IS `C z`, NOT THE FIRST THREE STATES. `Receptance.rotated`
    moves the input and output maps and leaves `A` alone, so a receptance written
    on the workpiece axes still carries its state in the basis it was built in -
    `C` is `[R, 0]`, not `[I, 0]`. Reading `z[:3]` as the deflection therefore
    silently mixes the axes, and closing the cut on it feeds `K_cut` a rotated
    displacement: the loop stops being the one `closed_loop_matrix` analysed and
    goes unstable where the eigenvalues say it is comfortably damped.

    So the loop is closed the general way the library does it, off the output
    maps rather than off state indices:

        dx  = C z        dxd = C zdot = (C A) z + (C B) f

    `C B` is zero for anything built from M/D/K, which is what makes `dxd` a
    function of the state alone and keeps this an ODE rather than an algebraic
    loop - the same condition `closed_loop_matrix` checks before inverting.

    The cut matrices are held over each step and averaged for the half-step,
    which is exact enough: they change on the `ds_mm` node spacing, thousands of
    simulation steps apart, while the state changes on the arm's ~23 Hz modes.
    """
    r = sch["receptance"]
    A, B, Cm = r.A, r.B, r.C
    if np.abs(Cm @ B).max() > 1e-12 or np.abs(r.D).max() > 1e-12:
        raise NotImplementedError(
            "this plant is biproper (C B or D nonzero), so the damping term "
            "closes through an algebraic loop and needs a dF/dt state")
    CA = Cm @ A
    t, K_t, C_t, drive_t = sch["t"], sch["K_t"], sch["C_t"], sch["drive_t"]
    n, n_state = len(t), A.shape[0]

    def rhs(z, K, Cc, u):
        f = u + K @ (Cm @ z) + Cc @ (CA @ z)
        return A @ z + B @ f

    z = np.zeros(n_state)
    out = np.zeros((n, n_state))
    for k in range(n - 1):
        dt = t[k + 1] - t[k]
        Ka, Ca, ua = K_t[k], C_t[k], drive_t[k]
        Kb, Cb, ub = K_t[k + 1], C_t[k + 1], drive_t[k + 1]
        Kh, Ch, uh = 0.5 * (Ka + Kb), 0.5 * (Ca + Cb), 0.5 * (ua + ub)

        k1 = rhs(z, Ka, Ca, ua)
        k2 = rhs(z + 0.5 * dt * k1, Kh, Ch, uh)
        k3 = rhs(z + 0.5 * dt * k2, Kh, Ch, uh)
        k4 = rhs(z + dt * k3, Kb, Cb, ub)
        z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out[k + 1] = z
        if not np.all(np.isfinite(z)):
            print(f"         ! the linear pass diverged at t = {t[k+1]:.4f} s "
                  f"({100*(k+1)/n:.0f}% through) - the rest is not integrated")
            out[k + 1:] = np.nan
            break

    dx = out @ Cm.T                       # (n, 3) deflection [m]
    dxd = out @ CA.T                      # (n, 3) deflection rate [m/s]
    f_lin = (sch["F0_t"]
             + np.einsum("kij,kj->ki", sch["K_t"], dx)
             + np.einsum("kij,kj->ki", sch["C_t"], dxd))
    return {"dx_um": dx * 1e6, "dxd": dxd, "f_lin": f_lin, "state": out}


def revolution_average(t, F, mill):
    """(t, F) averaged over one full spindle revolution - the quantity `F0` is."""
    per_rev = max(1, int(round(60.0 / mill.spindle_rpm / (t[1] - t[0]))))
    m = len(t) // per_rev
    if m < 2:
        return t, F
    return (t[:m * per_rev].reshape(m, per_rev).mean(1),
            F[:m * per_rev].reshape(m, per_rev, 3).mean(1))


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def summary(sch, lin) -> str:
    m = sch["mill"]
    sim, mod, qs = sch["dev_um"], lin["dx_um"], sch["qs_dev_um"]
    ok = np.isfinite(mod).all(axis=1)
    axes = "xyz"

    lines = [
        f"run      {sch['run_dir'].name}",
        f"         {m.spindle_rpm:g} rpm, {m.n_teeth} teeth, "
        f"{m.feed_mm_s:g} mm/s -> {m.feed_per_tooth_mm():.4f} mm/tooth | "
        f"TPF {m.spindle_rpm * m.n_teeth / 60:.0f} Hz",
        f"         drive {sch['drive']!r}: |drive| max "
        f"{np.linalg.norm(sch['drive_t'], axis=1).max():.1f} N of |F0| max "
        f"{np.linalg.norm(sch['F0_t'], axis=1).max():.1f} N",
        "",
    ]

    # SCORE THE ENGAGED SPAN, NOT THE WHOLE PASS. The coupled run also contains
    # the arm's tracking error under the feed ramps, which live entirely in the
    # lead-in and lead-out and have nothing to do with the cut; the linear model
    # is driven by the cutting force alone and has no such term. At a fixed chip
    # load the feed rises with rpm, so those ramps grow with it - including them
    # would charge the linearisation for an excursion it never modelled.
    cut = np.linalg.norm(sch["force_w"], axis=1) > 50.0
    for label, mask in (("ENGAGED span only", ok & cut), ("whole pass", ok)):
        lines += [
            f"TCP deviation from the command [um] - {label}",
            f"         {'axis':<6} {'DC sim':>9} {'DC lin':>9} {'DC qstat':>9} "
            f"{'rms sim':>9} {'rms lin':>9} {'corr':>7}",
        ]
        for j, a in enumerate(axes):
            s, l, q = sim[mask, j], mod[mask, j], qs[mask, j]
            c = (float(np.corrcoef(s, l)[0, 1])
                 if s.std() > 1e-12 and l.std() > 1e-12 else np.nan)
            lines.append(f"         {a:<6} {s.mean():9.2f} {l.mean():9.2f} "
                         f"{q.mean():9.2f} {np.sqrt((s**2).mean()):9.2f} "
                         f"{np.sqrt((l**2).mean()):9.2f} {c:7.3f}")
        lines.append("")
    lines.pop()

    t_r, F_r = revolution_average(sch["t"], sch["force_w"], m)
    _t, F_l = revolution_average(sch["t"], lin["f_lin"], m)
    lines += ["",
              "cutting force, revolution-averaged over the whole pass [N]",
              f"         {'axis':<6} {'engine':>9} {'linear':>9} {'diff':>9}"]
    for j, a in enumerate(axes):
        e, l = F_r[:, j].mean(), F_l[:, j].mean()
        lines.append(f"         {a:<6} {e:9.2f} {l:9.2f} {l - e:9.2f}")
    return "\n".join(lines)


def figure(sch, lin, save_dir=None, expected=False):
    """Forces and deviation, x/y/z, engine against the integrated linear model.

    `expected` adds the QUASI-STATIC pair: the scheduled `F0` on its own, and the
    deflection `G(0) drive` a steady load of the applied drive would hold. They
    are the prediction with the dynamics taken out, so the distance from them to
    the integrated curve is what closing `K_cut`/`C_cut` around the plant and
    letting it ring actually bought - and the distance from them to the
    simulation is what a purely static compliance estimate would have got wrong.
    """
    # Only force Agg when the figure is going to a file. Showing one has to keep
    # whatever interactive back end is configured, or `plt.show()` does nothing.
    plt = plots._mpl(backend="Agg" if save_dir else None)

    t, m = sch["t"], sch["mill"]
    t_r, F_r = revolution_average(t, sch["force_w"], m)
    fig, axes = plt.subplots(3, 2, figsize=(13, 8), sharex=True)

    for j, a in enumerate("xyz"):
        # ── force ────────────────────────────────────────────────────────────
        ax = plots._clean(axes[j, 0])
        ax.plot(t, sch["force_w"][:, j], color=plots.SIM, lw=0.4, alpha=0.22,
                label="engine, every sample" if j == 0 else None)
        ax.plot(t_r, F_r[:, j], color=plots.SIM, lw=1.5,
                label="engine, per revolution" if j == 0 else None)
        ax.plot(t, lin["f_lin"][:, j], color=plots.MODEL, lw=1.4, ls=(0, (4, 2)),
                label="linear  F0 + K dx + C dxd" if j == 0 else None)
        if expected:
            ax.plot(t, sch["qs_F0"][:, j], color=plots.C4, lw=1.2,
                    ls=(0, (1, 1.6)),
                    label="expected  F0 alone" if j == 0 else None)
        ax.axhline(0.0, color=plots.MUTED, lw=0.8)
        ax.set_ylabel(f"F{a}  [N]")
        if j == 0:
            ax.set_title("cutting force", loc="left", color=plots.INK)
            ax.legend(loc="best", ncol=1)

        # ── deviation ────────────────────────────────────────────────────────
        ax = plots._clean(axes[j, 1])
        ax.plot(t, sch["dev_um"][:, j], color=plots.SIM, lw=1.0,
                label="coupled simulation" if j == 0 else None)
        ax.plot(t, lin["dx_um"][:, j], color=plots.MODEL, lw=1.4, ls=(0, (4, 2)),
                label="linear, integrated" if j == 0 else None)
        if expected:
            ax.plot(t, sch["qs_dev_um"][:, j], color=plots.C4, lw=1.2,
                    ls=(0, (1, 1.6)),
                    label="expected  G(0) . drive" if j == 0 else None)
        ax.axhline(0.0, color=plots.MUTED, lw=0.8)
        ax.set_ylabel(f"d{a}  [um]")
        if j == 0:
            ax.set_title("TCP deviation from the command", loc="left",
                         color=plots.INK)
            ax.legend(loc="best")

    for ax in axes[-1]:
        ax.set_xlabel("time  [s]")
    fig.suptitle(
        f"{sch['run_dir'].name}   -   {m.spindle_rpm:g} rpm, {m.n_teeth} teeth, "
        f"fz {m.feed_per_tooth_mm():.4f} mm, TPF "
        f"{m.spindle_rpm * m.n_teeth / 60:.0f} Hz   -   drive {sch['drive']!r}",
        x=0.005, ha="left", fontsize=11, color=plots.INK, fontweight="medium")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if save_dir:
        out = Path(save_dir) / f"{sch['run_dir'].name}_linear_vs_coupled.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        print(f"         figure -> {out}")
        return out
    plt.show()
    return None


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=None,
                   help="the run directory directly, instead of searching")
    p.add_argument("--rpm", type=float, default=None)
    p.add_argument("--teeth", type=int, default=None)
    p.add_argument("--fz", type=float, default=None, metavar="MM",
                   help="chip load [mm/tooth] - one of --fz / --feed")
    p.add_argument("--feed", type=float, default=None, metavar="MM_S")
    p.add_argument("--out", default=None, help="search root (default: out/)")
    p.add_argument("--expected", action="store_true",
                   help="also overlay the QUASI-STATIC expectation: F0 on its "
                        "own on the force panels, and G(0) . drive on the "
                        "deviation panels. The prediction with the dynamics "
                        "taken out, for reference against the integrated one")
    p.add_argument("--drive", default="as-run", choices=DRIVES,
                   help="'as-run' drives the linear model with F0 minus what "
                        "the run's motors carried; 'uncompensated' with all of F0")
    p.add_argument("--ds", type=float, default=2.0, metavar="MM",
                   help="arc-length spacing the cut matrices are built on")
    p.add_argument("--save-dir", default=None,
                   help="write the figure here instead of showing it "
                        "(off by default - the figure is shown, not saved)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    if a.run:
        d = Path(a.run)
    else:
        if a.rpm is None or a.teeth is None:
            raise SystemExit("give --rpm and --teeth (and --fz or --feed), "
                             "or point at a run with --run")
        d = find_run(Path(a.out) if a.out else analysis.OUT,
                     a.rpm, a.teeth, a.fz, a.feed)

    sch = schedule(d, ds_mm=a.ds, drive=a.drive, verbose=a.verbose)
    lin = integrate(sch)
    print(summary(sch, lin))
    figure(sch, lin, a.save_dir, expected=a.expected)
    return sch, lin


if __name__ == "__main__":
    main()
