"""One coupled robot milling pass across a rectangular workpiece, entry to exit.

    python main.py
    python main.py --no-compensate
    python main.py --ap 2.0 --rpm 2000 --name deep_slow
    python main.py --predict-only            the linear side, no simulation

WHAT IT DOES, IN ORDER

    1  build the job and plan the constant-feed toolpath      analysis.toolpath
    2  linearise the arm at the start pose -> M/D/K -> G(s)   analysis.plant
    3  measure the engagement geometry once along the path    analysis.stability
    4  predict stability at every node, and F0(s) with it     analysis.stability
    5  build the mean-force feedforward from that F0          analysis.feedforward
    6  run the dexel cut with the arm in the loop             analysis.sim_coupled
    7  score both sides and write everything to out/<name>/   analysis.save

Steps 3-4 are the linear surrogate; step 6 is the truth. They share the geometry
measurement, the plant and the cutting coefficients, so what separates them is
the LINEARISATION of the cut and nothing else.

THE COMPENSATION IS ON BY DEFAULT AND SHOULD STAY ON.

The mean cutting force presses this arm out of the workpiece by more than a
tenth of a millimetre against a 5 mm radial engagement. Uncompensated, the engine
cuts at that deflected position while the prediction measures the commanded one,
and the comparison is contaminated by a geometry error before the linearisation
is asked anything. `--no-compensate` runs it anyway, because seeing the size of
the effect is the point of having the flag.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis                                              # noqa: E402
from analysis import config as acfg                          # noqa: E402
from analysis import (feedforward, forces, plant, report,     # noqa: E402
                      save, sim_coupled, stability, toolpath)


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
    g.add_argument("--part-length", type=float, default=None, metavar="MM",
                   help="stock length [mm]; this is the knob that sets how much "
                        "CUT there is, and every dynamic measurement is bounded "
                        "by it - see analysis.growth")
    g.add_argument("--part-width", type=float, default=None, metavar="MM",
                   help="stock width [mm]; only an ae-wide strip is milled, so "
                        "trim this when lengthening the part - the raster is a "
                        "grid over the whole bounding box")
    g.add_argument("--rpm", type=float, default=None, help="spindle speed")
    g.add_argument("--feed", type=float, default=None, metavar="MM_S",
                   help="max feed [mm/s]; regenerates the toolpath")
    g.add_argument("--teeth", type=int, default=None, metavar="N",
                   help="flutes on the cutter")
    g.add_argument("--feed-profile", default="flying",
                   choices=("ramped", "flying"),
                   help="'flying' (default) opens at full feed and stays there, "
                        "so the deviation carries only the cut; 'ramped' puts "
                        "smoothstep feed ramps in the lead-in and lead-out, "
                        "which is what a real machine does but adds the arm's "
                        "commanded-acceleration tracking error on top")
    g.add_argument("--fz", type=float, default=None, metavar="MM",
                   help="chip load [mm/tooth]; sets the feed from the rpm and "
                        "the tooth count instead of the other way round. "
                        "Mutually exclusive with --feed")
    g.add_argument("--steps-per-tooth", type=float, default=None, metavar="N",
                   help="set sim_dt from the TOOTH PERIOD, dt = 60/(rpm N spt), "
                        "so a tooth-passing sweep stays equally resolved at "
                        "every point. Overrides --sim-dt; capped at "
                        f"{acfg.MAX_SIM_DT:g} s so the arm's own modes stay "
                        "resolved at the slow corner")
    g.add_argument("--ktc", type=float, default=None, help="Ktc [N/mm^2]")
    g.add_argument("--krc", type=float, default=None, help="Krc [N/mm^2]")
    g.add_argument("--sim-dt", type=float, default=None, help="integration step [s]")
    g.add_argument("--raster", type=float, default=None, metavar="MM",
                   help="dexel raster [mm]; the cost driver of the engine")
    g.add_argument("--robot-model", default=None,
                   help="which arm (default: joints123)")

    g = p.add_argument_group("the model")
    g.add_argument("--ds", type=float, default=2.0, metavar="MM",
                   help="arc-length spacing of the stability nodes [mm]")
    g.add_argument("--coupling", default="both",
                   choices=("both", "damping", "stiffness", "none"),
                   help="which cut terms the prediction closes")

    g.add_argument("--linearize-at", type=float, default=None, metavar="FRAC",
                   help="linearise the arm at the commanded pose FRAC of the way "
                        "along the path (0.5 = the halfway mark) instead of at the "
                        "start pose, and report the prediction at that node")

    g = p.add_argument_group("the growth-rate tap")
    g.add_argument("--pulse-at", type=float, default=None, metavar="FRAC",
                   help="tap the TCP at FRAC of the path during the coupled pass. "
                        "Run once with and once without it and subtract - see "
                        "analysis.pulse. Default off")
    g.add_argument("--pulse-force", type=float, default=20.0, metavar="N",
                   help="tap magnitude [N]")
    g.add_argument("--pulse-ms", type=float, default=5.0, metavar="MS",
                   help="tap duration [ms]")
    g.add_argument("--pulse-dir", default="auto", metavar="X,Y,Z|auto",
                   help="tap direction, workpiece frame (normalised). 'auto' "
                        "(default) taps along the input that best excites the "
                        "LEAST-DAMPED closed-loop mode at the tap node, so the "
                        "response is dominated by the mode whose rate is being "
                        "measured; needs --linearize-at, else falls back to 1,1,1")

    g = p.add_argument_group("the DC compensation")
    g.add_argument("--no-compensate", action="store_true",
                   help="leave the motors idle; the joint springs carry the "
                        "whole mean cutting force and the tool is pressed out "
                        "of the workpiece")
    g.add_argument("--comp-gain", type=float, default=1.0,
                   help="scale the feedforward (1.0 = carry all of F0)")
    g.add_argument("--comp-axes", default="xy",
                   help="which force components the motors carry (default xy)")

    g = p.add_argument_group("running")
    g.add_argument("--predict-only", action="store_true",
                   help="the linear side only - no time-domain pass")
    g.add_argument("--steady-state", action="store_true",
                   help="pre-carve the entry slot and open already cutting; "
                        "deletes the entry transient this job is about")
    g.add_argument("--csv-decimate", type=int, default=10, metavar="N",
                   help="keep every N-th sample in timeseries.csv (the npz "
                        "always holds every sample)")
    g.add_argument("--no-plots", action="store_true",
                   help="skip the figures (nothing is imported from matplotlib)")
    g.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def default_name(cfg, compensated) -> str:
    """The run directory, named after the operating point it is.

    The TOOTH COUNT is in the name because it is a swept variable, not a fixed
    property of the shop: two runs that differ only in `N` are a different cut
    at the same feed, and without `z` in the name they would collide in `out/`.
    """
    m = cfg.milling()
    return (f"ap{cfg.part.height_mm:g}_ae{m.radial_engagement_mm:g}"
            f"_rpm{m.spindle_rpm:g}_z{m.n_teeth:g}_f{m.feed_mm_s:g}"
            f"_{'comp' if compensated else 'plain'}").replace(".", "p")


def main(argv=None):
    a = parse_args(argv)
    lines = []

    def say(text=""):
        print(text, flush=True)
        lines.append(str(text))

    # ── 1. the job ───────────────────────────────────────────────────────────
    cfg = acfg.load_base(a.config)
    cfg = acfg.apply_operating_point(
        cfg, ap_mm=a.ap, ae_mm=a.ae, rpm=a.rpm, feed_mm_s=a.feed,
        n_teeth=a.teeth, fz_mm=a.fz, steps_per_tooth=a.steps_per_tooth,
        part_length_mm=a.part_length, part_width_mm=a.part_width,
        Ktc=a.ktc, Krc=a.krc, sim_dt=a.sim_dt, raster_mm=a.raster,
        robot_model=a.robot_model)
    acfg.check_compliant(cfg)

    compensated = not a.no_compensate
    name = a.name or default_name(cfg, compensated)
    out_root = Path(a.out) if a.out else analysis.OUT
    d = save.run_dir(out_root, name)

    # A new max feed needs a new constant-feed profile, not a rescaled one, and
    # each feed gets its own file so a sweep cannot overwrite the path another
    # run is replaying. `ensure` returns the config pointed at what it wrote.
    cfg = toolpath.ensure(cfg, profile=a.feed_profile, verbose=True)

    setup = stability.prepare(cfg, ds_mm=a.ds, verbose=a.verbose)
    say(setup.summary())

    # ── 2. the linear replacement for the arm ────────────────────────────────
    pose_info = {"plant_pose_frac": None}
    if a.linearize_at is not None:
        receptance, pose_info = plant.receptance_at_fraction(
            setup, a.linearize_at, verbose=a.verbose)
        say(f"plant    linearised at {100 * a.linearize_at:g}% of the path "
            f"(s = {pose_info['plant_pose_s_mm']:.1f} mm, "
            f"t = {pose_info['plant_pose_t_s']:.3f} s)")
    else:
        receptance = plant.receptance_from_robot(cfg)
    say(plant.compliance_report(receptance, setup.scene))

    # ── 3-4. the prediction, node by node along the trajectory ───────────────
    stab = stability.predict(setup, receptance, coupling=a.coupling,
                             verbose=a.verbose)
    ap_crit = stability.critical_depth(stab)
    row = {"run": name, "compensated": compensated,
           **acfg.operating_point_row(cfg),
           **stability.prediction_row(stab, ap_crit)}
    # The model's own truncation parameter at this operating point. Recorded, not
    # judged: both cut terms drop the regenerative delay after one order, so this
    # is the axis the tooth-passing sweep is really about.
    wT = np.asarray(stab.omega_T(), float)
    row["omega_T_max"] = float(np.nanmax(wT)) if np.isfinite(wT).any() else np.nan
    row["tooth_trunc_frac"] = 0.5 * row["omega_T_max"] ** 2
    row.update(pose_info)
    tap = _tap_spec(a)
    if a.linearize_at is not None:
        row.update(stability.mid_node_row(stab, ap_crit, a.linearize_at))
        extra, tap = _linear_tap(stab, receptance, setup, tap,
                                 auto_dir=str(a.pulse_dir).lower() == "auto")
        row.update(extra)
    say(_prediction_block(row))

    # ── 5. the mean-force feedforward ────────────────────────────────────────
    ff = feedforward.build(setup, receptance, axes=a.comp_axes,
                           gain=a.comp_gain if compensated else 0.0,
                           verbose=a.verbose)
    say("comp     TCP DC compensation: tau_ff = -J^T F0(s), carried on the motors")
    say(ff.summary(ae_mm=setup.mill.radial_engagement_mm))
    row.update(ff.row())

    # ── 6. the truth ─────────────────────────────────────────────────────────
    run, force_rows = None, []
    if a.predict_only:
        say("\nsim      skipped (--predict-only)")
    else:
        say("")
        run = sim_coupled.simulate(cfg, ff if compensated else None,
                                   steady_state=a.steady_state,
                                   pulse=(tap if a.pulse_at is not None else None),
                                   verbose=a.verbose)
        row.update(report.chatter_metrics(run, setup.mill,
                                          modes_hz=receptance.modes_hz))
        row.update(forces.force_error(run, stab, setup.mill))
        row.update(report.agreement(row))
        force_rows = forces.table(run, stab, setup.mill)
        say("")
        say(_measurement_block(row))
        say(forces.summary(row))
        say(_verdict_block(row))

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
        from analysis import plots          # imported only when it is wanted
        for k, v in plots.run_figures(d, stab, ff, run, setup, receptance).items():
            written[k] = v
    say("")
    say(f"out      {d}")
    for k, v in written.items():
        say(f"         {k:<18} {Path(v).relative_to(d)}")
    (d / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return row


def _tap_spec(a):
    """The `Pulse` this run would apply (always built - its timing also sets the
    linear side's fit window, pulse or not)."""
    from analysis.pulse import Pulse
    if str(a.pulse_dir).lower() == "auto":
        d = (1.0, 1.0, 1.0)            # resolved from the modes in `_linear_tap`
    else:
        d = tuple(float(v) for v in str(a.pulse_dir).split(","))
        if len(d) != 3:
            raise SystemExit(f"--pulse-dir needs three components, got {a.pulse_dir!r}")
    at = a.pulse_at if a.pulse_at is not None else (
        a.linearize_at if a.linearize_at is not None else 0.5)
    return Pulse(at_frac=float(at), force_N=float(a.pulse_force),
                 duration_s=1e-3 * float(a.pulse_ms), dir_w=d)


def _dominant_input(A, B):
    """(3,) unit input that best excites the least-damped mode of `A`.

    With left eigenvector `w` of the dominant eigenvalue, an input `u` enters that
    mode as `w^H B u`. For a real `u` the magnitude is maximised by the top right
    singular vector of `[Re(w^H B); Im(w^H B)]`. Tapping along it makes the
    response one mode rather than a beat between several, which is what lets an
    envelope fit over ~3 time constants recover that mode's rate.
    """
    from scipy.linalg import eig
    lam, wl = eig(A, left=True, right=False)
    k = int(np.argmax(lam.real))
    c = wl[:, k].conj() @ B
    _u, _s, vt = np.linalg.svd(np.vstack([c.real, c.imag]))
    return vt[0] / np.linalg.norm(vt[0])


def _linear_tap(stab, receptance, setup, tap, *, auto_dir=False):
    """(row fields, tap) - sigma_lin_fit, the fit window, and the tap direction.

    The tap through the linear closed loop at the node nearest the pulse,
    fitted with the same estimator as the coupled pass - so the sweep can tell
    an estimator limit from a model error. The window is stored in the row and
    reused, unchanged, for the twin-run fit.
    """
    from dataclasses import replace as _replace

    from analysis import pulse as pmod
    from stabsim.stability import closed_loop_matrix

    s = np.asarray(stab.s_mm, float)
    eng = np.asarray(stab.engaged, bool)
    out = {"pulse_at_frac": tap.at_frac, "pulse_force_N": tap.force_N,
           "pulse_ms": 1e3 * tap.duration_s, "pulse_dir_mode": (
               "auto" if auto_dir else "fixed"),
           "pred_pulse_fit_1_s": np.nan, "pred_pulse_fit_r2": np.nan}
    if not eng.any():
        out["pulse_dir_w"] = ",".join(f"{v:.4g}" for v in tap.dir_w)
        return out, tap
    t_pulse = pmod.time_at_fraction(setup.path, tap.at_frac)
    t_exit = pmod.time_at_s(setup.path, float(s[eng].max()))
    lam = float(np.linalg.eigvals(receptance.A).real.max())
    tau = -1.0 / lam if lam < 0 else 0.32
    modes = np.asarray(receptance.modes_hz, float)
    modes = modes[np.isfinite(modes) & (modes > 0)]
    f_low = float(modes.min())
    band = (0.4 * f_low, 2.5 * float(modes.max()))
    t0, t1, t_min = pmod.fit_window(t_pulse, tap.duration_s, t_exit, tau, f_low)
    out.update({"pulse_t_s": t_pulse, "pulse_fit_t0_s": t0, "pulse_fit_t1_s": t1,
                "pulse_fit_ok": bool(t1 - t0 >= t_min),
                "pulse_band_lo_hz": band[0], "pulse_band_hi_hz": band[1],
                "pulse_f_lowest_hz": f_low, "plant_tau_s": tau})
    i = int(np.argmin(np.abs(s - pmod.path_arclength_mm(setup.path)[-1]
                             * tap.at_frac)))
    K = None if stab.K_p is None else stab.K_p[i]
    C = None if stab.C_p is None else stab.C_p[i]
    # plant frame <-> workpiece frame for the force
    R_pw = (np.eye(3) if receptance.frame == "workpiece"
            else np.asarray(setup.scene.R_iw, float))
    if auto_dir:
        u_plant = _dominant_input(closed_loop_matrix(receptance, K, C),
                                  receptance.B)
        tap = _replace(tap, dir_w=tuple(float(v) for v in R_pw.T @ u_plant))
    out["pulse_dir_w"] = ",".join(f"{v:.4g}" for v in tap.dir_w)
    if t1 - t0 < t_min:
        return out, tap
    f_plant = R_pw @ tap.force_w()
    sig, r2 = pmod.linear_pulse_rate(
        receptance, K, C, f_plant, tap, fit_t0_rel=t0 - t_pulse,
        fit_t1_rel=t1 - t_pulse, band=band, f_lowest_hz=f_low)
    out.update({"pred_pulse_fit_1_s": sig, "pred_pulse_fit_r2": r2})
    return out, tap


def _prediction_block(row) -> str:
    return (
        "pred     linear stability along the trajectory "
        f"({row['pred_engaged_nodes']} engaged nodes)\n"
        f"         growth   {row['pred_growth_max_trim_1_s']:+7.2f} 1/s trimmed | "
        f"{row['pred_growth_max_1_s']:+7.2f} untrimmed | "
        f"{row['pred_growth_entry_1_s']:+7.2f} at the entry node\n"
        f"         verdict  {'UNSTABLE' if row['pred_unstable_trim'] else 'stable'}"
        f" over the trimmed span, "
        f"{row['pred_unstable_fraction'] * 100:.0f}% of engaged nodes above zero"
        f" | mode {row['pred_mode_hz']:.1f} Hz\n"
        f"         ap_crit  {row['pred_ap_crit_trim_mm']:.2f} mm is the depth the "
        f"whole trimmed cut survives (running at "
        f"{row['axial_depth_mm']:g} mm)")


def _measurement_block(row) -> str:
    if not row.get("sim_valid", False):
        return ("meas     the pass produced nothing measurable - it never engaged, "
                "or it was cut short before an envelope could be fitted")
    return (
        "meas     the simulated pass\n"
        f"         deviation from the command  DC {row['sim_dc_dev_um']:7.1f} um | "
        f"AC {row['sim_ac_rms_um']:6.1f} um rms | peak {row['sim_peak_dev_um']:7.1f} um\n"
        f"         growth   {row['sim_growth_1_s']:+7.2f} 1/s "
        f"(+/- {row['sim_growth_se_1_s']:.2f}, R2 {row['sim_growth_r2']:.2f}) "
        f"at {row['sim_chatter_hz']:.1f} Hz over {row['sim_engaged_s']:.2f} s engaged\n"
        f"         verdict  "
        f"{'DIVERGED' if row['sim_diverged'] else ('UNSTABLE' if row['sim_unstable'] else 'stable')}"
        f" | peak |F| {row['sim_peak_force_N']:.0f} N")


def _verdict_block(row) -> str:
    agree = row.get("agree_trim")
    mark = "-" if agree is None else ("they agree" if agree else "THEY DISAGREE")
    err = row.get("growth_err_trim_1_s", np.nan)
    dc = row.get("sim_dc_dev_um", np.nan)
    note = ""
    if np.isfinite(dc):
        note = (f"\n         residual DC deviation {dc:.1f} um - what the "
                f"feedforward did not cancel" if row.get("ff_applied") else
                f"\n         DC deviation {dc:.1f} um - uncompensated, so the "
                f"engine cut a different geometry than the prediction assumed")
    return (f"verdict  {mark}: prediction "
            f"{'unstable' if row['pred_unstable_trim'] else 'stable'}, "
            f"simulation "
            f"{'unstable' if row.get('sim_unstable') else 'stable'}"
            f" | growth rate differs by {err:+.2f} 1/s{note}")


if __name__ == "__main__":
    main()
