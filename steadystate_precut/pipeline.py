"""One steady-state pass, prediction and truth, written out like `main.py` does.

    row = run(job, name="steady_s40", out_root=OUT)
    row = run_twin(job, name="steady_s40", out_root=OUT)     + the tapped twin

`main.main` in the same seven steps, with three differences:

  - the geometry is measured on the PRECUT stock (`precut.prepare`), not on the
    virgin part along a path that starts in the air;
  - the pass is always `steady_state=True` and always compensated - the engine
    carves and warms up to s0, and the arm opens preloaded on its tracking
    equilibrium with F0 carried on the motors;
  - `analysis.toolpath.ensure` is never called: it would re-point the config at
    the stock lead-in path, and the steady path is the whole point.

It is a copy rather than a set of options on `main.main` for the reason
`fastsim.coupled` gives: the loop belongs in the script that changes it.

WHAT TO READ STABILITY FROM

A preloaded steady cut has almost nothing in it to decay: no entry transient,
only the forced tooth-passing ripple, whose envelope is flat however damped the
machine is. `sim_growth_1_s` over such a pass is mostly a fit to a flat line
(low R^2) unless the cut is actually unstable. The decay rate of a deliberate
perturbation is the measurement - `run_twin` runs the pass twice, once with a
short tap on the TCP, and subtracts (`analysis.pulse`).
"""

import numpy as np

import main as run_main
from analysis import (feedforward, forces, plant, report, save, sim_coupled,
                      stability)
from analysis import config as acfg
from analysis import pulse as pmod
from analysis.pulse import Pulse
from steadystate_precut import precut


def _plant(job, setup, linearize_at, full: bool, verbose: bool):
    if linearize_at is None:
        fn = plant.full_receptance_from_robot if full else plant.receptance_from_robot
        return fn(job.cfg), {"plant_pose_frac": None}
    fn = plant.full_receptance_at_fraction if full else plant.receptance_at_fraction
    return fn(setup, linearize_at, verbose=verbose)


def _start_state(run, ff, mill) -> dict:
    """Whether the pass really opened in the steady state it was prepared for.

    The TCP is not ON the command at t = 0 and should not be: the motors carry
    only `comp_axes` of F0, the springs the rest, and the steady cut runs at
    that residual deflection. What matters is that it STARTS there, so the start
    is scored against the pass's own mean deflection: `ss_dev0_um` at t = 0,
    `ss_dev_rev1_max_um` over the first revolution (the latter includes the
    tooth-passing ripple). `F_rev1_N` is the engine's mean force over the first
    revolution against the F0 the preset assumed: close means the carved and
    warmed-up stock gave the tool the chip it expected.
    """
    d = run.result.deflection_w_um                     # (N, 3), vs the command
    d_mean = d.mean(axis=0)
    off = np.linalg.norm(d - d_mean, axis=1)
    n_rev = max(1, int(round(mill.steps_per_rev)))
    f = run.result.force_w[:n_rev, :2].mean(axis=0)
    f0 = ff.F0_w[0, :2]
    return {"ss_dev0_um": float(off[0]),
            "ss_dev_rev1_max_um": float(off[:n_rev].max()),
            "ss_defl_mean_um": float(np.linalg.norm(d_mean)),
            "ss_F_rev1_N": float(np.linalg.norm(f)),
            "ss_F0_start_N": float(np.linalg.norm(f0)),
            "ss_F_rev1_err_pct": float(100.0 * np.linalg.norm(f - f0)
                                       / max(np.linalg.norm(f0), 1e-9))}


def run(job, *, name, out_root, plant_kind="reduced",
        coupling="both", linearize_at=None, tap: Pulse = None,
        tap_at: float = None, tapped=False, auto_dir=True, comp_axes="xy",
        csv_decimate=10, no_plots=False, verbose=False) -> dict:
    """Predict and simulate one steady pass; everything to `out_root/name/`.

    With `tap_at` set, the tap's fit window and direction are resolved from the
    prediction; `tapped` decides whether the pulse is actually applied. Resolving
    them either way is what lets the plain and the tapped run of a twin share
    them.
    """
    lines = []

    def say(text=""):
        print(text, flush=True)
        lines.append(str(text))

    cfg = job.cfg
    d = save.run_dir(out_root, name)

    # ── 1. the job, on the precut stock ──────────────────────────────────────
    setup = precut.prepare(job, verbose=verbose)
    say(setup.summary())
    say(job.summary())

    # ── 2. the plant: at the start pose (= s0) unless told otherwise ─────────
    receptance, pose_info = _plant(job, setup, linearize_at,
                                   plant_kind == "full", verbose)
    if linearize_at is None:
        say("plant    linearised at the start pose - the tool centre on s0")
    else:
        say(f"plant    linearised at {100 * linearize_at:g}% of the steady path "
            f"(s = {job.s0_mm + pose_info['plant_pose_s_mm']:.1f} mm along the edge)")
    say(plant.compliance_report(receptance, setup.scene))

    # ── 3-4. the prediction ──────────────────────────────────────────────────
    stab = stability.predict(setup, receptance, coupling=coupling, verbose=verbose)
    ap_crit = stability.critical_depth(stab)
    row = {"run": name, "compensated": True, "steady_state": True,
           **acfg.operating_point_row(cfg), **job.row(),
           **stability.prediction_row(stab, ap_crit)}
    wT = np.asarray(stab.omega_T(), float)
    row["omega_T_max"] = float(np.nanmax(wT)) if np.isfinite(wT).any() else np.nan
    row["tooth_trunc_frac"] = 0.5 * row["omega_T_max"] ** 2
    row.update(pose_info)
    if tap_at is not None:
        tap = tap or Pulse()
        tap = Pulse(at_frac=float(tap_at), force_N=tap.force_N,
                    duration_s=tap.duration_s, dir_w=tap.dir_w)
        row.update(stability.mid_node_row(stab, ap_crit, tap_at))
        extra, tap = run_main._linear_tap(stab, receptance, setup, tap,
                                          auto_dir=auto_dir)
        row.update(extra)
    say(run_main._prediction_block(row))

    # ── 5. the feedforward: F0 on the motors from t = 0 ──────────────────────
    ff = feedforward.build(setup, receptance, axes=comp_axes, gain=1.0,
                           verbose=verbose)
    say("comp     TCP DC compensation: tau_ff = -J^T F0(s), carried on the motors")
    say(ff.summary(ae_mm=setup.mill.radial_engagement_mm))
    row.update(ff.row())

    # ── 6. the truth, opened mid-cut ─────────────────────────────────────────
    say("")
    pulse = tap if (tapped and tap_at is not None) else None
    row["tapped"] = pulse is not None
    run_ = sim_coupled.simulate(cfg, ff, steady_state=True, pulse=pulse,
                                verbose=verbose)
    row.update(_start_state(run_, ff, setup.mill))
    row.update(report.chatter_metrics(run_, setup.mill,
                                      modes_hz=receptance.modes_hz))
    row.update(forces.force_error(run_, stab, setup.mill))
    row.update(report.agreement(row))
    force_rows = forces.table(run_, stab, setup.mill)
    say("")
    say(f"start    TCP {row['ss_dev0_um']:.2f} um from the pass's own mean "
        f"deflection ({row['ss_defl_mean_um']:.1f} um) at t = 0, "
        f"{row['ss_dev_rev1_max_um']:.2f} um max over the first revolution\n"
        f"         first-revolution mean force {row['ss_F_rev1_N']:.0f} N against "
        f"the preset F0 {row['ss_F0_start_N']:.0f} N "
        f"({row['ss_F_rev1_err_pct']:.1f}% apart)")
    say(run_main._measurement_block(row))
    say(forces.summary(row))
    say(run_main._verdict_block(row))

    # ── 7. everything to disk ────────────────────────────────────────────────
    cfg.save(d / "config.json")
    receptance.in_workpiece(setup.scene).save(d / "plant.npz")
    np.savez_compressed(d / "precut.npz", part_pred_xy_mm=job.part_pred.boundary_xy_mm,
                        virgin_xy_mm=cfg.part.build().boundary_xy_mm,
                        start_xy_mm=job.start_xy_mm, end_xy_mm=job.end_xy_mm,
                        s0_mm=job.s0_mm, length_mm=job.length_mm,
                        s_cut_mm=job.s_cut_mm, warmup_mm=job.warmup_mm,
                        radius_mm=job.edge.radius)
    written = {"config": d / "config.json", "plant": d / "plant.npz",
               "precut": d / "precut.npz"}
    written.update(save.save_stability(d, stab, stability.table(stab, ap_crit)))
    written.update(save.save_forces(d, ff, force_rows))
    written.update(save.save_run(d, run_, decimate=csv_decimate))

    row.update({"engine": "dexel", "feed_profile": "flying",
                "csv_decimate": int(csv_decimate), "coupling": coupling,
                "plant": plant_kind, "ds_mm": float(job.ds_mm)})
    save.write_json(d / "summary.json", row)

    if not no_plots:
        from analysis import plots
        from steadystate_precut import plot_precut
        written.update(plots.run_figures(d, stab, ff, run_, setup, receptance))
        written["precut_png"] = plot_precut.figure(d / "figures" / "precut.png",
                                                   job)
    say("")
    say(f"out      {d}")
    for k, v in written.items():
        say(f"         {k:<18} {v.relative_to(d) if hasattr(v, 'relative_to') else v}")
    (d / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return row


def run_twin(job, *, name, out_root, tap: Pulse = None, tap_at=0.5, **kw) -> dict:
    """The plain pass, the tapped pass, and the tap's decay rate between them.

    Both passes resolve the same fit window and tap direction from the same
    prediction, so they differ in the pulse and nothing else. The combined row
    (the plain pass's, plus `sim_sigma_*`) is written back to its summary.json.
    """
    from pathlib import Path

    tap = tap or Pulse()
    base = run(job, name=name, out_root=out_root, tap=tap, tap_at=tap_at,
               tapped=False, **kw)
    pulsed = run(job, name=f"{name}_pulse", out_root=out_root, tap=tap,
                 tap_at=tap_at, tapped=True, **kw)

    row = dict(base)
    row["pulse_sim_diverged"] = bool(pulsed.get("sim_diverged"))
    needed = ("pulse_t_s", "pulse_fit_t0_s", "pulse_fit_t1_s",
              "pulse_band_lo_hz", "pulse_band_hi_hz", "pulse_f_lowest_hz")
    if all(np.isfinite(float(row.get(k, np.nan))) for k in needed):
        row.update(pmod.twin_rate(
            Path(out_root) / name, Path(out_root) / f"{name}_pulse",
            t_pulse=row["pulse_t_s"], duration_s=1e-3 * row["pulse_ms"],
            fit_t0=row["pulse_fit_t0_s"], fit_t1=row["pulse_fit_t1_s"],
            band=(row["pulse_band_lo_hz"], row["pulse_band_hi_hz"]),
            f_lowest_hz=row["pulse_f_lowest_hz"]))
    row["lin_state"] = pmod.lin_state(row.get("pred_growth_mid_1_s"))
    row["sim_state"] = pmod.sim_state(row)
    save.write_json(Path(out_root) / name / "summary.json", row)
    print(f"\ntap      sigma_sim {row.get('sim_sigma_1_s', np.nan):+.2f} 1/s "
          f"(R2 {row.get('sim_sigma_r2', np.nan):.2f}, floor "
          f"{row.get('sim_sigma_floor_1_s', np.nan):.2f}) | sigma_lin "
          f"{row.get('pred_growth_mid_1_s', np.nan):+.2f} 1/s | linear tap fit "
          f"{row.get('pred_pulse_fit_1_s', np.nan):+.2f} 1/s -> "
          f"sim {row['sim_state']}, lin {row['lin_state']}", flush=True)
    return row
