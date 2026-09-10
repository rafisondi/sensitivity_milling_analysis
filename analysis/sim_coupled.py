"""The truth: the dexel cut with the flexible-joint arm in the loop.

    run = simulate(cfg, ff)          ff from `analysis.feedforward.build`
    run = simulate(cfg, None)        the same pass with the motors idle

IK onto the commanded path, joint-space RK4 integration, the joint springs the
arm actually has. The regenerative loop is the ten lines under THE COUPLED LOOP:
the DEFLECTED tool centre decides the chip thickness, the chip decides the force,
the force deflects the joints again, and the dexel surface record turns the
previous tooth's displacement into this tooth's chip.

WHY THE LOOP IS REPEATED HERE INSTEAD OF CALLING `run_pass`

`fastsim.coupled.run_pass` already supports a feedforward torque, but only a
CONSTANT one: it builds `tau_ff` once from `carry_force_w` and applies the same
wrench at every step. The mean cutting force is not constant along this job — it
ramps in at the entry, holds through the edge and ramps out — and carrying it
properly needs `tau_ff` scheduled along the path. There is no hook for that, so
the loop is repeated with the schedule added. Every piece it uses is imported
from `fastsim`/`robotsim`; nothing is reimplemented. If `run_pass` ever takes a
callable `tau_ff`, this module collapses back into it.

THE BAIL GUARD

A pass that has left the cut costs hours of raster work and says nothing after
the moment it left: once the tool is clear of the material the force vanishes,
the motion stops growing, and a runaway reads as a flat pass. `BAIL_MM` stops it
and records where. A truncated run is data — `analysis.report` classifies it as
diverged — not an error.
"""

import numpy as np

from fastsim.coupled import MillingPass, ProcessAdapter, solve_joints
from fastsim.prepare import build_process, prepare_steady_state_cut
from robotsim.dynamics import Simulator
from robotsim.scene import Scene

NAME = "coupled"
ARTIFACT = "coupled.npz"

#: Stop once the tool centre is this far from the commanded path [mm]. Past a
#: full radial engagement the tool is clear of the metal it was supposed to be
#: cutting and the pass is diverged by any definition.
BAIL_MM = 5.0


def simulate(cfg, feedforward=None, *, path=None, preload_i=None,
             steady_state=False, seed_rad=None, pulse=None,
             verbose=False) -> MillingPass:
    """One coupled pass -> `MillingPass`.

    `feedforward` is an `analysis.feedforward.Feedforward`, or None for a pass
    with the motors idle and the joint springs carrying the whole mean force.

    `steady_state` pre-carves the entry slot and opens the run on the tracking
    equilibrium of the load at t = 0, so the pass starts already cutting. It is
    off by default here: the entry transient is one of the two events this job
    exists to look at, and pre-carving deletes it.

    `pulse` is an `analysis.pulse.Pulse`: a short force on the TCP at a fraction
    of the path, for measuring the decay rate of a perturbation during the steady
    cut. It goes in through the motor-torque channel as `+J^T W` - mechanically
    the same as an external force on the tool, but the recorded CUTTING force
    stays clean. Run the pass twice, with and without it, and subtract: see
    `analysis.pulse`.
    """
    mill, part, scene, _waypoints, cfg_path = cfg.build()
    path = cfg_path if path is None else path
    scene = Scene() if scene is None else scene

    robot, joints = solve_joints(scene, path, seed_rad=seed_rad, verbose=verbose)
    t, theta_cmd, thetaD_cmd = joints.resample(mill.sim_dt)
    f0 = None if feedforward is None else feedforward.schedule(t)

    sim = Simulator(robot, scene, sim_dt=mill.sim_dt)

    pulse_t0 = pulse_t1 = None
    if pulse is not None:
        from analysis.pulse import time_at_fraction
        pulse_t0 = time_at_fraction(path, pulse.at_frac)
        pulse_t1 = pulse_t0 + float(pulse.duration_s)
        pulse_w = sim.wrench_from_force_w(pulse.force_w())
        print(f"pulse    {pulse.force_N:g} N for {1e3 * pulse.duration_s:g} ms at "
              f"t = {pulse_t0:.3f} s ({100 * pulse.at_frac:g}% of the path), "
              f"dir_w {tuple(round(float(v), 3) for v in pulse.dir_w)}", flush=True)

    def tau_at(i):
        """-J(theta)^T F0 at step `i`, or None when nothing is carried."""
        if f0 is None:
            return None
        return sim.tau_ff(theta_cmd[i], sim.wrench_from_force_w(f0[i]))

    # open ON the tracking equilibrium of whatever load t = 0 carries, so the run
    # does not start by springing to it
    f_cut0_w = (f0[0] if (steady_state and f0 is not None) else None)
    # `thetaDD_at(0)` rather than `thetaDD[0]` - the first sample of the double
    # finite difference carries an IK-seeding artefact that would preset the
    # springs for a load that is not there. See `JointTrajectory.thetaDD_at`.
    sim.reset(theta_cmd[0], thetaD_cmd[0], joints.thetaDD_at(0),
              f_ext=sim.wrench_from_force_w(f_cut0_w, preload_i),
              tau_ff=tau_at(0))

    if steady_state:
        heading = path.xy_mm[min(len(path) - 1, 1)] - path.xy_mm[0]
        engine = prepare_steady_state_cut(mill, part, sim.tcp_w_mm(theta_cmd[0]),
                                          heading, verbose=verbose)
    else:
        engine = build_process(mill, part)
    process = ProcessAdapter(engine)

    if verbose:
        print(f"engine   {process.n_slices} slices, raster {mill.raster_mm:g} mm")
    print(f"sim      {len(t)} steps of {mill.sim_dt:g} s = {t[-1]:.3f} s"
          f"{'' if f0 is None else '  | F0 carried on the motors'}", flush=True)

    # the commanded path on the SIMULATION grid — `path.xy_mm` is on the plan
    # grid, and indexing it with a sim index compares points seconds apart
    nom = np.column_stack([np.interp(t, path.t, path.xy_mm[:, k])
                           for k in range(2)])

    def _with_pulse(tau, i, ti):
        """Add the tap as `+J^T W` while it is on. `tau_ff` returns `-J^T W`."""
        if pulse_t0 is None or not (pulse_t0 <= ti < pulse_t1):
            return tau
        tap = -sim.tau_ff(theta_cmd[i], pulse_w)
        return tap if tau is None else tau + tap

    # ── THE COUPLED LOOP ─────────────────────────────────────────────────────
    mark = max(1, len(t) // 10)
    n_done = len(t)
    for i, ti in enumerate(t):
        xy_mm = sim.tcp_w_mm(theta_cmd[i])                  # deflected tool centre
        f_w = process.step(xy_mm, mill.omega_rad_s * ti)    # -> cutting force [N]
        # the command at the END of this step too, so the RK4 stages track the
        # moving command instead of holding it - see robotsim.solver._rk4_step
        j = min(i + 1, len(t) - 1)
        sim.step(theta_cmd[i], thetaD_cmd[i],               # -> deflects the joints
                 theta_end=theta_cmd[j], thetaD_end=thetaD_cmd[j],
                 f_ext=sim.wrench_from_force_w(f_w, preload_i),
                 tau_ff=_with_pulse(tau_at(i), i, ti))

        if np.linalg.norm(xy_mm - nom[i]) > BAIL_MM:
            n_done = i + 1
            print(f"  BAILED at step {n_done}/{len(t)} ({ti:.3f} s): more than "
                  f"{BAIL_MM:g} mm off the commanded path", flush=True)
            break
        if (i + 1) % mark == 0:
            print(f"  sim {round(100 * (i + 1) / len(t)):3d}%  "
                  f"|F| = {np.linalg.norm(f_w):7.1f} N", flush=True)
    # ─────────────────────────────────────────────────────────────────────────

    return MillingPass(cfg=mill, scene=scene, part=part, path=path, robot=robot,
                       joints=joints, process=process,
                       result=sim.result(path, nominal_force_w=(
                           None if f0 is None else f0.mean(axis=0))))
