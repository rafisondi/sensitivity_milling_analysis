"""The truth, a second way: the shapely polygon cut with the flexible-joint arm
in the loop.

    run = simulate(cfg, ff)          ff from `analysis.feedforward.build`

A line-for-line copy of `analysis.sim_coupled.simulate`, with exactly one
change: the engine `fastsim.prepare.build_process(mill, part)` constructs is
swapped for `shapely_engine.build_shapely_process(mill, part, scene.T_iw)`. That
module's own docstring says why a copy rather than an option: the loop is meant
to be forked when what happens per step changes, and the process backend is
exactly that. Everything else - IK, the feedforward schedule, the RK4 arm, the
bail guard, the result object - is `fastsim`/`robotsim`/`analysis` unchanged.

NOT SUPPORTED HERE: `steady_state=True`. `fastsim.prepare.prepare_steady_state_cut`
pre-carves the raster directly (`process._grids`), which has no meaning for a
polygon workpiece - the shapely engine has its own erase history instead. This
workspace's default pipeline runs with `steady_state=False` throughout (the
"flying" feed profile plus a real lead-in through air), so this is not a
practical gap, only a documented one.
"""

import numpy as np

from fastsim.coupled import MillingPass, ProcessAdapter, solve_joints
from robotsim.dynamics import Simulator
from robotsim.scene import Scene

from shapely_engine import build_shapely_process

NAME = "coupled_shapely"
ARTIFACT = "coupled.npz"

#: Same bail distance `analysis.sim_coupled` uses - see its own docstring.
BAIL_MM = 5.0


def simulate(cfg, feedforward=None, *, path=None, preload_i=None,
             seed_rad=None, pulse=None, verbose=False) -> MillingPass:
    """One coupled pass, shapely engine -> `MillingPass`. See
    `analysis.sim_coupled.simulate` for what every argument means; this differs
    only in which engine `build_process` calls."""
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
        if f0 is None:
            return None
        return sim.tau_ff(theta_cmd[i], sim.wrench_from_force_w(f0[i]))

    sim.reset(theta_cmd[0], thetaD_cmd[0], joints.thetaDD_at(0),
              f_ext=sim.wrench_from_force_w(None, preload_i),
              tau_ff=tau_at(0))

    engine = build_shapely_process(mill, part, scene.T_iw)
    process = ProcessAdapter(engine)

    if verbose:
        print(f"engine   shapely, {process.n_slices} slice(s), "
              f"{mill.slice_angle_deg:g} deg/slice")
    print(f"sim      {len(t)} steps of {mill.sim_dt:g} s = {t[-1]:.3f} s"
          f"{'' if f0 is None else '  | F0 carried on the motors'}", flush=True)

    nom = np.column_stack([np.interp(t, path.t, path.xy_mm[:, k])
                           for k in range(2)])

    def _with_pulse(tau, i, ti):
        if pulse_t0 is None or not (pulse_t0 <= ti < pulse_t1):
            return tau
        tap = -sim.tau_ff(theta_cmd[i], pulse_w)
        return tap if tau is None else tau + tap

    # ── THE COUPLED LOOP ─────────────────────────────────────────────────────
    mark = max(1, len(t) // 10)
    n_done = len(t)
    for i, ti in enumerate(t):
        xy_mm = sim.tcp_w_mm(theta_cmd[i])
        f_w = process.step(xy_mm, mill.omega_rad_s * ti)
        j = min(i + 1, len(t) - 1)
        sim.step(theta_cmd[i], thetaD_cmd[i],
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
