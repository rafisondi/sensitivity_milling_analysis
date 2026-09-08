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
             steady_state=False, seed_rad=None, verbose=False) -> MillingPass:
    """One coupled pass -> `MillingPass`.

    `feedforward` is an `analysis.feedforward.Feedforward`, or None for a pass
    with the motors idle and the joint springs carrying the whole mean force.

    `steady_state` pre-carves the entry slot and opens the run on the tracking
    equilibrium of the load at t = 0, so the pass starts already cutting. It is
    off by default here: the entry transient is one of the two events this job
    exists to look at, and pre-carving deletes it.
    """
    mill, part, scene, _waypoints, cfg_path = cfg.build()
    path = cfg_path if path is None else path
    scene = Scene() if scene is None else scene

    robot, joints = solve_joints(scene, path, seed_rad=seed_rad, verbose=verbose)
    t, theta_cmd, thetaD_cmd = joints.resample(mill.sim_dt)
    f0 = None if feedforward is None else feedforward.schedule(t)

    sim = Simulator(robot, scene, sim_dt=mill.sim_dt)

    def tau_at(i):
        """-J(theta)^T F0 at step `i`, or None when nothing is carried."""
        if f0 is None:
            return None
        return sim.tau_ff(theta_cmd[i], sim.wrench_from_force_w(f0[i]))

    # open ON the tracking equilibrium of whatever load t = 0 carries, so the run
    # does not start by springing to it
    f_cut0_w = (f0[0] if (steady_state and f0 is not None) else None)
    sim.reset(theta_cmd[0], thetaD_cmd[0], joints.thetaDD[0],
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

    # ── THE COUPLED LOOP ─────────────────────────────────────────────────────
    mark = max(1, len(t) // 10)
    n_done = len(t)
    for i, ti in enumerate(t):
        xy_mm = sim.tcp_w_mm(theta_cmd[i])                  # deflected tool centre
        f_w = process.step(xy_mm, mill.omega_rad_s * ti)    # -> cutting force [N]
        sim.step(theta_cmd[i], thetaD_cmd[i],               # -> deflects the joints
                 f_ext=sim.wrench_from_force_w(f_w, preload_i),
                 tau_ff=tau_at(i))

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
