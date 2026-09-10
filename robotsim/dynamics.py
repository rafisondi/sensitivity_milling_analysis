"""The flexible-joint arm, stepped one dt at a time.

There is no batch "run" function here on purpose: the loop belongs in your
script, where you can see it and change it. This module gives you the one step.

    sim = Simulator(robot, scene, sim_dt)
    sim.reset(joints.theta[0], joints.thetaD[0], joints.thetaDD[0])

    for i, ti in enumerate(t):
        f_w = process.step(sim.tcp_w_mm(theta_cmd[i]), omega_rad_s * ti)
        sim.step(theta_cmd[i], thetaD_cmd[i], f_ext=sim.wrench_from_force_w(f_w))

    result = sim.result(path)

That loop IS the coupled physics: the deflected TCP decides the chip thickness,
the chip decides the force, the force deflects the joints again. Drop the
`process.step` line and you have the robot on its own; hold `theta_cmd` constant
and you have a hold-pose run.

No sign flip on the force: `process.step` already returns what the workpiece
exerts on the tool. Negating x/y turns the regenerative loop into positive
feedback and the run diverges.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from robotsim.robot import Robot
from robotsim.solver import Solver
from robotsim.trajectory import OperationalPath
from robotsim.transforms import invert_transform, rotate_vectors


@dataclass
class SimResult:
    """Everything a run produces, with the frame conversions attached.

    t          (N,)    simulation time [s]
    theta_cmd  (N, n)  commanded motor angles [rad]
    q_hist     (N, n)  actual (deflected) link angles [rad]
    fk_hist    (N, 6)  TCP pose, base frame: xyz [m] + xyz Euler [rad]
    f_hist_i   (N, 6)  applied TCP wrench, base frame [N, Nm]
    T_iw       (4, 4)  workpiece -> base

    Two different quantities, kept apart: ERROR is against the nominal path (the
    geometry the part should end up with), DEFLECTION is against the command
    (how far the joints gave way under load).
    """

    t: np.ndarray
    theta_cmd: np.ndarray
    q_hist: np.ndarray
    fk_hist: np.ndarray
    f_hist_i: np.ndarray
    T_iw: np.ndarray
    path: OperationalPath
    nominal_force_w: Optional[np.ndarray] = None

    @property
    def force_i(self) -> np.ndarray:
        """Cutting force at the TCP, base frame [N] — the primary record."""
        return self.f_hist_i[:, :3]

    @property
    def force_w(self) -> np.ndarray:
        """The same force in the workpiece frame. x/y are workpiece axes, not
        path-relative — z is the tool axis, so only it is always "axial"."""
        return rotate_vectors(self.T_iw[:3, :3].T, self.force_i)

    @property
    def peak_force_n(self) -> float:
        return float(np.linalg.norm(self.force_i, axis=1).max())

    @property
    def tcp_i(self) -> np.ndarray:
        """Simulated TCP position, base frame [m]."""
        return self.fk_hist[:, :3]

    @property
    def planned_i(self) -> np.ndarray:
        """Commanded TCP position on the simulation grid [m]."""
        return self.path.resample_i(self.t)

    @property
    def nominal_i(self) -> np.ndarray:
        """Nominal (uncompensated) TCP position on the simulation grid [m]."""
        return self.path.resample_nominal_i(self.t)

    @property
    def tcp_error_i_um(self) -> np.ndarray:
        return (self.tcp_i - self.nominal_i) * 1e6

    @property
    def tcp_error_w_um(self) -> np.ndarray:
        return rotate_vectors(self.T_iw[:3, :3].T, self.tcp_error_i_um)

    @property
    def deflection_i_um(self) -> np.ndarray:
        return (self.tcp_i - self.planned_i) * 1e6

    @property
    def deflection_w_um(self) -> np.ndarray:
        return rotate_vectors(self.T_iw[:3, :3].T, self.deflection_i_um)

    def steady_state_stats(self, window: float = 0.5) -> dict:
        """Mean and p-p of force and TCP error over the last `window` of the run."""
        i0 = int(len(self.t) * (1.0 - window))
        f_i, f_w = self.force_i[i0:], self.force_w[i0:]
        d_i, d_w = self.tcp_error_i_um[i0:], self.tcp_error_w_um[i0:]
        g_i, g_w = self.deflection_i_um[i0:], self.deflection_w_um[i0:]
        return {
            "window": window, "t_start_s": float(self.t[i0]),
            "force_i_mean_N": f_i.mean(axis=0), "force_i_ptp_N": np.ptp(f_i, axis=0),
            "force_w_mean_N": f_w.mean(axis=0), "force_w_ptp_N": np.ptp(f_w, axis=0),
            "tcp_err_i_mean_um": d_i.mean(axis=0), "tcp_err_i_ptp_um": np.ptp(d_i, axis=0),
            "tcp_err_w_mean_um": d_w.mean(axis=0), "tcp_err_w_ptp_um": np.ptp(d_w, axis=0),
            "defl_i_mean_um": g_i.mean(axis=0), "defl_i_ptp_um": np.ptp(g_i, axis=0),
            "defl_w_mean_um": g_w.mean(axis=0), "defl_w_ptp_um": np.ptp(g_w, axis=0),
        }

    def report(self, window: float = 0.5) -> str:
        s = self.steady_state_stats(window)
        r3 = lambda v: np.array2string(v, precision=1, suppress_small=True)
        return (
            f"Steady state (last {window:.0%} of the run, from t = {s['t_start_s']:.3f} s)\n"
            f"  force, base frame      [N]   mean {r3(s['force_i_mean_N'])}  "
            f"p-p {r3(s['force_i_ptp_N'])}\n"
            f"  force, workpiece frame [N]   mean {r3(s['force_w_mean_N'])}  "
            f"p-p {r3(s['force_w_ptp_N'])}\n"
            f"  TCP error vs NOMINAL, wp    [um]  mean {r3(s['tcp_err_w_mean_um'])}  "
            f"p-p {r3(s['tcp_err_w_ptp_um'])}\n"
            f"  deflection vs command, wp   [um]  mean {r3(s['defl_w_mean_um'])}  "
            f"p-p {r3(s['defl_w_ptp_um'])}\n"
            f"  peak |F| over the run  [N]   {self.peak_force_n:.0f}")

    def save_npz(self, filepath) -> None:
        np.savez_compressed(
            filepath, t=self.t, theta_cmd=self.theta_cmd, q_hist=self.q_hist,
            fk_hist=self.fk_hist, f_hist_i=self.f_hist_i, force_i=self.force_i,
            force_w=self.force_w, tcp_error_i_um=self.tcp_error_i_um,
            tcp_error_w_um=self.tcp_error_w_um,
            deflection_i_um=self.deflection_i_um,
            deflection_w_um=self.deflection_w_um,
            s_traj_i=self.path.s_i, s_traj_w=self.path.s_w,
            nominal_i=self.nominal_i, T_iw=self.T_iw,
            nominal_force_w=(np.zeros(3) if self.nominal_force_w is None
                             else np.asarray(self.nominal_force_w)))


class Simulator:
    """One flexible-joint arm and the RK4 solver behind it. You drive it.

    `reset` presets the springs on their load equilibrium, so the run starts
    tracking instead of ringing toward it. `step` advances one `sim_dt` and
    records the state AS IT WAS at the start of that step. `result` packages what
    was recorded.

    The recording convention is that sample `k` of every array — time, command,
    wrench, joint angles, TCP pose — belongs to the same instant `t[k]`. `step`
    explains why the alternative silently biases every deviation measurement.
    """

    def __init__(self, robot: Robot, scene, sim_dt: float, record: bool = True):
        self.robot = robot
        self.scene = scene
        self.dt = float(sim_dt)
        self.ee = scene.ee_frame
        self.T_iw = scene.T_iw
        self.T_wi = invert_transform(self.T_iw)
        self.R_iw = self.T_iw[:3, :3]

        self.solver = Solver(robot, dt=self.dt)
        self.record = bool(record)
        self.time = 0.0
        self._t, self._theta, self._q, self._fk, self._f = [], [], [], [], []

    # ── where the tool is RIGHT NOW, before you step ─────────────────────────

    def q_full(self, theta) -> np.ndarray:
        """Link-side (deflected) joint angles for the motor command `theta`."""
        return self.solver.get_current_q_full(
            np.asarray(theta, dtype=float).reshape(-1)).reshape(-1)

    def tcp_i(self, theta) -> np.ndarray:
        """Deflected TCP position, base frame [m]."""
        return self.robot.fkine(self.q_full(theta), self.ee)[:3, 3]

    def tcp_w_mm(self, theta) -> np.ndarray:
        """Deflected tool centre (x, y), workpiece frame [mm].

        What the milling process is stepped with — deflected, not commanded,
        which is what closes the regenerative loop.
        """
        p_i = self.tcp_i(theta)
        return (self.T_wi[:3, :3] @ p_i + self.T_wi[:3, 3])[:2] * 1e3

    # ── building the wrench and the feedforward torque ───────────────────────

    def wrench_from_force_w(self, force_w, preload_i=None) -> np.ndarray:
        """(6, 1) TCP wrench in the BASE frame from a workpiece-frame force [N].

        A pure force at the TCP, no moment. `preload_i` adds a constant external
        base-frame wrench on top.
        """
        F = (np.zeros((6, 1)) if preload_i is None
             else np.asarray(preload_i, dtype=float).reshape(6, 1).copy())
        if force_w is not None:
            F[:3, 0] += self.R_iw @ np.asarray(force_w, dtype=float).reshape(3)
        return F

    def tau_ff(self, theta, wrench_i) -> np.ndarray:
        """Motor torque -J(theta)^T F that carries `wrench_i` (6,) so the springs
        do not have to.

        The mean cutting force is ~1 kN; left uncompensated it lands on the
        joints as a step at t = 0 and rings for revolutions.
        """
        F = np.asarray(wrench_i, dtype=float).reshape(6, 1)
        return -(self.robot.jacobian(theta, self.ee).T @ F).ravel()

    # ── the two calls that matter ────────────────────────────────────────────

    def reset(self, theta, thetaD=None, thetaDD=None, f_ext=None, tau_ff=None,
              equilibrium: bool = True):
        """Preset the springs on their tracking equilibrium at t = 0.

        The link angle q and the motor angle theta differ by the spring
        deflection. At equilibrium the link tracks the command (qdd = thetaDD,
        qd = thetaD), so the damper drops out and the flexible rows give

            K (q - theta) = -( M thetaDD + bias - J^T f_ext - tau_ff )

        i.e. the spring carries whatever inertia and gravity demand that the
        external wrench and the feedforward torque do not. `Solver.reset` solves
        it by fixed point — M, bias and J all depend on q — which converges in
        about two sweeps because the deflection is ~1 mrad.

        Skip it and the springs start straight while the load instantly demands
        that deflection: the joints swing to it, overshoot to roughly double, and
        ring at their natural frequency for seconds. On this arm under gravity
        that is 2.8 mm p-p of TCP wobble instead of a flat 1.42 mm sag.

        Gravity and the Coriolis bias are always in `bias`, so the preset
        happens whether or not you pass a load. `equilibrium=False` opts out and
        starts the springs straight — a cold start, not a tracking one.
        """
        theta = np.asarray(theta, dtype=float).reshape(-1)
        thetaD = np.zeros_like(theta) if thetaD is None else thetaD
        self.solver.reset(theta, thetaD, thetaDD, f_ext=f_ext, tau_ff=tau_ff,
                          equilibrium=equilibrium)
        self.time = 0.0
        self._t, self._theta, self._q, self._fk, self._f = [], [], [], [], []
        return self

    def step(self, theta, thetaD=None, f_ext=None, tau_ff=None, theta_end=None,
             thetaD_end=None):
        """Advance one `sim_dt`. Returns (q_full, fk) — the deflected state after.

        f_ext      (6, 1) external TCP wrench, base frame — see `wrench_from_force_w`
        tau_ff     (n,)   feedforward motor torque — see `tau_ff`
        theta_end  (n,)   the command at the END of the step. Pass it on any
                          moving path: without it the integrator holds `theta`
                          across the step and the link trails the command by
                          about 0.65 dt of travel - see `robotsim.solver._rk4_step`

        WHAT GETS RECORDED IS THE STATE AT `self.time`, NOT AFTER THE STEP.

        Sample `k` of every recorded array belongs to `t[k]`: the command applied
        over `[t_k, t_k + dt]`, the wrench applied over it, and the pose the arm
        was IN at `t_k`. The returned pair is still the post-step state, because
        a caller driving the loop wants the state it just produced.

        Recording the post-step pose against the pre-step stamp - which is what
        this did - pairs the pose at `t_k + dt` with the command at `t_k`, so
        every deviation carries a spurious `feed * dt` along the feed direction.
        That is a fixed offset, not noise, and it is invisible in a sweep that
        holds the chip load and the steps per tooth fixed, because `feed * dt` is
        then exactly `fz / steps_per_tooth` in every cell.

        The cost is the final post-step state: `n` steps still record `n` samples,
        now spanning `t[0] .. t[n-1]` rather than `t[1] .. t[n]`.
        """
        theta = np.asarray(theta, dtype=float).reshape(-1)
        thetaD = np.zeros_like(theta) if thetaD is None else np.asarray(
            thetaD, dtype=float).reshape(-1)
        f_ext = np.zeros((6, 1)) if f_ext is None else f_ext

        if self.record:
            q_now, fk_now = self.solver.current_state(theta)
            self._t.append(self.time)
            self._theta.append(theta.copy())
            self._q.append(q_now)
            self._fk.append(fk_now)
            self._f.append(np.asarray(f_ext, dtype=float).ravel().copy())

        _, q_full, _, fk = self.solver.step(
            theta, thetaD, f_ext, tau_ff=tau_ff,
            theta_end=(None if theta_end is None
                       else np.asarray(theta_end, dtype=float).reshape(-1)),
            thetaD_end=(None if thetaD_end is None
                        else np.asarray(thetaD_end, dtype=float).reshape(-1)))
        self.time += self.dt
        return q_full, fk

    # ── what came out ────────────────────────────────────────────────────────

    @property
    def n_steps(self) -> int:
        return len(self._t)

    def result(self, path: OperationalPath,
               nominal_force_w=None) -> SimResult:
        """Package the recorded history. `path` is what the command was aiming at.

        Sample `k` of every array is at `t[k]` — see `step` for why that alignment
        is load-bearing rather than a detail: `SimResult.deflection_i_um` compares
        `tcp_i` against `path.resample_i(t)`, so a one-sample skew between them
        shows up as a constant deviation along the feed.
        """
        if not self._t:
            raise RuntimeError("nothing recorded — step the simulator first")
        return SimResult(
            t=np.asarray(self._t), theta_cmd=np.vstack(self._theta),
            q_hist=np.vstack(self._q), fk_hist=np.vstack(self._fk),
            f_hist_i=np.vstack(self._f), T_iw=self.T_iw, path=path,
            nominal_force_w=nominal_force_w)
