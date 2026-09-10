import warnings

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R
from robotsim.robot import Robot


def _rk4_step(f, t, dt, x, theta, thetaD, tau_ext, theta_end=None,
              thetaD_end=None):
    """One RK4 step of the flexible dynamics, driven by the motor command.

    THE COMMAND IS EVALUATED AT EACH STAGE'S OWN TIME when `theta_end` is given:
    `theta` at `t`, the midpoint at `t + dt/2` for k2 and k3, `theta_end` at
    `t + dt`. Without it every stage sees `theta` - a zero-order hold, under which
    the spring `K (q - theta)` is driven by a motor angle that goes stale across
    the step, and the link trails the command by a fixed fraction of it.

    That trail is not small on a moving path. With no cutting force and no
    gravity, tracking a straight line at 40 mm/s left the TCP 2.86 um behind the
    command at dt = 109 us, 1.49 um at 55 us and 0.80 um at 27 us - first order
    in dt, about 0.65 dt of travel, and entirely along the feed. It scales with
    feed * dt, so it reached ~17 um at 240 mm/s and read as deflection.

    `tau_ext` stays held across the step: the cutting force is sampled once per
    step by the process engine, and interpolating it is a different question.
    """
    if theta_end is None:
        th = (theta, theta, theta, theta)
        thD = (thetaD, thetaD, thetaD, thetaD)
    else:
        te = np.asarray(theta_end, dtype=float)
        tdE = thetaD if thetaD_end is None else np.asarray(thetaD_end, dtype=float)
        th_mid = 0.5 * (theta + te)
        thD_mid = 0.5 * (thetaD + tdE)
        th = (theta, th_mid, th_mid, te)
        thD = (thetaD, thD_mid, thD_mid, tdE)

    k1 = f(t, x, th[0], thD[0], tau_ext)
    k2 = f(t + dt/2, x + dt/2 * k1, th[1], thD[1], tau_ext)
    k3 = f(t + dt/2, x + dt/2 * k2, th[2], thD[2], tau_ext)
    k4 = f(t + dt, x + dt * k3, th[3], thD[3], tau_ext)

    return x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


class Solver:
    """Fixed-step RK4 integrator for the flexible-joint robot dynamics.

    Integrates only the flexible joints; the rigid joints track the motor
    command exactly. Call reset() with the initial command, then step() once
    per SIM_DT with the current motor command and external TCP wrench.
    """

    def __init__(self, robot: Robot, dt: float):
        self.robot = robot
        self.reference_system = 'TCP'
        self.dt = dt

        # The stiffest joint sets the RK4 step. Left as a warning rather than an
        # error because the real limit moves with the pose, but crossing the
        # measured one means the run WILL blow up — better said here than as a
        # bare "NaN detected" thirty seconds in.
        dt_max = getattr(robot, "sim_dt_max", None)
        if dt_max is not None and dt > dt_max:
            warnings.warn(
                f"sim_dt {dt:g} s exceeds {dt_max:g} s, the measured RK4 "
                f"stability limit for this compliance model ({len(robot.idx_flex)} "
                "flexible joints) — the integration is expected to diverge",
                RuntimeWarning, stacklevel=2)

        self.noj = self.robot.n
        self.x_flex = None
        self.n_flex = len(self.robot.idx_flex)
        self.time = 0
        # A zero configuration with no command is a genuine cold start, and
        # the caller resets properly before stepping anyway.
        self.reset(np.zeros(self.noj), np.zeros(self.noj), equilibrium=False)

    @staticmethod
    def _assemble_q_full(robot: Robot, q_flex, theta, n):
        q_full = np.full((robot.n, 1), np.nan)
        q_full[robot.idx_flex, :] = q_flex
        q_full[robot.idx_rigid] = theta[robot.idx_rigid].reshape(-1, 1)
        return q_full

    @staticmethod
    def _compute_q_flex(x_flex, n):
        return x_flex[:n].reshape(-1, 1)

    def get_current_q_full(self, theta: np.ndarray) -> np.ndarray:
        q_flex = self._compute_q_flex(x_flex=self.x_flex, n=self.n_flex)
        return self._assemble_q_full(self.robot, q_flex=q_flex, theta=theta, n=self.n_flex)

    def get_current_qd_full(self, thetaD: np.ndarray) -> np.ndarray:
        qD_full = np.full((self.robot.n, 1), np.nan)
        qD_full[self.robot.idx_flex, :] = self.x_flex[self.n_flex:].reshape(-1, 1)
        qD_full[self.robot.idx_rigid, :] = thetaD[self.robot.idx_rigid].reshape(-1, 1)
        return qD_full

    def _ode_system(self, t, x_flex, theta, thetaD, tau_ext):
        """
        Defines dy/dt = f(t, y, u), specifically
        qDD = inv(M) (-g - c + tau_j + tau_ext)
        for the flexible joints. q, qD, qDD are angular position, velocity,
        acceleration of the rigid link; M, g, c are mass matrix, gravity
        vector, and Coriolis terms; tau_j and tau_ext are joint and external
        torques, with tau_j = k(q - theta) + d(qD - thetaD).

        inputs:
        x_flex: states of flexible coordinates [2 x len(idx_flex)]
        t:      time
        theta:  joint trajectory                [noj]
        thetaD: joint trajectory time derivative [noj]
        """
        n = self.n_flex
        q_flex = self._compute_q_flex(x_flex, n)
        qD_flex = x_flex[n:2*n].reshape(-1, 1)

        q_full = self._assemble_q_full(self.robot, q_flex, theta, n)

        qD_full = np.full((self.robot.n, 1), np.nan)
        qD_full[self.robot.idx_flex, :] = qD_flex.reshape(-1, 1)
        qD_full[self.robot.idx_rigid, :] = thetaD[self.robot.idx_rigid].reshape(-1, 1)

        tau_flex_joint = (
            - self.robot.K_m_diag_flex * (q_flex - theta[self.robot.idx_flex].reshape(-1, 1))
            - self.robot.D_m_diag_flex * (qD_flex - thetaD[self.robot.idx_flex].reshape(-1, 1))
        ).reshape(-1, 1)

        qDD_flex = self.robot.forward_dynamics(q_full, qD_full, tau_ext, tau_u=tau_flex_joint)

        return np.vstack([qD_flex, qDD_flex]).flatten()

    def _calc_fkine_vec(self, q_full, ee_name):
        T = self.robot.fkine(q_full, ee_name)
        rot = R.from_matrix(T[:3, :3])
        return np.hstack([T[:3, -1], rot.as_euler('xyz', degrees=False)])

    def current_state(self, theta: np.ndarray):
        """(q_full (n,), fk (6,)) as the arm stands NOW, before any further step.

        `step` returns the state AFTER integrating one dt. Anything recording a
        history against a time stamp needs the state AT that stamp instead, which
        is this one - see `robotsim.dynamics.Simulator.step`.
        """
        q_full = self.get_current_q_full(theta).reshape(-1)
        return q_full, self._calc_fkine_vec(q_full, self.reference_system)

    def step(self, theta: np.ndarray, thetaD: np.ndarray, f_ext: np.ndarray,
             tau_ff: np.ndarray = None, theta_end: np.ndarray = None,
             thetaD_end: np.ndarray = None):
        """Advance one step. tau_ff is an optional feedforward motor torque
        [noj] added on the flexible joints (e.g. -J(theta)^T f_ext to
        compensate a known external preload).

        `theta_end` / `thetaD_end` are the command at the END of the step. Pass
        them and the RK4 stages see the command at their own times instead of a
        hold at `theta` - see `_rk4_step` for what the hold costs. `theta` is
        still the command at the START, and is what the force mapping and the
        returned rigid-joint angles are built from."""
        # Assemble current full state before integration for force mapping / FK
        q_full_current = self.get_current_q_full(theta)
        tau_ext = self.robot.jacobian(q_full_current, "TCP")[:, self.robot.idx_flex].transpose() @ f_ext

        tau_drive = tau_ext
        if tau_ff is not None:
            tau_drive = tau_ext + np.asarray(tau_ff).reshape(-1)[self.robot.idx_flex].reshape(-1, 1)

        # Integrate flexible dynamics
        self.x_flex = _rk4_step(
            self._ode_system, self.time, self.dt, self.x_flex, theta, thetaD, tau_drive,
            theta_end=theta_end, thetaD_end=thetaD_end,
        )
        self.time += self.dt

        if np.isnan(self.x_flex).any():
            raise Exception("NaN detected")

        # Return properly assembled full vectors in robot joint index order
        q_full = self.get_current_q_full(theta).reshape(-1)
        qD_full = self.get_current_qd_full(thetaD).reshape(-1)
        fk = self._calc_fkine_vec(q_full, self.reference_system)

        return tau_ext, q_full, qD_full, fk

    def reset(self, theta: np.ndarray, thetaD: np.ndarray,
              thetaDD: np.ndarray = None, f_ext: np.ndarray = None,
              tau_ff: np.ndarray = None, equilibrium: bool = True):
        """Initialise the flexible state on the motor command.

        equilibrium=True (default) presets the quasi-static spring deflection

            q = theta - K^-1 (M thetaDD + bias - J^T f_ext - tau_ff)

        so the run starts ON its tracking equilibrium. The optional arguments
        each remove one term: omit thetaDD and the command is taken as
        unaccelerated, omit f_ext / tau_ff and nothing external carries the
        load. Gravity and the Coriolis bias are always included — they are in
        `bias` whether or not anything else is passed.

        equilibrium=False starts the springs undeflected (q = theta), which is
        only a true equilibrium at rest under no load. Anything else and the
        joints swing to the deflection the load demands, overshoot to roughly
        double, and ring at their natural frequency for seconds.

        qd starts at thetaD either way: a constant deflection means the link
        moves at the commanded speed.
        """
        idx = self.robot.idx_flex
        q_flex = theta[idx].astype(float).copy()
        if equilibrium:
            # Acceleration as seen by the sim model: flexible rows follow the
            # command, rigid-row coupling is neglected (as in _ode_system).
            a_full = np.zeros(self.robot.n)
            if thetaDD is not None:
                a_full[idx] = thetaDD[idx]
            q_full = np.asarray(theta, dtype=float).copy()
            k_flex = self.robot.K_m_diag[idx]
            for _ in range(3):                      # fixed-point on the deflection
                q_full[idx] = q_flex
                tau = pin.rnea(self.robot.model, self.robot.data,
                               q_full, np.asarray(thetaD, dtype=float), a_full)
                tau_load = tau[idx].copy()          # M qDD + bias on flex rows
                if f_ext is not None:
                    tau_load -= (self.robot.jacobian(q_full, "TCP")[:, idx].T
                                 @ f_ext).ravel()
                if tau_ff is not None:
                    tau_load -= np.asarray(tau_ff).reshape(-1)[idx]
                q_flex = theta[idx] - tau_load / k_flex
        self.x_flex = np.hstack([q_flex, thetaD[idx]])
        self.time = 0
