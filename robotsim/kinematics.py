"""Load the robot and solve the IK.

Joints 1-3 are FLEXIBLE: a spring + damper between motor and link. The motor
angle theta is what the IK produces and the controller commands; the link angle
q is what the dynamics integrate, and the two differ by the spring deflection.

    build_robot(gravity_on, robot_model)     the robot model
    load_robot(scene)                        the same, from a Scene
    solve_ik(robot, path, seed_rad)          path -> JointTrajectory
    solve_ik_pose(robot, p_i, R, seed_rad)   one pose -> joint angles
    hold_pose_trajectory(theta, duration_s)  a command that holds still
    place_workpiece_origin(p0_mm, R_iw)      where to put the job in the cell
    joint_limit_report(robot, theta)         which joints are out of range

eps=1e-7 and edge_order=2 are deliberate: at looser tolerance the first point
converges from a different direction than the warm-started rest, and the
resulting velocity step at t=0 excites the joints.
"""

import warnings
from dataclasses import dataclass

import numpy as np

from robotsim.robot import Robot
from robotsim.scene import DEFAULT_START_DEG
from robotsim.settings import (
    SettingsAllAxesSDOF, SettingsRealParameter, SettingsTriJoint,
)
from robotsim.trajectory import OperationalPath

# Which ARM, and which of its joints are springs — a Settings class bundles the
# URDF, the joint names and the K/D tables, so both choices live in one entry.
# These are the Staeubli at Huynh's three identification levels; mind the smaller
# sim_dt the wider ones need (see settings.py).
#
# The upstream KUKA KR60 entries are NOT here. They set `K_m_diag = nan` on every
# joint, i.e. a fully RIGID arm — which has no compliance to deflect, so the
# coupled pass this workspace is built around would report a flat zero and the
# DC compensation would have nothing to cancel.
ROBOT_MODELS = {
    "joints123": SettingsRealParameter,   # Staeubli,  3 DOF, joints 1-3
    "all-axes": SettingsAllAxesSDOF,      # Staeubli,  6 DOF, one spring per axis
    "trijoint": SettingsTriJoint,         # Staeubli, 18 DOF, three per joint
}


def build_robot(gravity_on: bool = False, robot_model: str = "joints123",
                urdf_path=None) -> Robot:
    """Load the URDF and apply the flexible-joint parameters.

    Gravity off by default: a static sag of a few hundred um would otherwise
    dominate the deflection plots. Kinematics are unaffected either way.

    `urdf_path` overrides the file the robot model would pick. The joint
    tables are positional, so a URDF with different joint names raises rather
    than mis-assigning stiffnesses.
    """
    try:
        settings = ROBOT_MODELS[robot_model]
    except KeyError:
        raise ValueError(f"unknown robot model {robot_model!r} — expected one "
                         f"of {sorted(ROBOT_MODELS)}") from None
    robot = Robot(settings(urdf_path))
    if not gravity_on:
        robot.model.gravity.linear[:] = np.zeros(3)
    return robot


def load_robot(scene) -> Robot:
    """`build_robot` with the model, URDF and gravity from a `Scene`."""
    return build_robot(gravity_on=scene.gravity_on, robot_model=scene.robot_model,
                       urdf_path=getattr(scene, "urdf_path", None))


def place_workpiece_origin(first_point_mm, R_iw, *,
                           seed_deg=DEFAULT_START_DEG,
                           lift_mm: float = 5.0, ee_frame: str = "TCP",
                           robot: Robot = None, urdf_path=None,
                           robot_model: str = "joints123") -> np.ndarray:
    """Workpiece origin [m] placing `first_point_mm` on the TCP at the seed pose.

        origin_i = fkine(seed)[:3, 3] + [0, 0, lift] - R_iw @ [x0, y0, 0]

    Pass lift_mm = ap so the tool bites the full depth instead of scratching.

    The ORIGIN only: the orientation `R_iw` is yours to pick, and the lift is
    along the base +Z rather than the workpiece one. `Scene.placed_at_start`
    does the whole placement instead, taking the orientation from the tool as it
    is held at the start pose — reach for this one only when the part is squared
    with the cell rather than with the tool.

    The placement is FK of a specific arm, so pass `robot` (or the matching
    `robot_model` / `urdf_path`) — placing a job with the wrong robot's
    kinematics puts it somewhere that arm cannot reach.
    """
    p0 = np.zeros(3)
    p0[:2] = np.asarray(first_point_mm, dtype=float).ravel()[:2]
    robot = (build_robot(gravity_on=False, robot_model=robot_model,
                         urdf_path=urdf_path) if robot is None else robot)
    origin = robot.fkine(np.deg2rad(np.asarray(seed_deg, dtype=float)),
                         ee_frame)[:3, 3].copy()
    origin[2] += lift_mm * 1e-3
    return origin - np.asarray(R_iw, dtype=float) @ (p0 * 1e-3)


def tcp_pose_i(robot: Robot, q, ee_frame: str = "TCP") -> np.ndarray:
    """4x4 TCP pose in the base frame for the joint configuration q."""
    return robot.fkine(np.asarray(q, dtype=float).reshape(-1), ee_frame)


@dataclass
class JointTrajectory:
    """The motor command: theta [rad], thetaD, thetaDD, on the grid t at dt."""

    theta: np.ndarray
    thetaD: np.ndarray
    thetaDD: np.ndarray
    t: np.ndarray
    dt: float

    def __len__(self) -> int:
        return len(self.theta)

    @property
    def n_joints(self) -> int:
        return self.theta.shape[1]

    @property
    def duration_s(self) -> float:
        return float(self.t[-1])

    @property
    def max_joint_vel(self) -> float:
        return float(np.abs(self.thetaD).max())

    def resample(self, sim_dt: float):
        """(t_sim, theta_cmd, thetaD_cmd) on the finer simulation grid.

        The dynamics run at sim_dt to resolve tooth passing; the plan is sampled
        coarser because IK is the expensive part.
        """
        t_sim = np.arange(0.0, self.t[-1], sim_dt)
        theta_cmd = np.column_stack([np.interp(t_sim, self.t, self.theta[:, j])
                                     for j in range(self.n_joints)])
        thetaD_cmd = np.column_stack([np.interp(t_sim, self.t, self.thetaD[:, j])
                                      for j in range(self.n_joints)])
        return t_sim, theta_cmd, thetaD_cmd


def joint_limit_report(robot: Robot, theta) -> str:
    """Which joints leave their URDF limits, and by how much. "" if none do.

    `clik` integrates freely and never looks at the limits, so a solution can be
    geometrically perfect and mechanically impossible. The Staeubli URDF
    declares +-180 deg on every axis (i.e. no real limits), which is why this
    only starts to matter on an arm whose axis 2 spans just
    [-135, +35] deg.
    """
    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    lo, hi = robot.model.lowerPositionLimit, robot.model.upperPositionLimit
    over = np.maximum(theta - hi, lo - theta).max(axis=0)      # per joint, worst
    bad = np.where(over > 1e-9)[0]
    if len(bad) == 0:
        return ""
    rows = [f"joint {j + 1}: {np.rad2deg(over[j]):.2f} deg outside "
            f"[{np.rad2deg(lo[j]):.1f}, {np.rad2deg(hi[j]):.1f}]" for j in bad]
    return "; ".join(rows)


def solve_ik_points(robot: Robot, points_i, R_i_tcp, seed_rad,
                    ee_frame: str = "TCP", eps: float = 1e-7,
                    check_limits: bool = True) -> np.ndarray:
    """Motor angles (N, n) at each base-frame TCP position [m].

    Each point is warm-started from the previous one, so the whole set stays on
    one IK branch (no elbow flips mid-path). Warns when a solution leaves the
    URDF joint limits — `clik` does not enforce them.
    """
    points_i = np.atleast_2d(np.asarray(points_i, dtype=float))
    theta = np.zeros((len(points_i), robot.n))
    q = robot.expand_actuated(seed_rad)
    for k, p in enumerate(points_i):
        q = robot.clik(p, np.asarray(R_i_tcp, dtype=float), ee_frame,
                       q_init=q, eps=eps)
        theta[k] = q

    if check_limits:
        report = joint_limit_report(robot, theta)
        if report:
            warnings.warn(f"IK solution is outside the joint limits — "
                          f"{report}", RuntimeWarning, stacklevel=2)
    return theta


def solve_ik_pose(robot: Robot, p_i, R_i_tcp, seed_rad, ee_frame: str = "TCP",
                  eps: float = 1e-7) -> np.ndarray:
    """Joint angles (n,) for ONE base-frame TCP pose."""
    return solve_ik_points(robot, np.asarray(p_i, dtype=float).reshape(1, 3),
                           R_i_tcp, seed_rad, ee_frame=ee_frame, eps=eps)[0]


def solve_ik(robot: Robot, path: OperationalPath, seed_rad,
             ee_frame: str = "TCP", eps: float = 1e-7,
             max_joint_vel: float = 100.0, verbose: bool = True) -> JointTrajectory:
    """IK for every path point, then finite-difference the velocity command.

    Raises if the joint velocity exceeds `max_joint_vel`: `clik` returns its last
    iterate even when it does NOT converge, and non-converged points leave kinks
    whose velocity command explodes. This catches an unreachable path in seconds.
    """
    theta = solve_ik_points(robot, path.s_i, path.R_i_tcp, seed_rad,
                            ee_frame=ee_frame, eps=eps)
    thetaD = np.gradient(theta, path.dt, axis=0, edge_order=2)
    thetaDD = np.gradient(thetaD, path.dt, axis=0)
    traj = JointTrajectory(theta=theta, thetaD=thetaD, thetaDD=thetaDD,
                           t=path.t.copy(), dt=path.dt)

    if traj.max_joint_vel > max_joint_vel:
        raise ValueError(
            f"planned joint velocity {traj.max_joint_vel:.1f} rad/s exceeds "
            f"{max_joint_vel:.0f} rad/s — IK did not converge (the path is "
            "likely outside the reachable workspace at this workpiece pose).")

    if verbose:
        print(f"ik       {len(traj)} points | max |thetaD| "
              f"{traj.max_joint_vel:.3f} rad/s | {traj.duration_s:.3f} s")
    return traj


def hold_pose_trajectory(theta, duration_s: float,
                         dt: float = 1.0e-3) -> JointTrajectory:
    """A command that sits at `theta` — zero velocity, zero acceleration."""
    theta = np.asarray(theta, dtype=float).reshape(-1)
    n_pts = max(2, int(round(duration_s / dt)) + 1)
    return JointTrajectory(theta=np.tile(theta, (n_pts, 1)),
                           thetaD=np.zeros((n_pts, len(theta))),
                           thetaDD=np.zeros((n_pts, len(theta))),
                           t=np.arange(n_pts) * dt, dt=dt)
