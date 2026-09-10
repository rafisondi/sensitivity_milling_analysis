"""The linear replacement for the arm: pinocchio, linearised at the start pose.

Two objects, and they are the same arm written twice:

    LinearModel   M dx'' + D dx' + K dx = F, 3x3, at the tool tip
    Receptance    the same, as the 6-state second-order state space that
                  `stabsim.stability` closes the cut around

`tcp_linear_model` builds the first out of the flexible-joint arm's own mass
matrix, its joint springs and the Jacobian at one configuration; `Receptance.
from_mdk` is the bridge to the second. Nothing measured is read anywhere in this
workspace — the M/D/K here and the arm the coupled pass integrates are the same
model, one linearised and one not, which is what makes a disagreement between
them attributable to the LINEARISATION rather than to two different machines.

WHAT IS FROZEN, AND IT IS THE MAIN CAVEAT

The pose. `M`, `D` and `K` are evaluated once at `cfg.start` and used for the
whole prediction, while the arm traverses 150 mm of path. The Jacobian turns
along the way, so the tool-tip stiffness the real arm presents at the exit is not
the one at the entry. That is a slow drift across the pass rather than an
entry/exit effect, but it is unmodelled and it bounds how well any prediction
here can do — `deflection_report` prints the DC compliance so the number is at
least visible.

GRAVITY. `tcp_linear_model` defaults `gravity_on=True` while `Scene.gravity_on`
defaults False, and dG/dq is the largest term of dtau_dq. The scene's value is
therefore passed explicitly rather than allowed to default, so the linearisation
and the simulation agree about whether the arm is hanging in a gravity field.
"""

import numpy as np

from robotsim.kinematics import ROBOT_MODELS, load_robot
from robotsim.linear import LinearModel
from robotsim.linearize_robot_dynamics import tcp_linear_model
from robotsim.receptance import Receptance


def linear_model_from_robot(cfg, *, frame="base", name=None) -> LinearModel:
    """Linearise the configured arm at the config's start pose -> M/D/K at the TCP."""
    robot = load_robot(cfg.scene)
    theta = cfg.start.theta(robot, cfg.scene)
    return tcp_linear_model(
        robot, theta,
        gravity_on=bool(cfg.scene.gravity_on),
        ee_frame=cfg.scene.ee_frame,
        frame=frame,
        name=name or f"{cfg.scene.robot_model} @ start pose")


def commanded_theta_at(setup, frac=0.5, *, verbose=False):
    """(robot, theta (n,), t [s], s [mm]) - the COMMANDED joint angles where the
    path has covered `frac` of its own arc length.

    Solved by the same whole-path IK the coupled pass runs (`solve_joints`), so
    the pose is the one the simulation actually commands there, on the same
    branch - a single-point IK seeded from the start pose can land on another
    wrist configuration once the workpiece is turned far enough. `setup` is an
    `analysis.stability.Setup`, whose `scene` is the PLACED scene and `path` the
    replayed toolpath.
    """
    from analysis.pulse import path_arclength_mm, time_at_fraction
    from fastsim.coupled import solve_joints

    robot, joints = solve_joints(setup.scene, setup.path, verbose=verbose)
    t_star = time_at_fraction(setup.path, frac)
    theta = np.array([np.interp(t_star, joints.t, joints.theta[:, j])
                      for j in range(joints.theta.shape[1])])
    s_star = float(frac) * float(path_arclength_mm(setup.path)[-1])
    return robot, theta, t_star, s_star


def receptance_at_fraction(setup, frac=0.5, *, frame="base", verbose=False):
    """`Receptance` of the arm linearised at the halfway mark (or any `frac`).

    The start-pose model freezes M/D/K where the pass BEGINS, while the cut the
    stability verdict is about happens along the edge - and once the workpiece is
    turned, the arm configuration at the start and in the middle of the edge
    differ by the whole swing of the path. Linearising at the commanded pose half
    way along puts the plant where the steady cut is. Static linearisation
    (`thetaD = thetaDD = 0`), the same as the start-pose model, so the two differ
    only in WHERE they are taken.

    Returns `(receptance, info)`; `info` carries the pose for the run record.
    """
    robot, theta, t_star, s_star = commanded_theta_at(setup, frac, verbose=verbose)
    model = tcp_linear_model(
        robot, theta, gravity_on=bool(setup.scene.gravity_on),
        ee_frame=setup.scene.ee_frame, frame=frame,
        name=f"{setup.scene.robot_model} @ {100 * frac:g}% of path")
    info = {"plant_pose_frac": float(frac), "plant_pose_t_s": float(t_star),
            "plant_pose_s_mm": float(s_star),
            "plant_pose_deg": [float(v) for v in np.degrees(theta)]}
    return Receptance.from_mdk(model), info


def receptance_from_robot(cfg, **kw) -> Receptance:
    """`linear_model_from_robot` -> `Receptance.from_mdk`, the common path."""
    return Receptance.from_mdk(linear_model_from_robot(cfg, **kw))


def compliance_report(receptance, scene) -> str:
    """What a steady kilonewton does to this arm, in the workpiece frame.

    Printed before a run because it is the number that decides whether the
    feedforward is needed: `G(0)` times a typical `|F0|` IS the offset the tool
    would otherwise sit at, and on this arm that is hundreds of micrometres
    against a 5 mm radial engagement.
    """
    r = receptance.in_workpiece(scene)
    g = r.compliance_um_per_n
    return ("plant    M/D/K at the tool tip, workpiece frame\n"
            f"         modes    "
            + " ".join(f"{v:.2f}" for v in np.sort(np.asarray(r.modes_hz, float)))
            + " Hz\n"
            "         G(0) [um/N]   "
            + np.array2string(g, precision=3, prefix=" " * 23)
            + f"\n         1 kN along +x_w moves the tool "
            f"{np.linalg.norm(r.static_deflection_um([1e3, 0, 0])):.0f} um")
