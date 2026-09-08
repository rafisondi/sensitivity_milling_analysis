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
