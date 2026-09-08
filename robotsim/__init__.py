"""A flexible-joint serial arm: URDF, IK, trajectories, dynamics, linearisation.

No milling here. The only thing this package knows about a cutting process is a
duck type: `process.step(xy_mm, spindle_angle_rad) -> (3,) force [N]`.

    scene  = Scene()                                     where the job sits
    robot  = build_robot(gravity_on=False)               URDF + spring tables
    joints = kinematics.solve_ik(robot, path, seed_rad)

Then write the loop yourself — see `robotsim.dynamics.Simulator`. There is no
batch "run" function: the loop belongs in your script, not behind an import.

TWO VIEWS OF THE SAME ARM, and this workspace uses both:

    Simulator / Solver          the nonlinear flexible-joint arm, RK4, the TRUTH
    tcp_linear_model            that arm linearised at one pose into M/D/K at
                                the tool tip — `LinearModel`
    Receptance.from_mdk         the same M/D/K as a state space, which is what
                                `stabsim.stability` closes the cut around

`receptance.py` is the one module that is not upstream's: it is `gdsim/plant.py`
with the measured-data reader taken out, because every plant here is built from
the pinocchio linearisation rather than read from a shaker identification.

`plotting.py` and `viz.py` are not carried — this workspace writes its results to
disk rather than to a window.
"""

from robotsim import (
    dynamics, kinematics, linear, linearize_robot_dynamics, receptance, scene,
    trajectory, transforms,
)
from robotsim.dynamics import SimResult, Simulator
from robotsim.kinematics import (
    ROBOT_MODELS, JointTrajectory, build_robot, hold_pose_trajectory,
    joint_limit_report, load_robot, place_workpiece_origin, solve_ik,
    solve_ik_points, solve_ik_pose, tcp_pose_i,
)
from robotsim.linear import LinearModel, LinearPlant, default_model
from robotsim.linearize_robot_dynamics import (
    TrajectoryLinearization, linearize_trajectory, tau_partials, tcp_linear_model,
)
from robotsim.receptance import Receptance, ReceptancePlant
from robotsim.robot import Robot
from robotsim.scene import (
    DEFAULT_START_DEG, Scene, default_r_iw, default_r_w_tcp, workpiece_transform,
)
from robotsim.settings import (
    Settings, SettingsAllAxesSDOF, SettingsRealParameter, SettingsTriJoint,
)
from robotsim.solver import Solver
from robotsim.trajectory import (
    OperationalPath, Segment, chain, hold, linear_segment, plan_linear_move,
    plan_quintic_path, quintic_scaling, quintic_segment,
)

__all__ = [
    "Robot", "Solver", "Settings", "SettingsRealParameter", "SettingsAllAxesSDOF",
    "SettingsTriJoint", "build_robot", "load_robot", "ROBOT_MODELS",
    "Scene", "workpiece_transform", "default_r_iw", "default_r_w_tcp",
    "DEFAULT_START_DEG", "place_workpiece_origin",
    "OperationalPath", "Segment", "quintic_scaling", "quintic_segment",
    "linear_segment", "chain", "plan_quintic_path", "plan_linear_move", "hold",
    "JointTrajectory", "solve_ik", "solve_ik_points", "solve_ik_pose",
    "hold_pose_trajectory", "tcp_pose_i", "joint_limit_report",
    "Simulator", "SimResult",
    "LinearModel", "LinearPlant", "default_model",
    "Receptance", "ReceptancePlant",
    "tcp_linear_model", "tau_partials", "linearize_trajectory",
    "TrajectoryLinearization",
    "dynamics", "kinematics", "linear", "linearize_robot_dynamics", "receptance",
    "scene", "trajectory", "transforms",
]
