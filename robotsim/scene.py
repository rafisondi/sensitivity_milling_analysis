"""Where the job sits relative to the robot.

ONE homogeneous transform places the workpiece. `T` maps WORKPIECE -> BASE,

    p_i = R_iw @ p_w + origin_i,        T = [[R_iw, origin_i],
                                             [0, 0, 0,  1    ]]

so it carries the translation AND the orientation of the job in the robot base
frame. Hand it over and the job sits exactly there:

    Scene(T=make_transform(R_iw, origin_i))      placed by hand
    Scene(R_cut=default_r_iw())                  ORIENTED by hand, positioned at
                                                 the start pose
    Scene()                                      placed at the START POSE

Leave it out and the placement falls back to where the ARM ALREADY IS: forward
kinematics of `start_deg` gives the tool pose, the tool tip becomes the workpiece
origin, and the part is clocked so that the tool, held exactly as it is held
there, is `R_w_tcp` in the part frame — i.e. `scene.R_i_tcp` reproduces the start
orientation. With `R_w_tcp = I` the workpiece frame IS the EE frame. So the
default job is the one lying under the tool where the robot already stands, for
whichever arm, tool and start configuration the scene names — nothing is pinned
to a number that a change of any of the three would silently invalidate.

`R_cut` is the middle ground, and usually the one you want when the stock is
clamped in a fixture rather than presented to the tool: it PINS THE ORIENTATION —
the attitude the whole cut is run at, since `R_i_tcp = R_cut @ R_w_tcp` holds for
every point of the path — and lets the position keep following the start pose, so
the job stays where the arm can reach it. Pass a 3x3 `R_iw` or an xyz-Euler
triple in degrees.

`scene.T_iw` is the resolved 4x4 either way, and is what the rest of the package
reads. The workpiece frame is where the part and its toolpath are authored (the
milling engine uses mm there); everything here is in metres.
"""

from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from robotsim.transforms import invert_transform, make_transform

# The configuration the default placement hangs off. Also the IK seed used
# everywhere else in the project (`runconfig.StartPose.ik_seed_deg`).
DEFAULT_START_DEG = (-90.0, 30.0, 95.0, 0.0, 0.0, 0.0)


def default_r_w_tcp() -> np.ndarray:
    """Tool in the workpiece frame: TCP z -> -Z_w, i.e. pointing down into the stock.

    The clocking is not a free wrist motion here — the TCP sits 277 mm off the
    wrist roll axis, so it picks the IK branch and with it the compliance. It is
    also what the default placement factors out of the start orientation, so it
    has to match the TCP convention of the URDF in use: `Scene.summary` prints
    where the part +Z_w lands in the base frame, which is where a tool clocked
    the wrong way shows up.
    """
    return np.diag([-1.0, 1.0, -1.0])


def default_r_iw() -> np.ndarray:
    """Flat in the cell: +X_w -> -Y_i, +Y_w -> +X_i, +Z_w -> +Z_i. rpy (0, 0, -90).

    The stock lying level on the table with the tool straight down — the Stäubli
    cell as it was set up before the placement followed the tool. Not the default
    any more, but one argument away:

        Scene(R_cut=default_r_iw())     level, positioned at the start pose
    """
    return np.array([[0.0, 1.0, 0.0],
                     [-1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0]])


def as_rotation(R) -> np.ndarray:
    """A 3x3 rotation from a 3x3, or from an xyz-Euler triple in degrees."""
    R = np.asarray(R, dtype=float)
    if R.shape == (3,):
        return Rotation.from_euler("xyz", R, degrees=True).as_matrix()
    if R.shape != (3, 3):
        raise ValueError("an orientation is a 3x3 rotation (or an xyz-Euler "
                         f"triple in degrees), got shape {R.shape}")
    return R.copy()


def as_transform(T) -> np.ndarray:
    """A 4x4 homogeneous transform from a 4x4, or from a bare 3x3 rotation."""
    T = np.asarray(T, dtype=float)
    if T.shape == (3, 3):
        return make_transform(T, np.zeros(3))
    if T.shape != (4, 4):
        raise ValueError("a placement is a 4x4 homogeneous transform (or a 3x3 "
                         f"rotation), got shape {T.shape}")
    return T.copy()


def workpiece_transform(T_i_ee, R_w_tcp=None, *, R_iw=None, first_point_mm=None,
                        lift_mm: float = 0.0) -> np.ndarray:
    """WORKPIECE -> BASE placing the job under a tool held at `T_i_ee`.

        R_iw     = R_i_ee @ R_w_tcp.T                  part clocked to the tool
        origin_i = p_i_ee + R_iw @ ([0, 0, lift] - p0)

    Pass `R_iw` instead and that orientation is used verbatim — the cut runs at
    the attitude you name, and only the POSITION comes from the tool.

    `first_point_mm` is the workpiece-frame point [mm] that lands on the tool
    tip — pass the first waypoint of a toolpath and the arm starts the cut where
    it already stands; None puts the workpiece ORIGIN there. `lift_mm` raises the
    stock along its OWN +Z_w: pass ap and the tool bites the full depth instead
    of scratching the surface.
    """
    T_i_ee = np.asarray(T_i_ee, dtype=float)
    if R_iw is None:
        if R_w_tcp is None:
            raise ValueError("give the placement an orientation: either R_iw, or "
                             "R_w_tcp to take it from the tool")
        R_iw = T_i_ee[:3, :3] @ np.asarray(R_w_tcp, dtype=float).T
    R_iw = as_rotation(R_iw)

    p0_w = np.zeros(3)
    if first_point_mm is not None:
        p = np.asarray(first_point_mm, dtype=float).ravel()
        p0_w[:min(3, p.size)] = p[:3]
    origin_i = T_i_ee[:3, 3] + R_iw @ (np.array([0.0, 0.0, lift_mm]) - p0_w) * 1e-3
    return make_transform(R_iw, origin_i)


def _fkine_deg(robot, theta_deg, ee_frame: str) -> np.ndarray:
    """FK at a MOTOR command in degrees, padded out for the passive-DOF models."""
    q = robot.expand_actuated(np.deg2rad(np.asarray(theta_deg, dtype=float).ravel()))
    return robot.fkine(q, ee_frame)


@lru_cache(maxsize=32)
def _ee_pose_cached(start_deg: tuple, ee_frame: str, robot_model: str,
                    urdf_path: Optional[str]) -> np.ndarray:
    # Imported here rather than at module scope: a Scene is a placement, and
    # reading one should not require pinocchio to be importable. Cached because
    # otherwise every default-placed Scene re-parses the URDF.
    from robotsim.kinematics import build_robot

    robot = build_robot(gravity_on=False, robot_model=robot_model,
                        urdf_path=urdf_path)
    return _fkine_deg(robot, start_deg, ee_frame)


@dataclass(frozen=True)
class Scene:
    """Where the workpiece sits, plus the robot model and the tool clocking.

    T           4x4 WORKPIECE -> BASE, the placement. None (the default) puts
                the job at the start pose — see the module docstring.
    R_cut       the attitude to cut at, as R_iw (3x3) or xyz-Euler [deg]. Pins
                the ORIENTATION while the position keeps following the start
                pose. Ignored once `T` is given, which already fixes both.
    R_w_tcp     the tool in the workpiece frame, 3x3 or xyz-Euler [deg]; what
                the placement factors out of the start orientation when `R_cut`
                does not decide it. Belongs to the URDF's TCP convention, so it
                changes with the robot model
    start_deg   the configuration the default placement hangs off [deg], and
                the IK seed that goes with it
    ee_frame    the URDF frame that IS the tool tip
    """

    T: Optional[np.ndarray] = None
    R_cut: Optional[np.ndarray] = None
    R_w_tcp: np.ndarray = field(default_factory=default_r_w_tcp)
    start_deg: tuple = DEFAULT_START_DEG

    ee_frame: str = "TCP"
    gravity_on: bool = False
    robot_model: str = "joints123"          # see kinematics.ROBOT_MODELS
    urdf_path: Optional[str] = None          # compatible override; None -> model default

    def __post_init__(self):
        object.__setattr__(self, "R_w_tcp", as_rotation(self.R_w_tcp))
        object.__setattr__(self, "start_deg",
                           tuple(float(a) for a in
                                 np.asarray(self.start_deg, dtype=float).ravel()))
        if self.T is not None:
            object.__setattr__(self, "T", as_transform(self.T))
        if self.R_cut is not None:
            object.__setattr__(self, "R_cut", as_rotation(self.R_cut))

    # ── the placement ────────────────────────────────────────────────────────

    @property
    def placed_by_hand(self) -> bool:
        """True when `T` was given, False when the start pose supplies it."""
        return self.T is not None

    @property
    def T_iw(self) -> np.ndarray:
        """4x4 WORKPIECE -> BASE — the placement, resolved. Read this one."""
        T = self.__dict__.get("_T_iw")
        if T is None:
            T = (self.T if self.T is not None else
                 workpiece_transform(self.ee_pose(), self.R_w_tcp, R_iw=self.R_cut))
            object.__setattr__(self, "_T_iw", T)
        return T

    @property
    def T_wi(self) -> np.ndarray:
        """4x4 BASE -> WORKPIECE."""
        return invert_transform(self.T_iw)

    @property
    def R_iw(self) -> np.ndarray:
        return self.T_iw[:3, :3]

    @property
    def origin_i(self) -> np.ndarray:
        return self.T_iw[:3, 3]

    @property
    def R_i_tcp(self) -> np.ndarray:
        """Base-frame TCP orientation, tool clocked to the workpiece."""
        return self.R_iw @ self.R_w_tcp

    def ee_pose(self, robot=None, start_deg=None) -> np.ndarray:
        """4x4 base-frame pose of `ee_frame` at a joint configuration [deg].

        The `start_deg` of the scene unless told otherwise. Pass `robot` to reuse
        an arm you already built; without it the URDF is parsed once per (model,
        frame, configuration) and cached.
        """
        theta_deg = self.start_deg if start_deg is None else start_deg
        if robot is not None:
            return _fkine_deg(robot, theta_deg, self.ee_frame)
        key = tuple(float(a) for a in np.asarray(theta_deg, dtype=float).ravel())
        return _ee_pose_cached(key, self.ee_frame, self.robot_model,
                               self.urdf_path).copy()

    def placed(self, T) -> "Scene":
        """The same scene with the job pinned at `T` (4x4, or a 3x3 rotation)."""
        return replace(self, T=as_transform(T))

    def placed_at_ee(self, T_i_ee, *, first_point_mm=None,
                     lift_mm: float = 0.0) -> "Scene":
        """Place the job under a tool held at the base-frame pose `T_i_ee`.

        Honours `R_cut`: only the position comes from the tool when an attitude
        was named. See `workpiece_transform` for `first_point_mm` and `lift_mm`.
        """
        return replace(self, T=workpiece_transform(
            T_i_ee, self.R_w_tcp, R_iw=self.R_cut,
            first_point_mm=first_point_mm, lift_mm=lift_mm))

    def placed_at_start(self, robot=None, start_deg=None, *, first_point_mm=None,
                        lift_mm: float = 0.0) -> "Scene":
        """`placed_at_ee` on the FK of a start configuration [deg].

        With no arguments it pins exactly what an unplaced scene resolves to.
        """
        scene = self if start_deg is None else replace(self, start_deg=start_deg)
        return scene.placed_at_ee(scene.ee_pose(robot),
                                  first_point_mm=first_point_mm, lift_mm=lift_mm)

    def unplaced(self) -> "Scene":
        """Drop the placement — back to following the start pose."""
        return replace(self, T=None)

    # ── moving the job around ────────────────────────────────────────────────

    def point_i(self, xy_mm, z_mm: float = 0.0) -> np.ndarray:
        """Workpiece-frame point [mm] -> base-frame position [m]."""
        p = np.asarray(xy_mm, dtype=float).ravel()
        p_w = np.array([p[0], p[1], float(z_mm) if p.size < 3 else p[2]]) * 1e-3
        return self.R_iw @ p_w + self.origin_i

    def point_w(self, p_i) -> np.ndarray:
        """Base-frame position [m] -> workpiece-frame point [mm]."""
        T = self.T_wi
        return (T[:3, :3] @ np.asarray(p_i, dtype=float).reshape(3) + T[:3, 3]) * 1e3

    def rotated(self, R_extra, pivot_w=None) -> "Scene":
        """Turn the stock about its own axes, holding `pivot_w` [m] fixed in base.

        Pins the placement: a scene you have turned stops following the start
        pose, as it must — the turn is relative to where it stood.
        """
        R0, o0 = self.R_iw, self.origin_i
        R_extra = np.asarray(R_extra, dtype=float)
        pivot = np.zeros(3) if pivot_w is None else np.asarray(pivot_w, dtype=float)
        return self.placed(make_transform(
            R0 @ R_extra, o0 + R0 @ (np.eye(3) - R_extra) @ pivot))

    def translated(self, d_i=None, d_w=None) -> "Scene":
        """Move the stock [m]: d_i along base axes, d_w along workpiece axes.

        Rigid — the cut geometry is untouched; what changes is the arm
        configuration and with it the compliance. Watch reachability. Pins the
        placement, like `rotated`.
        """
        origin = self.origin_i.copy()
        if d_i is not None:
            origin = origin + np.asarray(d_i, dtype=float).reshape(3)
        if d_w is not None:
            origin = origin + self.R_iw @ np.asarray(d_w, dtype=float).reshape(3)
        return self.placed(make_transform(self.R_iw, origin))

    # ── reporting ────────────────────────────────────────────────────────────

    def summary(self) -> str:
        o, R = self.origin_i, self.R_iw
        rpy = np.round(Rotation.from_matrix(R).as_euler("xyz", degrees=True), 2)
        at_start = f"at the start pose {np.round(np.asarray(self.start_deg), 2)} deg"
        how = ("given" if self.placed_by_hand else
               f"R_cut given, positioned {at_start}" if self.R_cut is not None
               else at_start)
        urdf = Path(self.urdf_path).name if self.urdf_path else "(model default)"
        return (f"scene    workpiece origin ({o[0]:.3f}, {o[1]:.3f}, {o[2]:.3f}) m "
                f"| rpy ({rpy[0]:g}, {rpy[1]:g}, {rpy[2]:g}) deg  [{how}]\n"
                f"         +Z_w in base ({R[0, 2]:+.3f}, {R[1, 2]:+.3f}, "
                f"{R[2, 2]:+.3f}) | ee '{self.ee_frame}'\n"
                f"         robot model '{self.robot_model}' | gravity "
                f"{'on' if self.gravity_on else 'off'}\n"
                f"         urdf {urdf}")
