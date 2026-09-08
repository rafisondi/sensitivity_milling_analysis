"""One place for every knob. Edit the defaults here; override per run from JSON.

    from runconfig import RunConfig

    cfg = RunConfig()                          # the defaults below
    cfg = RunConfig.load("configs/deep.json")  # a saved variation
    cfg.save(cfg.out() / "config.json")        # what this run actually used

It composes the dataclasses that already exist — `robotsim.Scene` and
`fastsim.MillConfig` — rather than restating their fields, so there is one
source of truth for every number.

    cfg.sim      sim_dt, plan_dt, duration
    cfg.scene    robot model, URDF override, base <- workpiece transform, gravity
    cfg.mill     cutter, ap, ae, spindle, raster                (fastsim.MillConfig)
    cfg.part     the stock: rectangle dimensions, or a saved pickle
    cfg.path     the contour: which edges, lead-in, feed — or a saved .npz
    cfg.start    starting joints or TCP pose; also supplies the path IK seed
    cfg.io       where results go

Frames. `scene.T_iw` maps WORKPIECE -> BASE (`p_i = R_iw @ p_w + origin_i`);
`base_to_workpiece()` is its inverse. Base frame is metres, workpiece frame is
millimetres — the milling engine's unit. Where the job sits is ONE transform,
`cfg.scene.T`: leave it None and the part is placed under the tool at `cfg.start`
— the arm mills where it already stands — set it and the part sits exactly there.
"""

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from fastsim import geometry
from fastsim.config import MillConfig
from robotsim import Scene, kinematics, trajectory
from robotsim.scene import default_r_iw
from robotsim.transforms import make_transform

REPO = Path(__file__).resolve().parent


def rpy_to_matrix(rpy_deg) -> np.ndarray:
    return Rotation.from_euler("xyz", np.asarray(rpy_deg, float), degrees=True).as_matrix()


def matrix_to_rpy(R) -> np.ndarray:
    """xyz-Euler [deg]. Rounded: the decomposition carries ~1e-14 deg of noise,
    which would otherwise make a saved config differ from the one it came from."""
    rpy = Rotation.from_matrix(np.asarray(R, float)).as_euler("xyz", degrees=True)
    return np.round(rpy, 12)


# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SimParams:
    """Time stepping. `sim_dt` must resolve the tooth-passing period."""

    sim_dt: float = 1.0e-3          # dynamics + process step [s]
    plan_dt: float = 1.0e-3         # trajectory / IK sampling step [s]
    duration_s: float = 2.0        # hold-pose runs; a path run uses its own length


@dataclass(frozen=True)
class StartPose:
    """Where the robot starts — given either as a TCP pose or as joint angles.

    joints_deg  (n,) joint angles [deg]. When set this WINS: it IS the starting
                configuration and `pos_m` / `rpy_deg` are ignored. FK then says
                where the TCP ended up. No IK runs, so there is no branch
                ambiguity, no reachability question and no joint-limit surprise.
    pos_m       (3,) TCP position [m], BASE frame
    rpy_deg     (3,) TCP xyz-Euler [deg], BASE frame; None = the tool clocked
                with the workpiece, i.e. `scene.R_i_tcp`

    Ask for the start through `theta()` / `pose()` rather than reading the
    fields — those resolve whichever of the two ways was used.
    """

    pos_m: tuple = (-0.0250, -1.6224, 0.4857)
    rpy_deg: Optional[tuple] = None
    joints_deg: Optional[tuple] = None
    ik_seed_deg: tuple = (-90.0, 30.0, 95.0, 0.0, 0.0, 0.0)

    @property
    def from_joints(self) -> bool:
        return self.joints_deg is not None

    @property
    def seed_deg(self) -> np.ndarray:
        """IK seed, derived from an explicit joint start when supplied."""
        values = self.joints_deg if self.from_joints else self.ik_seed_deg
        return np.asarray(values, dtype=float)

    @property
    def seed_rad(self) -> np.ndarray:
        return np.deg2rad(self.seed_deg)

    def position(self) -> np.ndarray:
        """The REQUESTED TCP position [m]. Meaningless when `from_joints`."""
        return np.asarray(self.pos_m, dtype=float).reshape(3)

    def rotation(self, scene: Scene) -> np.ndarray:
        """The REQUESTED TCP orientation. Meaningless when `from_joints`."""
        return scene.R_i_tcp if self.rpy_deg is None else rpy_to_matrix(self.rpy_deg)

    def anchor_pose(self, robot, ee_frame: str = "TCP") -> np.ndarray:
        """The 4x4 TCP pose an UNPLACED scene hangs the workpiece off.

        The start pose itself whenever it is known without a placement: the
        requested pose when an explicit `rpy_deg` came with it, otherwise FK of
        the starting configuration. `rpy_deg=None` means "clocked to the
        workpiece", which would need the very placement it is about to define —
        so that case falls back to the seed configuration, and the tool ends up
        clocked to the part by construction.
        """
        if not self.from_joints and self.rpy_deg is not None:
            return make_transform(rpy_to_matrix(self.rpy_deg), self.position())
        return robot.fkine(robot.expand_actuated(self.seed_rad), ee_frame)

    def theta(self, robot, scene: Scene) -> np.ndarray:
        """The starting joint configuration [rad] — the motor command.

        Straight from `joints_deg` when given (padded to the full joint vector
        for the compliance models that have passive DOFs), otherwise solved by
        IK from the requested pose.
        """
        if self.from_joints:
            return robot.expand_actuated(
                np.deg2rad(np.asarray(self.joints_deg, dtype=float).reshape(-1)))
        return kinematics.solve_ik_pose(robot, self.position(),
                                        self.rotation(scene),
                                        self.seed_rad,
                                        ee_frame=scene.ee_frame)

    def pose(self, robot, scene: Scene) -> np.ndarray:
        """4x4 TCP pose the start ACTUALLY lands on — FK of `theta`.

        Equal to the requested pose (to the IK tolerance) in the pose case; the
        thing you actually want to know in the joints case.
        """
        return robot.fkine(self.theta(robot, scene), scene.ee_frame)

    def describe(self, robot, scene: Scene) -> str:
        """Both halves of the start: the one you gave, and the one it implies."""
        T = self.pose(robot, scene)
        p, rpy = T[:3, 3], matrix_to_rpy(T[:3, :3])
        pos = f"({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}) m"

        if self.from_joints:
            joints = np.round(np.asarray(self.joints_deg, dtype=float), 2)
            return (f"start    joints {joints} deg\n"
                    f"         -> TCP {pos}, rpy {np.round(rpy, 2)} deg  (FK)")

        residual = np.linalg.norm(p - self.position()) * 1e6
        rpy_txt = ("clocked to workpiece" if self.rpy_deg is None
                   else f"rpy {tuple(self.rpy_deg)} deg")
        return (f"start    TCP {pos} | {rpy_txt}\n"
                f"         -> joints by IK, residual {residual:.2f} um")


@dataclass(frozen=True)
class PartSpec:
    """The stock. A rectangle by default; `pickle` loads a saved outline instead."""

    length_mm: float = 200.0        # +X_w extent
    width_mm: float = 200.0          # +Y_w extent
    height_mm: float = 5.0          # extrusion = axial depth ap
    name: str = "rect_contour_large"
    pickle: Optional[str] = None    # data/workpieces/<name>.pickle

    def build(self) -> geometry.Workpiece:
        if self.pickle:
            return geometry.load_workpiece(self.pickle, height_mm=self.height_mm)
        return geometry.rectangle(self.length_mm, self.width_mm,
                                  height_mm=self.height_mm, name=self.name)


@dataclass(frozen=True)
class PathSpec:
    """The contour to mill, or a saved toolpath to replay.

    speed_mm_s is the MEAN speed of each quintic segment; the peak is 1.875x.
    `toolpath` names a .npz written by `make_path.py --save` and, when set,
    every other field here is ignored.

    `place_by_fk` only bites while the scene is unplaced (`scene.T is None`).
    It then decides WHICH workpiece point lands on the tool tip at the start
    pose: the first waypoint, so the arm opens the cut where it already stands,
    or — when False — the workpiece origin. A scene with an explicit `T` is left
    exactly where it was put either way.
    """

    n_edges: int = 2                # 2 -> lead-in, A, B, C
    start_edge: int = 0
    lead_in_mm: float = 25.0
    lead_out_mm: float = 25.0
    speed_mm_s: float = 40.0
    place_by_fk: bool = True        # register waypoint 0 on the tool tip
    toolpath: Optional[str] = None  # data/toolpaths/<name>.npz


@dataclass(frozen=True)
class IO:
    """Where things are read from and written to."""

    out_dir: str = "out"
    workpiece_dir: str = "data/workpieces"
    toolpath_dir: str = "data/toolpaths"
    stamp_runs: bool = True         # out/<name>_<YYYYmmdd-HHMMSS>/

    def _abs(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else REPO / p

    def run_dir(self, name: str) -> Path:
        stamp = f"_{datetime.now():%Y%m%d-%H%M%S}" if self.stamp_runs else ""
        out = self._abs(self.out_dir) / f"{name}{stamp}"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def toolpath_file(self, name: str) -> Path:
        p = Path(name)
        return p if p.suffix else self._abs(self.toolpath_dir) / f"{name}.npz"


# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RunConfig:
    """Everything one run needs."""

    name: str = "run"
    sim: SimParams = field(default_factory=SimParams)
    scene: Scene = field(default_factory=lambda: Scene(R_cut=default_r_iw()))
    mill: MillConfig = field(default_factory=MillConfig)
    part: PartSpec = field(default_factory=PartSpec)
    path: PathSpec = field(default_factory=PathSpec)
    start: StartPose = field(default_factory=StartPose)
    io: IO = field(default_factory=IO)

    def __post_init__(self):
        """One start for the whole config.

        An unplaced scene hangs the job off a starting configuration, and this
        config already has one — so hand `start.seed_deg` down rather than let
        `Scene.start_deg` keep its own default. Without it `cfg.scene` on its own
        would report a placement `build()` never uses.
        """
        if (not self.scene.placed_by_hand
                and self.scene.start_deg != tuple(self.start.seed_deg)):
            object.__setattr__(self, "scene",
                               replace(self.scene, start_deg=tuple(self.start.seed_deg)))

    # ── frames ───────────────────────────────────────────────────────────────

    def base_to_workpiece(self) -> np.ndarray:
        """4x4 BASE -> WORKPIECE. The inverse of `scene.T_iw`."""
        return self.scene.T_wi

    # ── the pieces ───────────────────────────────────────────────────────────

    def robot(self):
        return kinematics.load_robot(self.scene)

    def milling(self) -> MillConfig:
        """`mill` with the fields that live elsewhere in this config applied."""
        return replace(self.mill, workpiece=self.part.name,
                       height_mm=self.part.height_mm,
                       feed_mm_s=self.path.speed_mm_s,
                       sim_dt=self.sim.sim_dt, plan_dt=self.sim.plan_dt)

    def build(self):
        mill = self.milling()
        part = self.part.build()

        if self.path.toolpath:
            waypoints, xy_mm, speed = self._load_toolpath()
        else:
            waypoints = part.contour_waypoints(
                mill.tool_offset_mm, start_edge=self.path.start_edge,
                n_edges=self.path.n_edges, lead_in_mm=self.path.lead_in_mm,
                lead_out_mm=self.path.lead_out_mm)
            xy_mm = speed = None

        scene = self.scene
        if not scene.placed_by_hand:
            # Nobody said where the job sits, so it goes where the arm already
            # is: under the tool at the start pose, lifted by ap so the cut runs
            # at full depth. The scene's OWN robot — placing a job with another
            # arm's kinematics puts it where this one cannot reach.
            robot = kinematics.load_robot(scene)
            scene = scene.placed_at_ee(
                self.start.anchor_pose(robot, scene.ee_frame),
                first_point_mm=waypoints[0] if self.path.place_by_fk else None,
                lift_mm=part.height_mm)

        if xy_mm is None:
            path = trajectory.plan_quintic_path(
                waypoints, self.path.speed_mm_s, T_iw=scene.T_iw,
                R_i_tcp=scene.R_i_tcp, dt=self.sim.plan_dt)
        else:
            path = _path_from_samples(xy_mm, speed, scene, self.sim.plan_dt)
        return mill, part, scene, waypoints, path

    def _load_toolpath(self):
        f = self.io.toolpath_file(self.path.toolpath)
        if not f.exists():
            raise FileNotFoundError(f"no saved toolpath at {f}")
        d = np.load(f, allow_pickle=False)
        return d["waypoints_mm"], d["xy_mm"], d["t"]

    # ── output ───────────────────────────────────────────────────────────────

    def out(self) -> Path:
        """The run's output folder, created."""
        return self.io.run_dir(self.name)

    # ── persistence ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        d = asdict(self)
        s = self.scene
        # The placement T is written split into its translation and its xyz-Euler
        # rotation — the same numbers, in the form a human edits. `placed_by_hand`
        # says whether they ARE the placement or just a record of where the start
        # pose put the job: False re-derives them on load, so a config follows the
        # robot it is loaded with.
        d["scene"] = {
            "placed_by_hand": s.placed_by_hand,
            "origin_i_m": np.asarray(s.origin_i, float).tolist(),
            "R_iw_rpy_deg": matrix_to_rpy(s.R_iw).tolist(),
            "R_cut_rpy_deg": (None if s.R_cut is None
                              else matrix_to_rpy(s.R_cut).tolist()),
            "R_w_tcp_rpy_deg": matrix_to_rpy(s.R_w_tcp).tolist(),
            "start_deg": list(s.start_deg),
            "ee_frame": s.ee_frame,
            "gravity_on": s.gravity_on,
            "robot_model": s.robot_model,
            "urdf_path": s.urdf_path,
        }
        st = self.start
        d["start"] = {
            "pos_m": tuple(st.pos_m),
            "rpy_deg": st.rpy_deg,
            "joints_deg": st.joints_deg,
        }
        if not st.from_joints:
            d["start"]["ik_seed_deg"] = tuple(st.ik_seed_deg)
        return _jsonable(d)

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        d = dict(d)
        sc = d.pop("scene", None)
        sc = {} if sc is None else dict(sc)
        legacy_seed = sc.pop("seed_pose_deg", None)
        # A file written before the placement was one transform has no
        # `placed_by_hand` flag; its origin was derived, not chosen, so it is
        # only honoured when the file says so.
        T = None
        if sc.get("placed_by_hand") and "origin_i_m" in sc:
            T = make_transform(rpy_to_matrix(sc.get("R_iw_rpy_deg", (0.0, 0.0, 0.0))),
                               np.asarray(sc["origin_i_m"], float))
        # A pre-T file fixed `R_iw` outright and only ever re-derived the origin,
        # so its orientation carries over as the cut attitude — otherwise loading
        # one would silently re-clock the job to the tool.
        R_cut = sc.get("R_cut_rpy_deg")
        if R_cut is None and "placed_by_hand" not in sc:
            R_cut = sc.get("R_iw_rpy_deg")
        scene = Scene(
            T=T,
            R_cut=None if R_cut is None else rpy_to_matrix(R_cut),
            R_w_tcp=(rpy_to_matrix(sc["R_w_tcp_rpy_deg"]) if "R_w_tcp_rpy_deg" in sc
                     else Scene().R_w_tcp),
            start_deg=tuple(sc.get("start_deg", Scene().start_deg)),
            ee_frame=sc.get("ee_frame", "TCP"),
            gravity_on=sc.get("gravity_on", False),
            robot_model=sc.get("robot_model", sc.get("compliance", "joints123")),
            urdf_path=sc.get("urdf_path"))

        sub = {"sim": SimParams, "mill": MillConfig, "part": PartSpec,
               "path": PathSpec, "start": StartPose, "io": IO}
        if legacy_seed is not None:
            start = dict(d.get("start", {}))
            start.setdefault("ik_seed_deg", tuple(legacy_seed))
            d["start"] = start
        kwargs = {k: sub[k](**d.pop(k)) for k in list(sub) if k in d}
        return cls(scene=scene, **kwargs, **d)

    @classmethod
    def load(cls, path) -> "RunConfig":
        p = Path(path)
        p = p if p.is_absolute() else REPO / p
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))

    # ── reporting ────────────────────────────────────────────────────────────

    def summary(self) -> str:
        st = self.start
        start_txt = (f"start    joints {np.round(np.asarray(st.joints_deg, float), 2)} deg"
                     if st.from_joints else
                     f"start    TCP base ({st.pos_m[0]:.4f}, {st.pos_m[1]:.4f}, "
                     f"{st.pos_m[2]:.4f}) m | "
                     + ("clocked to workpiece" if st.rpy_deg is None
                        else f"rpy {tuple(st.rpy_deg)} deg"))
        source = (f"toolpath '{self.path.toolpath}'" if self.path.toolpath
                  else f"{self.path.n_edges} edges from edge {self.path.start_edge}")
        return "\n".join([
            f"run      '{self.name}' -> {self.io._abs(self.io.out_dir)}",
            f"sim      dt {self.sim.sim_dt:g} s | plan_dt {self.sim.plan_dt:g} s "
            f"| duration {self.sim.duration_s:g} s",
            self.scene.summary(),
            start_txt,
            self.milling().summary(),
            f"path     {source} | lead-in {self.path.lead_in_mm:g} mm | "
            f"{self.path.speed_mm_s:g} mm/s mean",
        ])


# ─────────────────────────────────────────────────────────────────────────────

def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return o


def _path_from_samples(xy_mm, t, scene: Scene, dt: float):
    """Rebuild an OperationalPath from saved samples.

    `s_i` is re-derived from `s_w` through the CURRENT scene rather than read
    back, so a saved toolpath follows the part when the job is moved.
    """
    from robotsim.trajectory import OperationalPath
    from robotsim.transforms import rotate_vectors, transform_points

    t = np.asarray(t, float)
    xy_mm = np.asarray(xy_mm, float)
    grid = np.arange(0.0, t[-1], dt)
    xy = np.column_stack([np.interp(grid, t, xy_mm[:, k]) for k in (0, 1)])

    s_w = np.column_stack([xy * 1e-3, np.zeros(len(grid))])
    v_w = np.gradient(s_w, dt, axis=0, edge_order=2)
    return OperationalPath(
        s_w=s_w, s_i=transform_points(scene.T_iw, s_w),
        R_i_tcp=scene.R_i_tcp, t=grid, dt=dt,
        v_w=v_w, v_i=rotate_vectors(scene.T_iw[:3, :3], v_w))


# ─────────────────────────────────────────────────────────────────────────────
# The workspace default: the Staeubli, cutting one edge of the rectangular part.
#
# Upstream carried several more example configs here, all of them KUKA KR60
# variants. That arm is fully RIGID in `robotsim.settings` and is not part of
# this workspace (see `robotsim.kinematics.ROBOT_MODELS`), so they are gone.
# `configs/base.json` is what the runs actually load; this is the object it was
# written from.
# ─────────────────────────────────────────────────────────────────────────────

STAEUBLI_ORIGIN_I = (0.005, -1.677400, 0.485698)   # what that resolves to with
                                                   # the default part and path
default = RunConfig(
    name="Staeubli_v0",
    scene=Scene(
        R_cut=default_r_iw()        # or an xyz-Euler triple: (0.0, 0.0, -90.0)
    ),
)

if __name__ == "__main__":
    print(default.summary())
    out = default.save(REPO / "configs" / "default.json")
    print(f"\nwrote {out}")
    assert RunConfig.load(out).to_dict() == default.to_dict(), "round-trip failed"
    print("round-trip OK")
