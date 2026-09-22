"""A job that opens mid-cut: the stock already milled up to s0, the arm already
moving and already carrying the load.

    job   = build_job(cfg, s0_mm=40.0, length_mm=50.0)
    setup = prepare(job)                  the linear side, on the precut stock

WHAT "PRECUT" MEANS HERE

`s0` is measured along the edge being cut, from the part corner the edge starts
at: the tool centre, running on the offset contour `R - ae` outside the edge, is
abreast of edge point `p0 + s0 u`. Everything the tool would have removed on its
way there is already gone, and the run starts with the tool centre at `s0` and
drives `length_mm` further along the edge, then stops, still cutting. No entry,
no exit, nothing but the steady cut.

Two sides need that stock, and each gets it in the form it can use:

  the PREDICTION (Shapely engagement -> eigenvalues, F0 -> feedforward) gets an
    explicit outline, `precut_part`: the rectangle minus the channel the tool
    disc swept up to `s0 - ds`. That is exactly what node `s0` sees in the full
    entry-to-exit run, whose erosion removes the disc at every PREVIOUS node, so
    every node here is a steady node from the first one on.

  the ENGINE (dexel raster) gets the virgin rectangle and prepares it itself,
    through `fastsim.prepare.prepare_steady_state_cut`: carve the swept channel
    up to `warmup_revs` revolutions short of `s0`, then mill that last stretch
    rigidly at feed so the surface ahead of the tool carries the real tooth
    scallops and the engine's one-tooth lookback buffer is full at t = 0. A raster
    precut to `s0` itself would leave a smooth arc in front of the tool instead,
    and the first tooth period would ramp the chip up from zero.

THE ARM

The workpiece is placed so the TCP at `cfg.start` sits on the `s0` point at full
depth - the robot's start pose IS where the steady cut starts, and the plant
`analysis.plant.receptance_from_robot` linearises there. The toolpath is at full
feed from its first sample ("flying"), so `Simulator.reset` opens on the tracking
equilibrium at that velocity, and with `steady_state=True` it presets the springs
for the mean cutting force F0 while the motors carry `-J^T F0`: preloaded and
compliance-compensated at t = 0. See `analysis.sim_coupled.simulate`.
"""

from dataclasses import dataclass, replace

import numpy as np

from analysis import config as acfg
from analysis import feedplan
from analysis.stability import Setup
from fastsim.geometry import Workpiece
from robotsim import kinematics
from stabsim.engagement import (engagement_angles_along_path,
                                engagement_gradients_along_path)


@dataclass(frozen=True)
class EdgeFrame:
    """Edge `index` of the part, and the line the tool centre runs on beside it.

    p0      (2,) the part corner the edge starts at [mm]
    u       (2,) unit direction along the edge
    n_out   (2,) outward normal, out of the material
    length  edge length [mm]
    offset  `R - ae`, how far outside the edge the tool centre runs [mm]
    radius  tool radius R [mm]
    """

    index: int
    p0: np.ndarray
    u: np.ndarray
    n_out: np.ndarray
    length: float
    offset: float
    radius: float

    def point(self, s_mm) -> np.ndarray:
        """Tool-centre position [mm] abreast of edge point `p0 + s u`."""
        return self.p0 + self.n_out * self.offset + self.u * float(s_mm)

    @property
    def reach_mm(self) -> float:
        """How far AHEAD of the tool centre the disc still touches material.

        The leading wall crossing sits `sqrt(R^2 - off^2)` ahead of the centre
        while the centre runs outside the edge; once it runs inside (ae >= R) the
        disc's own front point is in the material, a full R ahead.
        """
        if self.offset <= 0.0:
            return self.radius
        return float(np.sqrt(max(self.radius ** 2 - self.offset ** 2, 0.0)))


def edge_frame(cfg, edge_index: int = None) -> EdgeFrame:
    edge_index = cfg.path.start_edge if edge_index is None else int(edge_index)
    mill = cfg.milling()
    e = cfg.part.build().edge(edge_index)
    return EdgeFrame(index=e.index, p0=e.p0, u=e.direction, n_out=e.outward_normal,
                     length=e.length_mm, offset=float(mill.tool_offset_mm),
                     radius=float(mill.radius_mm))


def precut_part(part: Workpiece, edge: EdgeFrame, s_cut_mm: float,
                quad_segs: int = 64) -> Workpiece:
    """`part` with the channel the tool disc swept up to `s_cut_mm` removed.

    The channel starts well behind the corner, so the whole approach is clear,
    and ends in the disc at `s_cut_mm` - the concave arc the next tooth meets.
    """
    from shapely.geometry import LineString, Polygon

    material = Polygon(part.boundary_xy_mm.T)
    s_from = -(edge.radius + abs(edge.offset) + 1.0)
    swept = LineString([edge.point(s_from), edge.point(s_cut_mm)]).buffer(
        edge.radius, quad_segs=quad_segs)
    left = material.difference(swept)
    if left.geom_type == "MultiPolygon":
        left = max(left.geoms, key=lambda g: g.area)
    xy = np.asarray(left.exterior.coords, float)[:-1].T      # (2, N), open ring
    x, y = xy
    if 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0.0:
        xy = xy[:, ::-1]                                     # CCW, like Workpiece
    return Workpiece(boundary_xy_mm=xy, height_mm=part.height_mm,
                     name=f"{part.name}_precut{s_cut_mm:g}")


def toolpath_name(cfg, edge: EdgeFrame, s0_mm, length_mm) -> str:
    """One file per stock, offset, edge, start, length and feed - no decimal
    points, for the same reason as `analysis.toolpath.name_for`."""
    return (f"precut_{cfg.part.length_mm:g}x{cfg.part.width_mm:g}"
            f"_off{edge.offset:g}_e{edge.index}_s{float(s0_mm):g}"
            f"_L{float(length_mm):g}_v{float(cfg.path.speed_mm_s):g}"
            ).replace(".", "p").replace("-", "m")


def write_toolpath(cfg, edge: EdgeFrame, s0_mm, length_mm) -> str:
    """The steady toolpath: straight along the edge from `s0`, at full feed from
    the first sample. Returns its name under `data/toolpaths/`."""
    v = float(cfg.path.speed_mm_s)
    dt = float(cfg.sim.plan_dt)
    t = np.arange(0.0, float(length_mm) / v + 0.5 * dt, dt)
    s = np.minimum(v * t, float(length_mm))
    start = edge.point(s0_mm)
    xy = start + np.outer(s, edge.u)
    waypoints = np.array([start, edge.point(float(s0_mm) + float(length_mm))])

    name = toolpath_name(cfg, edge, s0_mm, length_mm)
    out = acfg.TOOLPATH_DIR / f"{name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    feedplan.save_toolpath(out, waypoints, t, xy, np.full_like(t, v),
                           name=name, height_mm=cfg.part.height_mm)
    return name


def place_tcp_at(cfg, xy_mm):
    """Pin the job so the TCP at `cfg.start` sits on workpiece point `xy_mm`, at
    full depth. The robot does not move - only where the job sits relative to
    it (the same construction as `move_wp_constantRobPose.place_at_edge_midpoint`).

    The start's `pos_m` is then moved onto that point too. The job is anchored
    on FK of the IK seed, but with a cut attitude set (`R_cut`) the tool is
    re-clocked upright there, and `StartPose.theta` - the pose the plant is
    linearised at - solves IK for `pos_m`, which in `configs/base.json` is a
    point some 35 mm away. Pointing it at the path's first sample makes
    `cfg.start.theta` the commanded configuration at t = 0, on s0.
    """
    scene = cfg.scene.unplaced()
    robot = kinematics.load_robot(scene)
    anchor = cfg.start.anchor_pose(robot, scene.ee_frame)
    placed = scene.placed_at_ee(anchor, first_point_mm=np.asarray(xy_mm, float),
                                lift_mm=cfg.part.height_mm)
    start = replace(cfg.start, pos_m=tuple(float(v) for v in placed.point_i(xy_mm)))
    return replace(cfg, scene=placed, start=start)


@dataclass(frozen=True)
class SteadyJob:
    """One precut, steady-state job.

    cfg          runs it: pinned scene, the steady toolpath, the VIRGIN stock
                 (the engine carves its own - see the module docstring)
    part_pred    the precut stock the prediction measures its engagement on
    s0_mm        where the run starts, along the edge from its start corner
    length_mm    how much steady cut is simulated
    s_cut_mm     how far `part_pred` is precut (`s0 - ds`)
    """

    cfg: object
    edge: EdgeFrame
    part_pred: Workpiece
    s0_mm: float
    length_mm: float
    s_cut_mm: float
    ds_mm: float

    @property
    def start_xy_mm(self) -> np.ndarray:
        return self.edge.point(self.s0_mm)

    @property
    def end_xy_mm(self) -> np.ndarray:
        return self.edge.point(self.s0_mm + self.length_mm)

    @property
    def warmup_mm(self) -> float:
        return float(self.cfg.milling().warmup_distance_mm)

    def row(self) -> dict:
        return {"precut_s0_mm": float(self.s0_mm),
                "precut_length_mm": float(self.length_mm),
                "precut_s_cut_pred_mm": float(self.s_cut_mm),
                "precut_s_cut_engine_mm": float(self.s0_mm - self.warmup_mm),
                "precut_warmup_mm": self.warmup_mm,
                "precut_edge": int(self.edge.index),
                "precut_edge_length_mm": float(self.edge.length),
                "precut_reach_mm": float(self.edge.reach_mm)}

    def summary(self) -> str:
        s1 = self.s0_mm + self.length_mm
        a, b = self.start_xy_mm, self.end_xy_mm
        return (f"precut   edge {self.edge.index} ({self.edge.length:g} mm): stock "
                f"already milled up to s0 = {self.s0_mm:g} mm, the run cuts "
                f"s = {self.s0_mm:g} .. {s1:g} mm and stops\n"
                f"         tool centre ({a[0]:.2f}, {a[1]:.2f}) -> "
                f"({b[0]:.2f}, {b[1]:.2f}) mm | disc reaches "
                f"{self.edge.reach_mm:.2f} mm ahead, far corner at "
                f"{self.edge.length:g} mm\n"
                f"         prediction stock precut to {self.s_cut_mm:g} mm "
                f"(= s0 - ds); engine carves to {self.s0_mm - self.warmup_mm:.2f} "
                f"mm and mills the last {self.warmup_mm:.2f} mm rigidly")


def build_job(cfg, s0_mm: float, length_mm: float, *, edge_index: int = None,
              ds_mm: float = 2.0) -> SteadyJob:
    """Precut stock, steady toolpath and placement for one operating point.

    Call after the operating point is set (`analysis.config.apply_operating_point`):
    ae sets the tool-centre line, ap the lift, and the feed the toolpath.
    """
    if edge_index is not None:
        cfg = replace(cfg, path=replace(cfg.path, start_edge=int(edge_index)))
    edge = edge_frame(cfg)
    s0, L = float(s0_mm), float(length_mm)

    if L <= 0.0:
        raise ValueError(f"length_mm must be positive, got {L:g}")
    if s0 < ds_mm:
        raise ValueError(
            f"s0 = {s0:g} mm leaves no precut in front of the corner: the "
            f"prediction's stock is milled to s0 - ds, so s0 must be >= ds = "
            f"{ds_mm:g} mm")
    s_max = edge.length - edge.reach_mm
    if s0 + L > s_max + 1e-9:
        raise ValueError(
            f"s0 + length = {s0 + L:g} mm runs into the exit: the disc touches "
            f"material {edge.reach_mm:.2f} mm ahead of its centre, so on a "
            f"{edge.length:g} mm edge the steady cut ends at s = {s_max:.2f} mm. "
            f"Shorten --length or lower --s0.")

    name = write_toolpath(cfg, edge, s0, L)
    cfg = replace(cfg, path=replace(cfg.path, toolpath=name))
    cfg = place_tcp_at(cfg, edge.point(s0))

    s_cut = s0 - float(ds_mm)
    part_pred = precut_part(cfg.part.build(), edge, s_cut)
    return SteadyJob(cfg=cfg, edge=edge, part_pred=part_pred, s0_mm=s0,
                     length_mm=L, s_cut_mm=s_cut, ds_mm=float(ds_mm))


def prepare(job: SteadyJob, *, erode=True, h_mm=0.01, max_gap_mm=4.0,
            fill_k_gaps=False, verbose=False) -> Setup:
    """`analysis.stability.prepare`, measured on the precut stock.

    The same two Shapely sweeps on the same node spacing; the only difference is
    the outline they start from, which is why this is a copy rather than an
    option on the original.
    """
    mill, _virgin, scene, waypoints, path = job.cfg.build()
    part = job.part_pred
    xy = path.xy_mm

    s_all = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    step = max(1, int(round(job.ds_mm / max(np.median(np.diff(s_all)), 1e-9))))
    xy_s, s_s = xy[::step], s_all[::step]

    engagement = engagement_angles_along_path(
        xy_s, part, mill.radius_mm, erode=erode, s_mm=s_s, verbose=verbose)
    gradients = engagement_gradients_along_path(
        xy_s, part, mill.radius_mm, h_mm=h_mm, erode=erode, s_mm=s_s,
        nominal=engagement, fill_gaps=fill_k_gaps, max_gap_mm=max_gap_mm,
        verbose=verbose)
    return Setup(cfg=job.cfg, mill=mill, part=part, scene=scene, path=path,
                 waypoints=waypoints, engagement=engagement,
                 gradients=gradients, ds_mm=job.ds_mm)
