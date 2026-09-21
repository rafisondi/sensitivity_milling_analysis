"""Move or turn the workpiece under a fixed robot pose.

Three operations on a `RunConfig`, each returning a new one:

    place_at_edge_midpoint(cfg)          anchor the CURRENT start pose so the
                                         TCP sits on the middle of the edge
                                         about to be cut, at full engagement
                                         (ae) and depth (ap)
    move_workpiece(cfg, d_w_mm=...)      slide the stock under the tool
    attack_angle(cfg, angle_deg)         turn the stock about the point under
                                         the tool, so the same material point
                                         is cut from a different direction

All three leave `cfg.start` untouched, and all three leave the robot's actual
start JOINT configuration exactly where it was:

  - `move_workpiece` only ever changes `scene.origin_i`, never `scene.R_iw`,
    and `StartPose.theta` reads the workpiece placement at all only through
    `scene.R_i_tcp` (when `start.rpy_deg is None`) — pure translation cannot
    touch that, so there is nothing to compensate.

  - `attack_angle` pivots the stock, about the workpiece's own Z axis by
    default, through the midpoint of the edge being cut — a property of the
    PART and the operating point, not of wherever the TCP happens to be —
    and re-clocks `R_w_tcp` to cancel the turn's effect on `scene.R_i_tcp`:
    the same trick `sweep_rotation_fixedpose.py` uses to hold a coupled
    pass's receptance fixed while the workpiece turns, generalised from a
    turn about the workpiece origin to a turn about an arbitrary pivot.

Call `place_at_edge_midpoint` first; `move_workpiece` and `attack_angle` are
meant to perturb that placement, not to build one from scratch (an unplaced
scene has no fixed contact point to turn about).
"""

from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from robotsim import kinematics


def edge_offset_midpoint_mm(part, offset_mm: float, edge_index: int) -> np.ndarray:
    """(2,) midpoint [mm] of edge `edge_index` on the tool-centre offset ring.

    `offset_mm` is `MillConfig.tool_offset_mm` (`R - ae`) — the ring the tool
    CENTRE runs on for a cut at radial engagement `ae`, not the raw part
    boundary. Edge `i` of the offset ring still runs between offset vertices
    `i` and `i + 1`, same indexing as `part.edge(i)` (`fastsim.geometry`).
    """
    ring = part.offset_outline(offset_mm)
    n = part.n_edges
    i = int(edge_index) % n
    return 0.5 * (ring[:, i] + ring[:, (i + 1) % n])


def edge_pivot_mm(cfg, edge_index: int = None) -> np.ndarray:
    """(3,) midpoint of the edge to cut, at full depth [mm], workpiece frame.

    xy from the tool-centre offset ring (`edge_offset_midpoint_mm`); z = -ap,
    the depth `place_at_edge_midpoint` puts the tool at over that point. A
    pure Z-axis turn (`attack_angle`'s default `axis="z"`) does not actually
    care about z — only the xy of the pivot matters — but this is the
    physically meaningful point (the one actually being cut), so it is the
    one reported and pivoted about.

    This is a property of the PART, `ae` and `ap` alone — NOT of wherever the
    workpiece has since been moved to. Use this (the default), not
    `contact_point_w_mm`, to pivot `attack_angle` about the feature being
    cut rather than about wherever a start pose happens to have drifted to
    after a `move_workpiece` translation.
    """
    edge_index = cfg.path.start_edge if edge_index is None else int(edge_index)
    mill = cfg.milling()
    part = cfg.part.build()
    xy = edge_offset_midpoint_mm(part, mill.tool_offset_mm, edge_index)
    return np.array([xy[0], xy[1], -part.height_mm])


def contact_point_w_mm(cfg) -> np.ndarray:
    """(3,) workpiece-frame point [mm] currently under the TCP at `cfg.start`.

    A DIAGNOSTIC, not a pivot: reports whatever material point happens to
    coincide with the anchor pose right now, which drifts with every
    `move_workpiece` call (it is not fixed to any part feature) — right after
    `place_at_edge_midpoint` it equals `edge_pivot_mm`, but nothing keeps it
    there. Uses `cfg.start.anchor_pose` — the same pose `place_at_edge_midpoint`
    (and `RunConfig.build()`, for an unplaced scene) anchors the job to — not
    `cfg.start.theta()/.pose()`, which solve IK for a fixed `pos_m` and can
    land on a different pose entirely once `rpy_deg is None` (their target
    orientation is `scene.R_i_tcp`, which a placement is free to change).
    """
    robot = kinematics.load_robot(cfg.scene)
    T_tcp = cfg.start.anchor_pose(robot, cfg.scene.ee_frame)
    return cfg.scene.point_w(T_tcp[:3, 3])


def place_at_edge_midpoint(cfg, edge_index: int = None):
    """Anchor `cfg.start` so the TCP sits on the middle of the edge to cut.

    `edge_index` defaults to `cfg.path.start_edge` — the edge the planned
    contour already opens on — and, when given explicitly, is written BACK
    into `path.start_edge` too, so the planned contour always opens on the
    same edge the placement anchored to (passing `edge_index=2` used to leave
    the toolpath still starting on edge 0 — a latent mismatch between where
    the job is anchored and what it actually cuts). The stock is lifted by
    `ap` (`cfg.part.height_mm`) exactly as the default (pose-derived)
    placement does, so the tool sits at full depth over the middle of the
    engaged edge rather than scratching the surface — read `ae` and `ap` off
    `cfg.mill` / `cfg.part` beforehand (`analysis.config.apply_operating_point`
    is the usual way to set them).

    The robot itself does not move: this only changes WHERE THE JOB SITS
    relative to the pose already named by `cfg.start`. The resulting scene is
    `placed_by_hand` — pinned — so it stops following `cfg.start` and is ready
    for `move_workpiece` / `attack_angle` to perturb.
    """
    edge_index = cfg.path.start_edge if edge_index is None else int(edge_index)
    mill = cfg.milling()
    part = cfg.part.build()
    mid_xy_mm = edge_offset_midpoint_mm(part, mill.tool_offset_mm, edge_index)

    scene = cfg.scene.unplaced()
    robot = kinematics.load_robot(scene)
    anchor_pose = cfg.start.anchor_pose(robot, scene.ee_frame)
    placed = scene.placed_at_ee(anchor_pose, first_point_mm=mid_xy_mm,
                                lift_mm=part.height_mm)
    return replace(cfg, scene=placed, path=replace(cfg.path, start_edge=edge_index))


def move_workpiece(cfg, *, d_w_mm=None, d_i_mm=None):
    """Slide the stock under the tool. Robot pose: untouched, exactly.

    `d_w_mm` moves it along its OWN axes [mm] (e.g. along the edge, or into
    the material); `d_i_mm` moves it along the base/cell axes [mm]. Either or
    both. This is rigid — the part geometry and the cut plan are unchanged,
    only where the tool meets them — so where the edge midpoint used to sit
    under the TCP, it generally no longer does; that is the point of the move.
    """
    d_w = None if d_w_mm is None else np.asarray(d_w_mm, dtype=float) * 1e-3
    d_i = None if d_i_mm is None else np.asarray(d_i_mm, dtype=float) * 1e-3
    return replace(cfg, scene=cfg.scene.translated(d_i=d_i, d_w=d_w))


def attack_angle(cfg, angle_deg: float, *, axis: str = "z", pivot_w_mm=None,
                 edge_index: int = None):
    """Turn the stock about the edge midpoint. Robot pose: untouched, exactly.

    The pivot is `edge_pivot_mm(cfg, edge_index)` by default — the middle of
    the edge being cut, on the workpiece's own Z axis — NOT wherever the TCP
    happens to be right now: that point is fixed to the PART, so it stays put
    across `move_workpiece` calls, giving every `attack_angle` its own
    "approach the same feature from a different direction" reading regardless
    of where the stock has since been slid to. Pass `pivot_w_mm` [mm] to
    override it outright. `axis` is the workpiece axis to turn about ('z' for
    the usual case; the tool's spindle axis IS workpiece Z for the default
    `R_w_tcp`, so a 'z' turn is a wrist re-clocking, not a tilt).

    Requires an already-placed scene (`place_at_edge_midpoint`, or any other
    `placed_by_hand` scene) — an unplaced one has no fixed placement to pivot.
    """
    scene = cfg.scene
    if not scene.placed_by_hand:
        raise ValueError(
            "attack_angle needs a placed scene to pivot — call "
            "place_at_edge_midpoint(cfg) (or otherwise place the workpiece) first")
    if pivot_w_mm is None:
        pivot_w_mm = edge_pivot_mm(cfg, edge_index)

    R_i_tcp_ref = scene.R_i_tcp
    R_extra = Rotation.from_euler(axis, angle_deg, degrees=True).as_matrix()
    turned = scene.rotated(R_extra, pivot_w=np.asarray(pivot_w_mm, dtype=float) * 1e-3)
    # Re-clock the tool-in-workpiece so R_i_tcp is exactly what it was before
    # the turn — see the module docstring for why this is what keeps the
    # robot's start pose (position AND orientation) angle-invariant.
    fixed = replace(turned, R_w_tcp=turned.R_iw.T @ R_i_tcp_ref)
    return replace(cfg, scene=fixed)
