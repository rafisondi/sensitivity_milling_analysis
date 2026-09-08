"""Build the milling process, optionally already mid-cut at t = 0.

A virgin stock with the tool sitting inside it is not a physical state — the
tool would have had to mill its way there, and starting from it produces a large
artificial transient. `prepare_steady_state_cut` fixes that in two steps:
pre-carve the swept slot, then mill a few revolutions rigidly up to the start so
the surface and the chip-lookback history are those of an ongoing cut.

Skip it when the path has a lead-in through air: cutting in is then a real
transient and belongs in the result.
"""

import numpy as np

from fastsim.config import MillConfig
from fastsim.geometry import Workpiece
from fastsim.process import MillingProcess
from fastsim.raster import WorkpieceRaster


def build_process(cfg: MillConfig, part: Workpiece) -> MillingProcess:
    """The dexel process over the part outline — virgin, no cut history.

    The extrusion height IS the axial depth: the stacked-2-D raster has no other
    notion of depth. `omega_rad_s` is signed (the engine reads it through abs()
    for the tooth-period lookback); the angle passed to step() stays signed too.

    `process.step(xy_mm, angle_rad)` advances by `cfg.sim_dt` and returns the
    force in the workpiece frame [N]. Stateful — call once per step, in order.
    """
    raster = WorkpieceRaster(part.boundary_xy_mm, resolution=cfg.raster_mm)
    return MillingProcess(
        workpiece=raster, n_teeth=cfg.n_teeth, diameter_mm=cfg.diameter_mm,
        helix_angle_deg=cfg.helix_angle_deg, axial_depth_mm=part.height_mm,
        Ktc=cfg.Ktc, Krc=cfg.Krc, Kac=cfg.Kac, spindle_spin=cfg.spindle_spin,
        omega_rad_s=cfg.omega_rad_s, dt=cfg.sim_dt,
        raster_resolution=cfg.raster_mm, max_slice_angle_deg=cfg.slice_angle_deg,
        chip_mode=cfg.chip_mode, n_phi=cfg.n_phi, max_chip_mm=cfg.max_chip_mm,
        moment_about=cfg.moment_about, parallel=cfg.parallel)


def _feed_dir(feed_dir_xy) -> np.ndarray:
    u = np.asarray(feed_dir_xy, dtype=float).ravel()[:2]
    norm = np.linalg.norm(u)
    if norm < 1e-12:
        raise ValueError("feed_dir_xy must be a non-zero direction")
    return u / norm


def carve_entry_slot(process, cfg: MillConfig, start_xy_mm,
                     feed_dir_xy=(0.0, 1.0)) -> None:
    """Clear the slot the tool would already have milled: a 2R-wide channel
    trailing behind the start point plus the leading semicircle, every slice."""
    R = cfg.radius_mm
    p = np.asarray(start_xy_mm, dtype=float).ravel()[:2]
    u = _feed_dir(feed_dir_xy)
    n = np.array([-u[1], u[0]])

    res = process._raster_res
    _, NY, NX = process._grids.shape
    xs = process._raster_ox + np.arange(NX) * res
    ys = process._raster_oy + np.arange(NY) * res
    XX, YY = np.meshgrid(xs, ys)

    along = (XX - p[0]) * u[0] + (YY - p[1]) * u[1]       # + = ahead of the tool
    across = (XX - p[0]) * n[0] + (YY - p[1]) * n[1]
    process._grids[:, ((along <= 0.0) & (np.abs(across) <= R))
                   | (along ** 2 + across ** 2 <= R ** 2)] = 0


def warmup_process(process, cfg: MillConfig, start_xy_mm,
                   feed_dir_xy=(0.0, 1.0)) -> None:
    """Mill rigidly from `warmup_revs` revolutions behind the start point up to
    it, at t < 0, with the spindle angle running continuously into t = 0."""
    p = np.asarray(start_xy_mm, dtype=float).ravel()[:2]
    u = _feed_dir(feed_dir_xy)
    n_pre = int(round(cfg.warmup_revs * 60.0 / cfg.spindle_rpm / cfg.sim_dt))
    for i in range(n_pre):
        t_neg = (i - n_pre) * cfg.sim_dt
        process.step(p + u * cfg.feed_mm_s * t_neg, cfg.omega_rad_s * t_neg)


def prepare_steady_state_cut(cfg: MillConfig, part: Workpiece, start_xy_mm,
                             feed_dir_xy=(0.0, 1.0),
                             verbose: bool = True) -> MillingProcess:
    """Virgin -> carved -> warmed up, so the run opens mid-cut."""
    process = build_process(cfg, part)
    p = np.asarray(start_xy_mm, dtype=float).ravel()[:2]
    u = _feed_dir(feed_dir_xy)
    d_warm = cfg.warmup_distance_mm

    carve_entry_slot(process, cfg, p - u * d_warm, u)
    warmup_process(process, cfg, p, u)

    if verbose:
        entry = p - u * d_warm
        print(f"prepare  carved to ({entry[0]:.2f}, {entry[1]:.2f}) mm, then "
              f"{cfg.warmup_revs} rigid revs ({d_warm:.2f} mm) up to "
              f"({p[0]:.2f}, {p[1]:.2f}) mm")
    return process
