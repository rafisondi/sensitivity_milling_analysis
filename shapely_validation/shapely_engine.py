"""The polygon (Shapely) cutting engine, adapted to THIS workspace's config types.

    engine = build_shapely_process(mill, part, scene.T_iw)   virgin, no cut history
    f_w    = engine.step(tool_centre_xy_mm, spindle_angle_rad)

WHERE THIS COMES FROM

`mill_shaply/` (ported verbatim, see its own `__init__.py`) is the legacy
Shapely-based cutting engine from `MillingBenchmarkStability`, a sibling repo one
level up that runs the SAME robot-in-the-loop comparison this workspace does, but
with two interchangeable process backends: the dexel raster (`fastsim`, what this
workspace already has) and this one - exact polygon geometry per axial slice, no
raster quantisation, at roughly 60x the cost per step. `ShapelyMillingProcess`
below is that repo's own adapter (`millsim/milling/shapely_process.py`), copied
with one change: `build_shapely_process` there reads a `MillingSetup` this
workspace does not have, so it is rewritten here against `fastsim.config.MillConfig`
and `fastsim.geometry.Workpiece` - the same two types `fastsim.prepare.build_process`
already takes, so the two engines can be swapped in the same slot.

`mill_shaply/milling_data.py` was NOT ported - it is dead code even in the source
repo (its own `__init__.py` does not import it) and carries a stale
`from milling.milling_path import MillingPath` that predates the package rename.

THE INTERFACE MATCH, AND THE ONE GAP

Both engines expose `step(tool_centre_xy_mm, spindle_angle_rad) -> force_w [N]`,
stateful, one call per simulation step, workpiece frame - which is exactly what
`fastsim.coupled.ProcessAdapter` expects, so `ShapelyProcess` below needs no
special-casing anywhere the loop calls `.step()` or reads `.force_wp`. The one
gap is `.wrench_wp`: `ProcessAdapter` records a 6-D wrench for `moment_w`, and
this engine has no moment model - `ShapelyProcess.wrench_wp` zero-pads the force
into one rather than raising, so a run works with `ProcessAdapter(record=True)`
unchanged; `moment_w` on a shapely run is a declared zero, not a hole.

COST. ~60 ms/step against the dexel engine's ~1 ms/step (`MillingBenchmarkStability`
README), because every step re-slices real polygon geometry rather than reading a
raster. Budget accordingly - see `sweep_shapely.py`'s own docstring for the numbers
this workspace's grid actually costs.
"""

import numpy as np

from mill_shaply.eraser_of_matter import milling_workpiece

from fastsim.config import MillConfig
from fastsim.geometry import Workpiece


class ShapelyProcess:
    """Polygon (Shapely) milling process - drop-in for `fastsim.process.MillingProcess`.

    Stateful: call `step` once per time step, in order. All lengths in mm, forces
    in N, everything in the WORKPIECE frame - the same convention `MillingProcess`
    uses, so `ProcessAdapter` cannot tell the two apart.
    """

    def __init__(self, boundary_xy_mm: np.ndarray, *, n_teeth: int = 4,
                diameter_mm: float = 16.0, helix_angle_deg: float = 45.0,
                axial_depth_mm: float = 1.0, Ktc: float = 1930.4,
                Krc: float = 1159.6, Kac: float = 200.6, spindle_spin: int = -1,
                dt: float = None, max_slice_angle_deg: float = 10.0,
                T_iw: np.ndarray = None):
        xy = np.asarray(boundary_xy_mm, dtype=float)
        if xy.shape[0] != 2:
            xy = xy.T

        self.wp = milling_workpiece(
            xy_workpiece=xy, axial_cutting_depth=float(axial_depth_mm),
            T_base_workpiece=None if T_iw is None else np.asarray(T_iw, float),
            number_of_teeth=int(n_teeth), diameter_end_mill=float(diameter_mm),
            helix_angle_deg=float(helix_angle_deg),
            max_layer_rot_between_slices_deg=float(max_slice_angle_deg),
            Ktc=Ktc, Krc=Krc, Kac=Kac)

        self.spin = int(spindle_spin)
        self._dt = 0.0 if dt is None else float(dt)
        self.force_wp = np.zeros(3)
        self._n_steps = 0

    # ── the handful of attributes the run pipeline reports on ────────────────

    @property
    def n_slices(self) -> int:
        return self.wp.number_of_slices

    @property
    def n_teeth(self) -> int:
        return self.wp.number_of_teeth

    @property
    def slice_height(self) -> float:
        return self.wp.slice_height

    @property
    def radius(self) -> float:
        return self.wp.radius_tool

    # ── the interface `fastsim.coupled.ProcessAdapter` drives ────────────────

    def step(self, tool_center_xy, spindle_angle: float) -> np.ndarray:
        """Advance one timestep; return the cutting force (3,) [N], wp frame.

        The legacy `erase_step` needs three tool positions before it can build a
        swept area, so the first two calls remove nothing and return zero - the
        same warm-up the dexel model needs for its own tooth-period lookback.
        """
        center = np.asarray(tool_center_xy, dtype=float).ravel()[:2]
        self.wp.erase_step(center, float(spindle_angle),
                           direction=self.spin, t=self._n_steps * self._dt)
        self._n_steps += 1

        f = np.asarray(self.wp.total_milling_force, dtype=float).ravel()[:3].copy()
        self.force_wp = f
        return f.copy()

    @property
    def wrench_wp(self) -> np.ndarray:
        """(6,) - the force with ZERO moments; this engine has no moment model.

        Declared here rather than left absent so `ProcessAdapter(record=True)`
        (the pipeline's default) does not need a special case for which engine
        produced a run - `analysis.forces`/`report` never read `moment_w`, so
        this only affects code that explicitly asks for it.
        """
        return np.concatenate([self.force_wp, np.zeros(3)])

    # ── what the polygon model can show that the raster cannot ───────────────

    def remaining_area_mm2(self) -> float:
        """Material left in the bottom axial slice [mm^2]."""
        return float(self.wp.workpiece_slice[0].size())


def build_shapely_process(mill: MillConfig, part: Workpiece,
                          T_iw: np.ndarray) -> ShapelyProcess:
    """The shapely process over the part outline - virgin, no cut history.

    Mirrors `fastsim.prepare.build_process` field for field, so the two are
    interchangeable at the one call site that constructs the engine (see
    `sim_coupled_shapely.simulate`). `part.height_mm` IS the axial depth, same
    convention as the dexel side.
    """
    return ShapelyProcess(
        boundary_xy_mm=part.boundary_xy_mm, n_teeth=mill.n_teeth,
        diameter_mm=mill.diameter_mm, helix_angle_deg=mill.helix_angle_deg,
        axial_depth_mm=part.height_mm, Ktc=mill.Ktc, Krc=mill.Krc, Kac=mill.Kac,
        spindle_spin=mill.spindle_spin, dt=mill.sim_dt,
        max_slice_angle_deg=mill.slice_angle_deg, T_iw=T_iw)
