"""Entry/exit angles from the actual cut geometry.

    eng  = engagement_angles_along_path(xy_mm, part, radius_mm=10.0)
    grad = engagement_gradients_along_path(xy_mm, part, radius_mm=10.0, nominal=eng)
    eng.phi_en, eng.phi_ex        # (N,) rad, per path point
    grad.dphi                     # (N, 2, 2) rad/mm — what K_cut needs

Recovered from `millsim/engagement.py`, with `fastsim.geometry.Workpiece` in
place of the old outline class. The only geometry this repo has that is not the
dexel raster: the engine returns chip thickness and a force, never an engagement
ANGLE, and the ZOA terms are written in those angles.

WHY NOT THE CLOSED FORM. The textbook down-milling pair

    phi_en = pi - arccos(1 - ae/R),   phi_ex = pi

assumes a straight wall cut on one side, with the tool never breaking through
and nothing removed beforehand. Along a real part path none of that holds: the
boundary curves, the tool breaks out and re-enters, and by the time it comes
round again the material it would have cut is already gone. Both angles have to
come from the geometry. (`stabsim.cut.entry_angle` is the closed form, kept only
as a hand check.)

HOW THE ANGLES ARE MEASURED

  1. a local frame at each path point — x along the feed tangent `t`, y along
     the LEFT normal `n = (-t_y, t_x)`; `R_wc = [t n]` as columns,
  2. the tool disc of radius R at the tool centre,
  3. `engagement = material ^ disc`; empty means the tool is in air,
  4. the two points where the material BOUNDARY crosses the disc boundary; the
     vectors to them, rotated into the local frame, give
     `phi = -(atan2(p_y, p_x) - pi/2)` — measured from the local +y axis,
  5. `phi_en = min`, `phi_ex = max`, clipped to [0, pi],
  6. the disc is then SUBTRACTED from the material, so the next point sees what
     is actually left. This is what makes a second pass over the same ground
     read as air instead of a full-immersion cut.

More than two crossings means a fragmented engagement (an island, or entry and
exit in the same disc). That is flagged `contact=True` with the angles left NaN
rather than guessed at — `fill_short_gaps` is the separate, deliberate decision
to interpolate the short ones.

Cost is one polygon boolean per point, so downsample the path first.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fastsim.geometry import Workpiece


@dataclass(frozen=True, eq=False)
class Engagement:
    """Per-point engagement geometry along a tool path. Millimetres, radians.

    s_mm     (N,)      arc length
    xy_mm    (N, 2)    tool-centre position, workpiece frame
    phi_en   (N,)      entry angle [rad]; NaN where undefined
    phi_ex   (N,)      exit angle  [rad]; NaN where undefined
    contact  (N,) bool the tool overlaps material at all
    R_wc     (N, 2, 2) local cut frame -> workpiece frame, columns [t n_left]
    """

    s_mm: np.ndarray
    xy_mm: np.ndarray
    phi_en: np.ndarray
    phi_ex: np.ndarray
    contact: np.ndarray
    R_wc: np.ndarray
    radius_mm: float = 10.0
    eroded: bool = True

    def __len__(self) -> int:
        return len(self.s_mm)

    @property
    def valid(self) -> np.ndarray:
        """Points with a usable pair of angles."""
        return self.contact & np.isfinite(self.phi_en) & np.isfinite(self.phi_ex)

    @property
    def immersion_rad(self) -> np.ndarray:
        """Swept engagement arc `phi_ex - phi_en` [rad]; NaN where undefined."""
        return self.phi_ex - self.phi_en

    @property
    def radial_depth_mm(self) -> np.ndarray:
        """Radial engagement implied by the arc, `ae = R (1 - cos(phi_ex-phi_en))`
        for a straight wall — a readback for comparison with the closed form."""
        return self.radius_mm * (1.0 - np.cos(self.immersion_rad))

    def summary(self) -> str:
        n_c, n_v = int(self.contact.sum()), int(self.valid.sum())
        if not n_v:
            return f"  {len(self)} points, {n_c} in contact, none with valid angles"
        en = np.degrees(self.phi_en[self.valid])
        ex = np.degrees(self.phi_ex[self.valid])
        return (f"  {len(self)} points | {n_c} in contact | {n_v} with angles "
                f"({n_c - n_v} fragmented)\n"
                f"  phi_en  {en.min():6.1f} .. {en.max():6.1f} deg "
                f"(mean {en.mean():6.1f})\n"
                f"  phi_ex  {ex.min():6.1f} .. {ex.max():6.1f} deg "
                f"(mean {ex.mean():6.1f})\n"
                f"  swept   {np.degrees(self.immersion_rad[self.valid]).mean():6.1f} "
                f"deg mean ({'eroding' if self.eroded else 'static'} material)")

    def save_npz(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, s_mm=self.s_mm, xy_mm=self.xy_mm, phi_en=self.phi_en,
            phi_ex=self.phi_ex, contact=self.contact, R_wc=self.R_wc,
            radius_mm=self.radius_mm, eroded=self.eroded)
        return path

    @classmethod
    def load_npz(cls, path) -> "Engagement":
        with np.load(path) as d:
            return cls(s_mm=d["s_mm"], xy_mm=d["xy_mm"], phi_en=d["phi_en"],
                       phi_ex=d["phi_ex"], contact=d["contact"], R_wc=d["R_wc"],
                       radius_mm=float(d["radius_mm"]), eroded=bool(d["eroded"]))


def arclength_mm(xy_mm: np.ndarray) -> np.ndarray:
    """(N,) cumulative arc length of a tool-centre path [mm]."""
    xy_mm = np.atleast_2d(np.asarray(xy_mm, dtype=float))
    seg = np.linalg.norm(np.diff(xy_mm, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _local_frames(xy_mm: np.ndarray) -> np.ndarray:
    """(N, 2, 2) frames with columns [tangent, left normal], from the path."""
    d = np.gradient(xy_mm, axis=0)
    t = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    n = np.column_stack([-t[:, 1], t[:, 0]])
    R = np.zeros((len(xy_mm), 2, 2))
    R[:, :, 0] = t
    R[:, :, 1] = n
    return R


def engagement_angles_along_path(xy_mm: np.ndarray, workpiece, radius_mm: float,
                                 erode: bool = True, s_mm: np.ndarray = None,
                                 buffer_resolution: int = 128,
                                 verbose: bool = False) -> Engagement:
    """Entry/exit angles at every point of a tool-centre path.

    xy_mm       (N, 2) tool-centre positions, WORKPIECE frame [mm]
    workpiece   a `fastsim.geometry.Workpiece`, a shapely Polygon, or a ring
    erode       subtract the swept disc as the tool advances, so later points
                see the material that is actually left. Switch off to measure
                every point against the virgin outline.
    """
    from shapely.geometry import Point, Polygon

    xy_mm = np.atleast_2d(np.asarray(xy_mm, dtype=float))
    if s_mm is None:
        s_mm = arclength_mm(xy_mm)

    if isinstance(workpiece, Workpiece):
        material = Polygon(workpiece.boundary_xy_mm.T)
    elif isinstance(workpiece, Polygon):
        material = workpiece
    else:
        ring = np.asarray(workpiece, dtype=float)
        if ring.ndim == 2 and ring.shape[0] == 2 and ring.shape[1] != 2:
            ring = ring.T                     # (2, N) like Workpiece.boundary_xy_mm
        material = Polygon(ring)
    if not material.is_valid:
        material = material.buffer(0.0)

    R_wc = _local_frames(xy_mm)
    n = len(xy_mm)
    phi_en = np.full(n, np.nan)
    phi_ex = np.full(n, np.nan)
    contact = np.zeros(n, dtype=bool)
    n_fragmented = 0

    for i, (cx, cy) in enumerate(xy_mm):
        if material.is_empty:
            break
        if material.geom_type == "MultiPolygon":
            material = max(material.geoms, key=lambda p: p.area)

        disc = Point(cx, cy).buffer(radius_mm, resolution=buffer_resolution)
        if material.intersection(disc).is_empty:
            continue                                   # in air
        contact[i] = True

        crossings = material.exterior.intersection(disc.boundary)
        pts = [g for g in (crossings.geoms if hasattr(crossings, "geoms")
                           else [crossings]) if g.geom_type == "Point"]

        if len(pts) == 2:
            centre = np.array([cx, cy])
            local = [R_wc[i].T @ (np.array([p.x, p.y]) - centre) for p in pts]
            ang = [-(np.arctan2(p[1], p[0]) - 0.5 * np.pi) for p in local]
            phi_en[i] = np.clip(min(ang), 0.0, np.pi)
            phi_ex[i] = np.clip(max(ang), 0.0, np.pi)
        else:
            # Entry and exit inside one disc, or an island: not a single arc.
            n_fragmented += 1

        if erode:
            material = material.difference(disc)

    if verbose and n_fragmented:
        print(f"         {n_fragmented} points had a fragmented engagement "
              f"(not 2 boundary crossings) - angles left NaN")

    return Engagement(s_mm=np.asarray(s_mm, dtype=float), xy_mm=xy_mm,
                      phi_en=phi_en, phi_ex=phi_ex, contact=contact,
                      R_wc=R_wc, radius_mm=float(radius_mm), eroded=bool(erode))


# ─────────────────────────────────────────────────────────────────────────────
# Gradients of the angles with respect to the tool-centre position
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, eq=False)
class EngagementGradients:
    """d[phi_en, phi_ex] / d[t, n] at every path point. Millimetres, radians.

    s_mm   (N,)        arc length
    dphi   (N, 2, 2)   rows [phi_en, phi_ex], columns [d/dt, d/dn] [rad/mm];
                       zero where `valid` is False
    valid  (N,) bool   the nominal cut AND both offset cuts gave angles here
    h_mm   float       the lateral offset the normal derivative was taken with
    interpolated (N,)  points whose gradient uses at least one interpolated
                       angle from the nominal or offset engagement sweeps
    """

    s_mm: np.ndarray
    dphi: np.ndarray
    valid: np.ndarray
    h_mm: float = 0.01
    interpolated: np.ndarray = None

    def __len__(self) -> int:
        return len(self.s_mm)

    def summary(self) -> str:
        n_v = int(self.valid.sum())
        if not n_v:
            return f"  {len(self)} points, no usable angle gradients"
        d = np.degrees(self.dphi[self.valid])
        n_i = (0 if self.interpolated is None else
               int((self.valid & self.interpolated).sum()))
        interpolation = f" | {n_i} using filled angle gaps" if n_i else ""
        return (f"  {len(self)} points | {n_v} with gradients{interpolation} "
                f"(h = {self.h_mm:g} mm)\n"
                f"  dphi_en/dn  {d[:, 0, 1].min():8.2f} .. {d[:, 0, 1].max():8.2f} deg/mm\n"
                f"  dphi_ex/dn  {d[:, 1, 1].min():8.2f} .. {d[:, 1, 1].max():8.2f} deg/mm\n"
                f"  dphi_en/dt  {d[:, 0, 0].min():8.2f} .. {d[:, 0, 0].max():8.2f} deg/mm")


def engagement_gradients_along_path(xy_mm: np.ndarray, workpiece,
                                    radius_mm: float, h_mm: float = 0.01,
                                    erode: bool = True, s_mm: np.ndarray = None,
                                    nominal: Engagement = None,
                                    buffer_resolution: int = 128,
                                    fill_gaps: bool = False,
                                    max_gap_mm: float = 4.0,
                                    verbose: bool = False) -> EngagementGradients:
    """d[phi_en, phi_ex]/d[t, n] by central differences on offset paths.

    The NORMAL derivative re-runs the whole sweep on the path shifted +-h_mm
    along the left normal, each offset run eroding its OWN copy of the stock —
    tool and cut geometry move together. That is deliberate: it is the zero-order
    term of `x(t - T) ~ x(t) - T xd`, i.e. the sensitivity of the tooth-period
    AVERAGE force, valid while the tooth period is short against the vibration.
    The first-order remainder is process damping and lives in `cut.A_cut_0`, so
    the two terms partition the cut rather than overlap. (Holding the material
    fixed instead would be the static, zero-rpm limit and would double-count the
    regeneration `A_cut_0` already carries.)

    The TANGENTIAL derivative is d/ds of the nominal angles: offsetting along
    the path is the same as advancing along it, so it needs no extra sweep.

    Cost is two extra engagement sweeps. `nominal` is reused when given, and
    must have been computed on these same points with the same `erode`.

    When `fill_gaps` is true, short internal runs of unmeasurable angles in all
    three sweeps are interpolated BEFORE differentiation. This prevents one NaN
    from invalidating its neighbouring tangential derivatives and permits a
    nonzero K_cut there. Contact remains mandatory, so this never bridges air.
    """
    xy_mm = np.atleast_2d(np.asarray(xy_mm, dtype=float))
    if s_mm is None:
        s_mm = arclength_mm(xy_mm)
    s_mm = np.asarray(s_mm, dtype=float)

    if nominal is None:
        nominal = engagement_angles_along_path(
            xy_mm, workpiece, radius_mm, erode=erode, s_mm=s_mm,
            buffer_resolution=buffer_resolution)
    elif len(nominal) != len(xy_mm):
        raise ValueError(
            f"nominal engagement has {len(nominal)} points but the path has "
            f"{len(xy_mm)} — compute it on the same samples")

    n_hat = nominal.R_wc[:, :, 1]              # left normal, workpiece frame
    if verbose:
        print(f"         angle gradients: 2 offset sweeps at h = {h_mm:g} mm")

    sweep = lambda sign: engagement_angles_along_path(
        xy_mm + sign * h_mm * n_hat, workpiece, radius_mm, erode=erode,
        s_mm=s_mm, buffer_resolution=buffer_resolution)
    plus, minus = sweep(+1.0), sweep(-1.0)

    def angles_for_gradient(engagement):
        if not fill_gaps:
            return engagement.phi_en, engagement.phi_ex, engagement.valid, \
                np.zeros(len(engagement), dtype=bool)
        en, ok_en = fill_short_gaps(
            s_mm, engagement.phi_en, engagement.valid, max_gap_mm)
        ex, ok_ex = fill_short_gaps(
            s_mm, engagement.phi_ex, engagement.valid, max_gap_mm)
        usable = engagement.contact & ok_en & ok_ex
        filled = usable & ~engagement.valid
        return en, ex, usable, filled

    nom_en, nom_ex, nom_ok, nom_filled = angles_for_gradient(nominal)
    plus_en, plus_ex, plus_ok, plus_filled = angles_for_gradient(plus)
    minus_en, minus_ex, minus_ok, minus_filled = angles_for_gradient(minus)

    dphi = np.zeros((len(xy_mm), 2, 2))
    dphi[:, 0, 1] = (plus_en - minus_en) / (2.0 * h_mm)
    dphi[:, 1, 1] = (plus_ex - minus_ex) / (2.0 * h_mm)
    dphi[:, 0, 0] = np.gradient(nom_en, s_mm)
    dphi[:, 1, 0] = np.gradient(nom_ex, s_mm)

    # np.gradient spreads a NaN to its neighbours, so a fragmented point costs
    # its two neighbours their tangential derivative. They drop out here.
    valid = (nom_ok & plus_ok & minus_ok
             & np.isfinite(dphi).all(axis=(1, 2)))
    dphi[~valid] = 0.0
    # A central tangential derivative at either neighbour also consumes a
    # filled nominal angle. Mark those nodes as interpolation-dependent too.
    tangent_filled = nom_filled.copy()
    tangent_filled[:-1] |= nom_filled[1:]
    tangent_filled[1:] |= nom_filled[:-1]
    interpolated = tangent_filled | plus_filled | minus_filled

    if verbose:
        print(f"         {int(valid.sum())}/{len(xy_mm)} points have usable gradients")
        if fill_gaps and (valid & interpolated).any():
            print(f"         {int((valid & interpolated).sum())} gradient point(s) "
                  "use interpolated short angle gaps")
    return EngagementGradients(s_mm=s_mm, dphi=dphi, valid=valid,
                               h_mm=float(h_mm), interpolated=interpolated)


def fill_short_gaps(s_mm, phi, valid, max_gap_mm: float = 4.0):
    """Interpolate angles across SHORT runs of unmeasurable nodes.

    A fragmented engagement (not two boundary crossings) means the sweep could
    not measure the angles, NOT that the tool left the material. Dropping the
    cut there would report a stretch of free flight in the middle of a cut, and
    the stability verdict would read as the bare arm exactly where the geometry
    is most awkward. Interpolating from the neighbours is far closer to the truth.

    Only short runs are filled: a long one means the tool really is in air, and
    bridging it would invent a cut that never happened.
    """
    out = np.asarray(phi, dtype=float).copy()
    if valid.sum() < 2:
        return out, valid
    idx = np.flatnonzero(valid)
    lo = np.clip(np.searchsorted(idx, np.arange(len(s_mm)), "right") - 1,
                 0, len(idx) - 1)
    hi = np.clip(lo + 1, 0, len(idx) - 1)
    span = s_mm[idx[hi]] - s_mm[idx[lo]]
    fill = (~valid & (span <= max_gap_mm) & (np.arange(len(s_mm)) > idx[0])
            & (np.arange(len(s_mm)) < idx[-1]))
    out[fill] = np.interp(s_mm, s_mm[valid], phi[valid])[fill]
    return out, valid | fill
