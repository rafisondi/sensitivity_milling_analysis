"""
fastsim/metrics.py — path deviation measured the way the surface sees it
==========================================================================
    dev = deviation_from_path(driven_xy, nominal_xy, workpiece=wp)
    dev.signed_mm      # + tool sat outside the material (undercut)
    dev.lag_mm         # how far BEHIND the nominal point it was, along the path

Why not just subtract
---------------------
Comparing `driven(t)` against `nominal(t)` sample by sample charges the whole
difference as error, but two quite different things live in it:

    * a deviation ACROSS the path — the tool is nearer or further from the face
      than intended. This is the surface error; it is what the part records.
    * a deviation ALONG the path — the tool is at the right distance from the
      face but has not got as far as the plan said it would by now. A constant
      feed error, or a feed curve the robot cannot keep up with, produces this.
      It leaves the same surface, just later.

A run that tracks the face perfectly while running 10% slow has zero surface
error and a large same-time difference. So the deviation is measured against the
nominal path as a CURVE: drop a perpendicular from each driven point onto it and
take that distance. The along-path part is reported separately as `lag_mm`
rather than being folded into the error.
"""

from dataclasses import dataclass

import numpy as np

from fastsim.geometry import Workpiece


# ─────────────────────────────────────────────────────────────────────────────
# Orthogonal projection onto a polyline
# ─────────────────────────────────────────────────────────────────────────────

def polyline_arclength(polyline_xy: np.ndarray) -> np.ndarray:
    """Cumulative arc length at each vertex of an (M, 2) polyline [mm]."""
    seg = np.linalg.norm(np.diff(polyline_xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def project_onto_polyline(query_xy: np.ndarray, polyline_xy: np.ndarray,
                          search: int = 3) -> tuple:
    """Orthogonal projection of every query point onto a polyline.

    Returns (foot_xy (N, 2), distance (N,), s_foot (N,), tangent (N, 2)) — the
    perpendicular foot, its distance, its arc length along the polyline, and the
    unit tangent there.

    A KD-tree finds the nearest VERTEX, then the true foot is taken as the best
    of the `search` segments either side of it. The vertex alone is not enough:
    on a coarse polyline the real foot can lie mid-segment, which is exactly the
    sub-millimetre regime this is used to measure.
    """
    from scipy.spatial import cKDTree

    query_xy = np.atleast_2d(np.asarray(query_xy, dtype=float))
    poly = np.asarray(polyline_xy, dtype=float)
    if len(poly) < 2:
        raise ValueError("the polyline needs at least two vertices")

    s_vertex = polyline_arclength(poly)
    n_seg = len(poly) - 1
    nearest = cKDTree(poly).query(query_xy)[1]

    best_d = np.full(len(query_xy), np.inf)
    best_foot = np.zeros_like(query_xy)
    best_s = np.zeros(len(query_xy))
    best_tan = np.zeros_like(query_xy)

    for offset in range(-search, search + 1):
        i = np.clip(nearest + offset, 0, n_seg - 1)
        a, b = poly[i], poly[i + 1]
        ab = b - a
        ab_len2 = np.einsum("ij,ij->i", ab, ab)
        ab_len2 = np.where(ab_len2 > 0.0, ab_len2, 1.0)

        u = np.clip(np.einsum("ij,ij->i", query_xy - a, ab) / ab_len2, 0.0, 1.0)
        foot = a + u[:, None] * ab
        d = np.linalg.norm(query_xy - foot, axis=1)

        take = d < best_d
        best_d = np.where(take, d, best_d)
        best_foot[take] = foot[take]
        best_s = np.where(take, s_vertex[i] + u * np.sqrt(ab_len2), best_s)
        best_tan[take] = (ab / np.sqrt(ab_len2)[:, None])[take]

    return best_foot, best_d, best_s, best_tan


# ─────────────────────────────────────────────────────────────────────────────
# Deviation from a reference path
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, eq=False)
class PathDeviation:
    """Where a driven path sat relative to a reference path. Millimetres.

    foot_xy      (N, 2)  the perpendicular foot on the reference path
    distance_mm  (N,)    unsigned perpendicular distance to it
    signed_mm    (N,)    signed across-path deviation: + = away from the
                         material (undercut), - = into it (overcut)
    s_foot_mm    (N,)    arc length of the foot along the reference path
    lag_mm       (N,)    along-path shortfall: how far BEHIND the point the plan
                         called for at that instant the tool actually was.
                         Positive = behind. None when no timing was supplied.
    """

    foot_xy: np.ndarray
    distance_mm: np.ndarray
    signed_mm: np.ndarray
    s_foot_mm: np.ndarray
    lag_mm: np.ndarray = None

    def summary(self, mask: np.ndarray = None, label: str = "deviation") -> str:
        sel = slice(None) if mask is None else mask
        d, sg = self.distance_mm[sel], self.signed_mm[sel]
        if not len(d):
            return f"  {label}: no samples"
        out = [f"  {label} [mm], measured orthogonally:",
               f"    across-path |d|   mean {d.mean():6.3f}  max {d.max():6.3f}"
               f"   RMS {np.sqrt((d ** 2).mean()):6.3f}",
               f"    signed            mean {sg.mean():+6.3f}  "
               f"min {sg.min():+6.3f}  max {sg.max():+6.3f}"
               f"   ({100 * (sg > 0).mean():.0f}% undercut)"]
        if self.lag_mm is not None:
            lag = self.lag_mm[sel]
            out.append(f"    along-path lag    mean {lag.mean():+6.3f}  "
                       f"max {lag.max():+6.3f}   (excluded from the error above)")
        return "\n".join(out)


def _outward_normals(points_xy: np.ndarray, workpiece: Workpiece) -> np.ndarray:
    """Unit normals pointing from the material into free space at each point."""
    from matplotlib.path import Path as MplPath

    ring = workpiece.closed_xy_mm.T
    foot, _, _, _ = project_onto_polyline(points_xy, ring)
    u = foot - points_xy
    norm = np.linalg.norm(u, axis=1, keepdims=True)
    u = np.divide(u, norm, out=np.zeros_like(u), where=norm > 0)

    inside = MplPath(ring).contains_points(points_xy)
    # Inside the material, the way out is towards the boundary; outside it is
    # the other way.
    return np.where(inside[:, None], u, -u)


def deviation_from_path(driven_xy: np.ndarray, reference_xy: np.ndarray,
                        workpiece: Workpiece = None,
                        reference_at_time_xy: np.ndarray = None) -> PathDeviation:
    """Perpendicular deviation of `driven_xy` from the `reference_xy` curve.

    workpiece             signs the deviation by the outward surface normal, so
                          + means undercut. Without it the sign falls back to
                          the reference path's left-hand normal, which is
                          consistent but not tied to which side the material is.
    reference_at_time_xy  the reference sampled at the SAME instants as
                          `driven_xy`. Supply it to also get `lag_mm`, the
                          along-path shortfall — the part of the raw difference
                          that is timing rather than surface error.
    """
    driven_xy = np.atleast_2d(np.asarray(driven_xy, dtype=float))
    reference_xy = np.asarray(reference_xy, dtype=float)

    foot, dist, s_foot, tangent = project_onto_polyline(driven_xy, reference_xy)
    offset = driven_xy - foot

    if workpiece is not None:
        normal = _outward_normals(foot, workpiece)
    else:
        normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    signed = np.einsum("ij,ij->i", offset, normal)
    # `dist` is unsigned; keep the magnitude and take only the sense from above.
    signed = np.sign(signed) * dist

    lag = None
    if reference_at_time_xy is not None:
        _, _, s_want, _ = project_onto_polyline(
            np.atleast_2d(np.asarray(reference_at_time_xy, dtype=float)),
            reference_xy)
        lag = s_want - s_foot          # + = the tool has not got that far yet

    return PathDeviation(foot_xy=foot, distance_mm=dist, signed_mm=signed,
                         s_foot_mm=s_foot, lag_mm=lag)
