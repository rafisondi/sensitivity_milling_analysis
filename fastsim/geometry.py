"""The part: a closed 2-D outline extruded along +Z_w. Workpiece frame, mm.

    part = rectangle(100.0, 50.0, height_mm=5.0)
    part = load_workpiece("rectangular_workpiece", height_mm=10.0)

The ring is counter-clockwise and NOT closed, so edge k runs from vertex k to
k+1 and every edge normal points out of the material. `height_mm` is the
extrusion, which for a full-depth pass IS the axial depth ap.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WORKPIECE_DIR = Path(__file__).resolve().parents[1] / "data" / "workpieces"


# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, eq=False)
class Edge:
    """One boundary segment. `outward_normal` belongs to the face, so
    `flipped()` does not change it."""

    index: int
    p0: np.ndarray                # (2,) segment start [mm]
    p1: np.ndarray                # (2,) segment end   [mm]
    outward_normal: np.ndarray    # (2,) unit normal, out of the material

    @property
    def length_mm(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))

    @property
    def direction(self) -> np.ndarray:
        """Unit vector p0 -> p1."""
        return (self.p1 - self.p0) / self.length_mm

    @property
    def midpoint(self) -> np.ndarray:
        return 0.5 * (self.p0 + self.p1)

    def flipped(self) -> "Edge":
        """The same segment traversed the other way round."""
        return Edge(self.index, self.p1.copy(), self.p0.copy(), self.outward_normal)

    def __str__(self) -> str:
        return (f"edge {self.index}: ({self.p0[0]:7.2f}, {self.p0[1]:7.2f}) -> "
                f"({self.p1[0]:7.2f}, {self.p1[1]:7.2f}) mm | "
                f"length {self.length_mm:7.2f} mm | "
                f"outward ({self.outward_normal[0]:+.2f}, {self.outward_normal[1]:+.2f})")


@dataclass(frozen=True, eq=False)
class Workpiece:
    """boundary_xy_mm (2, N) counter-clockwise ring + the extrusion height."""

    boundary_xy_mm: np.ndarray
    height_mm: float = 10.0
    name: str = "workpiece"

    # ── outline ──────────────────────────────────────────────────────────────

    @property
    def n_edges(self) -> int:
        return self.boundary_xy_mm.shape[1]

    @property
    def closed_xy_mm(self) -> np.ndarray:
        """(2, N+1) with the first vertex repeated — for plotting."""
        return np.column_stack([self.boundary_xy_mm, self.boundary_xy_mm[:, :1]])

    @property
    def bbox_mm(self) -> tuple:
        """(x_min, y_min, x_max, y_max) [mm]."""
        lo = self.boundary_xy_mm.min(axis=1)
        hi = self.boundary_xy_mm.max(axis=1)
        return (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))

    @property
    def area_mm2(self) -> float:
        x, y = self.boundary_xy_mm
        return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

    def vertices_3d_m(self, z_mm: float = 0.0) -> np.ndarray:
        """(N, 3) ring at height `z_mm`, workpiece frame [m]."""
        n = self.n_edges
        pts = np.zeros((n, 3))
        pts[:, :2] = self.boundary_xy_mm.T
        pts[:, 2] = z_mm
        return pts * 1e-3

    # ── edges ────────────────────────────────────────────────────────────────

    def edge(self, index: int) -> Edge:
        """Boundary edge `index`, from vertex `index` to vertex `index + 1`."""
        n = self.n_edges
        i = int(index) % n
        p0 = self.boundary_xy_mm[:, i].copy()
        p1 = self.boundary_xy_mm[:, (i + 1) % n].copy()
        u = (p1 - p0) / np.linalg.norm(p1 - p0)
        # CCW ring: material on the left of p0 -> p1, so outward = u rotated -90.
        return Edge(i, p0, p1, np.array([u[1], -u[0]]))

    def edges(self) -> list:
        return [self.edge(i) for i in range(self.n_edges)]

    def longest_edge_index(self) -> int:
        lengths = [e.length_mm for e in self.edges()]
        return int(np.argmax(lengths))

    # ── reporting ────────────────────────────────────────────────────────────

    def summary(self) -> str:
        x0, y0, x1, y1 = self.bbox_mm
        return (f"Workpiece '{self.name}': {self.n_edges} edges, "
                f"{self.area_mm2:.0f} mm^2, height {self.height_mm:g} mm\n"
                f"  bounding box  x {x0:.1f} .. {x1:.1f} mm, "
                f"y {y0:.1f} .. {y1:.1f} mm")

    def edge_table(self, max_rows: int = 12) -> str:
        rows = [f"  {e}" for e in self.edges()[:max_rows]]
        if self.n_edges > max_rows:
            rows.append(f"  ... {self.n_edges - max_rows} more edges")
        return "\n".join(rows)

    # ── the toolpath around it ───────────────────────────────────────────────

    def offset_outline(self, offset_mm: float) -> np.ndarray:
        """(2, N) ring offset along the OUTWARD normals, miter-joined.

        Edges stay parallel to the originals, so a contour on this ring keeps
        constant radial engagement. The tool CENTRE runs on
        offset_mm = R - ae (`MillConfig.tool_offset_mm`).
        """
        edges = self.edges()
        shifted = [(e.p0 + e.outward_normal * float(offset_mm), e.direction)
                   for e in edges]

        pts = np.zeros((2, self.n_edges))
        for i in range(self.n_edges):
            # vertex i = where the offset of edge i-1 meets that of edge i
            (p_prev, u_prev), (p_cur, u_cur) = shifted[i - 1], shifted[i]
            cross = u_prev[0] * u_cur[1] - u_prev[1] * u_cur[0]
            if abs(cross) < 1e-12:
                raise ValueError(
                    f"edges {i - 1} and {i} of '{self.name}' are parallel — "
                    "there is no miter vertex between them")
            d = p_cur - p_prev
            s = (d[0] * u_cur[1] - d[1] * u_cur[0]) / cross
            pts[:, i] = p_prev + u_prev * s
        return pts

    def contour_waypoints(self, offset_mm: float, *, start_edge: int = 0,
                          n_edges: int = None, lead_in_mm: float = 0.0,
                          lead_out_mm: float = 0.0) -> np.ndarray:
        """(M, 2) corner waypoints for milling around the part [mm].

        n_edges None follows the whole loop. lead_in_mm/lead_out_mm add a point
        that far before/after, straight along the first/last edge, so the tool
        starts clear of the material. Feed the result to
        `robotsim.trajectory.plan_quintic_path`.

            part.contour_waypoints(cfg.tool_offset_mm, n_edges=2, lead_in_mm=25)
            # -> [lead-in, corner A, corner B, corner C]
        """
        ring = self.offset_outline(offset_mm)
        n = self.n_edges
        count = n if n_edges is None else int(n_edges)
        if count < 1:
            raise ValueError(f"n_edges must be at least 1, got {n_edges}")
        if count > n:
            raise ValueError(f"'{self.name}' has {n} edges, cannot follow {count}")

        idx = [(int(start_edge) + k) % n for k in range(count + 1)]
        pts = [ring[:, i] for i in idx]

        if lead_in_mm > 0.0:
            u = pts[1] - pts[0]
            pts.insert(0, pts[0] - u / np.linalg.norm(u) * float(lead_in_mm))
        if lead_out_mm > 0.0:
            u = pts[-1] - pts[-2]
            pts.append(pts[-1] + u / np.linalg.norm(u) * float(lead_out_mm))
        return np.array(pts)

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path=None) -> Path:
        """Pickle the ring to data/workpieces/<name>.pickle.

        Only the outline is stored; height_mm belongs to the CUT and is supplied
        again by `load_workpiece`.
        """
        path = Path(path) if path is not None else WORKPIECE_DIR / f"{self.name}.pickle"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self.boundary_xy_mm, fh)
        return path


# ─────────────────────────────────────────────────────────────────────────────
# Building one
# ─────────────────────────────────────────────────────────────────────────────

def rectangle(length_mm: float, width_mm: float, height_mm: float = 5.0,
              origin_mm=(0.0, 0.0), name: str = "rectangular_workpiece"
              ) -> Workpiece:
    """A rectangular block, `length_mm` along +X_w and `width_mm` along +Y_w.

    CCW from `origin_mm`, so edge 0 is the -Y face, 1 the +X, 2 the +Y, 3 the -X.
    """
    x0, y0 = (float(v) for v in origin_mm)
    x1, y1 = x0 + float(length_mm), y0 + float(width_mm)
    ring = np.array([[x0, x1, x1, x0],
                     [y0, y0, y1, y1]], dtype=float)
    return Workpiece(boundary_xy_mm=ring, height_mm=float(height_mm), name=name)


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

class _CompatUnpickler(pickle.Unpickler):
    """Tolerates classes from projects we do not have installed — only the
    geometry inside the pickle matters."""

    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError, ModuleNotFoundError):
            return type(name, (object,), {"__module__": module})


def _from_array(a) -> np.ndarray:
    """Normalise a point array to (2, N)."""
    a = np.asarray(a, dtype=float)
    if a.ndim != 2:
        raise TypeError(f"expected a 2-D point array, got shape {a.shape}")
    if a.shape[0] == 2 and a.shape[1] >= 3:
        return a
    if a.shape[1] == 2 and a.shape[0] >= 3:
        return a.T
    raise TypeError(f"cannot read an outline from an array of shape {a.shape}")


def _as_boundary_xy(obj, depth: int = 0) -> np.ndarray:
    """Dig the (2, N) outline out of whatever the pickle happens to hold."""
    if isinstance(obj, np.ndarray):
        return _from_array(obj)
    if hasattr(obj, "exterior"):                       # shapely Polygon
        return _from_array(np.asarray(obj.exterior.coords, dtype=float))
    if hasattr(obj, "coords"):                         # shapely LineString / ring
        return _from_array(np.asarray(obj.coords, dtype=float))
    if isinstance(obj, (list, tuple)) and obj:
        try:
            return _from_array(np.asarray(obj, dtype=float))
        except (TypeError, ValueError):
            pass

    if depth < 4:
        if isinstance(obj, dict):
            children = list(obj.values())
        elif isinstance(obj, (list, tuple)):
            children = list(obj)
        else:
            children = list(vars(obj).values()) if hasattr(obj, "__dict__") else []
        for child in children:
            try:
                return _as_boundary_xy(child, depth + 1)
            except (TypeError, ValueError):
                continue

    raise TypeError(f"no 2-D outline found in a {type(obj).__name__}")


def _clean_ring(xy: np.ndarray) -> np.ndarray:
    """Drop repeated vertices and orient the ring counter-clockwise."""
    keep = [0]
    for i in range(1, xy.shape[1]):
        if not np.allclose(xy[:, i], xy[:, keep[-1]]):
            keep.append(i)
    xy = xy[:, keep]
    if xy.shape[1] > 2 and np.allclose(xy[:, -1], xy[:, 0]):
        xy = xy[:, :-1]
    if xy.shape[1] < 3:
        raise ValueError("the outline needs at least 3 distinct vertices")

    x, y = xy
    signed_area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return xy if signed_area > 0.0 else xy[:, ::-1]


def available_workpieces() -> list:
    """Names that `load_workpiece` accepts without a path."""
    return sorted(p.stem for p in WORKPIECE_DIR.glob("*.pickle"))


def _resolve(source) -> Path:
    path = Path(source)
    if path.suffix or path.exists():
        return path
    candidate = WORKPIECE_DIR / f"{path.name}.pickle"
    if not candidate.exists():
        raise FileNotFoundError(
            f"no workpiece '{source}' in {WORKPIECE_DIR} — "
            f"available: {', '.join(available_workpieces())}")
    return candidate


def load_workpiece(source, height_mm: float = 10.0, name: str = None) -> Workpiece:
    """Load an outline from data/workpieces/<source>.pickle (or a full path).

    Nothing is re-centred — the coordinates are exactly what the file holds.
    """
    path = _resolve(source)
    with open(path, "rb") as fh:
        obj = _CompatUnpickler(fh).load()
    boundary = _clean_ring(_as_boundary_xy(obj))
    return Workpiece(boundary_xy_mm=boundary, height_mm=float(height_mm),
                     name=name or path.stem)
