import math

import numpy as np
from numba import njit


@njit(cache=True)
def _scanline_fill(grid, xs, ys, res, ox, oy):
    """Even-odd scanline fill of a closed polygon into a uint8 grid.

    A pixel is set when its centre is strictly inside the polygon. Edge
    crossings use the half-open test `(y0 <= wy) != (y1 <= wy)`, which counts a
    vertex exactly once and so keeps the parity correct where an edge endpoint
    lands on a scanline.
    """
    NY, NX = grid.shape
    n = xs.shape[0]
    xc = np.empty(n + 1)

    for py in range(NY):
        wy = py * res + oy

        m = 0
        for e in range(n):
            x0 = xs[e]; y0 = ys[e]
            nx_ = e + 1
            if nx_ == n:
                nx_ = 0
            x1 = xs[nx_]; y1 = ys[nx_]
            if (y0 <= wy) != (y1 <= wy):
                xc[m] = x0 + (wy - y0) / (y1 - y0) * (x1 - x0)
                m += 1
        if m < 2:
            continue

        # insertion sort: m is the number of edge crossings on this row, which
        # is 2 for a convex stock and stays small for realistic boundaries
        for a in range(1, m):
            v = xc[a]
            b = a - 1
            while b >= 0 and xc[b] > v:
                xc[b + 1] = xc[b]
                b -= 1
            xc[b + 1] = v

        for a in range(0, m - 1, 2):
            lo = int(math.ceil((xc[a] - ox) / res))
            hi = int(math.floor((xc[a + 1] - ox) / res))
            if lo < 0:
                lo = 0
            if hi > NX - 1:
                hi = NX - 1
            for px in range(lo, hi + 1):
                grid[py, px] = 1


class WorkpieceRaster:
    """2D workpiece as a NumPy uint8 grid (1=material, 0=removed).

    - Engagement check  : O(1) pixel lookup
    - Workpiece erosion : raster fill of the swept quadrilateral

    The boundary polygon is rasterized once at init. All coordinates in mm,
    workpiece frame.
    """

    def __init__(
        self,
        boundary_xy: np.ndarray,
        resolution: float = 0.1,
        padding: float = 1.0,
    ):
        """
        boundary_xy : (2, N) array — x-coords in row 0, y-coords in row 1
        resolution  : grid cell size [mm/pixel]
        padding     : extra margin beyond workpiece bbox [mm]
        """
        xs = np.ascontiguousarray(boundary_xy[0], dtype=np.float64)
        ys = np.ascontiguousarray(boundary_xy[1], dtype=np.float64)

        self.resolution = float(resolution)
        self.origin_x   = float(xs.min() - padding)
        self.origin_y   = float(ys.min() - padding)

        nx = int(np.ceil((xs.max() - self.origin_x + padding) / resolution)) + 1
        ny = int(np.ceil((ys.max() - self.origin_y + padding) / resolution)) + 1

        # Even-odd scanline fill: a pixel is material when its CENTRE lies
        # inside the boundary polygon.
        #
        # This replaces matplotlib's Path.contains_points, which built an
        # (nx*ny, 2) point array and tested every pixel individually -- 29 s and
        # ~4.7 GB of temporaries for a 600x300 mm stock at 0.025 mm. Scanline
        # touches each row once and allocates nothing per pixel.
        #
        # It also drops the old `radius=0.5*resolution` argument. That was
        # documented as shrinking the test by half a pixel; it in fact EXPANDED
        # the polygon, which is why the 120x40 stock rasterised to 4808.00 mm^2
        # against an exact 4800.00 (perimeter 320 mm x 0.025 mm = 8.00 mm^2).
        # The oracle uses the exact polygon, so dropping it removes a known
        # discrepancy rather than introducing one.
        self._grid = np.zeros((ny, nx), dtype=np.uint8)
        _scanline_fill(self._grid, xs, ys, self.resolution,
                       self.origin_x, self.origin_y)
        self._boundary_xy = boundary_xy.copy()  # original boundary for compatibility

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    @property
    def grid(self) -> np.ndarray:
        """(NY, NX) uint8 array — pass directly to the Numba step kernel."""
        return self._grid

    def is_material(self, x: float, y: float) -> bool:
        """Single pixel lookup — True if material is present at (x, y)."""
        ix = int(np.floor((x - self.origin_x) / self.resolution))
        iy = int(np.floor((y - self.origin_y) / self.resolution))
        if ix < 0 or iy < 0 or ix >= self._grid.shape[1] or iy >= self._grid.shape[0]:
            return False
        return bool(self._grid[iy, ix])

    def boundary_xy(self) -> np.ndarray:
        """Return the original workpiece boundary as (2, N) — for compatibility."""
        return self._boundary_xy
