import numpy as np
from shapely import LineString, Point
from scipy.interpolate import make_interp_spline, Akima1DInterpolator

class MillingPath():
    def __init__(self, path_coordinates: np.ndarray | LineString, axial_cutting_dept: float=5.0, feed_curve: np.ndarray | float=40.0, offset: float=None, offset_side: str='left'):
        if isinstance(path_coordinates, LineString):
            self.path_ls = path_coordinates
        else:
            self.path_ls = LineString(path_coordinates)
        
        if not offset is None:
            self.path_ls = self.path_ls.parallel_offset(distance=offset, side=offset_side)

        support_points, distances = self.n_support_points()
        self.spline_x = make_interp_spline(distances, support_points[:, 0], k=5) # C4 continuous
        self.spline_y = make_interp_spline(distances, support_points[:, 1], k=5)

        self.feed_curve = feed_curve    # in [mm/s]
        if isinstance(feed_curve, np.ndarray):
            self.feed_curve = make_interp_spline(feed_curve[:, 0], feed_curve[:, 1], k=3)
            # self.feed_curve = Akima1DInterpolator(feed_curve[:, 0], feed_curve[:, 1], method="makima")

        self.axial_cutting_depth = axial_cutting_dept

    def length(self) -> float:
        """Return length of milling path."""
        return self.path_ls.length
    
    def points_at_distances(self, distances: np.ndarray) -> np.ndarray:
        """Return coordinates of point at a certain distance on milling path."""
        distances[distances > self.length()] = self.length()
        x_spline = self.spline_x(distances)
        y_spline = self.spline_y(distances)

        return np.array([x_spline, y_spline])
    
    def feed_at_distances(self, distances: np.ndarray) -> np.ndarray:
        """Return the planned feedrate at a certain distance on milling path."""
        distances[distances > self.length()] = self.length()
        if isinstance(self.feed_curve, float):
            feed_speed = np.ones(len(distances)) * self.feed_curve
        else:
            feed_speed = self.feed_curve(distances)

        return feed_speed

    def support_points_distances(self) -> np.ndarray:
        """Return the positions of the support points along the milling path."""
        coords = np.array(self.path_ls.coords)
        diffs = np.diff(coords, axis=0)
        segment_distances = np.linalg.norm(diffs, axis=1)
        cumulative_distances = np.insert(np.cumsum(segment_distances), 0, 0)

        return cumulative_distances
    
    def n_support_points(self, n_points: int=np.inf):
        """Pick out fraction of support points."""
        coords = np.array(self.path_ls.coords)
        distances_all = self.support_points_distances()

        n = len(coords)

        if n_points >= n:
            picked_points = [Point(x, y) for x, y in coords]
            distances = list(distances_all)
        else:
            selected_indices = np.linspace(0, n - 1, n_points).astype(int)
            picked_points = [Point(coords[i]) for i in selected_indices]
            distances = [distances_all[i] for i in selected_indices]
    
        return  np.array([np.array(d.coords[0]) for d in picked_points]), np.array(distances)
        