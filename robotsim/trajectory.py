"""Operational-space paths and the quintic planner.

Paths are authored in the workpiece frame in MILLIMETRES and stored in both
frames in METRES. Each segment is a straight line with a quintic time scaling
s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5, so velocity and acceleration are zero at
both ends and segments chain safely around a sharp corner. speed_mm_s is the
MEAN speed; the quintic peak is 1.875x it.
"""

from dataclasses import dataclass

import numpy as np

from robotsim.transforms import rotate_vectors, transform_points

QUINTIC_PEAK = 15.0 / 8.0


@dataclass
class OperationalPath:
    """Tool-centre path in both frames.

    s_w/s_i  (N, 3) positions, workpiece / base frame [m]
    R_i_tcp  (3, 3) constant TCP orientation, base frame
    t, dt    time grid [s]
    v_w/v_i  (N, 3) velocities [m/s], or None
    nom_w/nom_i  the UNCOMPENSATED path, or None. Error is measured against
                 these; `s_*` may be aimed off-nominal to cancel deflection.
    """

    s_w: np.ndarray
    s_i: np.ndarray
    R_i_tcp: np.ndarray
    t: np.ndarray
    dt: float
    v_w: np.ndarray = None
    v_i: np.ndarray = None
    nom_w: np.ndarray = None
    nom_i: np.ndarray = None

    def __len__(self) -> int:
        return len(self.s_i)

    @property
    def duration_s(self) -> float:
        return float(self.t[-1])

    @property
    def length_m(self) -> float:
        """Arc length [m] — summed, not end-to-end."""
        return float(np.linalg.norm(np.diff(self.s_w, axis=0), axis=1).sum())

    @property
    def speed_m_s(self) -> float:
        return self.length_m / self.duration_s if self.duration_s > 0.0 else 0.0

    @property
    def xy_mm(self) -> np.ndarray:
        """(N, 2) tool centre, workpiece frame [mm] — what the process is stepped with."""
        return self.s_w[:, :2] * 1e3

    def velocity_i(self) -> np.ndarray:
        if self.v_i is not None:
            return self.v_i
        return np.gradient(self.s_i, self.dt, axis=0, edge_order=2)

    def speed_profile_mm_s(self) -> np.ndarray:
        v = (self.v_w if self.v_w is not None
             else np.gradient(self.s_w, self.dt, axis=0, edge_order=2))
        return np.linalg.norm(v, axis=1) * 1e3

    def resample_i(self, t_query) -> np.ndarray:
        """Commanded base-frame position on an arbitrary time grid."""
        return np.column_stack([np.interp(t_query, self.t, self.s_i[:, k])
                                for k in range(3)])

    @property
    def has_nominal(self) -> bool:
        return self.nom_i is not None

    @property
    def nominal_or_commanded_i(self) -> np.ndarray:
        return self.s_i if self.nom_i is None else self.nom_i

    def resample_nominal_i(self, t_query) -> np.ndarray:
        """Nominal base-frame position on an arbitrary time grid."""
        src = self.s_i if self.nom_i is None else self.nom_i
        return np.column_stack([np.interp(t_query, self.t, src[:, k])
                                for k in range(3)])

    def summary(self) -> str:
        v = self.speed_profile_mm_s()
        return (f"path     {self.length_m * 1e3:.1f} mm in {self.duration_s:.3f} s "
                f"| {len(self)} points at dt {self.dt:g} s\n"
                f"         speed mean {self.speed_m_s * 1e3:.1f}, "
                f"peak {v.max():.1f} mm/s")


# ── segments ─────────────────────────────────────────────────────────────────

def quintic_scaling(tau) -> tuple:
    """(s, ds/dtau, d2s/dtau2) of the rest-to-rest quintic on tau in [0, 1]."""
    tau = np.asarray(tau, dtype=float)
    s = tau ** 3 * (10.0 - 15.0 * tau + 6.0 * tau ** 2)
    sd = 30.0 * tau ** 2 * (1.0 - tau) ** 2
    sdd = 60.0 * tau * (1.0 - 3.0 * tau + 2.0 * tau ** 2)
    return s, sd, sdd


def _as_xyz_mm(point, z_mm: float) -> np.ndarray:
    p = np.asarray(point, dtype=float).ravel()
    if p.size == 3:
        return p.copy()
    if p.size == 2:
        return np.array([p[0], p[1], float(z_mm)])
    raise ValueError(f"expected a 2-D or 3-D point, got {p.size} components")


@dataclass
class Segment:
    """One straight A -> B move. Positions [mm], velocities [mm/s], t from 0."""

    start_mm: np.ndarray
    end_mm: np.ndarray
    xyz_mm: np.ndarray
    v_mm_s: np.ndarray
    t: np.ndarray
    speed_mm_s: float
    kind: str = "quintic"

    @property
    def length_mm(self) -> float:
        return float(np.linalg.norm(self.end_mm - self.start_mm))

    @property
    def duration_s(self) -> float:
        return float(self.t[-1])

    @property
    def peak_speed_mm_s(self) -> float:
        return float(np.linalg.norm(self.v_mm_s, axis=1).max())

    @property
    def peak_accel_mm_s2(self) -> float:
        speed = np.linalg.norm(self.v_mm_s, axis=1)
        return float(np.abs(np.gradient(speed, self.t, edge_order=2)).max())

    def __str__(self) -> str:
        a, b = self.start_mm, self.end_mm
        return (f"({a[0]:7.2f}, {a[1]:7.2f}) -> ({b[0]:7.2f}, {b[1]:7.2f}) mm | "
                f"{self.length_mm:7.2f} mm in {self.duration_s:5.3f} s | "
                f"{self.kind:7s} mean {self.speed_mm_s:6.1f}, "
                f"peak {self.peak_speed_mm_s:6.1f} mm/s")


def quintic_segment(start_mm, end_mm, speed_mm_s: float, dt: float = 1.0e-3,
                    z_mm: float = 0.0) -> Segment:
    """Straight A -> B at MEAN speed `speed_mm_s`, quintic in time. The primitive."""
    a, b = _as_xyz_mm(start_mm, z_mm), _as_xyz_mm(end_mm, z_mm)
    length = float(np.linalg.norm(b - a))
    if length <= 0.0:
        raise ValueError(f"segment has zero length: {a} -> {b} mm")
    if speed_mm_s <= 0.0:
        raise ValueError(f"speed must be positive, got {speed_mm_s} mm/s")

    duration = length / float(speed_mm_s)
    t = np.linspace(0.0, duration, max(2, int(round(duration / dt)) + 1))
    s, sd, _ = quintic_scaling(t / duration)

    direction = (b - a) / length
    return Segment(start_mm=a, end_mm=b, xyz_mm=a + np.outer(s, b - a),
                   v_mm_s=np.outer(sd * length / duration, direction), t=t,
                   speed_mm_s=float(speed_mm_s), kind="quintic")


def linear_segment(start_mm, end_mm, speed_mm_s: float, dt: float = 1.0e-3,
                   z_mm: float = 0.0) -> Segment:
    """The same move at CONSTANT speed — no ramp. Do not chain around a corner."""
    a, b = _as_xyz_mm(start_mm, z_mm), _as_xyz_mm(end_mm, z_mm)
    length = float(np.linalg.norm(b - a))
    if length <= 0.0:
        raise ValueError(f"segment has zero length: {a} -> {b} mm")

    duration = length / float(speed_mm_s)
    t = np.linspace(0.0, duration, max(2, int(round(duration / dt)) + 1))
    direction = (b - a) / length
    return Segment(start_mm=a, end_mm=b, xyz_mm=a + np.outer(t / duration, b - a),
                   v_mm_s=np.tile(direction * float(speed_mm_s), (len(t), 1)),
                   t=t, speed_mm_s=float(speed_mm_s), kind="linear")


# ── chaining ─────────────────────────────────────────────────────────────────

def chain(segments, *, T_iw, R_i_tcp, dt: float = 1.0e-3) -> OperationalPath:
    """Lay segments end to end and resample onto one uniform time grid."""
    segments = list(segments)
    if not segments:
        raise ValueError("no segments to chain")

    for prev, nxt in zip(segments, segments[1:]):
        gap = float(np.linalg.norm(nxt.start_mm - prev.end_mm))
        if gap > 1.0e-3:
            raise ValueError(f"segments do not join: {prev.end_mm} -> "
                             f"{nxt.start_mm} mm leaves a {gap:.4f} mm gap")

    offsets, running = [], 0.0
    for seg in segments:
        offsets.append(running)
        running += seg.duration_s
    t_src = np.concatenate([seg.t + off for seg, off in zip(segments, offsets)])
    xyz_src = np.vstack([seg.xyz_mm for seg in segments])
    v_src = np.vstack([seg.v_mm_s for seg in segments])

    # duplicate stamps at the joins would break np.interp's monotonicity
    keep = np.concatenate([[True], np.diff(t_src) > 1e-12])
    t_src, xyz_src, v_src = t_src[keep], xyz_src[keep], v_src[keep]

    t = np.arange(max(2, int(np.floor(running / dt)) + 1)) * dt
    xyz_mm = np.column_stack([np.interp(t, t_src, xyz_src[:, k]) for k in range(3)])
    v_mm_s = np.column_stack([np.interp(t, t_src, v_src[:, k]) for k in range(3)])

    s_w, v_w = xyz_mm * 1e-3, v_mm_s * 1e-3
    R_iw = np.asarray(T_iw, dtype=float)[:3, :3]
    return OperationalPath(s_w=s_w, s_i=transform_points(T_iw, s_w),
                           R_i_tcp=np.asarray(R_i_tcp, dtype=float), t=t, dt=dt,
                           v_w=v_w, v_i=rotate_vectors(R_iw, v_w))


def plan_quintic_path(waypoints_mm, speed_mm_s, *, T_iw, R_i_tcp,
                      dt: float = 1.0e-3, z_mm: float = 0.0) -> OperationalPath:
    """Quintic move through every waypoint in turn, resting at each.

    speed_mm_s is one mean speed for all segments, or one per segment.
    """
    pts = [_as_xyz_mm(w, z_mm)
           for w in np.atleast_2d(np.asarray(waypoints_mm, dtype=float))]
    if len(pts) < 2:
        raise ValueError("need at least two waypoints")

    speeds = np.atleast_1d(np.asarray(speed_mm_s, dtype=float))
    if speeds.size == 1:
        speeds = np.repeat(speeds, len(pts) - 1)
    if speeds.size != len(pts) - 1:
        raise ValueError(f"{len(pts)} waypoints need {len(pts) - 1} speeds, "
                         f"got {speeds.size}")

    return chain([quintic_segment(a, b, v, dt=dt)
                  for a, b, v in zip(pts, pts[1:], speeds)],
                 T_iw=T_iw, R_i_tcp=R_i_tcp, dt=dt)


def plan_linear_move(start_mm, end_mm, speed_mm_s: float, *, T_iw, R_i_tcp,
                     dt: float = 1.0e-3, z_mm: float = 0.0) -> OperationalPath:
    """One straight segment at constant speed."""
    return chain([linear_segment(start_mm, end_mm, speed_mm_s, dt=dt, z_mm=z_mm)],
                 T_iw=T_iw, R_i_tcp=R_i_tcp, dt=dt)


def hold(point_mm, duration_s: float, *, T_iw, R_i_tcp, dt: float = 1.0e-3,
         z_mm: float = 0.0) -> OperationalPath:
    """Stand still at one point — the commanded path of a hold-pose run."""
    p = _as_xyz_mm(point_mm, z_mm)
    n_pts = max(2, int(round(duration_s / dt)) + 1)
    s_w = np.tile(p * 1e-3, (n_pts, 1))
    return OperationalPath(s_w=s_w, s_i=transform_points(T_iw, s_w),
                           R_i_tcp=np.asarray(R_i_tcp, dtype=float),
                           t=np.arange(n_pts) * dt, dt=dt,
                           v_w=np.zeros_like(s_w), v_i=np.zeros_like(s_w))
