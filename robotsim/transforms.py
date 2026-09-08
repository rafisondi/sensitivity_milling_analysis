"""Rigid-transform helpers. T_iw maps workpiece -> base: p_i = R_iw @ p_w + t_iw.

Workpiece frame: the process lives here, in MILLIMETRES for the raster kernel
and metres everywhere else. Base frame: the robot, always metres.
"""

import numpy as np


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Assemble a 4x4 homogeneous transform from a rotation R and translation t."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Inverse of a 4x4 homogeneous transform (NOT the transpose)."""
    R, t = T[:3, :3], T[:3, 3]
    return make_transform(R.T, -R.T @ t)


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to (N, 3) POINTS."""
    return pts @ T[:3, :3].T + T[:3, 3]


def rotate_vectors(R: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    """Rotate (N, 3) FREE vectors — forces, deviations. No translation."""
    return vecs @ R.T


def rot_z(angle_rad: float) -> np.ndarray:
    """Rotation about +Z by angle [rad]."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def box_corners(length: float, width: float, height: float) -> np.ndarray:
    """8 corners of a block, starting corner at the origin."""
    dx, dy, dz = np.eye(3) * [length, width, height]
    return np.array([
        [0, 0, 0], dx, dx + dy, dy,           # bottom face (z = 0)
        dz, dx + dz, dx + dy + dz, dy + dz,   # top face    (z = height)
    ], dtype=float)
