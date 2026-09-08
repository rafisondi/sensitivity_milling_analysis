"""Which joints are compliant, and how much.

    Robot(SettingsRealParameter())    # joints 1-3, one spring each   (3 DOF)
    Robot(SettingsAllAxesSDOF())      # joints 1-6, one spring each   (6 DOF)
    Robot(SettingsTriJoint())         # joints 1-6, three springs each (18 DOF)

Following Huynh. K in N m/rad, D in N m s/rad. The 18-DOF model needs axes the
plain URDF lacks, so it uses a generated file (scripts/make_trijoint_urdf.py).

The added axes are much stiffer, so RK4 needs a smaller step. Measured limits
at a mid-path pose: 1.4e-2 s (3 DOF), 9.8e-4 s (6 DOF), 2.6e-4 s (18 DOF).
"""

import numpy as np
from pathlib import Path

# Repo root = one level above this `robotsim/` package. Resolving the URDF
# relative to the package (not the CWD) keeps the simulation runnable from any
# working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = str(_REPO_ROOT / 'data' / 'urdf' / 'v2' / 'Staeubli-Huynh 1.urdf')
URDF_PATH_TRIJOINT = str(_REPO_ROOT / 'data' / 'urdf' / 'v2'
                         / 'Staeubli-Huynh 1 - trijoint.urdf')

# ─────────────────────────────────────────────────────────────────────────────
# The identified parameters
#
# Each table is (joint name, K, D) in the order the corresponding URDF declares
# the joints — which is the order pinocchio numbers them, and which `Robot`
# checks against the model it built so a mis-ordered table cannot go unnoticed.
# ─────────────────────────────────────────────────────────────────────────────

# Huynh §5.3.3, the full 3-DOF-per-joint identification: (k, d) about the z, x
# and y axis of each joint frame. z is the actuation axis of every joint in this
# URDF, so x and y are exactly the two bending directions `make_trijoint_urdf.py`
# adds — the mapping is direct, no reordering.
_HUYNH_KD_ZXY = {
    #        (kz, kx, ky) [N m/rad]      (dz, dx, dy) [N m s/rad]
    1: ((2.3e6, 13.1e6, 17.0e6), (4.6e3, 0.10e3, 0.20e3)),
    2: ((8.6e6,  4.1e6, 19.9e6), (5.0e3, 1.00e3, 2.20e3)),
    3: ((2.8e6, 18.8e6,  2.9e6), (2.5e3, 2.60e3, 0.10e3)),
    4: ((3.0e6,  2.3e6,  3.1e6), (0.3e3, 1.00e3, 0.01e3)),
    5: ((0.2e6,  2.1e6,  0.9e6), (0.01e3, 0.02e3, 0.02e3)),
    6: ((0.9e6, 14.8e6,  8.3e6), (0.01e3, 0.01e3, 0.01e3)),
}

# One torsional spring per joint, about its own actuation axis: the (kz, dz)
# column of the table above, for all six main axes.
#
# These are NOT the numbers in `SettingsRealParameter` — that class carries an
# earlier identification for joints 1-3 (1.15/3.76/1.29 e6 against 2.3/8.6/2.8
# e6 here). Both are kept as they are: mixing the two would give an arm that is
# neither identification.
_SDOF_AXIAL = tuple(
    (f"Joint {i}", _HUYNH_KD_ZXY[i][0][0], _HUYNH_KD_ZXY[i][1][0])
    for i in range(1, 7)
)

# Every joint compliant about all three of its axes. Per joint the rows are the
# actuation axis (z), then the two bending axes (x, y) — the order
# `make_trijoint_urdf.py` emits them in.
_TRIJOINT = tuple(
    entry
    for i in range(1, 7)
    for entry in (
        (f"Joint {i}",    _HUYNH_KD_ZXY[i][0][0], _HUYNH_KD_ZXY[i][1][0]),
        (f"Joint {i} bx", _HUYNH_KD_ZXY[i][0][1], _HUYNH_KD_ZXY[i][1][1]),
        (f"Joint {i} by", _HUYNH_KD_ZXY[i][0][2], _HUYNH_KD_ZXY[i][1][2]),
    )
)


def _unpack(table):
    """(names, K, D) as a tuple of str and two float arrays."""
    names = tuple(row[0] for row in table)
    K = np.array([row[1] for row in table], dtype=float)
    D = np.array([row[2] for row in table], dtype=float)
    return names, K, D


def _flexible_where_identified(K, D, rigid=()):
    """`JOINT_STATE` (True = rigid) marking every joint without a value rigid.

    A joint whose stiffness is nan cannot be integrated, so it is held rigid
    rather than poisoning the state vector — the model then degrades to the
    subset that IS identified instead of returning nan. `rigid` names extra
    joints to pin regardless (by index).
    """
    state = ~(np.isfinite(K) & np.isfinite(D))
    state[list(rigid)] = True
    return state


# ─────────────────────────────────────────────────────────────────────────────
# The models
# ─────────────────────────────────────────────────────────────────────────────

class Settings:
    """Base class. Every model takes `urdf_path=None` to override the default.

    The K/D/JOINT_STATE tables are POSITIONAL, so a URDF whose joints are named
    or ordered differently would silently put each stiffness on the wrong axis.
    `Robot._check_joint_order` compares JOINT_NAMES against the parsed model and
    raises rather than let that happen.
    """


class SettingsRealParameter(Settings): # joints 1-3 compliant (real parameters from Huynh)
    SIM_DT_MAX = 1.4e-2

    def __init__(self, urdf_path=None):
        self.URDF_PATH = str(urdf_path) if urdf_path else URDF_PATH
        self.JOINT_NAMES = tuple(f"Joint {i}" for i in range(1, 7))
        self.JOINT_STATE = np.array([False, False, False, True, True, True])  # True = rigid
        self.K_m_diag = np.array([1.15e6, 3.76e6, 1.29e6, np.nan, np.nan, np.nan])
        self.D_m_diag = np.array([2.82e3, 0.61e3, 1.87e3, np.nan, np.nan, np.nan])


class SettingsAllAxesSDOF(Settings):
    """All six main axes compliant, one spring per joint (single-DOF joints).

    The plain URDF, six flexible DOFs. The TCP reduction is no longer square
    (J is 3x6), so `linearization` uses the operational-space projection.
    """

    SIM_DT_MAX = 9.8e-4

    def __init__(self, urdf_path=None):
        self.URDF_PATH = str(urdf_path) if urdf_path else URDF_PATH
        self.JOINT_NAMES, self.K_m_diag, self.D_m_diag = _unpack(_SDOF_AXIAL)
        self.JOINT_STATE = _flexible_where_identified(self.K_m_diag, self.D_m_diag)


class SettingsTriJoint(Settings):
    """Huynh §5.3.3: every joint compliant about all three of its axes.

    Needs the generated 18-joint URDF — run `scripts/make_trijoint_urdf.py`
    once if `data/urdf/v2/Staeubli-Huynh 1 - trijoint.urdf` is missing.
    """

    SIM_DT_MAX = 2.6e-4
    # The virtual bending joints are springs, not motors: the IK may not use
    # them to reach a pose, and the motor command is zero in those slots.
    ACTUATED_JOINTS = tuple(f"Joint {i}" for i in range(1, 7))

    def __init__(self, urdf_path=None):
        self.URDF_PATH = str(urdf_path) if urdf_path else URDF_PATH_TRIJOINT
        if not Path(self.URDF_PATH).exists():
            raise FileNotFoundError(
                f"{self.URDF_PATH} does not exist — generate it with "
                "`python scripts/make_trijoint_urdf.py`")
        self.JOINT_NAMES, self.K_m_diag, self.D_m_diag = _unpack(_TRIJOINT)
        self.JOINT_STATE = _flexible_where_identified(self.K_m_diag, self.D_m_diag)
