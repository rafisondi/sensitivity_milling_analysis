"""One coupled robot milling pass, and the linear model that is meant to replace it.

The question this workspace exists to answer, on one job:

    a rectangular workpiece, one edge, 25 mm of air before and after, so the pass
    contains an ENTRY and an EXIT — the two places a steady-state model has
    nothing to say about

    TRUTH        the dexel cutting engine closed on the flexible-joint arm,
                 integrated in joint space          `analysis.sim_coupled`
    PREDICTION   the same cut linearised into F0 + K_cut + C_cut, closed on the
                 arm's M/D/K at the tool tip        `analysis.stability`

and two things are read off the pair: whether the linear criterion calls the
stability of the cut the way the simulation does, node by node along the
trajectory, and how close its revolution-averaged force `F0` is to the force the
engine actually produced.

    python main.py                       one pass, compensated, results in out/
    python main.py --no-compensate       the same pass with the motors idle

THE ONE THING THAT MUST BE ON

Left alone, the coupled pass does not cut the geometry the prediction assumes.
The mean cutting force is of the order of a kilonewton and the arm is compliant,
so following the nominal trajectory presses the tool out of the workpiece by
more than a tenth of a millimetre; the engine then cuts at that deflected
position while the linear model measures its engagement on the commanded one,
and the two are no longer describing the same cut.

`analysis.feedforward` cancels it: the revolution-average force `F0(s)` is
carried on the MOTOR TORQUES, `tau_ff = -J(theta)^T F0(s)`, scheduled along the
path. The joint springs are then left carrying only the fluctuation about the
mean, which is the object the linearisation is about. See that module for why
the torque route is preferred here over aiming the command off the nominal.

LAYOUT

    fastsim/      the dexel cutting engine and the coupled loop
    robotsim/     the arm: URDF, IK, RK4, its linearisation, its receptance
    stabsim/      the cut linearised, and the eigenvalues of the closed loop
    analysis/     everything above, wired into one job          <- you are here
    out/          one directory per run: raw histories, stability, forces

UNITS are millimetres, seconds and newtons at the interfaces, as upstream:
`ap`/`ae`/`fz` in mm, feed in mm/s, `Ktc`/`Krc`/`Kac` in N/mm^2. Growth rates are
1/s and frequencies Hz. Internally `M`/`D`/`K` are SI and deflections are
reported in micrometres.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
OUT = ROOT / "out"

#: Every run defaults to this arm. It is the only model in `ROBOT_MODELS` whose
#: joints 1-3 carry an identified stiffness and damping; the wider ones
#: (`all-axes`, `trijoint`) are the same arm at Huynh's other identification
#: levels and need a smaller `sim_dt`.
DEFAULT_ROBOT_MODEL = "joints123"

#: The operating point every flag on `main.py` starts from.
DEFAULT_CONFIG = CONFIGS / "base.json"

__all__ = ["ROOT", "CONFIGS", "DATA", "OUT",
           "DEFAULT_ROBOT_MODEL", "DEFAULT_CONFIG"]
