"""mill_shaply — the older Shapely-based milling simulation environment.

Kept as-is (legacy code from the previous project) apart from two changes:
the intra-package import in `eraser_of_matter` was `from milling.…`, and
`milling_workpiece.__init__` now takes the cutter/coefficients as optional
keyword arguments instead of hard-coding them. Defaults reproduce the old
behaviour exactly.

`millsim.milling.shapely_process.ShapelyMillingProcess` wraps `milling_workpiece`
in the same `step(xy_mm, spindle_angle) -> force_w` interface the dexel
`MillingProcess` exposes, so either can drive `simulation.run_coupled_simulation`.
"""

from .eraser_of_matter import milling_workpiece
from .milling_path import MillingPath

__all__ = ["milling_workpiece", "MillingPath"]
