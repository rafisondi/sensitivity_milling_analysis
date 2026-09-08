"""The optimised dexel milling engine and the coupled loop that drives it.

Stacked uint8 raster + Numba step kernel, 6-D wrench out. Workpiece frame,
MILLIMETRES. The robot lives in `robotsim` (metres); the two meet only in
`fastsim.coupled`.

    cfg  = MillConfig(spindle_rpm=3333.0, raster_mm=0.02)
    part = geometry.rectangle(100.0, 60.0, height_mm=1.0)
    run  = coupled.run_pass(cfg, part, path, scene)     # the flexible arm in it

The regenerative loop is the ten lines at the bottom of `coupled.py`: the
DEFLECTED tool centre decides the chip thickness, the chip decides the force, the
force deflects the joints again.

Note the raster is a stacked 2-D, CONSTANT-DEPTH model — axial compliance is not
fed back. `MillingPass.depth_check()` says whether that held.

WHAT IS NOT HERE. Upstream also carries `nominal.py` (the rigid pass, which
obeys the command exactly), `run_linear.py` (the cut on a reduced M/D/K plant),
`plotting.py` and the `run_*.py` scripts. This workspace runs one thing — the
coupled robot pass — so they were left behind rather than carried unused.
"""

from fastsim import geometry, metrics, prepare
from fastsim.config import MillConfig
from fastsim.geometry import (
    Edge, WORKPIECE_DIR, Workpiece, available_workpieces, load_workpiece, rectangle,
)
from fastsim.prepare import (
    build_process, carve_entry_slot, prepare_steady_state_cut, warmup_process,
)
from fastsim.process import MillingProcess
from fastsim.raster import WorkpieceRaster

__all__ = [
    "MillConfig",
    "Workpiece", "Edge", "rectangle", "load_workpiece", "available_workpieces",
    "WORKPIECE_DIR", "geometry",
    "MillingProcess", "WorkpieceRaster", "build_process",
    "prepare_steady_state_cut", "carve_entry_slot", "warmup_process", "prepare",
    "metrics",
]


def __getattr__(name):
    """`fastsim.coupled` is imported lazily — it is the one module that pulls in
    robotsim, pinocchio and a URDF parse, which the geometry alone does not need."""
    if name in ("coupled", "run_pass", "MillingPass", "ProcessAdapter"):
        from fastsim import coupled
        return coupled if name == "coupled" else getattr(coupled, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
