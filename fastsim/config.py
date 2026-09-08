"""The whole recipe for a cut, in one frozen dataclass.

    cfg = MillConfig(spindle_rpm=6000.0, raster_mm=0.03)
    cfg = replace(cfg, sim_dt=1e-4)

No robot fields — where the job sits is `robotsim.Scene`.

Two settings decide whether the numbers mean anything:
  raster_mm        the chip must be several pixels across. `chip_px()` reports
                   it; below ~2 px the binary raster loses the chip.
  slice_angle_deg  max tooth rotation between stacked axial slices, so
                   n_slices = 1 + floor(ap tan(helix) / (R * slice_angle)).
                   Changes the RIPPLE, not the mean, and sets the memory
                   (n_slices * NY * NX bytes).
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class MillConfig:
    """Everything the cutting side of a run needs."""

    # the part
    workpiece: str = "rectangular_workpiece"
    height_mm: float = 5.0                 # extrusion = axial depth of cut ap

    # the tool
    diameter_mm: float = 16.0
    n_teeth: int = 4
    helix_angle_deg: float = 45.0
    Ktc: float = 1930.4                    # tangential cutting coeff [N/mm^2]
    Krc: float = 1159.6                    # radial
    Kac: float = 200.6                     # axial
    spindle_spin: int = -1                 # +1 = CCW, -1 = CW (seen from +Z_w)
    spindle_rpm: float = 3333.0

    # the cut
    feed_mm_s: float = 40.0
    radial_engagement_mm: float = 5.0      # ae; ae = R centres the tool on the face

    # numerics
    sim_dt: float = 2.0e-4
    plan_dt: float = 1.0e-3
    raster_mm: float = 0.01
    slice_angle_deg: float = 10.0
    warmup_revs: int = 5

    # engine
    chip_mode: str = "dexel"               # "dexel" regenerative | "analytic"
    n_phi: int = 2048                      # angular bins of the surface record
    max_chip_mm: float = 1.0               # runaway bound
    moment_about: str = "workpiece_origin"  # or "tool_center"
    parallel: Optional[bool] = None

    @property
    def radius_mm(self) -> float:
        return self.diameter_mm / 2.0

    @property
    def tool_offset_mm(self) -> float:
        """Tool-centre offset from the finished face, R - ae. Offset the outline
        by this to get the toolpath."""
        return self.radius_mm - self.radial_engagement_mm

    @property
    def omega_mag_rad_s(self) -> float:
        """Spindle speed MAGNITUDE [rad/s]."""
        return 2.0 * math.pi * self.spindle_rpm / 60.0

    @property
    def omega_rad_s(self) -> float:
        """SIGNED spindle speed. The process is driven by omega_rad_s * t, so for
        a CW spindle that angle decreases. `spindle_spin` separately sets the
        tangential force direction — flipping one without the other is wrong."""
        return self.spindle_spin * self.omega_mag_rad_s

    @property
    def tooth_period_s(self) -> float:
        return 2.0 * math.pi / (self.n_teeth * self.omega_mag_rad_s)

    @property
    def steps_per_tooth(self) -> float:
        return self.tooth_period_s / self.sim_dt

    @property
    def steps_per_rev(self) -> float:
        return 60.0 / self.spindle_rpm / self.sim_dt

    @property
    def warmup_distance_mm(self) -> float:
        return self.warmup_revs * 60.0 / self.spindle_rpm * self.feed_mm_s

    def feed_per_tooth_mm(self, feed_mm_s=None):
        """Chip load fz [mm/tooth] from a feed speed [mm/s]."""
        feed = self.feed_mm_s if feed_mm_s is None else feed_mm_s
        return np.asarray(feed) / (self.spindle_rpm / 60.0 * self.n_teeth)

    def chip_px(self, feed_mm_s=None):
        """Chip load in PIXELS — check this before a long run."""
        return self.feed_per_tooth_mm(feed_mm_s) / self.raster_mm

    def summary(self) -> str:
        fz, px = float(self.feed_per_tooth_mm()), float(self.chip_px())
        warn = "   ! under 2 px, the raster loses the chip" if px < 2.0 else ""
        # Below ~8 steps/tooth the tooth-passing force is aliased: the mean is
        # still roughly usable, the peak reads low and the harmonics are noise.
        dt_warn = ("   ! under 8 steps/tooth, harmonics are aliased"
                   if self.steps_per_tooth < 8.0 else "")
        return (
            f"cutter   D {self.diameter_mm:g} mm, {self.n_teeth} teeth, helix "
            f"{self.helix_angle_deg:g} deg | spin "
            f"{'CW' if self.spindle_spin < 0 else 'CCW'} at {self.spindle_rpm:g} rpm\n"
            f"cut      ap {self.height_mm:g} mm | ae {self.radial_engagement_mm:g} mm "
            f"(centre offset {self.tool_offset_mm:+.1f} mm) | feed {self.feed_mm_s:g} mm/s\n"
            f"chip     fz {fz:.4f} mm/tooth = {px:.1f} px at raster "
            f"{self.raster_mm:g} mm{warn}\n"
            f"numerics dt {self.sim_dt:g} s ({self.steps_per_rev:.0f} steps/rev, "
            f"{self.steps_per_tooth:.1f} steps/tooth) | chip_mode "
            f"'{self.chip_mode}'{dt_warn}")
