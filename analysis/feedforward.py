"""The TCP DC compensation: carry the mean cutting force on the motors.

    tau_ff(t) = -J(theta(t))^T * F0(s(t))

WHY ANYTHING IS NEEDED AT ALL

The coupled loop steps the cutting engine with the DEFLECTED tool centre, and
this arm is compliant: the revolution-average cutting force is of the order of a
kilonewton and `G(0)` is hundreds of micrometres per kilonewton, so a tool asked
to follow the nominal trajectory is pushed OUT of the workpiece by more than a
tenth of a millimetre before it has done anything wrong. The radial engagement is
5 mm, so that is a several-percent error in `ae` — and it is a systematic one,
always in the same direction, growing with the cutting coefficients because the
force does.

That breaks the comparison this workspace exists to make. `analysis.stability`
measures its engagement on the COMMANDED path; the engine cuts at the deflected
one. Uncompensated, the two sides are not describing the same cut, and any
disagreement between them is contaminated by a geometry error before the
linearisation is even asked a question.

WHAT IS CANCELLED, AND WHAT IS NOT

`Solver.step` forms `tau_drive = tau_ext + tau_ff` with `tau_ext = J^T f_ext`, so
a feedforward of `-J^T F0` leaves the joint springs carrying `J^T (F_cut - F0)`
— the FLUCTUATION about the mean and nothing else. That fluctuation is the object
the linearisation is about: the tooth-passing ripple, the entry and exit
transients and any regenerative growth all survive untouched, because none of
them is the revolution average.

It is feed-forward and offline. Nothing is measured during the cut, there is no
controller, and `F0` is the ZOA model's mean force rather than the engine's — so
whatever those two disagree by is a residual this cannot remove. That residual is
a result, not a defect: the plant is identical on both sides of it, so it
measures the force model.

WHY THE TORQUE ROUTE AND NOT AN OFFSET COMMAND

`stabsim.compensate` documents the other option: aim the command off the nominal
by `-G(0) F0`, so the mean force pushes the tool back onto the geometry it was
supposed to cut. Both cancel the same offset, and at DC they must agree wherever
`G(0) = J K_m^-1 J^T`. The torque route is used here for two reasons:

  * it moves no geometry. The commanded path stays the nominal one, so `F0`,
    `K_cut` and `C_cut` are evaluated at the path the tool is actually asked to
    follow, and there is no second path to keep track of.
  * an offset command can leave the material. The correction is `G(0) F0`, and
    once that exceeds `ae` the compensated command steers the tool clear of the
    workpiece and the pass never cuts. A torque feedforward has no such limit.

There is also a real check hiding in the pair: neither route is validated by its
own construction, and the two are built from different objects — one from the
Jacobian at each instant, one from the DC gain of the linearisation. `offset_um`
below reports what the path-offset route WOULD have applied, so the two can be
put side by side without running both.

THE SCHEDULE

`F0` is not constant along this job — it ramps in at the entry, holds through the
edge, and ramps out. `fastsim.coupled.run_pass` supports a feedforward torque but
only a CONSTANT one, built once from `carry_force_w`; carrying `F0` properly
needs it scheduled along the path, which is why `analysis.sim_coupled` repeats
the coupled loop rather than calling `run_pass`.
"""

from dataclasses import dataclass

import numpy as np

from stabsim.compensate import f0_along_path
from stabsim.engagement import arclength_mm

#: Order of the columns in `F0_w`, for the `axes` mask.
AXES = ("x", "y", "z")


@dataclass
class Feedforward:
    """The mean-force schedule, and what it is worth.

    t          (N,)     plan-grid time [s] — the grid `F0_w` is written on
    s_mm       (N,)     arc length along the commanded path [mm]
    F0_w       (N, 3)   ZOA revolution-average cutting force, workpiece frame [N]
    carried_w  (N, 3)   what the motors are actually asked to carry [N] — `F0_w`
                        masked to `axes` and scaled by `gain`
    offset_w   (N, 3)   `G(0) F0`, the static deflection the UNCOMPENSATED pass
                        would sit at [m]. Not applied to anything; reported so
                        the size of the problem is visible, and so the
                        path-offset route can be compared against this one.
    engaged    (N,)     where the tool is cutting
    """

    t: np.ndarray
    s_mm: np.ndarray
    F0_w: np.ndarray
    carried_w: np.ndarray
    offset_w: np.ndarray
    engaged: np.ndarray
    G0_w: np.ndarray
    axes: str = "xy"
    gain: float = 1.0
    ds_mm: float = 2.0

    @property
    def offset_um(self) -> np.ndarray:
        """(N,) magnitude of the deflection the mean force holds the tool at [um]."""
        return np.linalg.norm(self.offset_w, axis=1) * 1e6

    def row(self) -> dict:
        """The scalar summary, for the run summary."""
        eng = self.engaged
        if not eng.any():
            return {"ff_applied": bool(self.gain != 0.0), "ff_axes": self.axes,
                    "ff_gain": float(self.gain), "ff_F0_mean_N": 0.0,
                    "ff_F0_max_N": 0.0, "ff_offset_mean_um": 0.0,
                    "ff_offset_max_um": 0.0}
        f = np.linalg.norm(self.F0_w[eng], axis=1)
        d = self.offset_um[eng]
        return {"ff_applied": bool(self.gain != 0.0),
                "ff_axes": self.axes,
                "ff_gain": float(self.gain),
                "ff_F0_mean_N": float(f.mean()),
                "ff_F0_max_N": float(f.max()),
                "ff_offset_mean_um": float(d.mean()),
                "ff_offset_max_um": float(d.max())}

    def summary(self, ae_mm=None) -> str:
        eng = self.engaged
        if not eng.any():
            return "         the tool never engages - nothing to carry"
        f = np.linalg.norm(self.F0_w[eng], axis=1)
        d = self.offset_um[eng]
        note = ("" if ae_mm is None else
                f"  = {100.0 * d.max() * 1e-3 / float(ae_mm):.1f}% of ae")
        state = ("carried on the motors" if self.gain != 0.0
                 else "NOT applied - the springs carry it")
        return (f"         ZOA mean force  {f.min():6.0f} .. {f.max():6.0f} N "
                f"(mean {f.mean():.0f}) | axes '{self.axes}', gain {self.gain:g}\n"
                f"         it would hold the tool  {d.min():6.1f} .. {d.max():6.1f} um "
                f"off the nominal{note}\n"
                f"         {state}")

    def schedule(self, t_sim) -> np.ndarray:
        """`carried_w` resampled onto the simulation time grid -> (M, 3) [N].

        The plan grid is 1 ms and the simulation runs at 0.2 ms, so this is an
        interpolation onto a finer grid, not a decimation.
        """
        t_sim = np.asarray(t_sim, float).ravel()
        return np.column_stack([np.interp(t_sim, self.t, self.carried_w[:, k])
                                for k in range(3)])

    def save_npz(self, path):
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, t=self.t, s_mm=self.s_mm, F0_w=self.F0_w,
            carried_w=self.carried_w, offset_w=self.offset_w,
            engaged=self.engaged, G0_w=self.G0_w, axes=self.axes,
            gain=self.gain, ds_mm=self.ds_mm)
        return path


def build(setup, receptance, *, mill=None, axes="xy", gain=1.0,
          verbose=False) -> Feedforward:
    """The mean-force schedule for `setup`, on the engagement it already measured.

    `mill` overrides the cutting coefficients, so a run that perturbs Ktc/Krc
    carries the force IT produces rather than the nominal one. `axes` masks which
    components the motors carry — "xy" leaves the tool-axis component alone,
    which is the honest default here because the raster is a constant-depth model
    and nothing in the cut responds to a `z` correction. `gain` scales the whole
    schedule; `gain=0` builds the same object and applies none of it, which is
    what `--no-compensate` uses so the two runs differ in one number.
    """
    m = setup.mill if mill is None else mill
    path = setup.path

    # chip load follows the planned feed profile, so F0 ramps with the lead-in
    # instead of stepping to its full value at t = 0
    fz = m.feed_per_tooth_mm(path.speed_profile_mm_s())
    F0_w, engaged, _ = f0_along_path(
        path.xy_mm, setup.part, m, ds_mm=setup.ds_mm, fz_mm=fz,
        axial_depth_mm=setup.part.height_mm,
        engagement=setup.engagement, verbose=verbose)

    G0_w = receptance.in_workpiece(setup.scene).dc_gain          # [m/N]
    offset_w = F0_w @ G0_w.T                                     # dx0 = G(0) F0
    mask = np.array([ax in axes for ax in AXES], dtype=float)
    carried_w = float(gain) * F0_w * mask

    return Feedforward(t=np.asarray(path.t, float),
                       s_mm=arclength_mm(path.xy_mm),
                       F0_w=F0_w, carried_w=carried_w, offset_w=offset_w,
                       engaged=engaged, G0_w=G0_w, axes=axes,
                       gain=float(gain), ds_mm=setup.ds_mm)
