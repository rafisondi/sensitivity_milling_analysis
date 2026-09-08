"""Aim the command off the nominal path so the mean force lands the tool ON it.

    F0(s)   the ZOA mean cutting force along the path        (stabsim.cut)
    dx0(s)  = G(0) F0(s), the deflection it holds            (the plant)
    cmd(s)  = nominal(s) - dx0(s)

An offline, feed-forward correction: nothing is measured during the cut and no
controller is involved. It removes exactly one thing — the QUASI-STATIC offset
the tooth-period-average force holds the tool at — and nothing else. The
tooth-passing ripple, the entry and exit transients and any chatter are all
still there afterwards, because none of them is `G(0) F0`.

WHY THE ENGAGEMENT IS MEASURED ON THE NOMINAL PATH, and why that is a fixed
point rather than an iteration. The force depends on where the tool ACTUALLY
cuts, not on where it is commanded. Uncompensated, the tool sits at
`nominal + dx0`, so its engagement is not the nominal one — but the whole point
of the correction is that the compensated tool lands back on the nominal, which
is therefore the geometry it cuts. Measuring `F0` there closes the loop on
itself: `F0(nominal) -> dx0 -> cmd = nominal - dx0 -> tool at nominal`. One
shot, no iteration, and the fixed point is the answer rather than the limit of
one.

WHAT LIMITS IT. Three things, in the order they bite:

* `G(0)` has to be right. On the `from_mdk` plant used here it is exactly
  `K^-1` of the linearisation, so it is not extrapolated the way a measured fit
  starting at 1 Hz would be — but it IS taken at one frozen pose, and the arm's
  compliance drifts along the path. `--gain` scales the whole correction, which
  is the honest knob when that DC gain is uncertain: 0.5 says "I believe half of
  it".
* The ZOA mean force is a model, not the dexel engine's force. Whatever the two
  disagree by is a residual this cannot remove — which is exactly what makes the
  compensated run a test OF the force model, since the plant is the same on both
  sides of the comparison.
* `dx0` is quasi-static. Where `F0` steps — an entry, an exit, a corner — the
  plant rings at its own modes, and a static correction cannot cancel a
  transient. Expect the residual to be concentrated there.

AXES. The in-plane correction is real; the axial one is not, in this simulator.
The raster is a constant-depth model, so a commanded `z` shift changes what the
report says about `z` and nothing about the cut. `axes="xy"` is the default for
that reason; `"xyz"` is there for exporting a command to a machine that does
respond in `z`.
"""

from dataclasses import dataclass, replace

import numpy as np

from fastsim.config import MillConfig
from fastsim.geometry import Workpiece
from robotsim.receptance import Receptance
from robotsim.trajectory import OperationalPath
from stabsim.cut import F0_cut
from stabsim.engagement import (
    Engagement, arclength_mm, engagement_angles_along_path, fill_short_gaps,
)

AXES = "xyz"


def _blend(s_q, s_nodes, values, valid):
    """(interpolated values, nearest node index) at each query arc length.

    Linear in `s` where BOTH bracketing nodes are engaged, otherwise the nearer
    node is held. The engagement is only measured every `ds_mm`, so a plain
    lookup would make every quantity a staircase at that spacing; interpolating
    across a breakout would instead invent geometry that was never measured,
    hence the hold.
    """
    s_q = np.asarray(s_q, dtype=float)
    hi = np.clip(np.searchsorted(s_nodes, s_q), 1, len(s_nodes) - 1)
    lo = hi - 1
    w = np.clip((s_q - s_nodes[lo]) / np.maximum(s_nodes[hi] - s_nodes[lo], 1e-12),
                0.0, 1.0)
    near = np.where(w < 0.5, lo, hi)

    shape = (-1,) + (1,) * (values.ndim - 1)
    wr = w.reshape(shape)
    lin = (1.0 - wr) * values[lo] + wr * values[hi]
    both = (valid[lo] & valid[hi]).reshape(shape)
    return np.where(both, lin, values[near]), near


def f0_along_path(xy_mm, part: Workpiece, mill: MillConfig, ds_mm: float = 2.0,
                  fz_mm=None, axial_depth_mm: float = None, s_mm=None,
                  erode: bool = True, max_gap_mm: float = 4.0,
                  engagement: Engagement = None, verbose: bool = False):
    """(F0_w (N, 3) [N], engaged (N,), engagement) on EVERY point of `xy_mm`.

    The engagement is swept every `ds_mm` — one polygon boolean each, so the
    spacing is what the cost is — and the angles are then interpolated onto the
    full path, where `F0` is rebuilt per sample. That way the force varies as
    smoothly as the geometry does instead of stepping at the sweep spacing.
    """
    xy_mm = np.atleast_2d(np.asarray(xy_mm, dtype=float))
    s_all = arclength_mm(xy_mm) if s_mm is None else np.asarray(s_mm, dtype=float)
    a_mm = part.height_mm if axial_depth_mm is None else float(axial_depth_mm)

    fz = mill.feed_per_tooth_mm() if fz_mm is None else fz_mm
    fz = np.asarray(fz, dtype=float)
    fz = np.full(len(xy_mm), float(fz)) if fz.ndim == 0 else fz
    if len(fz) != len(xy_mm):
        raise ValueError(f"fz_mm has {len(fz)} points, the path has {len(xy_mm)}")

    step = max(1, int(round(ds_mm / max(np.median(np.diff(s_all)), 1e-9))))
    xy, s = xy_mm[::step], s_all[::step]
    if engagement is None:
        engagement = engagement_angles_along_path(xy, part, mill.radius_mm,
                                                  erode=erode, s_mm=s,
                                                  verbose=verbose)
    phi_en, ok_en = fill_short_gaps(s, engagement.phi_en, engagement.valid, max_gap_mm)
    phi_ex, ok_ex = fill_short_gaps(s, engagement.phi_ex, engagement.valid, max_gap_mm)
    usable = engagement.contact & ok_en & ok_ex
    if verbose:
        print(f"         F0 from {len(xy)} nodes every {ds_mm:g} mm, "
              f"{int(usable.sum())} of them cutting")

    en, near = _blend(s_all, s, phi_en, usable)
    ex, _ = _blend(s_all, s, phi_ex, usable)
    R2, _ = _blend(s_all, s, engagement.R_wc, usable)
    engaged = usable[near]

    # F0_cut is elementwise, so the whole path goes through in one call; the
    # angles are neutralised before the gate so NaNs cannot leak into it.
    f0_cut = F0_cut(fz, a_mm, mill.n_teeth, np.nan_to_num(en), np.nan_to_num(ex),
                    mill.Ktc, mill.Krc, mill.Kac).T          # (N, 3), cut frame
    R = np.zeros((len(xy_mm), 3, 3))
    R[:, :2, :2] = R2
    R[:, 2, 2] = 1.0
    F0_w = np.where(engaged[:, None], np.einsum("kij,kj->ki", R, f0_cut), 0.0)
    return F0_w, engaged, engagement


@dataclass
class Compensation:
    """A command aimed off the nominal path, and what it was built from.

    path      the compensated command; its `nom_w`/`nom_i` are the nominal, so
              every error report downstream still measures against the geometry
              the part is supposed to end up with
    nominal   the path as planned
    F0_w      (N, 3) ZOA mean cutting force on the path grid, workpiece frame [N]
    offset_w  (N, 3) the static deflection it holds, `G_d(0) F0` [m]
    applied_w (N, 3) what was actually subtracted from the command [m] — the
              same thing masked to `axes` and scaled by `gain`
    G0_w      (3, 3) the DC gain used [m/N], workpiece frame
    """

    path: OperationalPath
    nominal: OperationalPath
    F0_w: np.ndarray
    offset_w: np.ndarray
    applied_w: np.ndarray
    engaged: np.ndarray
    G0_w: np.ndarray
    axes: str = "xy"
    gain: float = 1.0
    ds_mm: float = 2.0
    engagement: Engagement = None
    receptance: Receptance = None

    @property
    def offset_mm(self) -> np.ndarray:
        """(N,) magnitude of the correction actually applied [mm]."""
        return np.linalg.norm(self.applied_w, axis=1) * 1e3

    def summary(self) -> str:
        eng = self.engaged
        if not eng.any():
            return "  the tool never engages — nothing to compensate"
        f = np.linalg.norm(self.F0_w[eng], axis=1)
        d = self.offset_mm[eng]
        k = int(np.argmax(d))
        worst = np.where(eng)[0][k]
        full = np.linalg.norm(self.offset_w[eng], axis=1) * 1e3
        return (
            f"  correction           G_d(0) x ZOA mean force | axes '{self.axes}'"
            f" | gain {self.gain:g}\n"
            f"  ZOA mean force |F0|  {f.min():6.0f} .. {f.max():6.0f} N "
            f"(mean {f.mean():.0f})\n"
            f"  static deflection    {full.min():6.3f} .. {full.max():6.3f} mm "
            f"(mean {full.mean():.3f})   <- what the mean force holds\n"
            f"  command aimed off    {d.min():6.3f} .. {d.max():6.3f} mm "
            f"(mean {d.mean():.3f})   <- what is subtracted\n"
            f"  worst at             s = {self.arclength_mm[worst]:.1f} mm, "
            f"({self.nominal.xy_mm[worst, 0]:.1f}, "
            f"{self.nominal.xy_mm[worst, 1]:.1f}) mm\n"
            f"  G_d(0) [um/N], wp    "
            + np.array2string(self.G0_w * 1e6, precision=2, prefix=" " * 23))

    @property
    def arclength_mm(self) -> np.ndarray:
        return arclength_mm(self.nominal.xy_mm)

    def save_npz(self, path):
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, s_mm=self.arclength_mm, nominal_w=self.nominal.s_w,
            command_w=self.path.s_w, F0_w=self.F0_w, offset_w=self.offset_w,
            applied_w=self.applied_w, engaged=self.engaged, G0_w=self.G0_w,
            axes=self.axes, gain=self.gain, ds_mm=self.ds_mm)
        return path


def compensate(cfg, receptance: Receptance, ds_mm: float = 2.0,
               axes: str = "xy", gain: float = 1.0, erode: bool = True,
               verbose: bool = True) -> Compensation:
    """Build the off-nominal command for the cut `cfg` plans.

    The chip load follows the PLANNED feed profile, so `F0` ramps with the
    quintic instead of stepping to its full value at t = 0.
    """
    mill, part, scene, _, path = cfg.build()
    nominal = (replace(path, s_w=path.nom_w, s_i=path.nom_i, nom_w=None, nom_i=None)
               if path.has_nominal else path)

    fz = mill.feed_per_tooth_mm(nominal.speed_profile_mm_s())
    F0_w, engaged, eng = f0_along_path(nominal.xy_mm, part, mill, ds_mm=ds_mm,
                                       fz_mm=fz, erode=erode, verbose=verbose)

    G0_w = receptance.in_workpiece(scene).dc_gain              # [m/N]
    offset_w = F0_w @ G0_w.T                                   # dx0 = G(0) F0
    mask = np.array([ax in axes for ax in AXES], dtype=float)
    applied_w = float(gain) * offset_w * mask

    s_w = nominal.s_w - applied_w
    s_i = s_w @ np.asarray(scene.R_iw, dtype=float).T + scene.origin_i
    command = replace(nominal, s_w=s_w, s_i=s_i,
                      nom_w=nominal.s_w.copy(), nom_i=nominal.s_i.copy())

    return Compensation(path=command, nominal=nominal, F0_w=F0_w,
                        offset_w=offset_w, applied_w=applied_w, engaged=engaged,
                        G0_w=G0_w, axes=axes, gain=float(gain), ds_mm=ds_mm,
                        engagement=eng, receptance=receptance)
