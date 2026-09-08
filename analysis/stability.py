"""The linear prediction: engagement measured once, eigenvalues at every node.

    setup = prepare(cfg, ds_mm=2.0)          two Shapely sweeps of the path
    stab  = predict(setup, receptance)       one eigenproblem per node

WHY THE GEOMETRY IS A SEPARATE STEP

`engagement_gradients_along_path` costs two extra full sweeps of the path on top
of the angle sweep, and none of that geometry depends on the plant, on the
spindle speed or on the cutting coefficients — it is a property of where the tool
is relative to the part. `prepare` measures it once and `predict` reuses it,
which is exactly what the upstream docstring invites. A prediction then costs one
6x6 eigenproblem per path node, a fraction of a second for the whole pass, and
the same `Setup` also feeds `analysis.feedforward` so the mean force and the
stability verdict are built on one measurement of the geometry rather than two.

WHAT THE VERDICT IS, AND WHAT IT IS NOT

`growth_rate` is `max Re(eig)` of the cut closed on the plant at that node:
positive means the deviation from the nominal path grows. It is a verdict about
one operating point, evaluated node by node along the trajectory, and the two
places it moves fastest on this job are the entry and the exit, where the tool is
at partial radial engagement.

It is NOT a lobe diagram. Both cut terms truncate the regenerative delay after
one order, so there is no `e^{-jwT}` left in the loop and the criterion cannot
produce speed lobes. Read a growth rate as "at this speed, at this depth", never
as "at this speed rather than that one".

`ap_crit_mm` — the smallest depth at which each node goes unstable — is the more
useful output of the two, and `critical_depth` gets it almost free because both
cut terms are linear in the depth. It is in millimetres, it is what a planner
actually picks, and it needs no threshold.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from fastsim.config import MillConfig
from fastsim.geometry import Workpiece
from stabsim.engagement import (Engagement, EngagementGradients,
                                engagement_angles_along_path,
                                engagement_gradients_along_path)
from stabsim.stability import StabilityAlongPath, stability_along_path

#: Depth the critical-depth search scans to [mm]. `critical_depth_mm` returns NaN
#: for a node still stable at the cap, so the cap must sit ABOVE anything
#: reachable or the output silently censors itself — this arm's critical depth is
#: already well past the vendored default of 20 mm over most of the speed range.
AP_CRIT_MAX_MM = 120.0

#: Fraction of the engaged span trimmed at each end before the prediction is
#: reduced to a single number. Matches `analysis.report.TRIM`, so the prediction
#: and the measurement describe the same part of the cut — see `prediction_row`.
TRIM = 0.15


@dataclass
class Setup:
    """The job, with its engagement geometry already measured."""

    cfg: object
    mill: MillConfig
    part: Workpiece
    scene: object
    path: object
    waypoints: np.ndarray
    engagement: Engagement
    gradients: Optional[EngagementGradients]
    ds_mm: float = 2.0

    @property
    def xy_mm(self):
        return self.path.xy_mm

    def summary(self) -> str:
        eng = self.engagement
        x0, y0, x1, y1 = self.part.bbox_mm
        return (f"job      {self.part.name} {x1 - x0:g}x{y1 - y0:g}x"
                f"{self.part.height_mm:g} mm | "
                f"D {self.mill.diameter_mm:g} mm, {self.mill.n_teeth} teeth, "
                f"ae {self.mill.radial_engagement_mm:g} mm, "
                f"ap {self.part.height_mm:g} mm\n"
                f"         {self.mill.spindle_rpm:g} rpm, "
                f"{self.mill.feed_mm_s:g} mm/s -> "
                f"{self.mill.feed_per_tooth_mm():.4f} mm/tooth | "
                f"Ktc {self.mill.Ktc:g}, Krc {self.mill.Krc:g} N/mm^2\n"
                f"geometry {len(eng)} nodes every {self.ds_mm:g} mm, "
                f"{int(eng.contact.sum())} of them in contact")


def prepare(cfg, ds_mm=2.0, *, erode=True, h_mm=0.01, max_gap_mm=4.0,
            fill_k_gaps=False, verbose=True) -> Setup:
    """Build the job and measure its engagement geometry once."""
    mill, part, scene, waypoints, path = cfg.build()
    xy = path.xy_mm

    s_all = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    step = max(1, int(round(ds_mm / max(np.median(np.diff(s_all)), 1e-9))))
    xy_s, s_s = xy[::step], s_all[::step]

    engagement = engagement_angles_along_path(
        xy_s, part, mill.radius_mm, erode=erode, s_mm=s_s, verbose=verbose)
    gradients = engagement_gradients_along_path(
        xy_s, part, mill.radius_mm, h_mm=h_mm, erode=erode, s_mm=s_s,
        nominal=engagement, fill_gaps=fill_k_gaps, max_gap_mm=max_gap_mm,
        verbose=verbose)

    return Setup(cfg=cfg, mill=mill, part=part, scene=scene, path=path,
                 waypoints=waypoints, engagement=engagement,
                 gradients=gradients, ds_mm=ds_mm)


def predict(setup: Setup, receptance, *, mill: MillConfig = None,
            axial_depth_mm=None, coupling="both",
            verbose=False) -> StabilityAlongPath:
    """One stability verdict per node, on the cached geometry.

    `mill` overrides the config's cutting parameters as a whole rather than
    field by field. That is deliberate: `stability_along_path` derives chip load
    from the mill's OWN rpm, so changing the speed through a separate argument
    would move the process-damping term while leaving `fz` at the old speed.
    Replace the `MillConfig` and the two stay consistent by construction.
    """
    m = setup.mill if mill is None else mill
    return stability_along_path(
        setup.xy_mm, setup.part, receptance, setup.scene, m,
        ds_mm=setup.ds_mm, axial_depth_mm=axial_depth_mm,
        coupling=coupling, engagement=setup.engagement,
        gradients=setup.gradients, verbose=verbose)


def critical_depth(stab: StabilityAlongPath, a_max_mm=AP_CRIT_MAX_MM):
    """`ap_crit` per node [mm], NaN where the node is stable at the cap.

    Returns NaNs rather than raising when the prediction was run with
    `coupling='none'`, which stores no cut matrices to scale.
    """
    try:
        return np.asarray(stab.critical_depth(a_max_mm=a_max_mm), float)
    except (RuntimeError, np.linalg.LinAlgError):
        return np.full(len(stab), np.nan)


def table(stab: StabilityAlongPath, ap_crit=None) -> list:
    """One row per path node - the along-the-trajectory evaluation, for CSV.

    This is the raw prediction: everything the criterion said, where it said it,
    with no reduction to a verdict. `F0_x/y/z` is the surrogate linear model's
    revolution-average force at that node, which `analysis.forces` compares
    against what the engine actually produced.
    """
    n = len(stab)
    ap_crit = critical_depth(stab) if ap_crit is None else np.asarray(ap_crit, float)
    F0 = (np.zeros((n, 3)) if stab.F0_w is None else np.asarray(stab.F0_w, float))
    mode = (np.full(n, np.nan) if stab.mode_hz is None
            else np.asarray(stab.mode_hz, float))
    fz = (np.full(n, np.nan) if stab.fz_mm is None
          else np.broadcast_to(np.asarray(stab.fz_mm, float), (n,)))
    phi_en = np.full(n, np.nan) if stab.phi_en is None else stab.phi_en
    phi_ex = np.full(n, np.nan) if stab.phi_ex is None else stab.phi_ex

    rows = []
    for i in range(n):
        rows.append({
            "s_mm": float(stab.s_mm[i]),
            "x_mm": float(stab.xy_mm[i, 0]),
            "y_mm": float(stab.xy_mm[i, 1]),
            "ae_mm": float(stab.ae_mm[i]),
            "engaged": bool(stab.engaged[i]),
            "fz_mm": float(fz[i]),
            "phi_en_deg": float(np.degrees(phi_en[i])),
            "phi_ex_deg": float(np.degrees(phi_ex[i])),
            "growth_rate_1_s": float(stab.growth_rate[i]),
            "mode_hz": float(mode[i]),
            "unstable": bool(stab.unstable[i]),
            "chattering": bool(stab.chattering[i]),
            "diverging": bool(stab.diverging[i]),
            "ap_crit_mm": float(ap_crit[i]),
            "F0_x_N": float(F0[i, 0]),
            "F0_y_N": float(F0[i, 1]),
            "F0_z_N": float(F0[i, 2]),
            "F0_mag_N": float(np.linalg.norm(F0[i])),
        })
    return rows


def prediction_row(stab: StabilityAlongPath, ap_crit=None) -> dict:
    """The prediction reduced to scalars, for the run summary.

    THE ENTRY NODE OWNS THE MAXIMUM, SO THE TRIMMED PAIR IS REPORTED TOO.

    The first engaged node is the tool part-way into the material — partial `ae`,
    and reliably the largest growth rate on the path. The measurement side
    (`analysis.report.chatter_metrics`) throws that span away: it trims `TRIM`
    off each end of the engaged samples before fitting an envelope, because the
    entry and exit transients are forced motion and not the regenerative growth
    the eigenvalue describes. Pairing an untrimmed maximum with a trimmed
    measurement compares different parts of the cut, so `growth_max_trim` applies
    the same trim and `growth_entry` names the node responsible rather than
    quietly dropping it — the entry is a real risk moment, it just is not the one
    the time-domain fit reports on.
    """
    eng = np.asarray(stab.engaged, bool)
    worst = stab.worst() if eng.any() else {}
    g = stab.growth_rate[eng] if eng.any() else np.array([np.nan])

    k = int(TRIM * len(g))
    g_trim = g[k:len(g) - k] if len(g) - 2 * k >= 1 else g

    ap_crit = critical_depth(stab) if ap_crit is None else np.asarray(ap_crit, float)
    c = ap_crit[eng] if eng.any() else np.array([np.nan])
    c_trim = c[k:len(c) - k] if len(c) - 2 * k >= 1 else c
    ok = np.isfinite(c)

    return {
        # the decision variable: the depth the WHOLE engaged cut survives
        "pred_ap_crit_min_mm": float(np.nanmin(c)) if ok.any() else np.nan,
        "pred_ap_crit_trim_mm": (float(np.nanmin(c_trim))
                                 if np.isfinite(c_trim).any() else np.nan),
        "pred_ap_crit_median_mm": float(np.nanmedian(c)) if ok.any() else np.nan,
        "pred_ap_crit_found": int(ok.sum()),
        "pred_unstable": bool(stab.unstable.any()),
        "pred_unstable_fraction": float(stab.unstable_fraction),
        "pred_growth_max_1_s": float(np.nanmax(g)),
        "pred_growth_median_1_s": float(np.nanmedian(g)),
        "pred_growth_entry_1_s": float(g[0]),
        "pred_growth_max_trim_1_s": float(np.nanmax(g_trim)),
        # the criterion's own test — ANY engaged node above zero — applied to the
        # same span the time-domain measurement is fitted over
        "pred_unstable_trim": bool(np.nanmax(g_trim) > 0.0),
        "pred_mode_hz": float(worst.get("mode_hz", np.nan)),
        "pred_chattering_nodes": int(stab.chattering.sum()),
        "pred_diverging_nodes": int(stab.diverging.sum()),
        "pred_engaged_nodes": int(eng.sum()),
        "pred_open_loop_growth_1_s": float(stab.open_loop()),
        "fz_mm": float(np.nanmedian(stab.fz_mm)),
        "spindle_rpm": float(stab.spindle_rpm),
        "axial_depth_mm": float(stab.axial_depth_mm),
    }
