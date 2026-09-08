"""The identified plant closed on the linearised cut, point by point along a path.

    A_cl = A + B (I - Kc D - Cc C B)^-1 (Kc C + Cc C A)
    max Re(eig A_cl) > 0   ->   unstable

`robotsim.receptance` gives the machine as a state space, `dx = G(s) F` — force
in, TCP deflection out, n states. `stabsim.cut` gives the cut as two matrices
about the planned path, `dF = K dx + C dxd`. This module is the loop between them.

WHY IT IS NOT THE OLD 6x6. With an `M/D/K` arm the closed loop can be written by
inspection: `D_eff = D - a A_cut_0`, `K_eff = K - K_cut`, and the state is
`[x, xd]`. A general receptance has no mass, damping or stiffness matrix to
modify — it may have `n` internal states with no mechanical meaning — so the cut
is closed the way any output feedback is:

    xdot = A x + B dF          dx  = C x + D dF
    dF   = Kc dx + Cc dxd      dxd = C xdot = C A x + C B dF

    -> (I - Kc D - Cc C B) dF = (Kc C + Cc C A) x

which is the same system either way. In THIS workspace the plant is always
`Receptance.from_mdk`, so `D = 0` and `C B = 0` — the state is `[x, xd]`, the
inverse collapses to the identity, and this form is algebraically the classic
6x6. The general form is kept because it costs nothing and it is what makes the
module correct for a plant that is not second order; the `Cc C B` term is a
genuine algebraic loop for any model with relative degree one, and dropping it
there is not a lag but a different and violently unstable system.

The eigenvalues are then those of one `n x n` matrix — `n = 6` here — and they
ARE the poles of the cut-plus-machine system. With no cut closed they are the
poles of `G`, which is the check `--coupling none` runs.

WHAT THE VERDICT MEANS, AND WHAT IT DOES NOT. A positive growth rate at a
finite frequency is chatter; a positive one at ~0 Hz is DIVERGENCE — the cut has
softened the machine until a direction of `K - K_cut` goes negative and the tool
digs in. Only the stiffness term can produce the second kind.

Both cut terms truncate the regenerative delay after one order, so there is no
delay left in the loop: this says nothing about spindle-speed lobes. The
`from_mdk` plant IS passive and symmetric, unlike the measured closed-loop
receptance this module was written against, so an unstable verdict here has to
come from the cut rather than from the machine model — which is what makes the
comparison in this workspace a statement about the cut linearisation.
"""

from dataclasses import dataclass

import numpy as np

from fastsim.config import MillConfig
from fastsim.geometry import Workpiece
from fastsim.metrics import project_onto_polyline
from robotsim.receptance import Receptance
from stabsim.cut import C_cut, F0_cut, K_cut
from stabsim.engagement import (
    Engagement, EngagementGradients, arclength_mm, engagement_angles_along_path,
    engagement_gradients_along_path, fill_short_gaps,
)

#: A dominant eigenvalue below this frequency is divergence, not chatter.
DIVERGENCE_HZ = 0.1

COUPLINGS = ("both", "damping", "stiffness", "none")


# ─────────────────────────────────────────────────────────────────────────────
# The closed loop
# ─────────────────────────────────────────────────────────────────────────────

def closed_loop_matrix(r: Receptance, K=None, C=None) -> np.ndarray:
    """(n, n) state matrix of the plant with the cut closed around it.

    K  (3, 3) [N/m]    process stiffness, dF/ddx     — same frame as `r`
    C  (3, 3) [N s/m]  process damping,  dF/ddxd     — same frame as `r`

    Either may be None (that term open). With both None this is `r.A`, i.e. the
    identified plant on its own.
    """
    A, B, Cm, D = r.A, r.B, r.C, r.D
    if K is None and C is None:
        return A.copy()
    Kc = np.zeros((3, 3)) if K is None else np.asarray(K, dtype=float)
    Cc = np.zeros((3, 3)) if C is None else np.asarray(C, dtype=float)

    if np.any(D) and np.any(Cc):
        raise NotImplementedError(
            "this model is biproper (D != 0), so closing the DAMPING term needs "
            "a dF/dt state — the loop through D is a descriptor system, not an "
            "ODE. Use a fit made with ENFORCE_D0, or coupling='stiffness'.")

    M = np.eye(3) - Kc @ D - Cc @ (Cm @ B)
    if np.linalg.cond(M) > 1e12:
        raise np.linalg.LinAlgError(
            "the algebraic loop I - Kc D - Cc C B is singular: at this depth the "
            "instantaneous feedthrough of the cut cancels the plant's. Reduce ap "
            "or check the cut matrices.")
    return A + B @ np.linalg.solve(M, Kc @ Cm + Cc @ (Cm @ A))


def closed_loop_eigs(r: Receptance, K=None, C=None) -> np.ndarray:
    """(n,) eigenvalues of the closed loop [1/s]."""
    return np.linalg.eigvals(closed_loop_matrix(r, K, C))


def growth_rate(r: Receptance, K=None, C=None) -> float:
    """`max Re(eig)` of the closed loop [1/s]. > 0 means the cut is unstable."""
    return float(np.max(closed_loop_eigs(r, K, C).real))


def dominant_mode(r: Receptance, K=None, C=None) -> tuple:
    """(growth rate [1/s], frequency [Hz]) of the fastest-growing eigenvalue.

    A positive growth rate at ~0 Hz is DIVERGENCE — the cut softens the machine
    until `K - K_cut` loses a positive direction and the tool digs in — which is
    a different failure from chatter and wants a different fix.
    """
    w = closed_loop_eigs(r, K, C)
    k = int(np.argmax(w.real))
    return float(w[k].real), float(abs(w[k].imag)) / (2.0 * np.pi)


def open_loop_growth_rate(r: Receptance) -> float:
    """Growth rate with no cut — the identified plant alone. Should be negative."""
    return float(np.asarray(np.linalg.eigvals(r.A)).real.max())


def critical_depth_mm(r: Receptance, K1, C1, a_max_mm: float = 20.0,
                      n_scan: int = 40, tol_mm: float = 1e-3) -> float:
    """Smallest axial depth [mm] at which this point goes unstable.

    `K1`, `C1` are the cut matrices per MILLIMETRE of depth — both terms are
    linear in `a`, so a depth sweep costs one eigenproblem per trial and no new
    geometry. Coarse scan first, then bisection: the growth rate is not
    guaranteed monotonic in `a` (the stiffness term can restabilise a mode
    before the damping term takes it), so stepping into the first sign change is
    safer than assuming one crossing.

    Returns NaN when the point is stable all the way to `a_max_mm`.
    """
    def g(a):
        try:
            return growth_rate(r, None if K1 is None else a * K1,
                               None if C1 is None else a * C1)
        except (np.linalg.LinAlgError, NotImplementedError):
            return np.inf

    grid = np.linspace(0.0, float(a_max_mm), int(n_scan) + 1)[1:]
    lo = 0.0
    for a in grid:
        if g(a) > 0.0:
            hi = a
            while hi - lo > tol_mm:
                mid = 0.5 * (lo + hi)
                lo, hi = (lo, mid) if g(mid) > 0.0 else (mid, hi)
            return float(hi)
        lo = a
    return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Along a trajectory
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, eq=False)
class StabilityAlongPath:
    """Per-sample stability verdict along a tool path. Millimetres, seconds.

    s_mm         (N,)      arc length of each evaluated point
    xy_mm        (N, 2)    its tool-centre position, workpiece frame
    ae_mm        (N,)      radial engagement there
    engaged      (N,) bool the tool cuts AND the angles are usable
    phi_en/ex    (N,)      entry / exit angle [rad], from the cut geometry
    theta_deg    (N,)      cut orientation in the plant's frame [deg]
    growth_rate  (N,)      max Re(eig) [1/s]; > 0 = unstable
    mode_hz      (N,)      frequency of that eigenvalue [Hz]; ~0 = divergence
    F0_w         (N, 3)    ZOA operating-point force, workpiece frame [N]
    K_p / C_p    (N, 3, 3) the cut matrices as closed, in the PLANT frame
    ap_crit_mm   (N,)      smallest unstable depth, when `critical_depth` ran
    """

    s_mm: np.ndarray
    xy_mm: np.ndarray
    ae_mm: np.ndarray
    engaged: np.ndarray
    theta_deg: np.ndarray
    growth_rate: np.ndarray
    mode_hz: np.ndarray = None
    phi_en: np.ndarray = None
    phi_ex: np.ndarray = None
    F0_w: np.ndarray = None
    K_p: np.ndarray = None
    C_p: np.ndarray = None
    ap_crit_mm: np.ndarray = None
    receptance: Receptance = None
    axial_depth_mm: float = 0.0
    spindle_rpm: float = 0.0
    n_teeth: float = 0.0
    fz_mm: np.ndarray = None
    coupling: str = "both"
    ds_mm: float = 0.0
    engagement: Engagement = None
    gradients: EngagementGradients = None

    def __len__(self) -> int:
        return len(self.s_mm)

    # ── the model's own validity ─────────────────────────────────────────────

    @property
    def tooth_period_s(self) -> float:
        """One tooth period [s] — the delay the two cut terms expand around."""
        if not (self.spindle_rpm and self.n_teeth):
            return float("nan")
        return 60.0 / (self.spindle_rpm * self.n_teeth)

    def omega_T(self) -> np.ndarray:
        """`omega T` per identified mode — the truncation parameter of the cut
        model. The neglected term is roughly `(omega T)^2 / 2` of the cut force."""
        f = np.asarray(self.receptance.modes_hz, dtype=float)
        return 2.0 * np.pi * f * self.tooth_period_s

    def open_loop(self) -> float:
        """Growth rate of the identified plant with no cut [1/s]."""
        return open_loop_growth_rate(self.receptance)

    # ── the verdict ──────────────────────────────────────────────────────────

    @property
    def unstable(self) -> np.ndarray:
        """Engaged points whose dominant mode grows, either way."""
        return self.engaged & (self.growth_rate > 0.0)

    @property
    def diverging(self) -> np.ndarray:
        """Unstable points whose dominant eigenvalue is REAL — the tool digs in
        rather than chatters. Only the stiffness term can produce these."""
        if self.mode_hz is None:
            return np.zeros(len(self), dtype=bool)
        return self.unstable & (self.mode_hz < DIVERGENCE_HZ)

    @property
    def chattering(self) -> np.ndarray:
        """Unstable points that oscillate — the classic verdict."""
        return self.unstable & ~self.diverging

    @property
    def unstable_fraction(self) -> float:
        """Share of the ENGAGED path predicted unstable."""
        n = int(self.engaged.sum())
        return float(self.unstable.sum()) / n if n else 0.0

    def worst(self) -> dict:
        """The most unstable engaged point."""
        if not self.engaged.any():
            return {}
        g = np.where(self.engaged, self.growth_rate, -np.inf)
        k = int(np.argmax(g))
        return {"index": k, "s_mm": float(self.s_mm[k]),
                "xy_mm": self.xy_mm[k].tolist(), "ae_mm": float(self.ae_mm[k]),
                "theta_deg": float(self.theta_deg[k]),
                "growth_rate": float(self.growth_rate[k]),
                "mode_hz": (float("nan") if self.mode_hz is None
                            else float(self.mode_hz[k])),
                "F0_N": (None if self.F0_w is None
                         else float(np.linalg.norm(self.F0_w[k])))}

    def critical_depth(self, a_max_mm: float = 20.0, verbose: bool = False
                       ) -> np.ndarray:
        """(N,) smallest unstable depth at every engaged point [mm], NaN if none.

        Both cut terms are linear in `a`, so this reuses the matrices already
        built at `axial_depth_mm` — no new geometry, one eigenproblem per trial.
        The minimum over the path is the depth the WHOLE cut survives.
        """
        if self.K_p is None and self.C_p is None:
            raise RuntimeError("no cut matrices were stored — coupling='none'")
        a0 = self.axial_depth_mm
        out = np.full(len(self), np.nan)
        for i in np.flatnonzero(self.engaged):
            K1 = None if self.K_p is None else self.K_p[i] / a0
            C1 = None if self.C_p is None else self.C_p[i] / a0
            out[i] = critical_depth_mm(self.receptance, K1, C1, a_max_mm)
        if verbose:
            ok = np.isfinite(out)
            print(f"         critical depth found at {int(ok.sum())}/"
                  f"{int(self.engaged.sum())} engaged points "
                  f"(the rest are stable past {a_max_mm:g} mm)")
        object.__setattr__(self, "ap_crit_mm", out)
        return out

    # ── reporting ────────────────────────────────────────────────────────────

    def summary(self) -> str:
        n_eng = int(self.engaged.sum())
        if not n_eng:
            return "  the tool never engages — nothing to assess"
        g = self.growth_rate[self.engaged]
        w = self.worst()
        ae = self.ae_mm[self.engaged]
        slotting = float(np.mean(ae >= 0.999 * ae.max())) if ae.max() > 0 else 0.0

        angles = ""
        if self.phi_en is not None:
            en = np.degrees(self.phi_en[self.engaged])
            ex = np.degrees(self.phi_ex[self.engaged])
            angles = (f"  phi_en / phi_ex      {en.min():5.1f} .. {en.max():5.1f} / "
                      f"{ex.min():5.1f} .. {ex.max():5.1f} deg   "
                      f"(swept mean {(ex - en).mean():.1f} deg)\n")
        feed = ""
        if self.fz_mm is not None:
            f = self.fz_mm[self.engaged]
            feed = (f"  chip load fz         {f.min():.4f} .. {f.max():.4f} mm/tooth"
                    f"{'  (constant)' if np.ptp(f) < 1e-12 else '  (scheduled)'}\n")
        force = ""
        if self.F0_w is not None:
            n0 = np.linalg.norm(self.F0_w[self.engaged], axis=1)
            force = (f"  ZOA mean force |F0|  {n0.min():6.0f} .. {n0.max():6.0f} N "
                     f"(mean {n0.mean():.0f})\n")

        wT = self.omega_T()
        wT_max = float(np.nanmax(wT)) if np.isfinite(wT).any() else float("nan")
        valid_txt = (f"  omega T (worst mode) {wT_max:.2f} rad  -> delay truncation "
                     f"~{50 * wT_max ** 2:.0f}% of the cut force"
                     f"{'   <-- MARGINAL' if wT_max > 0.7 else ''}\n")

        split = ""
        n_div, n_cht = int(self.diverging.sum()), int(self.chattering.sum())
        if n_div or n_cht:
            split = (f"    of which            {n_cht} chatter, "
                     f"{n_div} DIVERGENCE (real eigenvalue, tool digs in)\n")

        crit = ""
        if self.ap_crit_mm is not None:
            c = self.ap_crit_mm[self.engaged]
            fin = np.isfinite(c)
            # The MINIMUM is the number that matters: below it every point on
            # the path is stable, so it is the depth the whole cut survives.
            crit = (f"  critical depth       {np.nanmin(c):.2f} mm minimum over "
                    f"the path (median {np.nanmedian(c):.2f}), "
                    f"{int((~fin).sum())}/{n_eng} stable past the cap\n"
                    if fin.any() else
                    f"  critical depth       none found - stable past the cap "
                    f"everywhere\n")

        return (
            f"  {len(self)} points evaluated, {n_eng} engaged "
            f"({100 * n_eng / len(self):.0f}% of the path)\n"
            f"  cut terms closed     {self.coupling}\n"
            f"  ae over the cut      {ae.min():5.2f} .. {ae.max():5.2f} mm"
            f"   ({100 * slotting:.0f}% at full immersion)\n" + angles + feed +
            force + valid_txt +
            f"  growth rate          {g.min():+8.3f} .. {g.max():+8.3f} 1/s  "
            f"(open loop {self.open_loop():+.3f})\n"
            f"  PREDICTED UNSTABLE   {100 * self.unstable_fraction:.1f}% of the "
            f"engaged path\n" + split + crit +
            f"  worst point          s = {w['s_mm']:.1f} mm, "
            f"({w['xy_mm'][0]:.1f}, {w['xy_mm'][1]:.1f}) mm, "
            f"ae = {w['ae_mm']:.2f} mm, growth {w['growth_rate']:+.3f} 1/s "
            f"@ {w['mode_hz']:.1f} Hz")

    def save_npz(self, path):
        """Everything a plot or a comparison needs, without the objects."""
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, s_mm=self.s_mm, xy_mm=self.xy_mm, ae_mm=self.ae_mm,
            engaged=self.engaged, theta_deg=self.theta_deg,
            growth_rate=self.growth_rate, mode_hz=self.mode_hz,
            phi_en=self.phi_en, phi_ex=self.phi_ex, F0_w=self.F0_w,
            fz_mm=self.fz_mm, coupling=self.coupling,
            axial_depth_mm=self.axial_depth_mm, spindle_rpm=self.spindle_rpm,
            n_teeth=self.n_teeth, model=str(self.receptance.name),
            ap_crit_mm=(np.zeros(0) if self.ap_crit_mm is None
                        else self.ap_crit_mm))
        return path


def _radial_engagement(xy_mm: np.ndarray, workpiece: Workpiece,
                       radius_mm: float) -> np.ndarray:
    """(N,) radial engagement `ae` [mm] at each tool-centre position.

    Overlap of the tool disc with the material, saturating at full slotting.
    Reported for context — the eigenvalues run off the measured ANGLES.
    """
    from matplotlib.path import Path as MplPath

    ring = workpiece.closed_xy_mm.T
    _, dist, _, _ = project_onto_polyline(xy_mm, ring)
    inside = MplPath(ring).contains_points(xy_mm)
    signed = np.where(inside, -dist, dist)   # + outside the material, - inside
    return np.clip(radius_mm - signed, 0.0, 2.0 * radius_mm)


def _plant_rotation(receptance: Receptance, scene) -> np.ndarray:
    """(3, 3) WORKPIECE -> the frame the receptance is written on."""
    if receptance.frame == "base":
        return np.asarray(scene.R_iw, dtype=float)
    if receptance.frame == "workpiece":
        return np.eye(3)
    raise ValueError(
        f"the plant is in the {receptance.frame!r} frame — the cut is measured in "
        "the workpiece frame, so give it as 'base' (as identified) or convert it "
        "with `in_workpiece(scene)` first")


def stability_along_path(path_xy_mm, part: Workpiece, receptance: Receptance,
                         scene, mill: MillConfig, *, ds_mm: float = 2.0,
                         axial_depth_mm: float = None, spindle_rpm: float = None,
                         fz_mm=None, coupling: str = "both", s_mm=None,
                         erode: bool = True, h_mm: float = 0.01,
                         max_gap_mm: float = 4.0, engagement: Engagement = None,
                         gradients: EngagementGradients = None,
                         fill_k_gaps: bool = False,
                         verbose: bool = True) -> StabilityAlongPath:
    """Growth rate at every `ds_mm` along a tool-centre path.

    path_xy_mm   (N, 2) tool-centre positions, WORKPIECE frame [mm] — the
                 NOMINAL path, which is the geometry the cut is linearised about
    part         the outline, for the engagement geometry
    receptance   the identified plant, `frame="base"` as loaded (or "workpiece")
    scene        supplies `R_iw`, which carries the cut into the plant's frame
    mill         the cutter and the cut: D, teeth, Ktc/Krc/Kac, rpm, feed
    ds_mm        arc-length spacing of the evaluated points — one engagement
                 sweep step and one eigenproblem each
    fz_mm        chip load: a scalar, or one value per FULL-length path point to
                 follow the scheduled feed. Defaults to `mill.feed_per_tooth_mm()`
    coupling     which cut terms close the loop: "both" | "damping" |
                 "stiffness" | "none"
    engagement, gradients
                 precomputed sweeps on the SAME downsampled points — pass them
                 to sweep rpm or ap without re-measuring the geometry

    The angles come from the actual tool-disc/material intersection, so a curved
    boundary, a breakout or ground already cut are all handled.
    """
    if coupling not in COUPLINGS:
        raise ValueError(f"coupling must be one of {COUPLINGS}, got {coupling!r}")
    use_damping = coupling in ("both", "damping")
    use_stiffness = coupling in ("both", "stiffness")

    xy_all = np.atleast_2d(np.asarray(path_xy_mm, dtype=float))
    s_all = arclength_mm(xy_all) if s_mm is None else np.asarray(s_mm, dtype=float)
    step = max(1, int(round(ds_mm / max(np.median(np.diff(s_all)), 1e-9))))
    xy, s = xy_all[::step], s_all[::step]

    R_mm = mill.radius_mm
    a_mm = part.height_mm if axial_depth_mm is None else float(axial_depth_mm)
    rpm = mill.spindle_rpm if spindle_rpm is None else float(spindle_rpm)

    fz = mill.feed_per_tooth_mm() if fz_mm is None else fz_mm
    fz = np.asarray(fz, dtype=float)
    if fz.ndim == 0:
        fz = np.full(len(xy), float(fz))
    elif len(fz) == len(xy_all):
        fz = fz[::step]
    elif len(fz) != len(xy):
        raise ValueError(f"fz_mm has {len(fz)} points but the path has "
                         f"{len(xy_all)} ({len(xy)} evaluated)")

    if verbose:
        print(f"sweep    {len(xy)} points every {ds_mm:g} mm along "
              f"{s_all[-1]:.1f} mm of path (stride {step})")

    if engagement is None:
        engagement = engagement_angles_along_path(xy, part, R_mm, erode=erode,
                                                  s_mm=s, verbose=verbose)
    elif len(engagement) != len(xy):
        raise ValueError(f"engagement has {len(engagement)} points, the sweep "
                         f"has {len(xy)} — compute it on the same samples")
    if use_stiffness and gradients is None:
        gradients = engagement_gradients_along_path(xy, part, R_mm, h_mm=h_mm,
                                                    erode=erode, s_mm=s,
                                                    nominal=engagement,
                                                    fill_gaps=fill_k_gaps,
                                                    max_gap_mm=max_gap_mm,
                                                    verbose=verbose)

    # A node in CONTACT whose angles could not be measured is still cutting —
    # fill it from its neighbours rather than calling it free flight.
    phi_en, ok_en = fill_short_gaps(s, engagement.phi_en, engagement.valid, max_gap_mm)
    phi_ex, ok_ex = fill_short_gaps(s, engagement.phi_ex, engagement.valid, max_gap_mm)
    engaged = engagement.contact & ok_en & ok_ex
    if verbose and (engaged & ~engagement.valid).any():
        print(f"         filled {int((engaged & ~engagement.valid).sum())} "
              f"unmeasurable node(s) in contact by interpolation")

    R_pw = _plant_rotation(receptance, scene)       # workpiece -> plant frame
    ae = _radial_engagement(xy, part, R_mm)

    n = len(xy)
    growth = np.zeros(n)
    mode_hz = np.zeros(n)
    theta = np.full(n, np.nan)
    F0_w = np.zeros((n, 3))
    K_p = np.zeros((n, 3, 3)) if use_stiffness else None
    C_p = np.zeros((n, 3, 3)) if use_damping else None

    g_air, hz_air = dominant_mode(receptance)       # in air: the machine alone
    for i in range(n):
        if not engaged[i]:
            growth[i], mode_hz[i] = g_air, hz_air
            continue

        R_wc = np.eye(3)                            # cut -> workpiece
        R_wc[:2, :2] = engagement.R_wc[i]
        R = R_pw @ R_wc                             # cut -> plant frame
        args = (phi_en[i], phi_ex[i], mill.Ktc, mill.Krc)

        F0_w[i] = R_wc @ F0_cut(fz[i], a_mm, mill.n_teeth, *args, mill.Kac)
        if use_damping:
            C_p[i] = R @ C_cut(a_mm, rpm, *args, mill.Kac) @ R.T
        if use_stiffness and gradients.valid[i]:
            K_p[i] = R @ K_cut(fz[i], a_mm, mill.n_teeth, *args[:2],
                               gradients.dphi[i], *args[2:], mill.Kac) @ R.T

        growth[i], mode_hz[i] = dominant_mode(
            receptance, None if K_p is None else K_p[i],
            None if C_p is None else C_p[i])
        theta[i] = np.degrees(np.arctan2(R[1, 0], R[0, 0]))

    return StabilityAlongPath(
        s_mm=s, xy_mm=xy, ae_mm=ae, engaged=engaged, theta_deg=theta,
        growth_rate=growth, mode_hz=mode_hz, phi_en=phi_en, phi_ex=phi_ex,
        F0_w=F0_w, K_p=K_p, C_p=C_p, receptance=receptance, axial_depth_mm=a_mm,
        spindle_rpm=rpm, n_teeth=mill.n_teeth, fz_mm=fz, coupling=coupling,
        ds_mm=float(ds_mm), engagement=engagement, gradients=gradients)
