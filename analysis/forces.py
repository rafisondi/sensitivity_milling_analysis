"""The surrogate's mean force against the engine's, revolution by revolution.

    F0(s)   the linear model's revolution-average cutting force  [N]
    F_sim   the dexel engine's force, averaged over one spindle revolution [N]

WHAT IS BEING COMPARED, AND WHY THIS IS THE FAIR COMPARISON

`F0` is the tooth-period average the cut is linearised about — one vector per
path node, with no tooth passing in it at all. The engine's force is the full
history at several kHz, dominated by the tooth-passing ripple. Comparing them
sample against sample would charge the model for a feature it never claimed to
have, and gating on `|F| > threshold` would make it worse: keeping only the upper
part of each tooth cycle biases the ENGINE's mean upward while the surrogate,
being already an average, keeps its own.

Averaging the simulated force over ONE FULL SPINDLE REVOLUTION removes exactly
that feature and leaves the quantity `F0` estimates. A revolution rather than one
tooth period also absorbs tooth-to-tooth variation. It must be a MEAN, not a
rolling median: the median of a skewed in-cut waveform is not its mean, and a
component can read 0 N under a median while its revolution mean is tens of
newtons.

WHAT THIS TESTS, AND WHAT IT DOES NOT

It tests the force model and the engagement geometry, and nothing else. It does
NOT test `K_cut`, `C_cut` or the delay truncation — those live in the growth
rate. It is the low-noise half of the comparison for that reason: on the plateau
of this job the two agree to a fraction of a percent componentwise, so a
disagreement is informative rather than lost in scatter.

MAGNITUDE AND DIRECTION ARE REPORTED SEPARATELY. A force that is 10 % too large
and one that is 10 degrees off point at different causes — the first at the
coefficients or the depth, the second at the engagement angles or the Ktc/Krc
ratio. Collapsing them into one norm hides which.

THE PLATEAU, AND THEN THE ENDS. `plateau_error` scores the middle of the engaged
span, trimmed at both ends, so a transient error is not reported as a steady one.
`entry_exit_error` scores the two trimmed-away spans separately, because on this
job they are the interesting part: they are where `F0` changes, where the
engagement is partial, and where a steady-state model has the least to say.
"""

import numpy as np

from analysis.report import TRIM

#: The tool counts as cutting where the revolution-averaged force exceeds this
#: fraction of the pass's OWN peak. A fraction rather than an absolute newton
#: threshold, so the mask adapts to the operating point instead of defining a
#: light cut away as air.
ENGAGED_FRAC = 0.5

#: Below this the pass never cut at all, and a fraction of its peak would be a
#: fraction of numerical noise.
ENGAGED_FLOOR_N = 1.0


def revolution_average(run, mill):
    """(t (M,), F (M, 3) [N], s (M,) [mm]) - the simulated force, one sample per rev.

    The arc length is measured along the COMMANDED tool centre, which is the same
    coordinate `StabilityAlongPath` indexes `F0` by. That is what lets the two be
    put on one axis.
    """
    res = run.result
    t = np.asarray(res.t, float)
    F = np.asarray(res.force_w, float)
    if len(t) < 2:
        return None, None, None
    n = max(1, int(round(float(mill.tooth_period_s) * mill.n_teeth
                         / (t[1] - t[0]))))          # samples per REVOLUTION
    m = len(t) // n
    if m < 2:
        return None, None, None

    xy = run.commanded_w_mm()
    s = np.concatenate([[0.0],
                        np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    return (t[:m * n].reshape(m, n).mean(1),
            F[:m * n].reshape(m, n, 3).mean(1),
            s[:m * n].reshape(m, n).mean(1))


def _paired(run, stab, mill):
    """(t, s, F_sim, F_lin, cutting mask) with `F0` interpolated onto the rev grid."""
    t_r, F_r, s_r = revolution_average(run, mill)
    if t_r is None:
        return None
    F0 = np.column_stack([np.interp(s_r, stab.s_mm, stab.F0_w[:, j])
                          for j in range(3)])
    mag = np.linalg.norm(F_r, axis=1)
    peak = float(mag.max()) if mag.size else 0.0
    cut = (mag > ENGAGED_FRAC * peak) if peak >= ENGAGED_FLOOR_N \
        else np.zeros(len(mag), bool)
    return t_r, s_r, F_r, F0, cut


def table(run, stab, mill) -> list:
    """One row per spindle revolution: both forces, side by side, along the path.

    This is the along-the-trajectory version of the force comparison, and it is
    where the entry and exit show up — the two curves separate exactly where `F0`
    is changing.
    """
    paired = _paired(run, stab, mill)
    if paired is None:
        return []
    t_r, s_r, F_r, F0, cut = paired
    rows = []
    for i in range(len(t_r)):
        err = F0[i] - F_r[i]
        ns = float(np.linalg.norm(F_r[i]))
        rows.append({
            "t_s": float(t_r[i]),
            "s_mm": float(s_r[i]),
            "cutting": bool(cut[i]),
            "F_sim_x_N": float(F_r[i, 0]), "F_sim_y_N": float(F_r[i, 1]),
            "F_sim_z_N": float(F_r[i, 2]), "F_sim_mag_N": ns,
            "F_lin_x_N": float(F0[i, 0]), "F_lin_y_N": float(F0[i, 1]),
            "F_lin_z_N": float(F0[i, 2]),
            "F_lin_mag_N": float(np.linalg.norm(F0[i])),
            "err_x_N": float(err[0]), "err_y_N": float(err[1]),
            "err_z_N": float(err[2]),
            "err_mag_N": float(np.linalg.norm(err)),
            "rel_err": float(np.linalg.norm(err) / ns) if ns > 0 else np.nan,
        })
    return rows


def _score(F_sim, F_lin, prefix) -> dict:
    """Magnitude, direction and componentwise error over a set of revolutions."""
    blank = {f"{prefix}_{k}": np.nan for k in
             ("F_sim_N", "F_lin_N", "abs_err_N", "rel_err", "mag_err",
              "angle_err_deg", "err_x_N", "err_y_N", "err_z_N")}
    blank[f"{prefix}_n_rev"] = 0
    if len(F_sim) < 1:
        return blank

    fs, fl = F_sim.mean(axis=0), F_lin.mean(axis=0)
    ns, nl = np.linalg.norm(fs), np.linalg.norm(fl)
    if ns <= 0 or nl <= 0:
        return blank
    cos = float(np.clip(fs @ fl / (ns * nl), -1.0, 1.0))
    return {
        f"{prefix}_F_sim_N": float(ns),
        f"{prefix}_F_lin_N": float(nl),
        f"{prefix}_abs_err_N": float(np.linalg.norm(fl - fs)),
        f"{prefix}_rel_err": float(np.linalg.norm(fl - fs) / ns),
        f"{prefix}_mag_err": float((nl - ns) / ns),
        f"{prefix}_angle_err_deg": float(np.degrees(np.arccos(cos))),
        f"{prefix}_err_x_N": float(fl[0] - fs[0]),
        f"{prefix}_err_y_N": float(fl[1] - fs[1]),
        f"{prefix}_err_z_N": float(fl[2] - fs[2]),
        f"{prefix}_n_rev": int(len(F_sim)),
    }


def force_error(run, stab, mill) -> dict:
    """The surrogate's mean-force error, on the plateau and at the two ends.

    `plateau_*` is the steady comparison — the middle of the engaged span, `TRIM`
    trimmed off each end, the same span the growth-rate fit uses. `ends_*` is the
    two trimmed spans taken together: the entry and the exit, where the model is
    expected to be worst and where this job spends its interest.
    """
    out = {**_score(np.zeros((0, 3)), np.zeros((0, 3)), "plateau"),
           **_score(np.zeros((0, 3)), np.zeros((0, 3)), "ends"),
           "force_valid": False}
    paired = _paired(run, stab, mill)
    if paired is None:
        return out
    _t, _s, F_r, F0, cut = paired

    idx = np.flatnonzero(cut)
    if idx.size < 8:
        return out
    k = int(TRIM * idx.size)
    mid = idx[k:idx.size - k] if idx.size - 2 * k >= 4 else idx
    ends = np.setdiff1d(idx, mid)

    out.update(_score(F_r[mid], F0[mid], "plateau"))
    if ends.size:
        out.update(_score(F_r[ends], F0[ends], "ends"))
    out["force_valid"] = True
    return out


def summary(row) -> str:
    """The force comparison as a console block."""
    if not row.get("force_valid", False):
        return "force    the pass never cut long enough to average a revolution"

    def block(prefix, title):
        return (f"         {title:<9} sim {row[f'{prefix}_F_sim_N']:7.1f} N | "
                f"lin {row[f'{prefix}_F_lin_N']:7.1f} N | "
                f"{row[f'{prefix}_mag_err'] * 100:+6.2f} % magnitude, "
                f"{row[f'{prefix}_angle_err_deg']:5.2f} deg direction "
                f"({row[f'{prefix}_n_rev']} revs)")

    return ("force    revolution-averaged engine force vs the surrogate's F0\n"
            + block("plateau", "plateau") + "\n"
            + block("ends", "entry+exit"))
