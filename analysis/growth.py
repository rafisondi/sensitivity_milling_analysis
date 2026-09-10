"""Three numbers for a coupled pass, where `chatter_metrics` fits one.

    score(run_dir)   -> ring-down rate | plateau growth + its floor | peak excursion

WHY NOT ONE NUMBER

`analysis.report.chatter_metrics` trims 15% off each end of the engaged span and
fits a single exponential to what is left. That is one model for a signal made of
three different things:

    the ENTRY TRANSIENT   free decay at the closed-loop rate - the only part of
                          the record an eigenvalue can be compared against
    the FORCED PLATEAU    the tooth-passing response, whose envelope is FLAT no
                          matter how damped the machine is; fitting a slope to it
                          returns ~0 and calls it a growth rate
    whatever else         on this arm's slowest cell, a 3.5x rise and fall over
                          13 s that no exponential describes at all

Fitting them together gives the average of a decay, a zero and an excursion, and
which one wins depends on how long the pass happens to be - so the answer moves
with the feed rather than with the physics.

LONGER PASSES ARE BETTER, AND THE OLD METRIC HID IT

Measured over this workspace's tooth-passing grid: the ring-down fit needs `4 tau`
of record and cells shorter than that cannot contain one (R2 0.10-0.40 under
1.3 s, against 0.59-0.86 over 4 s). The growth test resolves `ln(1.2)/T`, which
is 0.015 1/s over a 35 s pass and 1.2 1/s over a 0.44 s one - eighty times
better. And the largest event in the whole grid, a rise from 73 to 260 um and
back at 1000 rpm with one tooth, is a 13 s feature: no short pass could show it,
and every single-exponential fit reported ~0 for it.

WHAT EACH NUMBER IS FOR

`ring_g` is the one to compare against `stabsim.stability`'s eigenvalue - same
quantity, a free decay rate, measured where the record actually contains one.

`plateau_g` is the INSTABILITY TEST, and it is reported with the smallest rate
that window could have resolved. A growth rate below its own floor is not a
measurement of zero, it is the absence of one, and the pair says so.

`peak_pct_ae` needs no fitting and is the severity measure: it is what separates
a cut that runs away from one that does not, and on this grid it separates the
one violent cell from its neighbours by 3-5x where every fitted rate did not.
"""

from pathlib import Path

import numpy as np

#: Band around the structural modes, as multiples of the lowest / highest mode.
#: The same window `analysis.report` uses, so the two remain comparable.
BAND_LO, BAND_HI = 0.4, 2.5

#: A sample counts as cutting above this force [N].
ENGAGED_N = 50.0

#: Ring-down window, in plant time constants. Four is where a free decay has
#: fallen to 2% and the fit has enough dynamic range to be worth trusting.
RING_TAUS = 4.0

#: The plateau window, as a fraction of the engaged span. Starts past the entry
#: transient and stops before the exit, whose ramp-up would otherwise read as
#: growth - it is the one part of the record that is neither.
PLATEAU = (0.50, 0.85)

#: The envelope must change by this factor for a slope to count as resolved.
#: `floor = ln(RESOLVE) / T`, which is what makes a long pass sensitive.
RESOLVE = 1.2

#: Plant time constants that must elapse between the entry and the start of the
#: plateau window before the plateau is a plateau. Below this the entry transient
#: is still decaying through it and the test reads that decay as a property of
#: the steady cut - which is exactly the confusion this module exists to end.
#: With PLATEAU starting at half the span, this needs `engaged_tau >= 2 * SETTLED`.
SETTLED_TAUS = 3.0

#: Default plant decay [s] when a run carries no `plant.npz`.
FALLBACK_TAU_S = 0.32


def _bandpass(x, dt, f_lo, f_hi):
    """Zero-phase band-pass by FFT masking. Columns are independent."""
    n = len(x)
    f = np.fft.rfftfreq(n, dt)
    X = np.fft.rfft(x, axis=0)
    X[~((f >= f_lo) & (f <= f_hi))] = 0.0
    return np.fft.irfft(X, n=n, axis=0)


def _envelope_fit(sig, t, n_windows=24):
    """(slope [1/s], R^2) of log(running rms) against time."""
    n = len(sig)
    w = max(16, n // n_windows)
    n_win = n // w
    if n_win < 4:
        return np.nan, np.nan
    rms = np.sqrt((sig[:n_win * w].reshape(n_win, w, -1) ** 2).sum(2).mean(1))
    tm = t[:n_win * w].reshape(n_win, w).mean(1)
    ok = rms > 0
    if ok.sum() < 4:
        return np.nan, np.nan
    y, x = np.log(rms[ok]), tm[ok]
    slope, intercept = np.polyfit(x, y, 1)
    resid = ((y - (slope * x + intercept)) ** 2).sum()
    total = ((y - y.mean()) ** 2).sum()
    return float(slope), float(1.0 - resid / total) if total > 0 else np.nan


def plant_tau_s(run_dir) -> float:
    """Plant decay constant [s] from the run's own `plant.npz`, else the default.

    `1 / |max Re(pole)|` of the arm with no cut closed - the rate a free
    vibration decays at once the tool is out of the material, and therefore the
    natural unit for how much record a ring-down needs.
    """
    f = Path(run_dir) / "plant.npz"
    if not f.exists():
        return FALLBACK_TAU_S
    with np.load(f, allow_pickle=False) as z:
        if "A" not in z:
            return FALLBACK_TAU_S
        lam = float(np.linalg.eigvals(np.asarray(z["A"], float)).real.max())
    return FALLBACK_TAU_S if lam >= 0 else float(-1.0 / lam)


def band_hz(run_dir):
    """(f_lo, f_hi) around the modes this run's plant actually has."""
    f = Path(run_dir) / "plant.npz"
    modes = None
    if f.exists():
        with np.load(f, allow_pickle=False) as z:
            if "modes_hz" in z:
                modes = np.asarray(z["modes_hz"], float)
    if modes is None or not np.isfinite(modes).any():
        modes = np.array([8.77, 23.14])
    modes = modes[np.isfinite(modes) & (modes > 0)]
    return BAND_LO * modes.min(), BAND_HI * modes.max()


def score(run_dir, *, ae_mm=5.0, ring_taus=RING_TAUS, plateau=PLATEAU,
          engaged_n=ENGAGED_N) -> dict:
    """The three measurements for one coupled pass.

    `ae_mm` only scales `peak_pct_ae`; everything else is independent of it.
    """
    run_dir = Path(run_dir)
    npz = run_dir / "raw" / "coupled.npz"
    blank = {"grow_valid": False}
    if not npz.exists():
        return blank

    with np.load(npz, allow_pickle=False) as z:
        t = np.asarray(z["t"], float)
        dev = np.asarray(z["deflection_w_um"], float)
        force = np.asarray(z["force_w"], float)

    cut = np.linalg.norm(force, axis=1) > engaged_n
    if cut.sum() < 500 or not np.all(np.isfinite(dev)):
        return blank
    i0, i1 = np.flatnonzero(cut)[[0, -1]]
    dt = float(np.median(np.diff(t)))
    span = float(t[i1] - t[i0])
    tau = plant_tau_s(run_dir)
    f_lo, f_hi = band_hz(run_dir)
    f_hi = min(f_hi, 0.4 / dt)

    def window(lo, hi):
        seg = dev[lo:hi]
        if len(seg) < 200 or hi <= lo:
            return np.nan, np.nan
        return _envelope_fit(_bandpass(seg - seg.mean(axis=0), dt, f_lo, f_hi),
                             t[lo:hi])

    # ── the ring-down: a free decay, where one exists ────────────────────────
    n_eng = i1 - i0
    ring_lo = i0 + int(0.02 * n_eng)                 # past the entry step itself
    ring_hi = min(i1, ring_lo + int(ring_taus * tau / dt))
    ring_g, ring_r2 = window(ring_lo, ring_hi)
    taus_seen = (t[ring_hi - 1] - t[ring_lo]) / tau if ring_hi > ring_lo else 0.0

    # ── the plateau: the instability test, with its own resolving power ──────
    a = i0 + int(plateau[0] * n_eng)
    b = i0 + int(plateau[1] * n_eng)
    plat_g, plat_r2 = window(a, b)
    T = float(t[b - 1] - t[a]) if b > a else 0.0
    floor = float(np.log(RESOLVE) / T) if T > 0 else np.inf
    settled_taus = float((t[a] - t[i0]) / tau) if b > a else 0.0
    settled = settled_taus >= SETTLED_TAUS
    if not np.isfinite(plat_g):
        verdict = "unmeasured"
    elif not settled:
        # The window opens while the entry transient is still ringing through it.
        # Whatever slope comes out is that transient, not the steady cut.
        verdict = "unsettled"
    elif plat_g > floor:
        verdict = "growing"
    elif plat_g < -floor:
        verdict = "decaying"
    else:
        verdict = "flat"

    # ── severity: no fitting at all ─────────────────────────────────────────
    d = dev[i0:i1]
    ac = d - d.mean(axis=0)
    peak = float(np.linalg.norm(d, axis=1).max())
    peak_ac_xy = float(np.linalg.norm(ac[:, :2], axis=1).max())

    return {
        "grow_valid": True,
        "engaged_s": span,
        "engaged_tau": span / tau,
        "plant_tau_s": tau,
        "band_lo_hz": float(f_lo), "band_hi_hz": float(f_hi),
        "ring_g_1_s": ring_g, "ring_r2": ring_r2, "ring_taus_seen": float(taus_seen),
        "ring_complete": bool(taus_seen >= 0.95 * ring_taus),
        "plateau_g_1_s": plat_g, "plateau_r2": plat_r2,
        "plateau_floor_1_s": floor, "plateau_window_s": T,
        "plateau_settled_taus": settled_taus, "plateau_settled": bool(settled),
        "plateau_verdict": verdict,
        "peak_dev_um": peak,
        "peak_ac_xy_um": peak_ac_xy,
        "peak_pct_ae": float(100.0 * peak_ac_xy / (1e3 * ae_mm)) if ae_mm else np.nan,
        "ac_rms_um": float(np.sqrt((ac ** 2).sum(1).mean())),
    }


def envelope(run_dir, n_slices=12, *, engaged_n=ENGAGED_N):
    """(t [s], rms [um]) of the band-passed deviation in `n_slices` slices.

    The shape of the record, for looking at rather than fitting - which is how
    the 1000 rpm one-tooth excursion was found, after three different exponential
    fits had each reported it as nothing.
    """
    run_dir = Path(run_dir)
    with np.load(run_dir / "raw" / "coupled.npz", allow_pickle=False) as z:
        t = np.asarray(z["t"], float)
        dev = np.asarray(z["deflection_w_um"], float)
        force = np.asarray(z["force_w"], float)
    cut = np.linalg.norm(force, axis=1) > engaged_n
    if cut.sum() < 500:
        return np.zeros(0), np.zeros(0)
    i0, i1 = np.flatnonzero(cut)[[0, -1]]
    dt = float(np.median(np.diff(t)))
    f_lo, f_hi = band_hz(run_dir)
    seg = dev[i0:i1]
    band = _bandpass(seg - seg.mean(axis=0), dt, f_lo, min(f_hi, 0.4 / dt))
    w = len(band) // int(n_slices)
    if w < 2:
        return np.zeros(0), np.zeros(0)
    rms = np.array([np.sqrt((band[j * w:(j + 1) * w] ** 2).sum(1).mean())
                    for j in range(int(n_slices))])
    tm = np.array([t[i0 + j * w + w // 2] - t[i0] for j in range(int(n_slices))])
    return tm, rms
