"""One growth rate, measured the same way on the linear model and on the coupled pass.

    sigma_lin      = max Re(eig A_cl)   at the halfway mark, from the linear model
    sigma_sim      = decay rate of a small tap's response in the coupled pass,
                     taken at the same halfway mark
    sigma_lin_fit  = the same tap through the linear closed loop, fitted with the
                     same estimator - the check that the estimator recovers the
                     eigenvalue over the window it is given

WHY A TAP, AND WHY TWO RUNS

The eigenvalue is the decay rate of a SMALL FREE PERTURBATION about the operating
point. The coupled pass never contains one on its own: its plateau is a forced
tooth-passing response, whose envelope is flat however damped the machine is,
and its entry transient belongs to partial immersion, not to the steady cut the
eigenvalue describes. So a perturbation is put in on purpose: a short force pulse
on the TCP at the halfway mark, during the steady cut.

The pass is run twice, identically except for the pulse, and subtracted:

    d(t) = dp_pulse(t) - dp_base(t)

Everything the two runs share - the forced tooth-passing response, the feed
tracking, the DC deflection - cancels exactly, and what is left is the response of
the full nonlinear system to the tap, regeneration included. Band-passed to the
structural modes and fitted as `log |d| ~ sigma t` once the fastest modes have died,
it converges to the least-damped closed-loop rate: the same number the eigenvalue
predicts, measured on the system the eigenvalue approximates.

    sigma > 0    the perturbation grows: unstable
    sigma < 0    it decays: stable, at that rate

The comparison is then like for like - one operating point, one pose, one quantity
- instead of an eigenvalue against an envelope slope that measures something else.

WHERE THE PULSE GOES

Into the motor-torque channel as `+J^T W`, which is mechanically identical to an
external force on the TCP but keeps the recorded CUTTING force clean, so the force
comparison of the pulsed run is not charged for a load that was never cut. It
deflects the tool, the tool leaves a different surface, the next tooth meets it:
the regenerative loop sees the tap exactly as it would see a vibration.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Ring-down length the fit is allowed, in plant time constants.
RING_TAUS = 4.0

#: The envelope must change by this factor for a slope to count as resolved.
RESOLVE = 1.2

#: Envelope windows below this multiple of the pre-pulse difference are noise and
#: are dropped before fitting. The two runs are deterministic, so this is ~0 unless
#: a parallel kernel reorders a reduction.
NOISE_FACTOR = 10.0


@dataclass(frozen=True)
class Pulse:
    """A rectangular force pulse on the TCP.

    at_frac     where, as a fraction of the path's own arc length - 0.5 is the
                halfway mark, the same pose the linear model is linearised at
    force_N     magnitude [N]. Small enough to stay in the linear range of the
                chip (a 20 N, 5 ms tap moves this arm ~20 um against a 180 um
                chip), large enough to fit over several time constants
    duration_s  width [s]. Short against the mode periods (43-114 ms here), so it
                excites the whole structural band
    dir_w       direction in the WORKPIECE frame. Equal weights here; `main.py`
                and the sweeps default to `--pulse-dir auto` instead, which taps
                along the input that best excites the LEAST-DAMPED closed-loop
                mode. An equal-weight tap excites several modes whose beating
                biases a short envelope fit - measured at 80% engagement, the
                estimator returned -2.38 1/s against an eigenvalue of -3.57
                (33% off); tapped along the dominant mode it returned -3.42
                (4% off)
    """

    at_frac: float = 0.5
    force_N: float = 20.0
    duration_s: float = 0.005
    dir_w: tuple = (1.0, 1.0, 1.0)

    def force_w(self) -> np.ndarray:
        d = np.asarray(self.dir_w, dtype=float)
        n = np.linalg.norm(d)
        if n <= 0:
            raise ValueError("pulse direction is zero")
        return float(self.force_N) * d / n


# ─────────────────────────────────────────────────────────────────────────────
# Where things are on the path
# ─────────────────────────────────────────────────────────────────────────────

def path_arclength_mm(path) -> np.ndarray:
    """(N,) cumulative arc length of an `OperationalPath` on its own time grid."""
    xy = np.asarray(path.xy_mm, float)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])


def time_at_s(path, s_mm: float) -> float:
    """Time [s] at which the commanded path has covered `s_mm` of arc length."""
    s = path_arclength_mm(path)
    return float(np.interp(float(s_mm), s, np.asarray(path.t, float)))


def time_at_fraction(path, frac: float) -> float:
    """Time [s] at which the commanded path has covered `frac` of its length."""
    s = path_arclength_mm(path)
    return float(np.interp(float(frac) * s[-1], s, np.asarray(path.t, float)))


def fit_window(t_pulse, duration_s, t_exit, tau_s, f_lowest_hz,
               ring_taus=RING_TAUS):
    """(t0, t1) [s] the decay is fitted over, and the shortest usable span.

    Starts one period of the lowest mode after the pulse ends, so the fastest
    modes have died and the slope is the least-damped one; ends after `ring_taus`
    plant time constants or one period before the exit, whichever comes first -
    the exit is a different event and the cut matrices change through it.
    """
    period = 1.0 / float(f_lowest_hz)
    t0 = float(t_pulse) + float(duration_s) + period
    t1 = min(t0 + float(ring_taus) * float(tau_s), float(t_exit) - period)
    return t0, t1, 2.0 * period


# ─────────────────────────────────────────────────────────────────────────────
# The estimator - one function, both sides
# ─────────────────────────────────────────────────────────────────────────────

def _bandpass(x, dt, f_lo, f_hi):
    n = len(x)
    f = np.fft.rfftfreq(n, dt)
    X = np.fft.rfft(x, axis=0)
    X[~((f >= f_lo) & (f <= f_hi))] = 0.0
    return np.fft.irfft(X, n=n, axis=0)


def decay_rate(t, d, t0, t1, *, band, f_lowest_hz, noise=0.0):
    """(sigma [1/s], R^2, floor [1/s], n_windows) of `d` over [t0, t1].

    `d` is (N, k) - any number of axes; the envelope is the rms over all of them,
    so the result does not depend on the frame the response is written in.
    The envelope is taken in windows of one period of the lowest mode, so a
    sinusoid's own phase does not modulate it, and windows under `noise` are
    dropped. `floor` is `ln(RESOLVE) / (t1 - t0)`: a rate below it is not resolved.
    """
    t = np.asarray(t, float)
    d = np.asarray(d, float).reshape(len(t), -1)
    blank = (np.nan, np.nan, np.nan, 0)
    if not (t1 > t0):
        return blank
    dt = float(np.median(np.diff(t)))
    f_lo, f_hi = band
    f_hi = min(f_hi, 0.4 / dt)
    db = _bandpass(d, dt, f_lo, f_hi)

    m = (t >= t0) & (t < t1)
    if m.sum() < 16:
        return blank
    seg, tt = db[m], t[m]
    w = max(8, int(round(1.0 / (float(f_lowest_hz) * dt))))
    n_win = len(seg) // w
    if n_win < 4:
        return blank
    rms = np.sqrt((seg[:n_win * w].reshape(n_win, w, -1) ** 2).sum(2).mean(1))
    tm = tt[:n_win * w].reshape(n_win, w).mean(1)
    keep = rms > max(float(noise) * NOISE_FACTOR, 0.0)
    keep &= rms > 0
    if keep.sum() < 4:
        return blank
    y, x = np.log(rms[keep]), tm[keep]
    slope, icpt = np.polyfit(x, y, 1)
    res = ((y - (slope * x + icpt)) ** 2).sum()
    tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - res / tot if tot > 0 else np.nan
    floor = float(np.log(RESOLVE) / (t1 - t0))
    return float(slope), float(r2), floor, int(keep.sum())


# ─────────────────────────────────────────────────────────────────────────────
# The linear side: the same tap through the closed loop at the halfway mark
# ─────────────────────────────────────────────────────────────────────────────

def linear_pulse_rate(receptance, K, C, force_plant, pulse: Pulse, *, fit_t0_rel,
                      fit_t1_rel, band, f_lowest_hz, dt=1.0e-4):
    """(sigma_fit, R^2) for the linear closed loop's response to the tap.

    `receptance` and `K`, `C` must share a frame, and `force_plant` (3,) [N] must
    be written in it. Times are RELATIVE to the pulse start. Exact ZOH
    discretisation of `xdot = A_cl x + B u`: with `D = 0` and `C B = 0` (true of
    any `Receptance.from_mdk`) an external force enters through `B` unchanged when
    the cut is closed, so the input map does not move.

    This is the estimator's own check: if it returns the eigenvalue, the window
    and the band are adequate for this cell; if not, the comparison with
    `sigma_sim` is limited by the estimator, not by the model.
    """
    from scipy.linalg import expm
    from stabsim.stability import closed_loop_matrix

    A = closed_loop_matrix(receptance, K, C)
    B, Cm = receptance.B, receptance.C
    n, m = A.shape[0], B.shape[1]
    aug = np.zeros((n + m, n + m))
    aug[:n, :n], aug[:n, n:] = A * dt, B * dt
    E = expm(aug)
    Ad, Bd = E[:n, :n], E[:n, n:]

    T = float(fit_t1_rel) + 2.0 / float(f_lowest_hz)
    N = int(np.ceil(T / dt)) + 1
    t = np.arange(N) * dt
    x = np.zeros(n)
    y = np.zeros((N, 3))
    u_on = np.asarray(force_plant, float)
    for k in range(1, N):
        u = u_on if t[k - 1] < pulse.duration_s else 0.0 * u_on
        x = Ad @ x + Bd @ u
        y[k] = Cm @ x
    if not np.all(np.isfinite(y)):
        return np.nan, np.nan
    sig, r2, _floor, _n = decay_rate(t, y * 1e6, fit_t0_rel, fit_t1_rel,
                                     band=band, f_lowest_hz=f_lowest_hz)
    return sig, r2


# ─────────────────────────────────────────────────────────────────────────────
# The nonlinear side: the twin-run difference
# ─────────────────────────────────────────────────────────────────────────────

def twin_rate(base_dir, pulse_dir, *, t_pulse, duration_s, fit_t0, fit_t1,
              band, f_lowest_hz) -> dict:
    """sigma_sim and its diagnostics from two coupled passes, base and pulsed.

    Returns NaNs rather than raising when either run is missing or cut short
    before the window - a pass that bailed is data, and `sim_state` reads the
    bail separately.
    """
    out = {"pulse_valid": False, "sim_sigma_1_s": np.nan, "sim_sigma_r2": np.nan,
           "sim_sigma_floor_1_s": np.nan, "pulse_peak_um": np.nan,
           "pulse_noise_um": np.nan, "pulse_windows": 0}
    fb = Path(base_dir) / "raw" / "coupled.npz"
    fp = Path(pulse_dir) / "raw" / "coupled.npz"
    if not (fb.exists() and fp.exists()):
        return out
    with np.load(fb, allow_pickle=False) as z:
        tb = np.asarray(z["t"], float)
        db = np.asarray(z["deflection_w_um"], float)
    with np.load(fp, allow_pickle=False) as z:
        tp = np.asarray(z["t"], float)
        dp = np.asarray(z["deflection_w_um"], float)

    n = min(len(tb), len(tp))
    if n < 100 or not np.allclose(tb[:n], tp[:n]):
        return out
    t, d = tb[:n], dp[:n] - db[:n]
    if not np.all(np.isfinite(d)):
        return out

    pre = t < float(t_pulse)
    noise = float(np.sqrt((d[pre] ** 2).sum(1).mean())) if pre.any() else 0.0
    post = t >= float(t_pulse)
    peak = float(np.linalg.norm(d[post], axis=1).max()) if post.any() else np.nan

    t1 = min(float(fit_t1), float(t[-1]))
    sig, r2, floor, nw = decay_rate(t, d, float(fit_t0), t1, band=band,
                                    f_lowest_hz=f_lowest_hz, noise=noise)
    out.update({"pulse_valid": bool(np.isfinite(sig)), "sim_sigma_1_s": sig,
                "sim_sigma_r2": r2, "sim_sigma_floor_1_s": floor,
                "pulse_peak_um": peak, "pulse_noise_um": noise,
                "pulse_windows": nw,
                "pulse_run_truncated": bool(len(tp) < len(tb))})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# The verdicts the polar plots colour by
# ─────────────────────────────────────────────────────────────────────────────

def lin_state(sigma_lin) -> str:
    """'unstable' | 'stable' | 'unmeasured' from the halfway-mark eigenvalue."""
    if sigma_lin is None or not np.isfinite(sigma_lin):
        return "unmeasured"
    return "unstable" if sigma_lin > 0.0 else "stable"


def sim_state(row) -> str:
    """'unstable' | 'stable' | 'marginal' | 'unmeasured' for a simulated cell.

    Unstable if the baseline pass left the cut (`sim_diverged`, which includes a
    `BAIL_MM` stop), if only the PULSED pass did - the tap alone was enough to
    throw it out - or if the tap's response grows by more than its own
    resolution floor. Stable if it decays by more than the floor. Marginal if the
    rate is inside the floor: not a measurement of zero, the absence of one.
    Without a pulse run, the baseline's plateau growth test stands in.
    """
    def truthy(v):
        return v is True or str(v).lower() == "true"

    if truthy(row.get("sim_diverged")):
        return "unstable"
    if truthy(row.get("pulse_sim_diverged")):
        return "unstable"
    sig = row.get("sim_sigma_1_s")
    floor = row.get("sim_sigma_floor_1_s")
    try:
        sig, floor = float(sig), float(floor)
    except (TypeError, ValueError):
        sig = floor = np.nan
    if np.isfinite(sig) and np.isfinite(floor):
        if sig > floor:
            return "unstable"
        if sig < -floor:
            return "stable"
        return "marginal"
    verdict = row.get("plateau_verdict")
    if verdict == "growing":
        return "unstable"
    if verdict in ("flat", "decaying"):
        return "stable"
    return "unmeasured"
