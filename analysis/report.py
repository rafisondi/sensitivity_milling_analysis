"""Turning a simulated pass into numbers the linear prediction can be scored against.

The prediction reports a growth rate in 1/s and a mean force in N. The simulation
reports a displacement history and a force history at several kHz. This module
reduces the second to the first.

THREE TIMESCALES, AND ONLY THE MIDDLE ONE IS CHATTER

This arm's structural modes sit around 9 / 15 / 23 Hz while tooth passing at 3333
rpm is 222 Hz. So, unusually, chatter here is BELOW the forcing, not above it:

    ~0.4 Hz    the quasi-static deflection following the engagement profile as
               the tool enters and leaves the cut. Forced, not chatter.
    9-23 Hz    the structural modes. Regenerative growth lives here.
    222 Hz+    tooth passing and its harmonics. Forced, not chatter.

`chatter_metrics` band-passes the deviation to a window around the modes, so the
entry transient and the tooth-passing forcing are both excluded, then fits
log(envelope) against time. A stable cut gives a negative slope, an unstable one
positive — the same quantity the eigenvalue criterion reports.

A DIVERGENCE GUARD sits on top. Deflection beyond a quarter of the nominal radial
engagement means the tool has left the cut it was supposed to be taking, and the
envelope fit cannot see that: once the tool swings clear of the material the force
vanishes, the motion stops growing, and a runaway pass reads as a flat one.

WHAT THE DEVIATION IS MEASURED AGAINST

The command. In this workspace the command IS the nominal path — the DC
compensation is carried on the motor torques and moves no geometry
(`analysis.feedforward`), so there is only one reference and no bookkeeping to
get wrong. `MillingPass.deviation()` would still measure against `path.nom_w` if
a path ever carried one, which is what the offset-command route would produce.

READ `sim_growth_r2` ALONGSIDE THE SLOPE. The deviation is a forced response with
a damped component, not a clean exponential, so R^2 runs 0.2-0.8 here. Trust the
sign and the order of magnitude. `sim_growth_se_1_s` is the standard error of the
slope in the same 1/s the prediction is quoted in — about 0.2-0.3 — so two
numbers less than roughly 0.5 1/s apart are not resolved by this measurement.
"""

import numpy as np

#: Envelope growth above this is called unstable [1/s]. A little above zero, so a
#: marginally decaying pass is not flipped by fit noise. The eigenvalue criterion
#: uses a hard zero; both lines are reported rather than reconciled.
GROWTH_THRESHOLD = 0.5

#: Band around the structural modes, as multiples of the lowest / highest mode.
BAND_LO, BAND_HI = 0.4, 2.5

#: Fraction of the engaged span trimmed at each end, to drop entry/exit
#: transients. Matches `analysis.stability.TRIM`.
TRIM = 0.15

#: Deviation beyond this fraction of the NOMINAL radial engagement means the tool
#: has left the cut it was supposed to be taking.
DIVERGE_FRACTION_OF_AE = 0.25

#: A sample counts as cutting above this force [N].
ENGAGED_N = 50.0


def deviation_um(run):
    """(N, 2) tool-centre deviation from the commanded path [um], simulation grid.

    This is the plant's response to the cutting force. With the feedforward on it
    is what is left after the mean has been carried by the motors; with it off it
    is dominated by the quasi-static offset that mean force holds.
    """
    return (run.tcp_w_mm - run.commanded_w_mm()) * 1.0e3


def _bandpass(x, dt, f_lo, f_hi):
    """Zero-phase band-pass by FFT masking. Columns are independent."""
    n = len(x)
    f = np.fft.rfftfreq(n, dt)
    keep = (f >= f_lo) & (f <= f_hi)
    X = np.fft.rfft(x, axis=0)
    X[~keep] = 0.0
    return np.fft.irfft(X, n=n, axis=0)


def _envelope_growth(sig, t):
    """(slope [1/s], R^2, standard error of the slope) of log|envelope| vs time.

    The envelope is the running RMS over ~32 windows; fitting its log is a robust
    stand-in for the dominant eigenvalue's real part without having to identify
    the mode.
    """
    n = len(sig)
    w = max(16, n // 32)
    n_win = n // w
    if n_win < 4:
        return np.nan, np.nan, np.nan
    blocks = sig[:n_win * w].reshape(n_win, w, -1)
    rms = np.sqrt((blocks ** 2).sum(axis=2).mean(axis=1))
    tm = t[:n_win * w].reshape(n_win, w).mean(axis=1)

    good = rms > 0.0
    if good.sum() < 4:
        return np.nan, np.nan, np.nan
    y, x = np.log(rms[good]), tm[good]
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    sxx = float(((x - x.mean()) ** 2).sum())
    se = (float(np.sqrt(ss_res / (len(x) - 2) / sxx))
          if len(x) > 2 and sxx > 0 else np.nan)
    return float(slope), float(r2), se


def chatter_metrics(run, mill, *, modes_hz=None, threshold_n=ENGAGED_N,
                    growth_threshold=GROWTH_THRESHOLD) -> dict:
    """Measure the pass: how far it moved, and whether the motion grew.

    Returns NaNs and `sim_valid=False` rather than raising when a pass diverged,
    was truncated, or never engaged — a run that blew up is data, and it still
    has to occupy its row in the summary.
    """
    blank = {"sim_growth_1_s": np.nan, "sim_growth_r2": np.nan,
             "sim_growth_se_1_s": np.nan, "sim_chatter_hz": np.nan,
             "sim_peak_dev_um": np.nan, "sim_rms_dev_um": np.nan,
             "sim_dc_dev_um": np.nan, "sim_ac_rms_um": np.nan,
             "sim_unstable": False, "sim_diverged": False, "sim_valid": False,
             "sim_engaged_s": 0.0, "sim_peak_force_N": np.nan}

    if run is None or run.result is None:
        return blank

    dev = deviation_um(run)
    peak_force = float(np.linalg.norm(run.result.force_w, axis=1).max())
    if not np.all(np.isfinite(dev)):
        return {**blank, "sim_unstable": True, "sim_diverged": True,
                "sim_valid": True, "sim_peak_force_N": peak_force}

    # Classify divergence BEFORE the sample-count gate. A pass that left the cut
    # is cut short by `sim_coupled.BAIL_MM`, so it can be both diverged and too
    # short to fit an envelope to — that must read as diverged, not as "no data".
    diverge_um = 1.0e3 * DIVERGE_FRACTION_OF_AE * mill.radial_engagement_mm
    left_the_cut = bool(np.abs(dev).max() > diverge_um)

    def _too_short():
        if not left_the_cut:
            return {**blank, "sim_peak_force_N": peak_force}
        return {**blank, "sim_unstable": True, "sim_diverged": True,
                "sim_valid": True, "sim_peak_force_N": peak_force,
                "sim_peak_dev_um": float(np.abs(dev).max()),
                "sim_engaged_s": float(run.result.t[-1] - run.result.t[0])}

    engaged = run.engaged(threshold_n)
    if engaged.sum() < 256:
        return _too_short()

    d_all, t_all = dev[engaged], run.result.t[engaged]
    k = int(TRIM * len(d_all))
    d, t = d_all[k:len(d_all) - k], t_all[k:len(t_all) - k]
    if len(d) < 256:
        return _too_short()

    dt = float(np.median(np.diff(t)))
    peak = float(np.abs(d_all).max())
    rms = float(np.sqrt((d_all ** 2).sum(axis=1).mean()))
    # Split the deviation. Uncompensated it is mostly the QUASI-STATIC offset the
    # mean force holds the tool at, and a peak read as a vibration amplitude
    # would be wrong by an order of magnitude. `sim_dc_dev_um` is the number the
    # feedforward is supposed to drive to zero; `sim_ac_rms_um` is what it leaves.
    dc_vec = d_all.mean(axis=0)
    ac = d_all - dc_vec
    dc = float(np.linalg.norm(dc_vec))
    ac_rms = float(np.sqrt((ac ** 2).sum(axis=1).mean()))

    modes = np.asarray(modes_hz if modes_hz is not None else [8.8, 23.2], float)
    modes = modes[np.isfinite(modes) & (modes > 0)]
    if modes.size == 0:
        modes = np.array([8.8, 23.2])
    f_lo = max(BAND_LO * modes.min(), 2.0 / (t[-1] - t[0]))   # >= 2 cycles observed
    f_hi = min(BAND_HI * modes.max(), 0.4 / dt)
    if not (f_hi > f_lo):
        return {**blank, "sim_peak_force_N": peak_force}

    band = _bandpass(d - d.mean(axis=0), dt, f_lo, f_hi)
    growth, r2, se = _envelope_growth(band, t)

    spec = (np.abs(np.fft.rfft(band * np.hanning(len(band))[:, None],
                               axis=0)) ** 2).sum(axis=1)
    freq = np.fft.rfftfreq(len(band), dt)
    inband = (freq >= f_lo) & (freq <= f_hi)
    chatter_hz = (float(freq[inband][np.argmax(spec[inband])])
                  if inband.any() else np.nan)

    return {"sim_growth_1_s": growth,
            "sim_growth_r2": r2,
            "sim_growth_se_1_s": se,
            "sim_chatter_hz": chatter_hz,
            "sim_peak_dev_um": peak,
            "sim_rms_dev_um": rms,
            "sim_dc_dev_um": dc,
            "sim_ac_rms_um": ac_rms,
            "sim_band_lo_hz": float(f_lo),
            "sim_band_hi_hz": float(f_hi),
            "sim_peak_force_N": peak_force,
            "sim_unstable": bool(left_the_cut or (np.isfinite(growth)
                                                  and growth > growth_threshold)),
            "sim_diverged": bool(left_the_cut),
            "sim_valid": True,
            "sim_engaged_s": float(t_all[-1] - t_all[0])}


def agreement(row) -> dict:
    """Does the linear prediction call the same cut as the simulation?

    Also carries the signed growth-rate error, which is the finer comparison —
    two models can agree on the verdict and still disagree by an order of
    magnitude on how fast the cut is running away. The trimmed pair is the
    like-for-like one: both sides measured over the same span of the cut.
    """
    if not row.get("sim_valid", False):
        return {"agree": None, "agree_trim": None, "growth_err_1_s": np.nan,
                "growth_err_trim_1_s": np.nan}
    sim = row.get("sim_growth_1_s", np.nan)
    out = {"agree": bool(row["pred_unstable"] == row["sim_unstable"]),
           "growth_err_1_s": float(sim - row.get("pred_growth_max_1_s", np.nan))}
    if "pred_unstable_trim" in row:
        out["agree_trim"] = bool(row["pred_unstable_trim"] == row["sim_unstable"])
        out["growth_err_trim_1_s"] = float(
            sim - row.get("pred_growth_max_trim_1_s", np.nan))
    return out


def deviation_table(run, decimate=1) -> list:
    """One row per (decimated) simulation sample: where the tool went, and why.

    `decimate` keeps every n-th sample. The raw npz always holds every sample;
    this is for the CSV, which is meant to be openable.
    """
    res = run.result
    t = np.asarray(res.t, float)
    tcp = run.tcp_w_mm
    cmd = run.commanded_w_mm()
    dev = (tcp - cmd) * 1.0e3
    F = np.asarray(res.force_w, float)
    z = run.z_excursion_mm()

    idx = np.arange(0, len(t), max(1, int(decimate)))
    return [{
        "t_s": float(t[i]),
        "cmd_x_mm": float(cmd[i, 0]), "cmd_y_mm": float(cmd[i, 1]),
        "tcp_x_mm": float(tcp[i, 0]), "tcp_y_mm": float(tcp[i, 1]),
        "dev_x_um": float(dev[i, 0]), "dev_y_um": float(dev[i, 1]),
        "dev_mag_um": float(np.hypot(dev[i, 0], dev[i, 1])),
        "tcp_dz_mm": float(z[i]),
        "Fx_N": float(F[i, 0]), "Fy_N": float(F[i, 1]), "Fz_N": float(F[i, 2]),
        "F_mag_N": float(np.linalg.norm(F[i])),
    } for i in idx]
