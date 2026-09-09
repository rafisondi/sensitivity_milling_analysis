"""Figures for one run, and for a sweep of them.

    run_figures(d, stab, ff, run, setup, receptance)   -> out/<run>/figures/
    sweep_figures(d, rows)                             -> out/<sweep>/figures/

WHY THIS IS THE ONE MODULE THAT IMPORTS MATPLOTLIB

Nothing else in this workspace does, and that is deliberate: a run's result is
the npz and the csv, which are full precision and machine readable, and a figure
is a view of them that can always be rebuilt. So this module is imported ONLY
when a figure is actually wanted (`main.py` does it inside the `if`), the backend
is forced to Agg, and no plot is ever shown - they are written to disk and the
process exits.

WHAT IS PLOTTED, AND WHAT IS DELIBERATELY NOT

The three per-run figures are the geometry the model measured, the force it
predicted against the force the engine produced, and the stability it predicted
along the path. All three are indexed by ARC LENGTH rather than time, because
that is the coordinate both sides share: `StabilityAlongPath` is written on it,
and a sweep that varies the feed puts the same cut at different times.

No figure here fits, thresholds or classifies anything. The measurement-side
verdict (`analysis.report`) is not drawn, because under a tooth-passing sweep its
band-pass window and the tooth-passing frequency collide - see `sweep_figures`,
which plots the collision as an operating-point fact rather than resolving it.

COLOUR. Series identity is carried by a fixed four-slot categorical order, never
cycled: tooth counts 1, 2, 4, 8 always get the same colour in every figure of a
sweep, so a line's identity survives a change of which cells were run. Where a
figure shows the same quantity from two sources, the SOURCE is linestyle and the
COMPONENT is colour, so neither has to be read off colour alone.
"""

from pathlib import Path

import numpy as np

# ── the palette ──────────────────────────────────────────────────────────────
# Validated as a categorical set on the light chart surface under the all-pairs
# rule: worst CVD dE 9.2, worst normal-vision dE 16.3. The aqua sits below 3:1
# against the surface, so every series in these figures is DIRECT-LABELLED as
# well as legended - identity is never colour alone.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#a3a29b"
GRID = "#e4e3de"

#: Categorical slots, in fixed assignment order. Never cycled, never reordered.
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"

#: Tooth count -> colour, fixed for the life of a sweep.
TEETH_COLOUR = {1: C1, 2: C2, 4: C3, 8: C4}

#: The two sides of the comparison, wherever they appear together.
SIM, MODEL = C1, C2


def _mpl():
    """matplotlib with the Agg backend and this module's house style."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "savefig.dpi": 150,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.titleweight": "medium", "axes.labelcolor": INK_2,
        "text.color": INK, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "grid.color": GRID, "grid.linewidth": 0.8,
        "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 1.6, "lines.solid_capstyle": "round",
    })
    return plt


def _clean(ax, *, grid="y"):
    """Recessive axes: no top/right spine, grid behind the marks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(True, axis=grid, alpha=0.9)
    return ax


def _label_end(ax, x, y, text, colour, dx=4):
    """Direct-label a series at its last finite point.

    Required, not decorative: one slot of the palette sits below 3:1 against the
    surface, so a legend swatch alone is not enough to identify it.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any():
        return
    ax.annotate(text, (x[ok][-1], y[ok][-1]), textcoords="offset points",
                xytext=(dx, 0), va="center", ha="left", fontsize=8,
                color=colour, fontweight="medium", clip_on=False)


def _label_at(ax, x, y, frac, text, colour, dy=7):
    """Direct-label a series part-way along it, rather than at its end.

    Series that all decay to zero at the exit - the force components, the
    engagement angles - collide into one illegible pile if labelled at the end.
    `frac` picks a point along the finite span where they are still apart.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any():
        return
    xs, ys = x[ok], y[ok]
    i = int(np.clip(round(float(frac) * (len(xs) - 1)), 0, len(xs) - 1))
    ax.annotate(text, (xs[i], ys[i]), textcoords="offset points",
                xytext=(0, dy), ha="center",
                va="bottom" if dy >= 0 else "top", fontsize=8,
                color=colour, fontweight="medium", clip_on=False)


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def _engaged_span(stab):
    """(s_start, s_end) of the engaged part of the path [mm], or None."""
    eng = np.asarray(stab.engaged, bool)
    if not eng.any():
        return None
    s = np.asarray(stab.s_mm, float)
    return float(s[eng].min()), float(s[eng].max())


def _shade_engaged(ax, stab, label=False):
    span = _engaged_span(stab)
    if span is None:
        return
    ax.axvspan(span[0], span[1], color=C3, alpha=0.06, lw=0, zorder=0)
    if label:
        ax.annotate("engaged", (0.5 * (span[0] + span[1]), 1.0),
                    xycoords=("data", "axes fraction"), xytext=(0, -11),
                    textcoords="offset points", ha="center", fontsize=8,
                    color=INK_2)


# ─────────────────────────────────────────────────────────────────────────────
# One run
# ─────────────────────────────────────────────────────────────────────────────

def figure_trajectory(path, stab, run, setup):
    """Where the tool went: the plan view, the deviation, and the engagement.

    The deviation panel is what the plant actually did; the engagement panel is
    the geometry the linear model measured to predict it. They are stacked on a
    shared arc length so a feature in one can be read against the other.
    """
    plt = _mpl()
    n = 3 if run is not None else 2
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.5 * n + 0.6))

    # ── the pass in the workpiece plane ──────────────────────────────────────
    ax = _clean(axes[0], grid="both")
    ring = np.asarray(setup.part.closed_xy_mm, float)
    ax.fill(ring[0], ring[1], color=MUTED, alpha=0.18, lw=0, zorder=0)
    ax.plot(ring[0], ring[1], color=MUTED, lw=1.0, zorder=1)
    xy = np.asarray(setup.xy_mm, float)
    ax.plot(xy[:, 0], xy[:, 1], color=INK_2, lw=1.2, zorder=2)
    eng = np.asarray(stab.engaged, bool)
    if eng.any():
        ax.plot(stab.xy_mm[eng, 0], stab.xy_mm[eng, 1], color=C3, lw=3.0,
                alpha=0.85, zorder=3, solid_capstyle="butt")
        _label_at(ax, stab.xy_mm[eng, 0], stab.xy_mm[eng, 1], 0.5, "cutting",
                  C3, dy=7)
    ax.set_aspect("equal")
    ax.set_xlabel("x  [mm]")
    ax.set_ylabel("y  [mm]")
    ax.set_title("the pass, workpiece frame", loc="left", color=INK)

    # ── deviation from the command ───────────────────────────────────────────
    if run is not None:
        ax = _clean(axes[1])
        cmd = run.commanded_w_mm()
        dev = (run.tcp_w_mm - cmd) * 1e3
        s = np.concatenate([[0.0], np.cumsum(
            np.linalg.norm(np.diff(cmd, axis=0), axis=1))])
        _shade_engaged(ax, stab, label=True)
        for j, (c, name) in enumerate(((C1, "dev x"), (C2, "dev y"))):
            ax.plot(s, dev[:, j], color=c, lw=1.0, alpha=0.9, label=name)
            _label_end(ax, s, dev[:, j], f" {name.split()[1]}", c)
        mag = np.linalg.norm(dev, axis=1)
        ax.plot(s, mag, color=INK, lw=1.2, label="|dev|")
        _label_end(ax, s, mag, " |dev|", INK)
        ax.axhline(0.0, color=MUTED, lw=0.8)
        ax.set_ylabel("TCP - command  [um]")
        ax.set_xlabel("arc length s  [mm]")
        ax.set_title("deviation from the commanded path", loc="left", color=INK)
        ax.legend(loc="upper left", ncol=3)

    # ── the engagement the model measured ────────────────────────────────────
    ax = _clean(axes[-1])
    _shade_engaged(ax, stab)
    s = np.asarray(stab.s_mm, float)
    phi_en = np.degrees(np.asarray(stab.phi_en, float))
    phi_ex = np.degrees(np.asarray(stab.phi_ex, float))
    phi_en[~eng], phi_ex[~eng] = np.nan, np.nan
    ax.plot(s, phi_en, color=C1, label="phi entry")
    ax.plot(s, phi_ex, color=C2, label="phi exit")
    ax.fill_between(s, phi_en, phi_ex, color=C1, alpha=0.10, lw=0)
    # Both angles collapse to zero at the exit, so an end label puts them on
    # top of each other; the plateau is where they are furthest apart.
    _label_at(ax, s, phi_en, 0.5, "entry", C1, dy=-7)
    _label_at(ax, s, phi_ex, 0.5, "exit", C2, dy=7)
    ax.set_ylabel("engagement angle  [deg]")
    ax.set_xlabel("arc length s  [mm]")
    ax.set_title("entry / exit angles, measured off the cut geometry",
                 loc="left", color=INK)
    ax.legend(loc="lower left", ncol=2)

    fig.tight_layout()
    return _save(fig, path)


def figure_forces(path, stab, run, setup):
    """The surrogate's mean force against the engine's, along the path.

    The engine's raw history is drawn faintly behind its own revolution average,
    because the ripple is the thing `F0` never claimed to have: comparing the two
    without averaging first would charge the model for a missing feature. The
    component panel encodes SOURCE as linestyle and COMPONENT as colour, so
    neither reading depends on colour alone.
    """
    from analysis import forces as fmod

    plt = _mpl()
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.0), sharex=True)
    s_lin = np.asarray(stab.s_mm, float)
    F0 = np.asarray(stab.F0_w, float)

    rev = (None, None, None) if run is None else fmod.revolution_average(run, setup.mill)
    t_r, F_r, s_r = rev

    # ── magnitude ────────────────────────────────────────────────────────────
    ax = _clean(axes[0])
    _shade_engaged(ax, stab, label=True)
    if run is not None:
        cmd = run.commanded_w_mm()
        s_raw = np.concatenate([[0.0], np.cumsum(
            np.linalg.norm(np.diff(cmd, axis=0), axis=1))])
        raw = np.linalg.norm(np.asarray(run.result.force_w, float), axis=1)
        ax.plot(s_raw, raw, color=SIM, lw=0.5, alpha=0.25,
                label="engine, every sample")
    if F_r is not None:
        m = np.linalg.norm(F_r, axis=1)
        ax.plot(s_r, m, color=SIM, lw=1.6, label="engine, per revolution")
        _label_end(ax, s_r, m, " engine", SIM)
    m0 = np.linalg.norm(F0, axis=1)
    ax.plot(s_lin, m0, color=MODEL, lw=1.6, label="surrogate F0")
    _label_end(ax, s_lin, m0, " F0", MODEL)
    ax.set_ylabel("|F|  [N]")
    ax.set_title("cutting force magnitude", loc="left", color=INK)
    ax.legend(loc="upper left", ncol=3)

    # ── components ───────────────────────────────────────────────────────────
    ax = _clean(axes[1])
    _shade_engaged(ax, stab)
    for j, (c, name) in enumerate(((C1, "x"), (C2, "y"), (C3, "z"))):
        if F_r is not None:
            ax.plot(s_r, F_r[:, j], color=c, lw=1.4)
        ax.plot(s_lin, F0[:, j], color=c, lw=1.4, ls=(0, (3, 2)))
        # Always above the curve: the feed-direction component runs along the
        # bottom of the panel, where a label below it lands on the tick labels.
        _label_at(ax, s_lin, F0[:, j], 0.5, name, c, dy=7)
    ax.axhline(0.0, color=MUTED, lw=0.8)
    ax.set_ylabel("force component  [N]")
    ax.set_xlabel("arc length s  [mm]")
    ax.set_title("components: engine per revolution (solid) vs surrogate F0 "
                 "(dashed)", loc="left", color=INK)
    handles = [plt.Line2D([], [], color=INK_2, lw=1.4, ls=ls, label=lab)
               for ls, lab in (("-", "engine"), ((0, (3, 2)), "F0"))]
    handles += [plt.Line2D([], [], color=c, lw=1.4, label=n)
                for c, n in ((C1, "x"), (C2, "y"), (C3, "z"))]
    # The band between the x and y plateaux is the one part of this panel that
    # is always empty, whatever the operating point.
    ax.legend(handles=handles, loc="center left", ncol=2)

    fig.tight_layout()
    return _save(fig, path)


def figure_prediction(path, stab, receptance):
    """What the linear model said, node by node: growth rate and critical depth.

    Both panels are the PREDICTION alone - nothing measured is drawn on them.
    The open-loop line is the bare arm, so the vertical distance to it is what
    closing the cut did: below the line the cut adds damping, above it the cut
    takes damping away.
    """
    plt = _mpl()
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
    s = np.asarray(stab.s_mm, float)
    eng = np.asarray(stab.engaged, bool)

    ax = _clean(axes[0])
    _shade_engaged(ax, stab, label=True)
    g = np.where(eng, np.asarray(stab.growth_rate, float), np.nan)
    ax.plot(s, g, color=MODEL, lw=1.6)
    _label_at(ax, s, g, 0.5, "cut closed on the arm", MODEL, dy=-7)
    open_loop = stab.open_loop()
    ax.axhline(open_loop, color=INK_2, lw=1.0, ls=(0, (4, 3)))
    # Left-hand side: the right is where the growth rate returns TO this line at
    # the exit, so a label there lands on the curve it is describing.
    ax.annotate(f"bare arm {open_loop:+.2f} 1/s", (0.01, open_loop),
                xycoords=("axes fraction", "data"), textcoords="offset points",
                xytext=(0, 5), fontsize=8, color=INK_2,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))
    ax.axhline(0.0, color=MUTED, lw=0.8)
    ax.set_ylabel("growth rate  [1/s]")
    ax.set_title("predicted growth rate  (> 0 would be unstable)",
                 loc="left", color=INK)

    ax = _clean(axes[1])
    _shade_engaged(ax, stab)
    ap = stab.ap_crit_mm
    if ap is not None:
        ap = np.where(eng, np.asarray(ap, float), np.nan)
        ax.plot(s, ap, color=MODEL, lw=1.6)
        _label_end(ax, s, ap, " ap_crit", MODEL)
        if np.isfinite(ap).any():
            ax.set_yscale("log")
    run_ap = float(stab.axial_depth_mm)
    ax.axhline(run_ap, color=SIM, lw=1.2)
    ax.annotate(f"running at ap = {run_ap:g} mm", (0.01, run_ap),
                xycoords=("axes fraction", "data"), textcoords="offset points",
                xytext=(0, 5), fontsize=8, color=SIM, fontweight="medium",
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))
    ax.set_ylim(bottom=0.5 * run_ap)
    ax.set_ylabel("critical depth  [mm]")
    ax.set_xlabel("arc length s  [mm]")
    ax.set_title("smallest unstable depth at each node  (gaps: stable past the "
                 "cap)", loc="left", color=INK)

    fig.tight_layout()
    return _save(fig, path)


def run_figures(d, stab, ff, run, setup, receptance) -> dict:
    """Every figure for one run, into `<run>/figures/`."""
    fig_dir = Path(d) / "figures"
    return {
        "fig_trajectory": figure_trajectory(fig_dir / "trajectory.png", stab,
                                            run, setup),
        "fig_forces": figure_forces(fig_dir / "forces.png", stab, run, setup),
        "fig_prediction": figure_prediction(fig_dir / "prediction.png", stab,
                                            receptance),
    }


# ─────────────────────────────────────────────────────────────────────────────
# A sweep of runs
# ─────────────────────────────────────────────────────────────────────────────

def _by_teeth(rows, x_key, y_key):
    """{n_teeth: (x, y)} sorted on x, dropping rows missing either value."""
    out = {}
    for n in sorted({int(r["n_teeth"]) for r in rows}):
        pts = [(r.get(x_key), r.get(y_key)) for r in rows
               if int(r["n_teeth"]) == n]
        pts = [(float(a), float(b)) for a, b in pts
               if a is not None and b is not None
               and np.isfinite(float(a)) and np.isfinite(float(b))]
        if pts:
            pts.sort()
            out[n] = (np.array([p[0] for p in pts]),
                      np.array([p[1] for p in pts]))
    return out


def _teeth_panel(ax, rows, x_key, y_key, *, logx=True, logy=False):
    series = _by_teeth(rows, x_key, y_key)
    for n, (x, y) in series.items():
        c = TEETH_COLOUR.get(n, INK_2)
        ax.plot(x, y, color=c, marker="o", ms=4.5, lw=1.6, label=f"{n} teeth")
        _label_end(ax, x, y, f" N={n}", c)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    _clean(ax)
    return series


def figure_sweep_prediction(path, rows, x_key="tpf_hz",
                            x_label="tooth-passing frequency  [Hz]"):
    """What the linear model predicts across the grid.

    READ THIS FIGURE AGAINST ITS TWIN. `F0`, `K_cut` and `C_cut` see `(rpm, N)`
    only through `N fz` and `rpm`, so the prediction is a function of the SPINDLE
    SPEED and the chip load - never of the tooth count on its own.

    Plotted against rpm, the four tooth-count curves therefore lie on top of one
    another. Plotted against tooth-passing frequency, `rpm N / 60`, the same four
    curves separate into parallel copies shifted by exactly `log N` - which is
    not a tooth-passing effect but its absence, drawn. If the model did respond
    to tooth passing, the collapse would be the other way round.
    """
    plt = _mpl()
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.0), sharex=True)

    ax = axes[0]
    _teeth_panel(ax, rows, x_key, "pred_ap_crit_min_mm", logy=True)
    aps = [float(r["ap_mm"]) for r in rows if r.get("ap_mm") is not None]
    if aps:
        ax.axhline(aps[0], color=INK_2, lw=1.0, ls=(0, (4, 3)))
        ax.annotate(f" running ap = {aps[0]:g} mm", (ax.get_xlim()[0], aps[0]),
                    textcoords="offset points", xytext=(4, 4), fontsize=8,
                    color=INK_2)
    ax.set_ylabel("predicted ap_crit, min over path  [mm]")
    ax.set_title("predicted critical depth", loc="left", color=INK)
    ax.legend(loc="best", ncol=4)

    ax = axes[1]
    _teeth_panel(ax, rows, x_key, "pred_growth_max_1_s")
    ax.axhline(0.0, color=MUTED, lw=0.8)
    ax.set_ylabel("predicted growth rate, max  [1/s]")
    ax.set_xlabel(x_label)
    ax.set_title("predicted growth rate  (> 0 would be unstable)",
                 loc="left", color=INK)

    fig.tight_layout()
    return _save(fig, path)


def figure_sweep_validity(path, rows, modes_hz=None):
    """Where each cell sits against the three limits that bound this sweep.

    None of these is a result - they are the conditions a result would have to be
    read under, and all three move when rpm and the tooth count move:

      omega_T          the linear model's delay truncation. Both cut terms drop
                       the regenerative delay after one order, so the neglected
                       term is roughly `(omega T)^2 / 2` of the cut force.
      steps per tooth  the engine's time resolution. Below ~8 the tooth-passing
                       harmonics alias, and an aliased engine looks exactly like
                       a model that broke down.
      chip pixels      the engine's space resolution. Below ~2 px the binary
                       raster loses the chip.
    """
    plt = _mpl()
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.0), sharex=True)

    ax = axes[0]
    _teeth_panel(ax, rows, "tpf_hz", "omega_T_max", logy=True)
    for y, txt in ((0.7, "marginal"), (1.0, "truncated term ~50% of the cut")):
        ax.axhline(y, color=INK_2, lw=1.0, ls=(0, (4, 3)))
        ax.annotate(f" {txt}", (ax.get_xlim()[0], y), textcoords="offset points",
                    xytext=(4, 3), fontsize=8, color=INK_2)
    ax.set_ylabel("omega_T at the top mode  [rad]")
    ax.set_title("the linear model's own truncation parameter", loc="left",
                 color=INK)
    ax.legend(loc="best", ncol=4)

    ax = axes[1]
    _teeth_panel(ax, rows, "tpf_hz", "steps_per_tooth", logy=True)
    ax.axhline(8.0, color=INK_2, lw=1.0, ls=(0, (4, 3)))
    ax.annotate(" harmonics alias below 8", (ax.get_xlim()[0], 8.0),
                textcoords="offset points", xytext=(4, 3), fontsize=8,
                color=INK_2)
    ax.set_ylabel("simulation steps per tooth")
    ax.set_title("engine time resolution", loc="left", color=INK)

    ax = axes[2]
    _teeth_panel(ax, rows, "tpf_hz", "chip_px", logy=True)
    ax.axhline(2.0, color=INK_2, lw=1.0, ls=(0, (4, 3)))
    ax.annotate(" the raster loses the chip below 2 px",
                (ax.get_xlim()[0], 2.0), textcoords="offset points",
                xytext=(4, 3), fontsize=8, color=INK_2)
    ax.set_ylabel("chip load  [raster pixels]")
    ax.set_xlabel("tooth-passing frequency  [Hz]")
    ax.set_title("engine space resolution", loc="left", color=INK)

    fig.tight_layout()
    return _save(fig, path)


def figure_sweep_force(path, rows):
    """The mean force: what the model predicts, and where a run has scored it.

    The lower panel is empty until coupled passes exist - `plateau_mag_err` is a
    simulated quantity. It is drawn on the same tooth-passing axis so a smoke
    subset can be read against the full predicted grid above it.
    """
    plt = _mpl()
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.0), sharex=True)

    ax = axes[0]
    _teeth_panel(ax, rows, "tpf_hz", "ff_F0_mean_N")
    ax.set_ylabel("surrogate |F0|, mean over the cut  [N]")
    ax.set_title("predicted mean cutting force", loc="left", color=INK)
    ax.legend(loc="best", ncol=4)

    ax = axes[1]
    scored = [r for r in rows if r.get("plateau_mag_err") is not None]
    if scored:
        series = _by_teeth(scored, "tpf_hz", "plateau_mag_err")
        for n, (x, y) in series.items():
            c = TEETH_COLOUR.get(n, INK_2)
            ax.plot(x, 100.0 * y, color=c, marker="o", ms=5.5, lw=1.6)
            _label_end(ax, x, 100.0 * y, f" N={n}", c)
        ax.set_xscale("log")
    else:
        ax.annotate("no coupled pass has been scored yet",
                    (0.5, 0.5), xycoords="axes fraction", ha="center",
                    fontsize=9, color=INK_2)
    _clean(ax)
    ax.axhline(0.0, color=MUTED, lw=0.8)
    ax.set_ylabel("plateau magnitude error  [%]")
    ax.set_xlabel("tooth-passing frequency  [Hz]")
    ax.set_title("surrogate F0 against the engine, revolution-averaged",
                 loc="left", color=INK)

    fig.tight_layout()
    return _save(fig, path)


def sweep_figures(d, rows) -> dict:
    """Every sweep-level figure, into `<sweep>/figures/`."""
    fig_dir = Path(d) / "figures"
    rows = [r for r in rows if r.get("n_teeth") is not None]
    if not rows:
        return {}
    return {
        "fig_sweep_prediction": figure_sweep_prediction(
            fig_dir / "sweep_prediction.png", rows),
        # The same two panels on the axis the model actually depends on. The
        # pair is the point: collapse here, parallel shift there.
        "fig_sweep_prediction_rpm": figure_sweep_prediction(
            fig_dir / "sweep_prediction_rpm.png", rows,
            x_key="spindle_rpm", x_label="spindle speed  [rpm]"),
        "fig_sweep_validity": figure_sweep_validity(
            fig_dir / "sweep_validity.png", rows),
        "fig_sweep_force": figure_sweep_force(fig_dir / "sweep_force.png", rows),
    }
