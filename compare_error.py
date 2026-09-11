"""RMS error, linear model against the coupled truth, on the INNER THIRD of the cut.

    python compare_error.py                          out/tooth_passing_v4, both designs
    python compare_error.py --design grid
    python compare_error.py --resume                 skip cells already scored
    python compare_error.py --cells grid_fz0p18_rpm5000_z4,rows_feed40_rpm3333_z2

WHY THE INNER THIRD, AND NOT THE PLATEAU `analysis.growth` ALREADY DEFINES

`analysis.growth.PLATEAU` (50-85% of the engaged span) and `analysis.forces`'s
`TRIM` (15% off each end) both exist to answer "has the entry transient finished
and the exit not yet started" - the least aggressive trim that still keeps the
transients out. This script asks a plainer question - how big is the two models'
disagreement in STEADY CUTTING - so it uses the plainer window: the middle third
of the engaged span, `[start + n/3, start + 2n/3)`, cutting mask
`|F_sim| > ENGAGED_N` exactly as `analysis.growth` defines it. Wider than the
plateau's 85% cap would still be, narrower than TRIM's 70%, and not tied to
either module's own reasons for its particular fractions.

TWO ERRORS, TWO DIFFERENT REFERENCES

`compare_linear.schedule` + `.integrate` already do the expensive part - build
this run's `K_cut(s)`/`C_cut(s)`/`F0(s)` and RK4 them along the path - so this
script calls them rather than re-deriving anything. Two comparisons come out of
that one integration, against two different references:

  TCP ERROR       `dev_um` (the coupled truth) against `dx_um` (the linear
                  model's own INTEGRATED trajectory - `F0 + K_cut dx + C_cut
                  dxd`, closed on the plant, exactly as `compare_linear`'s figure
                  overlays them). This is the sharp test: it has to reproduce the
                  actual deflection, not just its average.

  FORCE ERROR     the engine's force against `F0` alone - no `K_cut`, no
                  `C_cut`, no integration - but both REVOLUTION-AVERAGED first
                  (`compare_linear.revolution_average`, one spindle turn), the
                  same quantity `analysis.forces` compares and for the reason
                  given there: `F0` is a tooth-period average with no
                  tooth-passing ripple in it at all, so comparing it
                  sample-for-sample would charge the model for a feature it
                  never claimed. This is the LOW-NOISE half - on a settled cell
                  the two agree to under 1%, so a large number here is the
                  engagement geometry or the coefficients, not integration noise.

Both are reported as an RMS in their own units and as a RELATIVE RMS - the error
RMS divided by the reference signal's OWN RMS over the same window - because an
absolute number means nothing without knowing how large the signal it is a
fraction of was.

THE QUESTION THIS IS FOR

Does the error fall as tooth-passing frequency rises? The averaging the linear
model performs is exact in the limit TPF -> infinity (infinitely many teeth
averaged over infinitely fast passes) and exact at TPF -> 0 is not claimed at
all, so a monotonic fall is the expected shape, not a given - `figure_vs_tpf`
draws it for both designs so the trend can be read off directly, in the same
`grid` (fixed chip load) / `rows` (fixed feed, null test) split `sweep.py`
itself uses.

TPF IS A PRODUCT, rpm * teeth. A single number cannot say whether the model's
error is a function of that product alone or whether rpm and tooth count matter
separately at fixed TPF - two cells with the same TPF but different factors
(2000 rpm x 4 teeth and 4000 rpm x 2 teeth, say) are not in this sweep's own
grid, which holds `fz` fixed rather than TPF, but the EXISTING `grid` design's
(rpm, teeth) plane is exactly the right shape to look for it in some other way:
`figure_rpm_teeth_grid` plots the relative force error as a coloured grid over
(rpm, teeth) with iso-TPF diagonals overlaid, so a colour that tracks the
diagonals says TPF is the sufficient variable and a colour that tracks rpm or
teeth on their own says it is not.

TOOLPATH REGENERATION. A sweep produced on another machine (or checked out
without `data/toolpaths/`) is missing the `.npz` files `stability.prepare`
replays - `compare_linear.schedule` does not rebuild them, it expects them
already on disk exactly as `main.py`/`sweep.py` leave them. `_ensure_toolpaths`
does that rebuild once per unique toolpath name before any cell is scored, using
`analysis.toolpath.ensure` on the run's own (repinned) config - which is exact
and deterministic, not an approximation, since the feedplan is a pure function
of the part and the feed.

COST. Each cell RK4-integrates the linear model over its whole pass, ~15-20 s on
this machine regardless of `--design`. `--resume` skips a cell already in
`error.csv`, so an interrupted run does not restart from zero.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis                                                # noqa: E402
import compare_linear as cl                                    # noqa: E402
from analysis import plots, save, toolpath                     # noqa: E402
from analysis.forces import ENGAGED_FLOOR_N, ENGAGED_FRAC       # noqa: E402
from runconfig import RunConfig                                # noqa: E402

#: A raw SAMPLE counts as cutting above this force [N] - the same convention
#: `analysis.growth` uses, so "engaged" means the same thing there and here.
ENGAGED_N = 50.0

#: A REVOLUTION-AVERAGE, unlike a raw sample, cannot use a fixed newton floor:
#: a 1-tooth cut is a brief spike buried in mostly air, so its revolution mean
#: can sit under 50 N while the pass is unmistakably cutting - `analysis.forces`
#: has the same problem and the same fix, a threshold relative to the PASS'S
#: OWN peak rather than an absolute one. Reused here rather than restated.

#: The window this script scores: the middle third of the engaged span.
INNER_THIRD = (1.0 / 3.0, 2.0 / 3.0)


# ─────────────────────────────────────────────────────────────────────────────
# Finding cells and their toolpaths
# ─────────────────────────────────────────────────────────────────────────────

def _design_of(run_name: str) -> str:
    """"grid_..." or "rows_..." -> that tag; a prefixed variant like
    "shapely_grid_..." (a different engine driving the same cell naming) counts
    the same way, so a second engine's sweep groups with the first's rather than
    falling into the ungrouped bucket."""
    for tag in ("grid", "rows"):
        if run_name.startswith(tag + "_") or f"_{tag}_" in run_name:
            return tag
    return "?"


def find_cells(sweep_dir: Path, design: str = "both") -> list:
    """Run directories under `<sweep_dir>/runs/` with a coupled pass to score."""
    out = []
    for d in sorted((sweep_dir / "runs").glob("*")):
        if not (d / "raw" / "coupled.npz").exists():
            continue
        des = _design_of(d.name)
        if design != "both" and des != design:
            continue
        out.append(d)
    return out


def _ensure_toolpaths(dirs: list, *, verbose: bool = False) -> None:
    """Rebuild every toolpath these runs need, once per unique name.

    Deterministic from each run's own (repinned) config - see the module
    docstring. Cheap (a feedplan, not a simulation), so this always runs rather
    than being made conditional on a missing file.
    """
    seen = set()
    for d in dirs:
        cfg = cl._repin(RunConfig.load(str(d / "config.json")))
        name = cfg.path.toolpath
        if not name or name in seen:
            continue
        seen.add(name)
        profile = "flying" if "_flying" in name else "ramped"
        toolpath.ensure(cfg, profile=profile, verbose=verbose)
    if seen:
        print(f"toolpath {len(seen)} unique path(s) checked/rebuilt")


# ─────────────────────────────────────────────────────────────────────────────
# Scoring one cell
# ─────────────────────────────────────────────────────────────────────────────

def _inner_third(cut: np.ndarray) -> tuple:
    """(a, b) indices of the middle third of the True span of `cut`, or (0, 0)."""
    idx = np.flatnonzero(cut)
    if idx.size < 30:                       # not enough of a span to thirds it
        return 0, 0
    i0, i1 = int(idx[0]), int(idx[-1])
    n = i1 - i0
    return i0 + int(INNER_THIRD[0] * n), i0 + int(INNER_THIRD[1] * n)


def _rms(x: np.ndarray) -> float:
    """RMS of the row-wise Euclidean norm of an (n, 3) array."""
    return float(np.sqrt((x ** 2).sum(1).mean())) if len(x) else np.nan


def score(d: Path, *, ds_mm: float = 2.0, drive: str = "as-run",
          verbose: bool = False) -> dict:
    """TCP and (revolution-averaged) force RMS error, on the inner third."""
    name = d.name
    row = {"run": name, "design": _design_of(name), "error_valid": False}
    try:
        sch = cl.schedule(d, ds_mm=ds_mm, drive=drive, verbose=verbose)
        lin = cl.integrate(sch)
    except Exception as exc:                         # a cell is data, not a stop
        row["error"] = str(exc)
        return row

    m = sch["mill"]
    row.update({"spindle_rpm": float(m.spindle_rpm), "n_teeth": int(m.n_teeth),
                "tpf_hz": float(m.spindle_rpm * m.n_teeth / 60.0),
                "fz_mm": float(m.feed_per_tooth_mm())})

    # ── TCP error: coupled truth vs the linear model's OWN trajectory ────────
    cut = np.linalg.norm(sch["force_w"], axis=1) > ENGAGED_N
    a, b = _inner_third(cut)
    if b > a and np.isfinite(lin["dx_um"][a:b]).all():
        err = sch["dev_um"][a:b] - lin["dx_um"][a:b]
        row["tcp_n_samples"] = int(b - a)
        row["tcp_rms_err_um"] = _rms(err)
        row["tcp_rms_sim_um"] = _rms(sch["dev_um"][a:b])
        row["tcp_rel_err"] = (row["tcp_rms_err_um"] / row["tcp_rms_sim_um"]
                              if row["tcp_rms_sim_um"] > 0 else np.nan)
    else:
        row.update({"tcp_n_samples": 0, "tcp_rms_err_um": np.nan,
                    "tcp_rms_sim_um": np.nan, "tcp_rel_err": np.nan})

    # ── force error: engine vs F0, both revolution-averaged first ────────────
    t_r, F_r = cl.revolution_average(sch["t"], sch["force_w"], m)
    _t, F0_r = cl.revolution_average(sch["t"], sch["F0_t"], m)
    mag_r = np.linalg.norm(F_r, axis=1)
    peak_r = float(mag_r.max()) if mag_r.size else 0.0
    cut_r = (mag_r > ENGAGED_FRAC * peak_r if peak_r >= ENGAGED_FLOOR_N
             else np.zeros(len(mag_r), bool))
    ar, br = _inner_third(cut_r)
    if br > ar:
        ferr = F_r[ar:br] - F0_r[ar:br]
        row["force_n_rev"] = int(br - ar)
        row["force_rms_err_N"] = _rms(ferr)
        row["force_rms_sim_N"] = _rms(F_r[ar:br])
        row["force_rel_err"] = (row["force_rms_err_N"] / row["force_rms_sim_N"]
                                if row["force_rms_sim_N"] > 0 else np.nan)
    else:
        row.update({"force_n_rev": 0, "force_rms_err_N": np.nan,
                    "force_rms_sim_N": np.nan, "force_rel_err": np.nan})

    row["error_valid"] = bool(b > a and br > ar)
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def figure_vs_tpf(path, rows):
    """Relative TCP and force RMS error against tooth-passing frequency."""
    plt = plots._mpl(backend="Agg" if path else None)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    markers = {"grid": "o", "rows": "^", "?": "s"}

    for ax, key, title, ylabel in (
        (ax1, "tcp_rel_err", "TCP deviation: linear vs coupled", "relative RMS error"),
        (ax2, "force_rel_err", "revolution-mean force: F0 vs coupled",
         "relative RMS error"),
    ):
        plots._clean(ax)
        seen = set()
        for r in rows:
            if not r.get("error_valid") or not np.isfinite(r.get(key, np.nan)):
                continue
            c = plots.TEETH_COLOUR.get(int(r["n_teeth"]), plots.INK)
            mk = markers.get(r["design"], "s")
            label = None
            if r["design"] not in seen:
                label = r["design"]
                seen.add(r["design"])
            ax.plot(r["tpf_hz"], 100 * r[key], marker=mk, ms=7, color=c,
                    mec=c, mfc=c, ls="", zorder=3, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("tooth-passing frequency  [Hz]")
        ax.set_ylabel(f"{ylabel}  [%]")
        ax.set_title(title, loc="left", color=plots.INK)
        ax.set_ylim(bottom=0)
        ax.legend(title="design", fontsize=7.5)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", mfc=plots.TEETH_COLOUR[n],
                       mec=plots.TEETH_COLOUR[n], label=f"{n} tooth" + ("" if n == 1 else "s"))
               for n in (1, 2, 4, 8) if any(int(r.get("n_teeth", -1)) == n for r in rows)]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8,
              bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("error vs tooth-passing frequency, inner third of the engaged span",
                x=0.01, ha="left", fontsize=10, color=plots.INK_2, y=1.1)
    fig.tight_layout()
    if path:
        return plots._save(fig, path)
    plt.show()
    return None


def figure_rpm_teeth_grid(path, rows):
    """Relative force RMS error over (rpm, teeth), grid design only, with
    iso-TPF diagonals - the sufficiency test for TPF as the controlling
    variable."""
    grid = [r for r in rows if r.get("design") == "grid" and r.get("error_valid")
            and np.isfinite(r.get("force_rel_err", np.nan))]
    if not grid:
        return None
    plt = plots._mpl(backend="Agg" if path else None)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    plots._clean(ax, grid=None)

    rpms = sorted({r["spindle_rpm"] for r in grid})
    teeth = sorted({r["n_teeth"] for r in grid})
    err = np.full((len(teeth), len(rpms)), np.nan)
    for r in grid:
        i = teeth.index(r["n_teeth"])
        j = rpms.index(r["spindle_rpm"])
        err[i, j] = 100 * r["force_rel_err"]

    im = ax.imshow(err, aspect="auto", origin="lower", cmap="magma_r",
                   extent=(-0.5, len(rpms) - 0.5, -0.5, len(teeth) - 0.5))
    ax.set_xticks(range(len(rpms)), [f"{r:g}" for r in rpms])
    ax.set_yticks(range(len(teeth)), [f"{n:g}" for n in teeth])
    ax.set_xlabel("spindle speed  [rpm]")
    ax.set_ylabel("teeth")
    ax.set_title("relative force RMS error  [%]  -  is TPF the whole story?",
                loc="left", color=plots.INK)
    for i, n in enumerate(teeth):
        for j, rpm in enumerate(rpms):
            if np.isfinite(err[i, j]):
                ax.text(j, i, f"{err[i, j]:.1f}", ha="center", va="center",
                        fontsize=8, color=plots.SURFACE if err[i, j] > np.nanmax(err) * 0.5
                        else plots.INK)

    # iso-TPF diagonals: tpf = rpm * n / 60, drawn in (rpm-index, teeth-index)
    # space by interpolating each teeth row onto the fractional rpm-index its
    # tpf would sit at, so a straight line here is a straight line in log(rpm)
    # only where the rpm axis itself is log-spaced - drawn as connected points.
    log_rpm = np.log(rpms)
    tpf_all = sorted({r["tpf_hz"] for r in grid})
    for tpf in tpf_all:
        xs, ys = [], []
        for i, n in enumerate(teeth):
            want_rpm = 60.0 * tpf / n
            lw = np.log(want_rpm)
            if lw < log_rpm[0] - 1e-6 or lw > log_rpm[-1] + 1e-6:
                continue
            xf = np.interp(lw, log_rpm, np.arange(len(rpms)))
            xs.append(xf)
            ys.append(i)
        if len(xs) >= 2:
            ax.plot(xs, ys, color=plots.SURFACE, lw=1.0, ls=(0, (3, 2)),
                    alpha=0.55, zorder=2)

    fig.colorbar(im, ax=ax, label="relative RMS error  [%]")
    fig.tight_layout()
    if path:
        return plots._save(fig, path)
    plt.show()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", default=None,
                   help="sweep directory (default: out/tooth_passing_v4)")
    p.add_argument("--design", default="both", choices=("grid", "rows", "both"))
    p.add_argument("--cells", default=None, metavar="NAME,...",
                   help="only these run directory names")
    p.add_argument("--drive", default="as-run", choices=cl.DRIVES,
                   help="what drives the linear model - see compare_linear.py")
    p.add_argument("--ds", type=float, default=2.0, metavar="MM",
                   help="arc-length spacing the cut matrices are rebuilt on")
    p.add_argument("--resume", action="store_true",
                   help="skip any run already in error.csv")
    p.add_argument("--save-dir", default=None,
                   help="write figures/results here (default: <sweep>/figures)")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    sweep_dir = Path(a.sweep) if a.sweep else analysis.OUT / "tooth_passing_v4"
    save_dir = Path(a.save_dir) if a.save_dir else sweep_dir / "figures"
    csv_path = save_dir / "error.csv"

    dirs = find_cells(sweep_dir, a.design)
    if a.cells:
        want = {c.strip() for c in a.cells.split(",") if c.strip()}
        dirs = [d for d in dirs if d.name in want]
    if not dirs:
        raise SystemExit(f"no coupled run under {sweep_dir}/runs matches")

    cached = {}
    if a.resume and csv_path.exists():
        import csv
        with open(csv_path, newline="", encoding="utf-8") as f:
            cached = {r["run"]: r for r in csv.DictReader(f)}

    _ensure_toolpaths(dirs, verbose=a.verbose)

    print(f"score    {len(dirs)} cell(s) from {sweep_dir}, inner third of the "
          f"engaged span, drive {a.drive!r}")
    rows = []
    for i, d in enumerate(dirs, 1):
        if d.name in cached:
            r = cached[d.name]
            for k in list(r):
                if k not in ("run", "design", "error"):
                    try:
                        r[k] = float(r[k])
                    except (TypeError, ValueError):
                        pass
            r["error_valid"] = str(r.get("error_valid")).lower() == "true"
            rows.append(r)
            print(f"[{i:2d}/{len(dirs)}] {d.name}: reused")
            continue
        t0 = time.time()
        r = score(d, ds_mm=a.ds, drive=a.drive, verbose=a.verbose)
        dt = time.time() - t0
        if r.get("error_valid"):
            print(f"[{i:2d}/{len(dirs)}] {d.name}   tcp {r['tcp_rel_err']*100:5.2f} % "
                  f"| force {r['force_rel_err']*100:5.2f} %   ({dt:.0f} s)")
        else:
            print(f"[{i:2d}/{len(dirs)}] {d.name}   ! {r.get('error', 'no engaged span')}")
        rows.append(r)

    save.write_csv(csv_path, rows)
    print(f"\nresults  {csv_path}  ({len(rows)} rows)")

    if not a.no_plot:
        ok = [r for r in rows if r.get("error_valid")]
        if ok:
            out1 = figure_vs_tpf(save_dir / "error_vs_tpf.png", ok)
            print(f"figure   {out1}")
            out2 = figure_rpm_teeth_grid(save_dir / "force_error_rpm_teeth.png", ok)
            if out2:
                print(f"figure   {out2}")
    return rows


if __name__ == "__main__":
    main()
