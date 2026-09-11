"""Does the linear growth rate track the engine's, cell by cell across a sweep?

    python compare_growth.py                       out/tooth_passing, both designs
    python compare_growth.py --sweep out/tooth_passing --design grid
    python compare_growth.py --save-dir out/tooth_passing/figures

WHAT "GROWTH RATE" MEANS HERE, AND WHY THAT DEFINITION

The prediction (`analysis.stability`) is an eigenvalue: `max Re(eig)` of the
closed-loop system frozen at each node, in 1/s, no time in it. The engine has no
such number — only a deflection time series — so one has to be MEASURED out of
it, on a footing an eigenvalue's growth rate is comparable to.

`analysis.report.chatter_metrics` already does this, and it is reused here
rather than re-derived:

    1. band-pass the TCP deviation to a window around the arm's structural modes
       (0.4-2.5x the lowest/highest mode) — this is where regenerative growth
       lives on this rig, since tooth-passing (100+ Hz) sits ABOVE the modes
       (9-23 Hz) instead of below them as in most machine-tool chatter, so both
       the entry/exit transient (~0.4 Hz) and the tooth-passing ripple fall
       outside the band on their own;
    2. take the running RMS envelope of the band-passed signal over ~32 windows;
    3. fit log(envelope) vs time by least squares — the slope is the growth
       rate, `sim_growth_r2` is how well an exponential actually describes it;
    4. trim 15% off each end of the engaged span first, so the entry/exit ramp
       (forced motion, not regenerative growth) does not bias the slope.

That trim is also why THIS script compares against `pred_growth_max_trim_1_s`
and not `pred_growth_max_1_s`: the untrimmed prediction is dominated by the
entry node (partial radial engagement, reliably the largest eigenvalue on the
path), a span the time-domain fit throws away on purpose. Pairing an untrimmed
maximum with a trimmed measurement would be comparing different parts of the
cut; `analysis.report.agreement` makes the same choice for its `_trim` fields,
and `overlay_growth.png` (`analysis.plots.figure_sweep_overlay`) already draws
both sides against rpm, split by tooth count.

WHAT IS NEW HERE. The sweep figures show the two curves vs. rpm, one panel per
tooth count — good for seeing where they track and where they part ways, but
not for reading off, in one glance, how tight the correspondence is overall.
This script adds the complementary view: a PARITY plot, predicted vs. measured,
one point per cell, everything on one axis. Perfect agreement is the diagonal;
systematic bias is an offset from it; scatter around it is the noise floor of
this comparison. Pearson r, RMSE and MAE are computed on it, both for every
valid cell and for the subset the fit actually trusts (`sim_growth_r2` at or
above `R2_TRUSTED`, the same 0.30 line `analysis.plots` already draws hollow
markers at) — the untrusted points are shown, since where the measurement gives
out is itself part of the answer, but they should not move the headline number.

DATA. Read straight from `summary.json` under `<sweep>/runs/*/`, not from
`sweep.csv` — this only wants the handful of scalars either side already
reduced its run to, and reading the per-run files means it works on a sweep
directory that was assembled by hand or is missing its aggregate CSV.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis                        # noqa: E402
from analysis import plots             # noqa: E402
from analysis.plots import R2_TRUSTED  # noqa: E402

PRED_KEY = "pred_growth_max_trim_1_s"
SIM_KEY = "sim_growth_1_s"


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def _design_of(run_name: str) -> str:
    for tag in ("grid", "rows"):
        if run_name.startswith(tag + "_"):
            return tag
    return "?"


def load_rows(sweep_dir: Path, design: str = "both") -> list:
    """One dict per run under `<sweep_dir>/runs/*/summary.json`."""
    rows = []
    for f in sorted((sweep_dir / "runs").glob("*/summary.json")):
        row = json.loads(f.read_text(encoding="utf-8"))
        row.setdefault("run", f.parent.name)
        row["design"] = _design_of(row["run"])
        if design != "both" and row["design"] != design:
            continue
        rows.append(row)
    return rows


def paired(rows: list) -> list:
    """Rows with both a prediction and a trustworthy-enough measurement to plot.

    `sim_valid` is the gate the rest of the pipeline uses: it is true for any
    pass that ran long enough to fit an envelope to (including a diverged one),
    false for a cell never simulated at all or too short to measure. Only rows
    with both keys finite belong in a growth-rate comparison at all.
    """
    out = []
    for r in rows:
        if not r.get("sim_valid"):
            continue
        p, s = r.get(PRED_KEY), r.get(SIM_KEY)
        if p is None or s is None:
            continue
        if not (np.isfinite(p) and np.isfinite(s)):
            continue
        out.append(r)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────────────────────

def parity_stats(rows: list) -> dict:
    """Pearson r, RMSE, MAE and verdict agreement of pred vs. sim growth rate."""
    if len(rows) < 2:
        return {"n": len(rows), "r": np.nan, "rmse": np.nan, "mae": np.nan,
                "bias": np.nan, "verdict_agree_frac": np.nan}
    p = np.array([r[PRED_KEY] for r in rows], float)
    s = np.array([r[SIM_KEY] for r in rows], float)
    r = (float(np.corrcoef(p, s)[0, 1])
         if p.std() > 1e-12 and s.std() > 1e-12 else np.nan)
    err = s - p
    verdict_agree = [bool(r_.get("pred_unstable_trim")) == bool(r_.get("sim_unstable"))
                      for r_ in rows]
    return {"n": len(rows), "r": r,
            "rmse": float(np.sqrt((err ** 2).mean())),
            "mae": float(np.abs(err).mean()),
            "bias": float(err.mean()),
            "verdict_agree_frac": float(np.mean(verdict_agree))}


def report(rows: list) -> str:
    trusted = [r for r in rows if np.isfinite(r.get("sim_growth_r2", np.nan))
               and r["sim_growth_r2"] >= R2_TRUSTED]
    all_stats, trust_stats = parity_stats(rows), parity_stats(trusted)

    lines = [
        f"cells    {len(rows)} with both a prediction and a measurement "
        f"({len(trusted)} with sim_growth_r2 >= {R2_TRUSTED:g})",
        "",
        f"{'':<10}{'n':>4}{'r':>8}{'rmse':>9}{'mae':>9}{'bias':>9}{'verdict':>10}",
        f"{'all':<10}{all_stats['n']:>4}{all_stats['r']:>8.3f}"
        f"{all_stats['rmse']:>9.3f}{all_stats['mae']:>9.3f}"
        f"{all_stats['bias']:>9.3f}{all_stats['verdict_agree_frac']:>9.0%}",
        f"{'trusted':<10}{trust_stats['n']:>4}{trust_stats['r']:>8.3f}"
        f"{trust_stats['rmse']:>9.3f}{trust_stats['mae']:>9.3f}"
        f"{trust_stats['bias']:>9.3f}{trust_stats['verdict_agree_frac']:>9.0%}",
        "         (rmse/mae/bias in 1/s, sim - pred; verdict = pred_unstable_trim "
        "== sim_unstable)",
        "",
        f"{'run':<28}{'design':>7}{'rpm':>8}{'z':>3}{'pred_trim':>11}"
        f"{'sim':>9}{'err':>8}{'sim_r2':>8}",
    ]
    for r in sorted(rows, key=lambda r: (r["design"], r["spindle_rpm"], r["n_teeth"])):
        err = r[SIM_KEY] - r[PRED_KEY]
        weak = "*" if r.get("sim_growth_r2", 0.0) < R2_TRUSTED else " "
        lines.append(
            f"{r['run']:<28}{r['design']:>7}{r['spindle_rpm']:>8.0f}"
            f"{r['n_teeth']:>3d}{r[PRED_KEY]:>11.3f}{r[SIM_KEY]:>9.3f}"
            f"{err:>8.3f}{weak}{r.get('sim_growth_r2', float('nan')):>7.2f}")
    lines.append("         * sim_growth_r2 below the trusted line - shown, not evidence")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# The parity figure
# ─────────────────────────────────────────────────────────────────────────────

def figure_parity(path, rows, *, title="growth rate: linear model vs. the "
                                        "coupled simulation, all cells"):
    """One point per cell: predicted (trimmed) growth rate against measured.

    The diagonal is perfect agreement. Colour is tooth count, the same fixed
    assignment every other sweep figure uses; marker shape is the design (grid
    vs. rows), since the two designs probe the model differently
    (`sweep.py`'s own docstring); a hollow marker is a cell whose envelope fit
    does not trust its own slope (`sim_growth_r2 < R2_TRUSTED`) - drawn, because
    hiding it would hide where the measurement gives out, but excluded from the
    correlation headline.
    """
    plt = plots._mpl(backend="Agg" if path else None)
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    plots._clean(ax)

    markers = {"grid": "o", "rows": "^", "?": "s"}
    p_all = np.array([r[PRED_KEY] for r in rows], float)
    s_all = np.array([r[SIM_KEY] for r in rows], float)
    lo = min(p_all.min(), s_all.min())
    hi = max(p_all.max(), s_all.max())
    pad = 0.08 * (hi - lo) if hi > lo else 1.0
    lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], color=plots.MUTED, lw=1.0, ls=(0, (4, 2)),
            zorder=1, label="perfect agreement")
    ax.axhline(0.0, color=plots.GRID, lw=0.8, zorder=0)
    ax.axvline(0.0, color=plots.GRID, lw=0.8, zorder=0)

    seen_design = set()
    for r in rows:
        p, s = r[PRED_KEY], r[SIM_KEY]
        colour = plots.TEETH_COLOUR.get(int(r["n_teeth"]), plots.INK)
        weak = r.get("sim_growth_r2", 0.0) < R2_TRUSTED
        mk = markers.get(r["design"], "s")
        label = None
        if r["design"] not in seen_design:
            label = f"{r['design']}"
            seen_design.add(r["design"])
        ax.plot(p, s, marker=mk, ms=8 if not weak else 9,
                mfc=colour if not weak else plots.SURFACE,
                mec=colour, mew=1.6, ls="", zorder=3, label=label)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"predicted growth rate, trimmed  [1/s]   ({PRED_KEY})")
    ax.set_ylabel(f"measured growth rate  [1/s]   ({SIM_KEY})")
    ax.set_title(title, loc="left", color=plots.INK)

    st = parity_stats(rows)
    trusted = [r for r in rows if r.get("sim_growth_r2", 0.0) >= R2_TRUSTED]
    stt = parity_stats(trusted)
    ax.text(0.02, 0.98,
             f"all (n={st['n']}): r={st['r']:.2f}, rmse={st['rmse']:.2f} 1/s\n"
             f"trusted (n={stt['n']}): r={stt['r']:.2f}, rmse={stt['rmse']:.2f} 1/s",
             transform=ax.transAxes, ha="left", va="top", fontsize=8,
             color=plots.INK_2)

    # legend: tooth colour + marker shape, built by hand since both carry meaning
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=plots.MUTED, lw=1.0, ls=(0, (4, 2)),
                       label="perfect agreement")]
    for n, c in plots.TEETH_COLOUR.items():
        if any(int(r["n_teeth"]) == n for r in rows):
            handles.append(Line2D([0], [0], marker="o", ls="", mfc=c, mec=c,
                                   label=f"{n} tooth" + ("" if n == 1 else "s")))
    for d, mk in markers.items():
        if any(r["design"] == d for r in rows):
            handles.append(Line2D([0], [0], marker=mk, ls="", mfc=plots.INK_2,
                                   mec=plots.INK_2, label=f"design: {d}"))
    handles.append(Line2D([0], [0], marker="o", ls="", mfc=plots.SURFACE,
                           mec=plots.INK_2, mew=1.6,
                           label=f"hollow: sim_growth_r2 < {R2_TRUSTED:g}"))
    ax.legend(handles=handles, loc="lower right", fontsize=7.5)

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
                   help="sweep directory (default: out/tooth_passing)")
    p.add_argument("--design", default="both", choices=("grid", "rows", "both"))
    p.add_argument("--save-dir", default=None,
                   help="write the figure here instead of showing it "
                        "(default: <sweep>/figures/growth_parity.png)")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    sweep_dir = Path(a.sweep) if a.sweep else analysis.OUT / "tooth_passing"

    rows = paired(load_rows(sweep_dir, a.design))
    if not rows:
        raise SystemExit(f"no cell under {sweep_dir} has both a prediction and "
                          f"a valid simulated growth rate")

    print(report(rows))

    if not a.no_plot:
        save_dir = Path(a.save_dir) if a.save_dir else sweep_dir / "figures"
        out = figure_parity(save_dir / "growth_parity.png", rows)
        print(f"\nfigure   -> {out}")

    return rows


if __name__ == "__main__":
    main()
