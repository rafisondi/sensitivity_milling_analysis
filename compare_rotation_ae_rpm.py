"""Does spindle speed change the orientation x engagement stability map?

    python compare_rotation_ae_rpm.py --sweeps out/rotation_ae_rpm5000,out/rotation_ae_rpm10000

`sweep_rotation_ae.py` holds rpm fixed and asks how STABLE the halfway node is
across cut direction and radial engagement. Running it again at a different rpm
asks the next question directly: is that map a property of the geometry alone,
or does the rpm-dependent term (`C_cut ~ 1/Omega`) move it?

ONE FIGURE, FOUR PANELS - one per ae, `pred_growth_mid_1_s` against alpha, one
line per rpm sweep given. Where the lines sit on top of each other, rpm does not
matter there; where they split, it does. A cell with a resolved SIMULATED verdict
is marked - filled if it agrees with the model's sign, an open ring if it does
not - so the same figure shows both "does rpm move the prediction" and "was the
prediction checked, and did it hold up" at once.

WHAT THIS DOES NOT DO. It does not re-run anything - every sweep passed in must
already have a `sweep.csv` (i.e. finished, `--replot`-able). A sweep whose
coupled side used too short a stock for the tap to resolve at that rpm (see
`sweep_rotation_ae.py --part-length`) will simply show no simulated markers,
which is itself the fact this script exists to surface, not something it
silently works around.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis import plots                                    # noqa: E402

RPM_COLOUR_MAP = "viridis"


def _f(r, k):
    try:
        v = float(r[k])
        return v
    except (TypeError, ValueError, KeyError):
        return np.nan


def load(path) -> list:
    import csv
    return list(csv.DictReader(open(Path(path) / "sweep.csv", encoding="utf-8")))


def figure_growth_vs_alpha(path, sweeps: dict):
    """sweeps: {rpm: rows} - one polar-free line-plot per ae, coloured by rpm."""
    plt = plots._mpl(backend="Agg" if path else None)

    all_rows = [r for rows in sweeps.values() for r in rows]
    aes = sorted({_f(r, "ae_pct") for r in all_rows if np.isfinite(_f(r, "ae_pct"))})
    rpms = sorted(sweeps.keys())
    colours = {rpm: plt.get_cmap(RPM_COLOUR_MAP)(i / max(1, len(rpms) - 1))
              for i, rpm in enumerate(rpms)}

    fig, axes = plt.subplots(1, len(aes), figsize=(4.6 * len(aes), 4.6),
                             sharey=True)
    axes = np.atleast_1d(axes)

    for ax, ae in zip(axes, aes):
        plots._clean(ax)
        for rpm in rpms:
            rows = [r for r in sweeps[rpm] if _f(r, "ae_pct") == ae]
            rows.sort(key=lambda r: _f(r, "alpha_deg"))
            alpha = np.array([_f(r, "alpha_deg") for r in rows])
            g = np.array([_f(r, "pred_growth_mid_1_s") for r in rows])
            c = colours[rpm]
            ax.plot(alpha, g, "-", color=c, lw=1.6, label=f"{rpm:g} rpm", zorder=2)

            sim_rows = [r for r in rows if r.get("sim_state") not in
                       (None, "", "unmeasured")]
            for r in sim_rows:
                a = _f(r, "alpha_deg")
                gv = _f(r, "pred_growth_mid_1_s")
                agree = r["lin_state"] == r["sim_state"]
                ax.plot(a, gv, marker="o", ms=8, mfc=(c if agree else "none"),
                        mec=c, mew=1.8, ls="", zorder=3)

        ax.axhline(0.0, color=plots.INK, lw=0.9, zorder=1)
        ax.set_xlabel("workpiece rotation  alpha  [deg]")
        ax.set_title(f"ae = {ae:.0f}% D", loc="left", color=plots.INK)
    axes[0].set_ylabel("growth rate at the halfway node  [1/s]")
    axes[-1].legend(loc="upper right", fontsize=8, title="linear model")

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", mfc=plots.INK_2, mec=plots.INK_2,
                       label="simulated, agrees with the model's sign"),
               Line2D([0], [0], marker="o", ls="", mfc="none", mec=plots.INK_2,
                       mew=1.8, label="simulated, DISAGREES")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8,
              bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("orientation x engagement: does spindle speed move the map?",
                x=0.01, ha="left", fontsize=11, color=plots.INK, fontweight="medium")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    if path:
        return plots._save(fig, path)
    plt.show()
    return None


def report(sweeps: dict) -> str:
    lines = []
    for rpm, rows in sorted(sweeps.items()):
        sim_rows = [r for r in rows if r.get("sim_state") not in
                   (None, "", "unmeasured")]
        agree = sum(1 for r in sim_rows if r["lin_state"] == r["sim_state"])
        lines.append(f"{rpm:g} rpm: {len(rows)} cells predicted, "
                     f"{len(sim_rows)} simulated, {agree}/{len(sim_rows)} agree "
                     f"with the model's sign" if sim_rows else
                     f"{rpm:g} rpm: {len(rows)} cells predicted, "
                     f"0 simulated (no resolved verdict - check the stock "
                     f"length against this rpm's feed)")

    # rpm-sensitivity of ap_crit_mid, cell by cell, where every rpm has a value
    rpms = sorted(sweeps.keys())
    if len(rpms) >= 2:
        keyed = {rpm: {(_f(r, "alpha_deg"), _f(r, "ae_pct")): _f(r, "pred_ap_crit_mid_mm")
                       for r in sweeps[rpm]} for rpm in rpms}
        common = set(keyed[rpms[0]])
        for rpm in rpms[1:]:
            common &= set(keyed[rpm])
        moved, still = [], 0
        for cell in common:
            vals = [keyed[rpm][cell] for rpm in rpms]
            if not all(np.isfinite(vals)):
                continue
            spread = (max(vals) - min(vals)) / max(1e-9, min(vals))
            if spread > 0.05:
                moved.append((cell, vals))
            else:
                still += 1
        lines.append("")
        lines.append(f"ap_crit_mid: {still} of {still + len(moved)} common cells "
                     f"change by <5% across {rpms} rpm; {len(moved)} move more "
                     f"than that")
        for (a, ae), vals in sorted(moved, key=lambda kv: -max(kv[1])):
            txt = "  ".join(f"{rpm:g}rpm={v:.1f}mm" for rpm, v in zip(rpms, vals))
            lines.append(f"  alpha={a:5.0f} ae={ae:4.0f}%   {txt}")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweeps", required=True, metavar="DIR,DIR,...",
                   help="rotation_ae sweep directories to compare, each with its "
                        "own rpm - the rpm is read from sweep.json")
    p.add_argument("--save-dir", default=None,
                   help="write the figure here (default: out/rotation_ae_rpm_compare)")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    import json
    sweeps = {}
    for d in a.sweeps.split(","):
        d = Path(d.strip())
        meta = json.loads((d / "sweep.json").read_text(encoding="utf-8"))
        sweeps[float(meta["rpm"])] = load(d)

    print(report(sweeps))

    if not a.no_plot:
        save_dir = Path(a.save_dir) if a.save_dir else Path("out") / "rotation_ae_rpm_compare"
        out = figure_growth_vs_alpha(save_dir / "growth_vs_alpha_by_rpm.png", sweeps)
        print(f"\nfigure   {out}")
    return sweeps


if __name__ == "__main__":
    main()
