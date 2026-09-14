"""World-frame driven path and cutting force for the fixed-pose sweep's critical cells.

    python plot_fixedpose_critical.py
    python plot_fixedpose_critical.py --sweep out/rotation_fixedpose --alphas 0,90,180

One figure per cell (2 panels: the commanded XY path in the BASE frame, and the
cutting force's three base-frame components against time), plus a text report
comparing the predicted verdict/growth (`pred_growth_mid_1_s`, the eigenvalue at
the tap node) against what the twin pulse-tap actually measured
(`sim_sigma_1_s`) - the same pairing `sweep_rotation_ae.py` uses, read straight
off `sweep.csv` rather than re-computed.

Reads `raw/coupled.npz` from each cell's PLAIN (untapped) run - `nominal_i`
(commanded TCP position, base frame, m) and `force_i` (cutting force, base
frame, N) against `t` - so the path shown is the actual driven path, not the
tap perturbation.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sweep_rotation_fixedpose import cell_name                # noqa: E402


def _f(row, key):
    try:
        v = float(row.get(key))
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def load_rows(sweep_dir: Path) -> list:
    return list(csv.DictReader(open(sweep_dir / "sweep.csv", encoding="utf-8")))


def plot_cell(path: Path, run_dir: Path, row: dict):
    from analysis import plots
    plt = plots._mpl(backend="Agg")

    d = np.load(run_dir / "raw" / "coupled.npz", allow_pickle=True)
    t = d["t"]
    pos = d["nominal_i"] * 1000.0            # m -> mm, base frame
    force = d["force_i"]                     # N, base frame

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    plots._clean(ax1)
    plots._clean(ax2)

    ax1.plot(pos[:, 0], pos[:, 1], "-", color=plots.INK, lw=1.2)
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.set_xlabel("X  base frame  [mm]")
    ax1.set_ylabel("Y  base frame  [mm]")
    ax1.set_title("driven path, world frame", loc="left")

    for i, (label, colour) in enumerate(
            zip(("Fx", "Fy", "Fz"), ("#c62828", "#2e7d32", "#1565c0"))):
        ax2.plot(t, force[:, i], "-", color=colour, lw=0.8, label=label)
    t_pulse = _f(row, "pulse_t_s")
    t0 = _f(row, "pulse_fit_t0_s")
    t1 = _f(row, "pulse_fit_t1_s")
    if t_pulse is not None:
        ax2.axvline(t_pulse, color=plots.INK_2, lw=1.0, ls=":", label="tap")
    if t0 is not None and t1 is not None:
        ax2.axvspan(t0, t1, color=plots.INK_2, alpha=0.08, label="fit window")
    ax2.set_xlabel("t  [s]")
    ax2.set_ylabel("cutting force, base frame  [N]")
    ax2.set_title("force, world frame", loc="left")
    ax2.legend(fontsize=8, ncol=4, loc="upper right")

    alpha = _f(row, "alpha_deg")
    lin = _f(row, "pred_growth_mid_1_s")
    sim = _f(row, "sim_sigma_1_s")
    lin_state = row.get("lin_state", "?")
    sim_state = row.get("sim_state", "?")
    agree = "agree" if lin_state == sim_state else "DISAGREE"
    fig.suptitle(
        f"alpha {alpha:.0f} deg  |  predicted {lin_state} ({lin:+.2f} 1/s)  "
        f"vs measured {sim_state} ({'n/a' if sim is None else f'{sim:+.2f} 1/s'})  "
        f"- {agree}", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return plots._save(fig, path)


def report(rows: list) -> str:
    lines = [f"{'alpha':>6} {'predicted':>10} {'sigma_lin':>10} {'measured':>10} "
             f"{'sigma_sim':>10} {'verdict':>10}"]
    for r in rows:
        alpha = _f(r, "alpha_deg")
        lin = _f(r, "pred_growth_mid_1_s")
        sim = _f(r, "sim_sigma_1_s")
        lin_state = r.get("lin_state", "?")
        sim_state = r.get("sim_state", "?")
        verdict = "agree" if lin_state == sim_state else "DISAGREE"
        lines.append(f"{alpha:6.0f} {lin_state:>10} {lin:10.2f} {sim_state:>10} "
                     f"{'n/a' if sim is None else f'{sim:10.2f}'} {verdict:>10}")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sweep", default="out/rotation_fixedpose", metavar="DIR")
    p.add_argument("--alphas", default=None, metavar="A,A,...",
                   help="only these alphas (default: every simulated cell)")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    sweep_dir = Path(a.sweep)
    rows = load_rows(sweep_dir)
    want = ({float(x) for x in a.alphas.split(",")} if a.alphas else None)

    sim_rows = [r for r in rows if r.get("sim_state") not in (None, "", "unmeasured")
               and (want is None or _f(r, "alpha_deg") in want)]
    sim_rows.sort(key=lambda r: _f(r, "alpha_deg"))

    print(f"{len(sim_rows)} simulated cell(s) in {sweep_dir}\n")
    print(report(sim_rows))
    print()

    fig_dir = sweep_dir / "figures" / "critical_cells"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for r in sim_rows:
        alpha = _f(r, "alpha_deg")
        run_dir = sweep_dir / "runs" / cell_name(alpha)
        if not (run_dir / "raw" / "coupled.npz").exists():
            print(f"  ! alpha {alpha:.0f}: no raw/coupled.npz, skipped")
            continue
        out = plot_cell(fig_dir / f"alpha{alpha:03.0f}_world.png", run_dir, r)
        print(f"  figure  alpha {alpha:5.0f} deg  ->  {out}")
    return sim_rows


if __name__ == "__main__":
    main()
