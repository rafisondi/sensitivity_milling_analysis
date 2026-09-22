"""Plan view of a precut job: what was already milled, and where the run cuts.

    python steadystate_precut/plot_precut.py out/steady_precut_s40_rpm3333

Written into every run's `figures/precut.png` by `pipeline.run`; the command
line redraws it from a finished run directory's `precut.npz`. The simulated
tool centre is not drawn: it sits micrometres off the command, invisible at the
scale of the stock - `figures/trajectory.png` has the deviation.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis.plots import (C2, C3, INK, INK_2, MUTED, _clean,  # noqa: E402
                            _mpl, _save)


def _draw(path, *, virgin, pred, start, end, s0, length, warmup, radius):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    _clean(ax, grid="both")

    v = np.column_stack([virgin, virgin[:, :1]])
    p = np.column_stack([pred, pred[:, :1]])
    ax.fill(v[0], v[1], color=MUTED, alpha=0.12, lw=0, zorder=0)
    ax.plot(v[0], v[1], color=MUTED, lw=1.0, ls="--", zorder=1,
            label="virgin stock")
    ax.fill(p[0], p[1], color=MUTED, alpha=0.30, lw=0, zorder=1)
    ax.plot(p[0], p[1], color=INK_2, lw=1.2, zorder=2,
            label="precut stock (prediction)")

    u = (end - start) / max(np.linalg.norm(end - start), 1e-12)
    w0 = start - u * warmup
    ax.plot([w0[0], start[0]], [w0[1], start[1]], color=C2, lw=3.0,
            solid_capstyle="butt", zorder=3, label="rigid warm-up (engine)")
    ax.plot([start[0], end[0]], [start[1], end[1]], color=C3, lw=3.0,
            solid_capstyle="butt", zorder=3, label="simulated steady cut")

    th = np.linspace(0.0, 2.0 * np.pi, 200)
    ax.plot(start[0] + radius * np.cos(th), start[1] + radius * np.sin(th),
            color=INK, lw=1.0, zorder=5)                    # the tool at t = 0
    ax.plot(*start, "o", color=INK, ms=8, zorder=6)
    ax.text(start[0], start[1] - radius - 1.0, f"tool at t = 0, s0 = {s0:g} mm",
            ha="center", va="top", color=INK, fontsize=9)

    lo = np.minimum(virgin.min(axis=1), np.minimum(start, end) - radius - 8.0)
    hi = np.maximum(virgin.max(axis=1), np.maximum(start, end) + 12.0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_xlabel("x  [mm]")
    ax.set_ylabel("y  [mm]")
    ax.set_title(f"precut job, workpiece frame: opens at s0 = {s0:g} mm, cuts "
                 f"{length:g} mm and stops", loc="left", color=INK)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def figure(path, job):
    """From a live `SteadyJob`."""
    return _draw(path, virgin=job.cfg.part.build().boundary_xy_mm,
                 pred=job.part_pred.boundary_xy_mm, start=job.start_xy_mm,
                 end=job.end_xy_mm, s0=job.s0_mm, length=job.length_mm,
                 warmup=job.warmup_mm, radius=job.edge.radius)


def from_run_dir(d, out=None):
    d = Path(d)
    z = np.load(d / "precut.npz")
    return _draw(out or d / "figures" / "precut.png",
                 virgin=z["virgin_xy_mm"], pred=z["part_pred_xy_mm"],
                 start=z["start_xy_mm"], end=z["end_xy_mm"],
                 s0=float(z["s0_mm"]), length=float(z["length_mm"]),
                 warmup=float(z["warmup_mm"]), radius=float(z["radius_mm"]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir", help="a finished steadystate_precut run directory")
    p.add_argument("--out", default=None, help="output PNG path")
    a = p.parse_args(argv)
    print(f"wrote {from_run_dir(a.run_dir, a.out)}")


if __name__ == "__main__":
    main()
