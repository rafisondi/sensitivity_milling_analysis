"""Plot a milled pass in the ROBOT BASE FRAME — the fixed cell — rather than
the workpiece's own (moving) frame that `analysis.plots.figure_trajectory`
draws in.

    python move_wp_constantRobPose/plot_pass_base_frame.py configs/_move_wp_constantRobPose/02_moved.json

Since `place_at_edge_midpoint` / `move_workpiece` / `attack_angle` all move
the WORKPIECE, not the robot, "the pass in the workpiece frame" looks
identical across every step — the interesting picture is where the cut and
the toolpath land relative to the arm, which only shows up in the base
frame. No robot build is needed: a `placed_by_hand` scene's `T_iw` is read
straight off the saved config.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from runconfig import RunConfig                                # noqa: E402


def base_frame_geometry(cfg):
    """(ring_i_mm, s_i_mm): workpiece outline and toolpath, both base-frame mm."""
    mill, part, scene, waypoints, path = cfg.build()
    ring_w = part.closed_xy_mm                                 # (2, N+1) mm
    ring_i = np.array([scene.point_i(ring_w[:, k])
                       for k in range(ring_w.shape[1])]) * 1e3  # (N+1, 3) mm
    s_i_mm = path.s_i * 1e3                                     # (N, 3) mm
    return ring_i, s_i_mm, scene


def plot(config_path, out_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = RunConfig.load(config_path)
    ring_i, s_i_mm, scene = base_frame_geometry(cfg)

    # The robot base origin sits ~1.6 m away (this arm's reach) while the
    # stock is ~100 mm across — forcing both into one equal-aspect view would
    # make the stock a speck. Zoom to the stock/path instead and point at the
    # base with an annotation.
    pts = np.vstack([ring_i[:, :2], s_i_mm[:, :2]])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = 0.15 * max(hi[0] - lo[0], hi[1] - lo[1], 1.0)
    to_base = -0.5 * (lo + hi)                                  # base frame origin, relative

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.fill(ring_i[:, 0], ring_i[:, 1], color="#cfd8dc", alpha=0.55, zorder=0,
            label="stock")
    ax.plot(ring_i[:, 0], ring_i[:, 1], color="#607d8b", lw=1.3, zorder=1)
    ax.plot(s_i_mm[:, 0], s_i_mm[:, 1], color="#1565c0", lw=2.0, zorder=2,
            label="toolpath")
    ax.plot(s_i_mm[0, 0], s_i_mm[0, 1], "o", color="#2e7d32", ms=9, zorder=3,
            label="path start (lead-in)")

    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x, base frame  [mm]")
    ax.set_ylabel("y, base frame  [mm]")
    dist_m = np.linalg.norm(to_base) * 1e-3
    ax.set_title(f"pass in the ROBOT BASE FRAME — {Path(config_path).stem}\n"
                f"(robot base origin is {dist_m:.2f} m away, off-frame — arm reach)",
                loc="left", fontsize=10)

    # An arrow pointing (in DATA coordinates) toward where the robot base
    # actually sits — bearing only, since the base itself is ~1.6 m off-frame.
    direction = to_base / (np.linalg.norm(to_base) + 1e-12)
    center = 0.5 * (lo + hi)
    tip = center + direction * 0.42 * (hi - lo).max()
    tail = center + direction * 0.20 * (hi - lo).max()
    ax.annotate("", xy=tuple(tip), xytext=tuple(tail),
               arrowprops=dict(arrowstyle="-|>", color="black", lw=1.8))
    ax.text(*(tip + direction * 4.0), "toward\nrobot base", fontsize=8,
           ha="center", va="center")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    out = (Path(out_path) if out_path else
          REPO / "out" / "_move_wp_constantRobPose_plots" /
          f"{Path(config_path).stem}_base_frame.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("config", help="a saved RunConfig JSON")
    p.add_argument("--out", default=None, help="output PNG path")
    a = p.parse_args(argv)
    out = plot(a.config, a.out)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
