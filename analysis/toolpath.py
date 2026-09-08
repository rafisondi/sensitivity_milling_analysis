"""Writing the constant-feed toolpath the run replays.

One edge of the rectangular part, 25 mm of air before and after, and the feed
held at its maximum for the ENTIRE engaged span — the ramps live in the lead-in
and lead-out. `analysis.feedplan` explains why; this module builds it, checks it
and writes it to `data/toolpaths/<name>.npz`, which `cfg.path.toolpath` then
replays through `runconfig._path_from_samples`.

    ensure(cfg)                     build it if the feed has changed
    ensure(cfg, force=True)         build it unconditionally

`ensure` is what `main.py` calls, so a run at a new `--feed` cannot silently
replay a path planned for the old one.
"""

import numpy as np

from analysis import config as acfg
from analysis import feedplan


def build(cfg, v_max_mm_s=None, name=None):
    """(waypoints, t, xy, v, path_file) for the one-edge constant-feed job."""
    mill = cfg.milling()
    part = cfg.part.build()
    v_max = float(v_max_mm_s or cfg.path.speed_mm_s)

    waypoints = part.contour_waypoints(
        mill.tool_offset_mm, start_edge=cfg.path.start_edge, n_edges=1,
        lead_in_mm=cfg.path.lead_in_mm, lead_out_mm=cfg.path.lead_out_mm)

    t, xy, v = feedplan.constant_feed_path(waypoints, v_max, dt=cfg.sim.plan_dt)

    name = name or cfg.path.toolpath or acfg.BASE_TOOLPATH
    out = acfg.TOOLPATH_DIR / f"{name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    feedplan.save_toolpath(out, waypoints, t, xy, v,
                           name=name, height_mm=part.height_mm)
    return waypoints, t, xy, v, out


def ensure(cfg, *, force=False, verbose=True):
    """Build the toolpath unless one already on disk matches the config's feed.

    The saved file records the speed profile it was planned with, so "matches"
    is a comparison rather than a timestamp: a path whose plateau is not the
    config's `speed_mm_s` is rebuilt.
    """
    name = cfg.path.toolpath or acfg.BASE_TOOLPATH
    out = acfg.TOOLPATH_DIR / f"{name}.npz"
    want = float(cfg.path.speed_mm_s)

    if out.exists() and not force:
        with np.load(out) as z:
            have = float(np.max(z["speed_mm_s"])) if "speed_mm_s" in z else np.nan
        if np.isfinite(have) and abs(have - want) < 1e-9:
            if verbose:
                print(f"toolpath {out.name} (max feed {have:g} mm/s, reused)")
            return out

    waypoints, t, xy, v, out = build(cfg, want, name)
    if verbose:
        print(describe(cfg, waypoints, t, xy, v))
        print(f"toolpath {out}")
    return out


def describe(cfg, waypoints, t, xy, v) -> str:
    """What was planned, and the one check that has to pass."""
    mill = cfg.milling()
    s = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    eng = feedplan.engaged_mask(waypoints, s)
    v_max = v.max()
    seg = np.linalg.norm(np.diff(np.asarray(waypoints, float), axis=0), axis=1)

    fz = mill.feed_per_tooth_mm(v_max)
    lines = [
        f"path     lead-in {seg[0]:.1f} | edge {seg[1]:.1f} | lead-out {seg[2]:.1f} mm"
        f"  = {s[-1]:.1f} mm in {t[-1]:.3f} s",
        f"         feed {v_max:.1f} mm/s max, "
        f"{v[eng].min():.3f}..{v[eng].max():.3f} mm/s while ENGAGED",
        f"         chip {fz:.4f} mm/tooth = {mill.chip_px(v_max):.1f} px at "
        f"{mill.spindle_rpm:g} rpm | peak accel "
        f"{np.abs(np.gradient(v, t)).max():.1f} mm/s^2, entirely in air",
    ]
    flat = np.allclose(v[eng], v_max, rtol=0, atol=1e-9)
    if not flat:
        lines.append("         ! the ramps reach into the material - lengthen the "
                     "leads or lower the feed")
    return "\n".join(lines)
