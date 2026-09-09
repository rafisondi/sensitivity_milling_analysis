"""Loading the base config, and moving the operating point off it.

`configs/base.json` is the job: a 100 x 60 mm rectangular part, one edge milled,
25 mm of lead-in and 25 mm of lead-out so the pass contains a real entry and a
real exit. `ap = 1.0 mm`, `ae = 5 mm`, 3333 rpm, 40 mm/s.

That depth is chosen, not inherited. It is deep enough that the cut is close to
the stability boundary — so the prediction is saying something — and shallow
enough that the tool stays in the material: at the stock `ap = 5 mm` this arm
deflects far enough that the cut stops being one.

`load_base` returns the config with `io` pinned to absolute workspace paths, so a
run started from any working directory writes to the same place.
"""

from dataclasses import replace

import analysis
from analysis import DEFAULT_ROBOT_MODEL, DATA, OUT, ROOT

from runconfig import RunConfig

TOOLPATH_DIR = DATA / "toolpaths"
WORKPIECE_DIR = DATA / "workpieces"

#: Name of the constant-feed toolpath `analysis.toolpath` writes and every run
#: replays. Regenerated automatically whenever the feed changes.
BASE_TOOLPATH = "edge_100x60_constant_feed"

#: Ceiling on the step `steps_per_tooth` may ask for [s]. At 1000 rpm with one
#: tooth a 20-step tooth period is 3 ms, which resolves the tooth but leaves only
#: ~14 samples per cycle of the arm's 23 Hz mode. The clamp keeps the STRUCTURE
#: resolved at the slow corner, at the cost of oversampling the tooth there -
#: which is the harmless direction to err in.
MAX_SIM_DT = 1.0e-3


def load_base(path=None, *, robot_model=None, toolpath=BASE_TOOLPATH,
              out_dir=None, **overrides) -> RunConfig:
    """The base config, with io pinned to this workspace.

    `toolpath` names the .npz under `data/toolpaths/`; pass None for the stock
    quintic planner instead, which chains rest-to-rest segments and therefore
    brings the tool to a full stop at the edge entry and again at the exit.
    `overrides` are applied to the top-level `RunConfig` fields.
    """
    cfg = RunConfig.load(str(path or analysis.DEFAULT_CONFIG))

    cfg = replace(cfg, io=replace(
        cfg.io,
        out_dir=str(out_dir or OUT),
        toolpath_dir=str(TOOLPATH_DIR),
        workpiece_dir=str(WORKPIECE_DIR),
        stamp_runs=False))                  # deterministic, overwritable dirs

    cfg = replace(cfg, scene=replace(
        cfg.scene, robot_model=robot_model or DEFAULT_ROBOT_MODEL))

    if toolpath is not None:
        cfg = replace(cfg, path=replace(cfg.path, toolpath=toolpath))

    return replace(cfg, **overrides) if overrides else cfg


def apply_operating_point(cfg, *, ap_mm=None, ae_mm=None, rpm=None, feed_mm_s=None,
                          n_teeth=None, fz_mm=None, steps_per_tooth=None,
                          max_sim_dt=MAX_SIM_DT,
                          Ktc=None, Krc=None, Kac=None, sim_dt=None,
                          raster_mm=None, robot_model=None) -> RunConfig:
    """Move the operating point. Every argument left None keeps the config's value.

    Four of these reach further than the field they name:

    `ap_mm` is the PART HEIGHT. The raster is a stacked constant-depth model, so
    axial depth is a property of the workpiece rather than of the cutter, and
    `part.height_mm` is the single place it is set — `MillConfig.height_mm` is
    kept in step so anything reading the mill sees the same number.

    `feed_mm_s` invalidates the saved toolpath. The constant-feed profile holds
    `v_max` across the whole engaged span and puts its ramps in the lead-in and
    lead-out, so it has to be rebuilt for a new maximum rather than rescaled.
    `analysis.toolpath.ensure` does that; this function only records the change.

    `fz_mm` sets the feed from the CHIP LOAD instead of the other way round:
    `feed = fz * rpm * N / 60`. It is the invariant to hold when sweeping rpm and
    tooth count, because the model's whole dependence on those two runs through
    `N * fz` (which fixes `F0` and `K_cut`) and `rpm` (which fixes `C_cut`).
    Holding the FEED fixed instead makes `fz` swing as `1 / (rpm N)` - a factor
    of 80 across a 1000-10000 rpm, 1-8 tooth grid, which leaves the chip
    unphysical at one corner and below the raster's resolution at the other.

    `steps_per_tooth` sets `sim_dt` from the TOOTH PERIOD rather than absolutely,
    `dt = 60 / (rpm N spt)`. This is what a tooth-passing sweep has to hold
    fixed: at a constant `dt` the samples per tooth fall as `1 / (rpm N)` - below
    8 over a quarter of that grid, where `MillConfig.summary` warns that the
    harmonics alias - and an aliased engine is indistinguishable from a model
    that broke down. Applied after `sim_dt`, so it wins if both are given, and
    clamped at `max_sim_dt` so the slow corner still resolves the arm's modes.
    """
    if ap_mm is not None:
        cfg = replace(cfg, part=replace(cfg.part, height_mm=float(ap_mm)),
                      mill=replace(cfg.mill, height_mm=float(ap_mm)))
    if fz_mm is not None and feed_mm_s is not None:
        raise ValueError(
            "give either fz_mm or feed_mm_s, not both - the chip load and the "
            "feed are the same number seen from two ends, and fz only fixes a "
            "feed once the rpm and the tooth count are settled")

    mill_kw = {}
    if ae_mm is not None:
        mill_kw["radial_engagement_mm"] = float(ae_mm)
    if rpm is not None:
        mill_kw["spindle_rpm"] = float(rpm)
    if n_teeth is not None:
        mill_kw["n_teeth"] = int(n_teeth)
    if feed_mm_s is not None:
        mill_kw["feed_mm_s"] = float(feed_mm_s)
    for name, value in (("Ktc", Ktc), ("Krc", Krc), ("Kac", Kac),
                        ("raster_mm", raster_mm)):
        if value is not None:
            mill_kw[name] = float(value)
    if sim_dt is not None:
        mill_kw["sim_dt"] = float(sim_dt)
        cfg = replace(cfg, sim=replace(cfg.sim, sim_dt=float(sim_dt)))
    if mill_kw:
        cfg = replace(cfg, mill=replace(cfg.mill, **mill_kw))

    # Both of these read the RESOLVED rpm and tooth count, so they come after
    # the block above rather than inside it.
    if fz_mm is not None:
        feed_mm_s = (float(fz_mm) * cfg.mill.spindle_rpm
                     * cfg.mill.n_teeth / 60.0)
        cfg = replace(cfg, mill=replace(cfg.mill, feed_mm_s=float(feed_mm_s)))
    if feed_mm_s is not None:
        cfg = replace(cfg, path=replace(cfg.path, speed_mm_s=float(feed_mm_s)))

    if steps_per_tooth is not None:
        dt = 60.0 / (cfg.mill.spindle_rpm * cfg.mill.n_teeth
                     * float(steps_per_tooth))
        if max_sim_dt:
            dt = min(dt, float(max_sim_dt))
        cfg = replace(cfg, mill=replace(cfg.mill, sim_dt=dt),
                      sim=replace(cfg.sim, sim_dt=dt))

    if robot_model is not None:
        cfg = replace(cfg, scene=replace(cfg.scene, robot_model=str(robot_model)))
    return cfg


def operating_point_row(cfg) -> dict:
    """The cut's own coordinates, for a sweep to index its runs by.

    Descriptors, not verdicts: what was asked for, plus the two dimensionless
    numbers that say whether either side of the comparison can be believed here.

    `tpf_hz` is the tooth-passing frequency, `rpm N / 60` - the forcing the
    revolution average deletes, and the whole reason a tooth-count sweep is not
    a no-op. `omega_T_max` is the linear model's OWN truncation parameter,
    `2 pi f T` at the highest structural mode: both cut terms drop the
    regenerative delay after one order, so the neglected term is roughly
    `(omega T)^2 / 2` of the cut force and the model stops meaning much as it
    approaches 1. It is added by the caller, which is what holds the modes.

    `steps_per_tooth` and `chip_px` are the engine's two resolution limits, and
    both move under a tooth-passing sweep - so they are recorded per run rather
    than assumed constant across one.
    """
    m = cfg.milling()
    return {
        "spindle_rpm": float(m.spindle_rpm),
        "n_teeth": int(m.n_teeth),
        "feed_mm_s": float(m.feed_mm_s),
        "fz_mm": float(m.feed_per_tooth_mm()),
        "ap_mm": float(cfg.part.height_mm),
        "ae_mm": float(m.radial_engagement_mm),
        "tpf_hz": float(m.spindle_rpm * m.n_teeth / 60.0),
        "tooth_period_s": float(m.tooth_period_s),
        "sim_dt": float(m.sim_dt),
        "steps_per_tooth": float(m.steps_per_tooth),
        "chip_px": float(m.chip_px()),
        "raster_mm": float(m.raster_mm),
    }


def check_compliant(cfg) -> None:
    """Raise unless the configured arm has identified joint compliance.

    A fully rigid arm would report zero deflection, leave the feedforward with
    nothing to cancel, and make the whole comparison vacuous — so this fails
    loudly rather than producing a flat result that looks like a finding.
    """
    import numpy as np
    from robotsim.kinematics import ROBOT_MODELS

    name = cfg.scene.robot_model
    settings = ROBOT_MODELS[name](cfg.scene.urdf_path)
    finite = np.isfinite(settings.K_m_diag) & np.isfinite(settings.D_m_diag)
    if not finite.any():
        raise ValueError(
            f"robot model {name!r} is fully RIGID - K_m_diag is all nan, so there "
            f"is no compliance to deflect. Use {DEFAULT_ROBOT_MODEL!r} instead.")
    return None
