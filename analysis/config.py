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
                          Ktc=None, Krc=None, Kac=None, sim_dt=None,
                          raster_mm=None, robot_model=None) -> RunConfig:
    """Move the operating point. Every argument left None keeps the config's value.

    Two of these reach further than the field they name:

    `ap_mm` is the PART HEIGHT. The raster is a stacked constant-depth model, so
    axial depth is a property of the workpiece rather than of the cutter, and
    `part.height_mm` is the single place it is set — `MillConfig.height_mm` is
    kept in step so anything reading the mill sees the same number.

    `feed_mm_s` invalidates the saved toolpath. The constant-feed profile holds
    `v_max` across the whole engaged span and puts its ramps in the lead-in and
    lead-out, so it has to be rebuilt for a new maximum rather than rescaled.
    `analysis.toolpath.ensure` does that; this function only records the change.
    """
    if ap_mm is not None:
        cfg = replace(cfg, part=replace(cfg.part, height_mm=float(ap_mm)),
                      mill=replace(cfg.mill, height_mm=float(ap_mm)))
    mill_kw = {}
    if ae_mm is not None:
        mill_kw["radial_engagement_mm"] = float(ae_mm)
    if rpm is not None:
        mill_kw["spindle_rpm"] = float(rpm)
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

    if feed_mm_s is not None:
        cfg = replace(cfg, path=replace(cfg.path, speed_mm_s=float(feed_mm_s)))
    if robot_model is not None:
        cfg = replace(cfg, scene=replace(cfg.scene, robot_model=str(robot_model)))
    return cfg


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
