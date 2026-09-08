"""Chatter stability: the linearised cut closed around the arm.

Three packages meet here and none of them changes. `fastsim` supplies the cut —
`MillConfig`, the `Workpiece`, the planned path. `robotsim` supplies the machine
as a state space. This package supplies the cut as two matrices about the
nominal path and takes the eigenvalues of the loop between them.

    from robotsim.receptance import Receptance
    from stabsim import stability_along_path

    mill, part, scene, _, path = cfg.build()
    stab = stability_along_path(path.xy_mm, part, receptance, scene, mill, ds_mm=2.0)
    print(stab.summary())
    stab.critical_depth()          # smallest unstable ap at every point

    stabsim/cut.py         F0, K_cut, C_cut — the ZOA terms, recovered
    stabsim/engagement.py  entry/exit angles and their gradients, measured
    stabsim/stability.py   the closed loop and the sweep along the path
    stabsim/compensate.py  F0 along the path, and the command aimed off the
                           nominal by -G(0) F0 so the mean cutting force lands
                           the tool back on the geometry it was linearised about

Read `stabsim/stability.py` before quoting a verdict: the delay is expanded to
FIRST ORDER, so there are no spindle-speed lobes here — a growth rate is a
verdict about this operating point, not a lobe diagram.

WHAT IS NOT HERE. Upstream also carries `surrogate.py` (the linear cut model
DRIVING a time-domain pass in place of the engine), `delay.py` and
`compare_delay_*.py` (the exact-delay eigenvalue work), `plan_stable_feed.py`,
`plotting.py`, `interactive.py` and the `run_*.py` scripts. This workspace reads
the linear model's mean-force prediction straight off `StabilityAlongPath.F0_w`,
so none of them is on the import path.
"""

from stabsim.compensate import Compensation, compensate, f0_along_path
from stabsim.cut import (
    A_cut_0, C_cut, F0_cut, K_cut, K_cut_0, embed_2x2, entry_angle,
    zero_order_force_analytical,
)
from stabsim.engagement import (
    Engagement, EngagementGradients, engagement_angles_along_path,
    engagement_gradients_along_path,
)
from stabsim.stability import (
    COUPLINGS, StabilityAlongPath, closed_loop_eigs, closed_loop_matrix,
    critical_depth_mm, dominant_mode, growth_rate, open_loop_growth_rate,
    stability_along_path,
)

__all__ = [
    "zero_order_force_analytical", "entry_angle", "A_cut_0", "K_cut_0",
    "embed_2x2", "F0_cut", "K_cut", "C_cut",
    "Engagement", "EngagementGradients", "engagement_angles_along_path",
    "engagement_gradients_along_path",
    "compensate", "Compensation", "f0_along_path",
    "closed_loop_matrix", "closed_loop_eigs", "growth_rate", "dominant_mode",
    "open_loop_growth_rate", "critical_depth_mm", "stability_along_path",
    "StabilityAlongPath", "COUPLINGS",
]
