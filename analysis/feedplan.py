"""A feed profile that holds max velocity for the whole cut.

The stock planner chains rest-to-rest quintics (`robotsim.trajectory.
plan_quintic_path`), so the tool comes to a full stop at the edge entry and again
at the edge exit, and only touches its peak speed in the middle of each segment.
Chip load would then vary over the entire cut — which is the wrong boundary
condition here, because the entry and the exit are exactly what this job is for
and a stopped tool at each of them is a different event from a tool driving into
the material at feed.

Here the ramps live entirely in the 25 mm lead-in and lead-out, through air, and
the engaged span runs at a flat `v_max`:

    v(s)                _________________________
                       /                         \\
                      /                           \\
    ________________ /                             \\ ______________
                   |  lead-in |   EDGE    | lead-out |
                   0         25         125        150   s [mm]

Each ramp is a smoothstep in time, v(tau) = v_max * (10t^3 - 15t^4 + 6t^5), which
has zero acceleration at both ends so the arm is never asked for a velocity step.
That polynomial integrates to exactly 1/2 over [0, 1], so a ramp of duration

    T = 2 * L_ramp / v_max

covers exactly L_ramp — no search, no numerical integration, and the plateau
starts precisely at the material.

THE FLYING PROFILE, AND WHY IT EXISTS

`profile="flying"` drops the ramps entirely: the tool is at `v_max` at t = 0 and
stays there to the end of the lead-out. It cannot be run on a real machine, and
that is not what it is for.

The ramps are through air, so they change nothing about the cut - but they are
not free in a COMPARISON. Accelerating this arm takes joint torque, joint torque
deflects the springs, and the tool therefore sits tens of micrometres off the
command all through the lead-in and again all through the lead-out, with no
cutting force anywhere near it. At a fixed chip load the feed rises with the
spindle speed, so that excursion grows with rpm: 5 um at 24 mm/s, 26 um at
120 mm/s. Any linear model driven by the cutting force alone - `compare_linear`,
`stabsim.stability` - has no term for it and cannot reproduce it, so the ramps
show up as model error that is really commanded-acceleration tracking error.

A flying start removes them at the source. `Simulator.reset` is handed
`thetaD_cmd[0]` and solves the tracking equilibrium at that velocity, so the run
opens already moving and already deflected correctly - there is no step and no
startup transient to decay. What is left in the deviation is the cut.
"""

import numpy as np

_MIN_RAMP_MM = 1e-6

#: How the feed gets to `v_max`. "ramped" puts smoothstep ramps in the lead-in
#: and lead-out; "flying" is at `v_max` from the first sample - see the module
#: docstring for when each is the right one.
PROFILES = ("ramped", "flying")


def smoothstep(tau):
    """The quintic S-curve on [0, 1]: 0 and 1 at the ends, zero slope at both."""
    tau = np.clip(np.asarray(tau, float), 0.0, 1.0)
    return tau ** 3 * (10.0 - 15.0 * tau + 6.0 * tau ** 2)


def _smoothstep_integral(tau):
    """Integral of `smoothstep` from 0 to tau. Equals 1/2 at tau = 1."""
    tau = np.clip(np.asarray(tau, float), 0.0, 1.0)
    return tau ** 4 * (2.5 - 3.0 * tau + tau ** 2)


def trapezoid_profile(lead_in_mm, edge_mm, lead_out_mm, v_max_mm_s, dt=1.0e-3):
    """(t, s_mm, v_mm_s) for ramp-up / cruise / ramp-down along arclength.

    The plateau spans exactly [lead_in_mm, lead_in_mm + edge_mm].
    """
    if v_max_mm_s <= 0.0:
        raise ValueError(f"v_max must be positive, got {v_max_mm_s} mm/s")
    if edge_mm <= 0.0:
        raise ValueError(f"the cut needs positive length, got {edge_mm} mm")

    t_up = 2.0 * lead_in_mm / v_max_mm_s if lead_in_mm > _MIN_RAMP_MM else 0.0
    t_cr = edge_mm / v_max_mm_s
    t_dn = 2.0 * lead_out_mm / v_max_mm_s if lead_out_mm > _MIN_RAMP_MM else 0.0
    total = t_up + t_cr + t_dn

    t = np.arange(0.0, total + 0.5 * dt, dt)
    s = np.empty_like(t)
    v = np.empty_like(t)

    up = t < t_up
    cr = (t >= t_up) & (t < t_up + t_cr)
    dn = t >= t_up + t_cr

    if t_up > 0.0:
        tau = t[up] / t_up
        v[up] = v_max_mm_s * smoothstep(tau)
        s[up] = v_max_mm_s * t_up * _smoothstep_integral(tau)

    v[cr] = v_max_mm_s
    s[cr] = lead_in_mm + v_max_mm_s * (t[cr] - t_up)

    if t_dn > 0.0:
        tau = (t[dn] - t_up - t_cr) / t_dn
        # mirrored ramp: speed runs the smoothstep backwards
        v[dn] = v_max_mm_s * (1.0 - smoothstep(tau))
        s[dn] = (lead_in_mm + edge_mm
                 + v_max_mm_s * t_dn * (tau - _smoothstep_integral(tau)))
    else:
        v[dn] = v_max_mm_s
        s[dn] = lead_in_mm + edge_mm

    return t, s, v


def flying_profile(lead_in_mm, edge_mm, lead_out_mm, v_max_mm_s, dt=1.0e-3):
    """(t, s_mm, v_mm_s) at a constant `v_max` over the whole polyline.

    No ramps anywhere: `s = v t` from the first sample to the last. The arm is
    expected to be handed this as an already-moving initial condition, which is
    what `Simulator.reset(theta, thetaD, ...)` sets up.
    """
    if v_max_mm_s <= 0.0:
        raise ValueError(f"v_max must be positive, got {v_max_mm_s} mm/s")
    if edge_mm <= 0.0:
        raise ValueError(f"the cut needs positive length, got {edge_mm} mm")

    total_mm = lead_in_mm + edge_mm + lead_out_mm
    t = np.arange(0.0, total_mm / v_max_mm_s + 0.5 * dt, dt)
    s = np.minimum(v_max_mm_s * t, total_mm)
    return t, s, np.full_like(t, float(v_max_mm_s))


def constant_feed_path(waypoints_mm, v_max_mm_s, dt=1.0e-3, profile="ramped"):
    """(t, xy_mm, v_mm_s) along a 4-waypoint lead-in / edge / lead-out polyline.

    `waypoints_mm` is what `Workpiece.contour_waypoints(..., n_edges=1,
    lead_in_mm=..., lead_out_mm=...)` returns: [lead-in start, entry, exit,
    lead-out end].

    `profile` is "ramped" (smoothstep ramps through the air at each end) or
    "flying" (at `v_max` from the first sample) - see the module docstring.
    """
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}, got {profile!r}")
    w = np.asarray(waypoints_mm, float)
    if w.shape != (4, 2):
        raise ValueError(
            f"expected 4 waypoints (lead-in, entry, exit, lead-out), got {w.shape}. "
            "Build them with contour_waypoints(n_edges=1, lead_in_mm=.., "
            "lead_out_mm=..).")

    seg = np.linalg.norm(np.diff(w, axis=0), axis=1)
    lead_in, edge, lead_out = seg
    make = trapezoid_profile if profile == "ramped" else flying_profile
    t, s, v = make(lead_in, edge, lead_out, v_max_mm_s, dt=dt)

    knots = np.concatenate([[0.0], np.cumsum(seg)])
    xy = np.column_stack([np.interp(s, knots, w[:, k]) for k in (0, 1)])
    return t, xy, v


def engaged_mask(waypoints_mm, s_mm):
    """True where the tool centre is alongside the edge, i.e. actually cutting."""
    w = np.asarray(waypoints_mm, float)
    seg = np.linalg.norm(np.diff(w, axis=0), axis=1)
    return (s_mm >= seg[0]) & (s_mm <= seg[0] + seg[1])


def save_toolpath(path, waypoints_mm, t, xy_mm, v_mm_s, *, name, height_mm,
                  R_i_tcp=None):
    """Write the npz `RunConfig.path.toolpath` consumes.

    `_load_toolpath` (runconfig.py) reads only `waypoints_mm`, `xy_mm` and `t`;
    the rest is written so the file is self-describing.
    """
    path = str(path)
    np.savez_compressed(
        path, waypoints_mm=np.asarray(waypoints_mm, float), xy_mm=xy_mm, t=t,
        dt=float(t[1] - t[0]), speed_mm_s=v_mm_s,
        R_i_tcp=np.eye(3) if R_i_tcp is None else R_i_tcp,
        height_mm=float(height_mm), name=str(name))
    return path
