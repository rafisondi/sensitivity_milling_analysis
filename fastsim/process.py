import math
import numpy as np
from collections import deque
from numba import njit, prange

# Lengths are mm inside the kernel; moments leave the wrapper in SI N*m.
MM_TO_M = 1.0e-3

from fastsim.raster import WorkpieceRaster


# ---------------------------------------------------------------------------
# Numba step kernel — module-level so Numba can JIT-compile it.
#
# Two phases per step:
#   Phase 1 — engagement check + force accumulation (loop over slices × teeth)
#             Uses workpiece state BEFORE this step's erasure.
#   Phase 2 — swept quad erasure per tooth (slice × teeth).
#             Erases the quadrilateral swept by each tooth tip from the
#             previous position to the current one, matching the Shapely
#             reference erasure model.
#             Order of quad vertices (CCW): centre_now → centre_prev →
#             tip_prev → tip_now.
# ---------------------------------------------------------------------------

@njit(cache=True)
def _fill_triangle(grids, k, x0, y0, x1, y1, x2, y2, res, ox, oy, NX, NY):
    """Erase the pixels of slice `k` whose centres lie in triangle (0,1,2).

    Scanline: for each raster row, intersect the row's y with the three edges to
    get the x-span, then walk only that span. The previous code walked the whole
    quad bounding box, which for these wedges is ~3.5% full -- the sliver is
    ~0.35 mm of arc travel inside a box up to 10x10 mm.

    The per-pixel test is still the same-sign cross-product test, applied inside
    a span padded by one pixel each way. That keeps the erased set bit-identical
    to the bbox version rather than depending on the scanline arithmetic
    agreeing with the cross products at the boundary.
    """
    e0x = x1 - x0; e0y = y1 - y0
    e1x = x2 - x1; e1y = y2 - y1
    e2x = x0 - x2; e2y = y0 - y2

    # Scan along whichever axis needs FEWER lines, i.e. lines perpendicular to
    # the sliver's long axis. A wedge is ~10 mm long and ~0.35 mm wide at an
    # arbitrary angle; scanning a near-vertical one by rows gives ~200 lines of
    # ~3 px, where the per-line setup swamps the pixel work. Picking the short
    # axis bounds the number of setups by the sliver's width wherever the
    # orientation is close to axis-aligned.
    swap = (max(y0, max(y1, y2)) - min(y0, min(y1, y2))) > \
           (max(x0, max(x1, x2)) - min(x0, min(x1, x2)))

    if swap:      # scan by columns: a = x, b = y
        a0 = x0; b0 = y0; a1 = x1; b1 = y1; a2 = x2; b2 = y2
        oa = ox; ob = oy; NA = NX; NB = NY
    else:         # scan by rows: a = y, b = x
        a0 = y0; b0 = x0; a1 = y1; b1 = x1; a2 = y2; b2 = x2
        oa = oy; ob = ox; NA = NY; NB = NX

    # Per-edge inverse slopes, hoisted out of the scan loop -- three divisions
    # per triangle instead of per line.
    d0 = a1 - a0; s0 = (b1 - b0) / d0 if d0 != 0.0 else 0.0
    d1 = a2 - a1; s1 = (b2 - b1) / d1 if d1 != 0.0 else 0.0
    d2 = a0 - a2; s2 = (b0 - b2) / d2 if d2 != 0.0 else 0.0

    amin = min(a0, min(a1, a2))
    amax = max(a0, max(a1, a2))
    ia_lo = int(math.floor((amin - oa) / res))
    ia_hi = int(math.floor((amax - oa) / res)) + 1
    if ia_lo < 0:
        ia_lo = 0
    if ia_hi > NA - 1:
        ia_hi = NA - 1

    inv_res = 1.0 / res

    for ia in range(ia_lo, ia_hi + 1):
        wa = ia * res + oa

        blo = 1.0e30
        bhi = -1.0e30
        # edge 0
        if (a0 <= wa and wa <= a1) or (a1 <= wa and wa <= a0):
            if d0 != 0.0:
                bb = b0 + (wa - a0) * s0
                if bb < blo: blo = bb
                if bb > bhi: bhi = bb
            else:
                if b0 < blo: blo = b0
                if b1 < blo: blo = b1
                if b0 > bhi: bhi = b0
                if b1 > bhi: bhi = b1
        # edge 1
        if (a1 <= wa and wa <= a2) or (a2 <= wa and wa <= a1):
            if d1 != 0.0:
                bb = b1 + (wa - a1) * s1
                if bb < blo: blo = bb
                if bb > bhi: bhi = bb
            else:
                if b1 < blo: blo = b1
                if b2 < blo: blo = b2
                if b1 > bhi: bhi = b1
                if b2 > bhi: bhi = b2
        # edge 2
        if (a2 <= wa and wa <= a0) or (a0 <= wa and wa <= a2):
            if d2 != 0.0:
                bb = b2 + (wa - a2) * s2
                if bb < blo: blo = bb
                if bb > bhi: bhi = bb
            else:
                if b2 < blo: blo = b2
                if b0 < blo: blo = b0
                if b2 > bhi: bhi = b2
                if b0 > bhi: bhi = b0
        if bhi < blo:
            continue

        ib_lo = int(math.floor((blo - ob) * inv_res)) - 1
        ib_hi = int(math.floor((bhi - ob) * inv_res)) + 1
        if ib_lo < 0:
            ib_lo = 0
        if ib_hi > NB - 1:
            ib_hi = NB - 1

        if swap:
            wx = wa
            for ib in range(ib_lo, ib_hi + 1):
                wy = ib * res + ob
                c0 = e0x * (wy - y0) - e0y * (wx - x0)
                c1 = e1x * (wy - y1) - e1y * (wx - x1)
                c2 = e2x * (wy - y2) - e2y * (wx - x2)
                if (c0 >= 0.0 and c1 >= 0.0 and c2 >= 0.0) or \
                   (c0 <= 0.0 and c1 <= 0.0 and c2 <= 0.0):
                    grids[k, ib, ia] = 0
        else:
            wy = wa
            for ib in range(ib_lo, ib_hi + 1):
                wx = ib * res + ob
                c0 = e0x * (wy - y0) - e0y * (wx - x0)
                c1 = e1x * (wy - y1) - e1y * (wx - x1)
                c2 = e2x * (wy - y2) - e2y * (wx - x2)
                if (c0 >= 0.0 and c1 >= 0.0 and c2 >= 0.0) or \
                   (c0 <= 0.0 and c1 <= 0.0 and c2 <= 0.0):
                    grids[k, ia, ib] = 0


def _step_kernel_impl(
    grids,          # (n_slices, NY, NX) uint8 — modified in-place
    center,         # (2,) float64 — current tool centre [mm]
    angle,          # float64 — current spindle angle [rad]
    center_prev,    # (2,) float64 — tool centre 1 step ago (for erasure)
    angle_prev,     # float64 — spindle angle 1 step ago (for erasure)
    center_T,       # (2,) float64 — tool centre T_tooth steps ago (chip thickness)
    helix_offset,   # (n_slices,) float64
    n_slices, n_teeth, radius, spin,
    slice_height, Kt, Kr, Ka,
    res, ox, oy,    # raster: mm/pixel, origin_x, origin_y
    use_dexel,      # bool — if True, measure chip thickness from the dexel
    surf,           # (n_slices, n_phi) float64 — surface radius at deposit time
    surf_cx, surf_cy,  # (n_slices, n_phi) float64 — tool centre at deposit
    cos_phi, sin_phi,  # (n_phi,) float64 — precomputed bin directions
    n_phi, dphi,    # int, float — dexel angular resolution
    max_chip,       # float — physical chip bound; beyond it, use analytic fallback
    surf_step,      # (n_slices, n_phi) int64 — step index of each bin's last deposit
    step_idx,       # int64 — current step
    stale_steps,    # float — a bin older than this is stale, not a cut surface
):
    NY = grids.shape[1]
    NX = grids.shape[2]
    two_pi = 2.0 * math.pi
    two_pi_over_n = two_pi / n_teeth
    # Per-(slice, tooth) wrench contributions, summed serially at the end.
    # [Fx, Fy, Fz, Mx, My, Mz] — force first (pinocchio spatial-force order).
    # Moments are about the TOOL CENTRE at the slice-0 plane, expressed in
    # workpiece-frame orientation; MillingProcess transports them.
    #
    # Buffering rather than accumulating in place is what lets the slice loop go
    # parallel while staying BIT-IDENTICAL: floating-point addition is not
    # associative, so per-thread subtotals would regroup the sum and change the
    # last bits. Replaying in the original (k, then j) order reproduces the
    # serial result exactly, and the replay is ~n_slices*n_teeth adds against a
    # parallel region doing orders of magnitude more work.
    contrib = np.zeros((n_slices, n_teeth, 6))
    # Per-tooth engagement flags (filled in Phase 1, used by the Phase 2 deposit)
    engaged = np.zeros((n_slices, n_teeth), dtype=np.uint8)

    dcx = center[0] - center_prev[0]
    dcy = center[1] - center_prev[1]

    fz = math.sqrt((center[0] - center_T[0]) ** 2 + (center[1] - center_T[1]) ** 2)
    erase_depth = 4.0 * fz
    if erase_depth < 2.0 * res:
        erase_depth = 2.0 * res
    if erase_depth > radius:
        erase_depth = radius

    # ------------------------------------------------------------------ #
    # Phase 0 + 1, per slice. Slices share nothing: grids[k], surf[k] and
    # engaged[k] are only ever indexed at k, so this loop is embarrassingly
    # parallel. Phase 0 (dexel surface advection) is folded in here rather
    # than run as its own pass, to keep one parallel region instead of two.
    #
    # Phase 0: a surface point is fixed in space but stored relative to the
    # moving centre, so shift every bin radius by the centre displacement
    # projected on that bin's direction. This is what produces the
    # regenerative chip thickness (feed plus any dynamic displacement).
    # Phase 1: engagement + force, on the PRE-erasure grid.
    # ------------------------------------------------------------------ #
    for k in prange(n_slices):
        for j in range(n_teeth):
            tooth_angle = angle + helix_offset[k] + j * two_pi_over_n
            ct = math.cos(tooth_angle)
            st = math.sin(tooth_angle)

            tip_x = center[0] + radius * ct
            tip_y = center[1] + radius * st

            ix = int(math.floor((tip_x - ox) / res))
            iy = int(math.floor((tip_y - oy) / res))

            if 0 <= ix < NX and 0 <= iy < NY and grids[k, iy, ix] > 0:
                engaged[k, j] = 1  # tip sits in material -> this tooth cuts

                # Displacement-projection chip thickness: the tool-centre motion
                # over one tooth period, projected on the tooth radial. Exact
                # ONCE A PREVIOUS TOOTH HAS PASSED, because then the surface
                # being cut is that tooth's arc, displaced by exactly the feed.
                h_lin = (center[0] - center_T[0]) * ct + (center[1] - center_T[1]) * st

                # ... but entering virgin stock no previous tooth has passed, so
                # the free surface is the STOCK BOUNDARY and the real chip is the
                # penetration depth, not the feed advance. Measured against the
                # Shapely oracle at first contact on the slot case: oracle
                # h = 0.00171 mm, displacement projection h = 0.11247 mm -- 65x
                # too large, decaying over ~0.3 rev, which is the entry force
                # overshoot (up to 2.80x peak in revolution 1).
                #
                # So cap h by the material actually present: march radially
                # inward from the tip until the raster goes empty. Phase 1 runs
                # before Phase 2's erasure, so the grid still holds the surface
                # left by the previous pass. Marching only as far as h_lin keeps
                # this to a handful of lookups -- and it is provably enough,
                # since the swept discs of past steps guarantee the free surface
                # never sits deeper than h_lin.
                step_r = 0.5 * res
                h_cap = h_lin
                if h_lin > 1e-9:
                    n_march = int(h_lin / step_r) + 1
                    for s in range(1, n_march + 1):
                        rr = radius - s * step_r
                        mx = center[0] + rr * ct
                        my = center[1] + rr * st
                        jx = int(math.floor((mx - ox) / res))
                        jy = int(math.floor((my - oy) / res))
                        if rr <= 0.0 or jx < 0 or jy < 0 or jx >= NX or jy >= NY \
                                or grids[k, jy, jx] == 0:
                            # Surface lies between step s-1 and s; take the midpoint.
                            h_cap = (s - 0.5) * step_r
                            break

                if use_dexel:
                    # Interpolated surface radius at this tooth angle
                    ph = tooth_angle % two_pi  # numba % -> [0, two_pi)
                    bf = ph / dphi
                    i0 = int(bf)
                    frac = bf - i0
                    i1 = i0 + 1
                    if i1 >= n_phi:
                        i1 = 0
                    # Lazy advection. A bin records the surface radius AND the
                    # tool centre it was deposited from, so the displacement
                    # since is applied here, on the two bins actually read,
                    # instead of Phase 0 walking all n_slices*n_phi bins every
                    # step (59392 of them at 29 slices x 2048). One difference
                    # also beats an accumulated running sum for drift.
                    s0 = (surf[k, i0]
                          - (center[0] - surf_cx[k, i0]) * cos_phi[i0]
                          - (center[1] - surf_cy[k, i0]) * sin_phi[i0])
                    s1 = (surf[k, i1]
                          - (center[0] - surf_cx[k, i1]) * cos_phi[i1]
                          - (center[1] - surf_cy[k, i1]) * sin_phi[i1])
                    sval = s0 * (1.0 - frac) + s1 * frac
                    # A bin only means "the surface a tooth left here" if it was
                    # actually deposited within the last tooth period. Phase 0
                    # advects EVERY bin every step, including ones no tooth has
                    # touched, so an unstamped bin drifts without bound. The
                    # deposit is gated on `engaged`, so bins at the edges of the
                    # engagement arc get stamped only intermittently -- and with
                    # a fractional steps_per_tooth the pattern repeats, giving
                    # periodic force spikes. Measured on a 62 mm slot pass:
                    # 35 spike clusters at exactly 2.00 tooth periods, peaking
                    # at 4744 N against a 2440 N mean, growing with distance as
                    # the drift accumulated. Their height was set purely by
                    # max_chip (1.0 mm = 8.9x the nominal chip): lowering it to
                    # 0.2 mm removed every spike, at any n_phi.
                    age = step_idx - surf_step[k, i0]
                    age1 = step_idx - surf_step[k, i1]
                    if age1 > age:
                        age = age1
                    if age > stale_steps or sval > 1.5 * radius \
                            or sval < radius - max_chip:
                        # Never cut, stale, or runaway -> use the
                        # boundary-limited estimate rather than nominal feed.
                        h = h_cap
                    else:
                        # A recorded surface is sub-pixel accurate and
                        # resolution-independent; do NOT cap it with the raster
                        # march, which would inject +/- half a march step.
                        h = radius - sval
                else:
                    h = h_lin
                    # No surface buffer in this mode, so the march would fire on
                    # every sample. The eroded surface it reads is itself
                    # quantised to +/- half a pixel, so the deadband must be a
                    # full pixel -- narrower and that noise trips the cap in
                    # steady state, biasing h downward (measured: 4.6% -> 16.6%
                    # on the slot case with a half-pixel deadband).
                    if h_cap < h_lin - res:
                        h = h_cap

                if h > 1e-9:
                    Ft = Kt * h * slice_height
                    Fr = Kr * h * slice_height
                    Fa = Ka * h * slice_height
                    # Tooth-frame force mapped to the workpiece frame. The
                    # rotation implied by the Shapely reference
                    #   T_WPtooth = Rz(theta) @ Rz(pi/2) @ Rx(pi) @ trans(0,r,-k*hs)
                    # is R = [[-s, c, 0], [c, s, 0], [0, 0, -1]], acting on
                    # f_local = (-spin*Kt*h*b, -Kr*h*b, -Ka*h*b).  The Rx(pi)
                    # factor inverts the local axial axis, so Fz comes out
                    # POSITIVE for a positive chip load — this is the sign
                    # convention defined by the Shapely oracle.
                    fx = spin * Ft * st - Fr * ct
                    fy = -spin * Ft * ct - Fr * st
                    fz = Fa
                    contrib[k, j, 0] = fx
                    contrib[k, j, 1] = fy
                    contrib[k, j, 2] = fz

                    # Moment about the tool centre: M = p x F, with p the tooth
                    # contact point relative to the tool axis at the slice-0
                    # plane.  (The full 6x6 adjoint of the Shapely reference
                    # reduces exactly to this cross product, because the local
                    # wrench carries no moment.)
                    px = radius * ct
                    py = radius * st
                    pz = k * slice_height
                    contrib[k, j, 3] = py * fz - pz * fy
                    contrib[k, j, 4] = pz * fx - px * fz
                    contrib[k, j, 5] = px * fy - py * fx

    # ------------------------------------------------------------------ #
    # Phase 2: swept quad erasure per tooth per slice
    #
    # For each tooth, erase the convex hull of:
    #   A = centre_now,  B = centre_prev,
    #   C = tip_prev,    D = tip_now
    # These are ordered CCW so we can use a consistent cross-product sign.
    # ------------------------------------------------------------------ #
    for k in prange(n_slices):
        for j in range(n_teeth):
            ta_now  = angle      + helix_offset[k] + j * two_pi_over_n
            ta_prev = angle_prev + helix_offset[k] + j * two_pi_over_n

            # Quad vertices CCW: A,B on the INNER radius, C=tip_prev, D=tip_now.
            #
            # The inner edge used to sit at the tool centre, so every tooth
            # re-cleared the whole sector every step -- 8 sectors are 27.9 mm^2
            # per slice per step at dt 2e-4, against ~0.08 mm^2 of genuinely new
            # material. But the disc canNOT simply be cleared wholesale: the
            # leading part of it has not been cut yet, and the teeth work
            # through it over the following revolution. Clearing it outright
            # drops the force by 55% (measured: peak 2403.7 -> 1071.6).
            #
            # What IS redundant is the inner part of each sector. Material
            # inside radius R - W was already swept when the centre was one
            # tooth period back, provided W exceeds the feed per tooth --
            # which is exactly |center - center_T|. Take 4x that, floored at two
            # pixels, and erase only the outer annulus.
            r_in = radius - erase_depth
            if r_in < 0.0:
                r_in = 0.0
            ax = center[0]      + r_in * math.cos(ta_now)
            ay = center[1]      + r_in * math.sin(ta_now)
            bx = center_prev[0] + r_in * math.cos(ta_prev)
            by = center_prev[1] + r_in * math.sin(ta_prev)
            cx = center_prev[0] + radius * math.cos(ta_prev)
            cy = center_prev[1] + radius * math.sin(ta_prev)
            dx = center[0]      + radius * math.cos(ta_now)
            dy = center[1]      + radius * math.sin(ta_now)

            # Split the quad on diagonal A-C into triangles (A,B,C) and
            # (A,C,D), and rasterise each independently.
            #
            # The previous code ran a single all-four-cross-products-same-sign
            # test over the quad, plus a heuristic C<->D swap to force convex
            # ordering. That silently failed: |AB| is the tool-centre motion in
            # one step (~0.005 mm) against |BC| ~ radius (10 mm), a 2000:1
            # ratio, so AB is numerically a doubled vertex -- yet it still
            # contributed a half-plane constraint slicing the wedge along the
            # horizontal through the tool centre. Pixels just off that line got
            # a tiny wrong-signed c0 while the other three were large and
            # correctly signed, so the same-sign test rejected them on EVERY
            # wedge, forever. Measured on the slot case: ~1.2 mm^2 of material
            # (470 px at 0.05 mm) inside the swept envelope that no wedge ever
            # erased, resolution-independent in area.
            #
            # A-C runs from the apex to a tip, so it is interior for any
            # |AB| << radius. Per-triangle sign tests are orientation-agnostic,
            # which also makes the convexity swap unnecessary. A degenerate
            # A == B (stationary tool) collapses triangle 1 to the segment A-C
            # and correctly matches nothing off that line.

            # Dexel deposit: the tooth leaves a surface at radius `radius` over
            # its swept arc — only where it was actually cutting this step.
            if use_dexel and engaged[k, j] == 1:
                d_ang = ta_now - ta_prev
                dcx = center[0] - center_prev[0]
                dcy = center[1] - center_prev[1]
                n_fill = int(math.ceil(abs(d_ang) / dphi)) + 1
                # Fill the half-open arc (ta_prev, ta_now]: skip s=0, since the
                # ta_prev bin was already stamped last step as its ta_now, so
                # each bin is stamped exactly once (lookback = one tooth period).
                # The tip crosses a bin mid-step, so deposit the surface radius
                # at the fractional crossing time: the centre was `remain` of
                # this step's motion behind the current centre.
                for s in range(1, n_fill + 1):
                    fr = s / n_fill
                    a_s = ta_prev + d_ang * fr
                    ph = a_s % two_pi  # numba % -> [0, two_pi)
                    bb = int(ph / dphi)
                    if bb >= n_phi:
                        bb = 0
                    # The tip crossed this bin mid-step, when the centre was
                    # `remain` of this step's motion back. Record that centre;
                    # the read above applies the displacement since.
                    remain = 1.0 - fr
                    surf[k, bb] = radius
                    surf_cx[k, bb] = center[0] - remain * dcx
                    surf_cy[k, bb] = center[1] - remain * dcy
                    surf_step[k, bb] = step_idx

            qx_lo = int(math.floor((min(ax, min(bx, min(cx, dx))) - ox) / res))
            qx_hi = int(math.floor((max(ax, max(bx, max(cx, dx))) - ox) / res)) + 1
            qy_lo = int(math.floor((min(ay, min(by, min(cy, dy))) - oy) / res))
            qy_hi = int(math.floor((max(ay, max(by, max(cy, dy))) - oy) / res)) + 1
            if qx_lo < 0:
                qx_lo = 0
            if qy_lo < 0:
                qy_lo = 0
            if qx_hi > NX - 1:
                qx_hi = NX - 1
            if qy_hi > NY - 1:
                qy_hi = NY - 1
            if qx_hi >= qx_lo and qy_hi >= qy_lo:
                _fill_triangle(grids, k, ax, ay, bx, by, cx, cy,
                               res, ox, oy, NX, NY)
                _fill_triangle(grids, k, ax, ay, cx, cy, dx, dy,
                               res, ox, oy, NX, NY)

    # Serial replay in the original (k, then j) order -- see `contrib` above.
    wrench = np.zeros(6)
    for k in range(n_slices):
        for j in range(n_teeth):
            wrench[0] += contrib[k, j, 0]
            wrench[1] += contrib[k, j, 1]
            wrench[2] += contrib[k, j, 2]
            wrench[3] += contrib[k, j, 3]
            wrench[4] += contrib[k, j, 4]
            wrench[5] += contrib[k, j, 5]
    return wrench


# Two compilations of the SAME body. Numba's prange degrades to a plain range
# when parallel=False, so the serial and parallel kernels cannot drift apart.
#
# Slices share nothing (grids[k], surf[k], engaged[k], contrib[k]), so the loop
# parallelises directly; the wrench reduction is buffered and replayed serially
# to keep results bit-identical rather than merely ULP-equivalent.
#
# Which one wins depends entirely on slice count: thread dispatch is a fixed
# cost paid on every timestep, and the kernel is called tens of thousands of
# times. See PARALLEL_MIN_SLICES.
_step_kernel = njit(cache=True)(_step_kernel_impl)
_step_kernel_par = njit(cache=True, parallel=True)(_step_kernel_impl)

# Measured on 22 cores, slot case, comparing the two kernels directly:
#
#   raster  slices   MB    serial      4 threads      22 threads
#   0.20      82     11     2799    3072 (1.10x)    2465 (0.88x)
#   0.20     239     31      659     812 (1.23x)     664 (1.01x)
#   0.05      82    168     1136     885 (0.78x)    1530 (1.35x)
#   0.05       3      6    18735          --        14302 (0.76x)
#
# Best case 1.35x, several cases below 1.0, and the ranking flips with thread
# count. Erasure is scattered writes over a grid far larger than cache, so the
# kernel is memory-bandwidth bound rather than compute bound -- extra threads
# contend for bandwidth instead of adding throughput. It is therefore OPT-IN,
# not automatic: pass parallel=True to try it on your own hardware, and measure.
PARALLEL_DEFAULT = False


def _warmup():
    """Trigger JIT compilation at import time (result cached to disk)."""
    g = np.ones((1, 5, 5), dtype=np.uint8)
    c = np.zeros(2)
    cp = np.zeros(2)
    h = np.zeros(1)
    surf = np.full((1, 1), 1e9)
    cphi = np.ones(1)
    sphi = np.zeros(1)
    sstep = np.zeros((1, 1), dtype=np.int64)
    for kern in (_step_kernel, _step_kernel_par):
        for dexel in (False, True):
            kern(g, c, 0.0, cp, 0.0, c, h, 1, 1, 1.0, -1, 1.0, 1.0, 1.0, 1.0,
                 0.1, 0.0, 0.0, dexel, surf, surf, surf, cphi, sphi, 1, 1.0, 1.0,
                 sstep, np.int64(0), 1.0)


_warmup()


# ---------------------------------------------------------------------------
# MillingProcess
# ---------------------------------------------------------------------------

class MillingProcess:
    """Mechanistic milling force model with helix slicing.

    Uses a NumPy uint8 raster workpiece and a Numba JIT kernel for the
    simulation hot path.

    Chip thickness is analytical (ring buffer lookback over one tooth period).
    Workpiece erosion matches the Shapely reference: per-tooth swept quads.
    All lengths in mm, forces in N.
    """

    def __init__(
        self,
        workpiece,
        n_teeth: int = 4,
        diameter_mm: float = 20.0,
        helix_angle_deg: float = 45.0,
        axial_depth_mm: float = 5.0,
        Ktc: float = 1930.4,
        Krc: float = 1159.6,
        Kac: float = 200.6,
        spindle_spin: int = -1,
        omega_rad_s: float = None,
        dt: float = None,
        raster_resolution: float = 0.5,
        max_slice_angle_deg: float = 0.5 , # 10.0,
        chip_mode: str = "analytic",   # "analytic" or "dexel"
        n_phi: int = 2048,             # dexel angular resolution (bins)
        max_chip_mm: float = 1.0,      # dexel chip bound (runaway -> analytic)
        moment_about: str = "workpiece_origin",  # or "tool_center"
        parallel: bool | None = None,  # None -> decide from slice count
    ):
        self.n_teeth     = n_teeth
        self.radius      = diameter_mm / 2.0
        self.helix_angle = math.radians(helix_angle_deg)
        self.axial_depth = axial_depth_mm
        self.Ktc         = Ktc
        self.Krc         = Krc
        self.Kac         = Kac
        self.spin        = spindle_spin

        # Axial slices for helix compensation
        max_slice_rot = math.radians(max_slice_angle_deg)
        self.n_slices = 1 + math.floor(
            axial_depth_mm * math.tan(self.helix_angle) / (self.radius * max_slice_rot)
        )
        self.slice_height = axial_depth_mm / self.n_slices
        self.helix_offset = np.array([
            k * self.slice_height / self.radius * math.tan(self.helix_angle)
            for k in range(self.n_slices)
        ])

        # Build raster grids — one per axial slice, stacked into a 3-D array
        ref = workpiece if isinstance(workpiece, WorkpieceRaster) else \
            WorkpieceRaster(np.asarray(workpiece), resolution=raster_resolution)

        self._raster_res = ref.resolution
        self._raster_ox  = ref.origin_x
        self._raster_oy  = ref.origin_y
        NY, NX = ref.grid.shape

        self._grids = np.empty((self.n_slices, NY, NX), dtype=np.uint8)
        for k in range(self.n_slices):
            self._grids[k] = ref.grid.copy()

        # Ring buffer: stores (centre [mm], spindle_angle [rad]) per step.
        # Chip thickness samples the tool centre exactly one tooth-passing
        # period back. steps_per_tooth is generally fractional, so we keep it
        # as a float and interpolate between the two bracketing history entries
        # (ages lb_floor and lb_ceil) — this removes the chip-load bias that a
        # rounded integer lookback would introduce.
        if omega_rad_s is not None and dt is not None and abs(omega_rad_s) > 1e-9:
            steps_per_tooth = 2.0 * math.pi / (n_teeth * abs(omega_rad_s) * dt)
        else:
            steps_per_tooth = 1.0
        self._steps_per_tooth = max(1.0, steps_per_tooth)
        self._lb_floor = int(math.floor(self._steps_per_tooth))
        self._lb_frac  = self._steps_per_tooth - self._lb_floor
        self._lb_ceil  = self._lb_floor + 1

        self._history: deque = deque(maxlen=self._lb_ceil + 1)
        self.force_wp = np.zeros(3)
        self.wrench_wp = np.zeros(6)

        # Both kernels compile from the same body and give bit-identical
        # results, so this is purely a speed choice. Defaults off -- see
        # PARALLEL_DEFAULT for the measurements.
        self.parallel = PARALLEL_DEFAULT if parallel is None else bool(parallel)
        self._kernel = _step_kernel_par if self.parallel else _step_kernel

        if moment_about not in ("workpiece_origin", "tool_center"):
            raise ValueError(
                f"moment_about must be 'workpiece_origin' or 'tool_center', got {moment_about!r}")
        self.moment_about = moment_about

        # Diagnostics for the "keep it honest" requirement: a ground-truth
        # reference should never approximate silently.
        self.n_steps = 0
        self.max_z_excursion_mm = 0.0   # populated by step(tool_z_mm=...)
        self._z_ref = None

        # Dexel surface buffer (chip thickness measured from the cut surface).
        # surf[k, i] = radius of last-cut surface along bin angle i, relative to
        # the current centre; sentinel 1e9 = "not yet cut" (analytic fallback).
        self.chip_mode = chip_mode
        self._max_chip = float(max_chip_mm)
        if chip_mode == "dexel":
            self._n_phi = int(n_phi)
            self._dphi  = 2.0 * math.pi / self._n_phi
            phi = np.arange(self._n_phi) * self._dphi
            self._cos_phi = np.cos(phi)
            self._sin_phi = np.sin(phi)
            self._surf = np.full((self.n_slices, self._n_phi), 1e9)
            self._surf_cx = np.zeros((self.n_slices, self._n_phi))
            self._surf_cy = np.zeros((self.n_slices, self._n_phi))
            self._surf_step = np.full((self.n_slices, self._n_phi),
                                      np.int64(-1 << 40), dtype=np.int64)
        else:
            # Minimal dummy buffers so the Numba signature stays consistent
            self._n_phi = 1
            self._dphi  = 1.0
            self._cos_phi = np.ones(1)
            self._sin_phi = np.zeros(1)
            self._surf = np.full((self.n_slices, 1), 1e9)
            self._surf_cx = np.zeros((self.n_slices, 1))
            self._surf_cy = np.zeros((self.n_slices, 1))
            self._surf_step = np.full((self.n_slices, 1),
                                      np.int64(-1 << 40), dtype=np.int64)

        # A surf bin is only a real cut surface if a tooth deposited it within
        # roughly the last tooth period; older than that and it is just an
        # advected ghost. 1.5x gives margin for the fractional steps_per_tooth
        # and for engagement flickering at the arc edges.
        self._stale_steps = 1.5 * self._steps_per_tooth


    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self, tool_center_xy: np.ndarray, spindle_angle: float,
             tool_z_mm: float | None = None) -> np.ndarray:
        """Advance one timestep.

        tool_center_xy : (2,) tool centre in workpiece frame [mm]
        spindle_angle  : cumulative spindle rotation [rad]
        tool_z_mm      : optional tool-axis height [mm]. Only logged — the
                         stacked-2D-raster model assumes constant depth of cut,
                         so z compliance is not fed back into the geometry.
                         Read `max_z_excursion_mm` afterwards to confirm the
                         assumption held.

        Returns force_wp (3,) in workpiece frame [N]. The full 6-D wrench
        [Fx,Fy,Fz,Mx,My,Mz] is available as `wrench_wp`.
        """
        center = np.asarray(tool_center_xy, dtype=np.float64).ravel()[:2]
        self.force_wp = np.zeros(3)
        self.wrench_wp = np.zeros(6)
        self.n_steps += 1

        if tool_z_mm is not None:
            if self._z_ref is None:
                self._z_ref = float(tool_z_mm)
            self.max_z_excursion_mm = max(
                self.max_z_excursion_mm, abs(float(tool_z_mm) - self._z_ref))

        if len(self._history) >= 1:
            center_prev, angle_prev = self._history[-1]

            # Chip thickness looks back one tooth-passing period; interpolate the
            # centre between the two bracketing history entries so a fractional
            # steps_per_tooth does not bias the chip load. Until enough history
            # exists, fall back to the current centre (h = 0, no forces).
            if len(self._history) >= self._lb_ceil:
                c_floor = self._history[-self._lb_floor][0]
                c_ceil  = self._history[-self._lb_ceil][0]
                center_T = (1.0 - self._lb_frac) * c_floor + self._lb_frac * c_ceil
            else:
                center_T = center

            w = self._kernel(
                self._grids,
                center, spindle_angle,
                center_prev, angle_prev,
                center_T,
                self.helix_offset,
                self.n_slices, self.n_teeth, self.radius, float(self.spin),
                self.slice_height, self.Ktc, self.Krc, self.Kac,
                self._raster_res, self._raster_ox, self._raster_oy,
                self.chip_mode == "dexel",
                self._surf, self._surf_cx, self._surf_cy,
                self._cos_phi, self._sin_phi,
                self._n_phi, self._dphi, self._max_chip,
                self._surf_step, np.int64(self.n_steps), self._stale_steps,
            )

            # The kernel returns moments about the tool centre at the slice-0
            # plane. Transport to the workpiece origin: M_o = M_c + c x F,
            # with c the tool-centre position (z = 0 at the slice-0 plane).
            if self.moment_about == "workpiece_origin":
                f = w[:3]
                w[3] += center[1] * f[2] - 0.0 * f[1]
                w[4] += 0.0 * f[0] - center[0] * f[2]
                w[5] += center[0] * f[1] - center[1] * f[0]

            # The kernel works in mm throughout, so its moments come out in
            # N*mm. Convert once, here, so everything downstream -- including
            # the robot coupling -- sees SI N*m and needs no further scaling.
            w[3:] *= MM_TO_M

            self.wrench_wp = w
            self.force_wp = w[:3].copy()

        self._history.append((center.copy(), spindle_angle))
        return self.force_wp.copy()

    def wrench(self) -> np.ndarray:
        """Last 6-D wrench [Fx,Fy,Fz,Mx,My,Mz], forces N, moments N*m.

        Force-first ordering matches pinocchio's spatial-force convention, so
        `J.T @ wrench` works directly with a LOCAL_WORLD_ALIGNED frame Jacobian.
        Note the Shapely oracle uses the opposite order ([M; F]).
        """
        return self.wrench_wp.copy()
