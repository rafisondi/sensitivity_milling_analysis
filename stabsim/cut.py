"""The cut, linearised about the planned path — the ZOA terms.

    F(t) = F0(s) + K_cut(s) dx(t) + C_cut(s) dxd(t)

Recovered from `millsim/forces.py`, `millsim/stability.py` and
`force_model_validation/linear_model.py` of the MillingBenchmarkStability tree,
unchanged in substance: same formulae, same units, same sign conventions. Only
the imports and the surrounding package moved.

WHERE THE THREE TERMS COME FROM. Expand the regenerative delay to first order,
`x(t - T) ~ x(t) - T xd(t)`, and the mechanistic cut splits cleanly in two:

  order 0   `K_cut`   the tool sits somewhere else, so the engagement angles
                      move, and with them the tooth-period AVERAGE force. A
                      STIFFNESS — the chain rule on `zero_order_force_analytical`
                      times the measured angle gradients.
  order 1   `C_cut`   the `-T xd` remainder: process DAMPING, set by the angles
                      and the spindle rate. `A_cut_0` per metre of depth.

They partition the cut rather than overlap, which is why either can be switched
off on its own. `F0` is the operating point they are the derivatives of.

VALIDITY. Both truncate the delay after one order, so the expansion parameter is
`omega T` with `T` the tooth period, and the neglected term is ~`(omega T)^2/2`
of the cut force. There is no delay left in the model, so nothing here predicts
the regenerative LOBES: it says nothing about spindle-speed selection.

FRAMES AND UNITS. Everything in this module is in the CUT frame — x along the
feed tangent, y along the LEFT normal, z along the tool axis — and every length
is in millimetres except where a docstring says otherwise. `stabsim.stability`
is what rotates the results into the frame the plant is written on.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# The operating point
# ─────────────────────────────────────────────────────────────────────────────

def zero_order_force_analytical(c, a: float, N: float, phi_st, phi_ex,
                                Ktc: float, Krc: float,
                                Kac: float = 0.0) -> np.ndarray:
    """Average force [Fx0, Fy0, Fz0] [N] for down-milling in the PAPER frame
    (feed = +x) — the zero-order (mean) term of the mechanistic force model.

    c: feed per tooth [mm], a: axial depth [mm], N: flutes,
    phi_st/phi_ex: entry/exit angles [rad], Ktc/Krc/Kac: coefficients [N/mm^2].

    Elementwise in `c`, `phi_st` and `phi_ex`, so a whole path goes through in
    one call. Reference: Eq. (9)/(10) of
    https://www.sciencedirect.com/science/article/pii/S0888327020307044
    """
    ax1_func = lambda phi: (Ktc * np.cos(2 * phi) - Krc * (2 * phi - np.sin(2 * phi)))
    ay1_func = lambda phi: (Ktc * (2 * phi - np.sin(2 * phi)) + Krc * np.cos(2 * phi))
    az1_func = lambda phi: (-Kac * np.cos(phi))

    ax1 = (N * a) / (8 * np.pi) * (ax1_func(phi_ex) - ax1_func(phi_st))
    ay1 = (N * a) / (8 * np.pi) * (ay1_func(phi_ex) - ay1_func(phi_st))
    az1 = (N * a) / (2 * np.pi) * (az1_func(phi_ex) - az1_func(phi_st))

    return np.array([ax1 * c, ay1 * c, az1 * c], dtype=float)


def entry_angle(ae_mm: float, radius_mm: float) -> float:
    """Down-milling entry angle [rad] for a radial engagement `ae_mm`.

    The straight-wall closed form, exit at pi. The path sweep does NOT use this
    — it measures both angles off the real cut geometry (`stabsim.engagement`).
    Kept for a sanity check against a hand calculation.
    """
    return float(np.pi - np.arccos(np.clip(1.0 - ae_mm / radius_mm, -1.0, 1.0)))


# ─────────────────────────────────────────────────────────────────────────────
# The two in-plane coupling matrices
# ─────────────────────────────────────────────────────────────────────────────

def A_cut_0(Ktc_N_m2: float, Krc_N_m2: float, rpm: float,
            phi_entry: float, phi_exit: float = np.pi) -> np.ndarray:
    """Velocity-coupling (process-damping) matrix, 2x2, in the CUT frame.

    With r = [sin(phi), cos(phi)], t = [cos(phi), -sin(phi)] and Omega the
    spindle rate [rad/s],

        A_cut_0 = -(1 / Omega) INT_{phi_en}^{phi_ex} (Ktc t + Krc r) r^T dphi

    the tooth-period average of the directional force matrix with the
    regenerative delay Taylor-expanded to first order, `x(t-T) - x(t) ~ -T xd`.
    The tooth count cancels between the average and the delay, which is why it
    is not an argument — and why the result scales strictly as 1/Omega, with no
    lobes.

    Cutting coefficients in N/m^2 (the usual N/mm^2 value times 1e6), so the
    result is N s/m per METRE of axial depth — multiply by `a` [m] to use it.
    """
    dphi = phi_exit - phi_entry
    ds2 = np.sin(phi_exit) ** 2 - np.sin(phi_entry) ** 2
    d2 = np.sin(2 * phi_exit) - np.sin(2 * phi_entry)

    S_ss = 0.5 * dphi - 0.25 * d2      # INT sin^2 phi  dphi
    S_cc = 0.5 * dphi + 0.25 * d2      # INT cos^2 phi  dphi
    S_sc = 0.5 * ds2                   # INT sin phi cos phi  dphi
    Kt, Kr = Ktc_N_m2, Krc_N_m2

    return (1.0 / (2.0 * np.pi * (rpm / 60.0))) * np.array([
        [-Kt * S_sc - Kr * S_ss, -Kt * S_cc - Kr * S_sc],
        [Kt * S_ss - Kr * S_sc, Kt * S_sc - Kr * S_cc],
    ])


def _alpha_prime(phi, Ktc: float, Krc: float):
    """d/dphi of the Fx kernel of `zero_order_force_analytical`."""
    return -2.0 * Ktc * np.sin(2 * phi) - Krc * (2.0 - 2.0 * np.cos(2 * phi))


def _beta_prime(phi, Ktc: float, Krc: float):
    """d/dphi of the Fy kernel."""
    return 2.0 * Ktc * (1.0 - np.cos(2 * phi)) - 2.0 * Krc * np.sin(2 * phi)


def K_cut_0(Ktc_N_mm2: float, Krc_N_mm2: float, feed_per_tooth_mm: float,
            n_teeth: float, axial_depth_mm: float, phi_entry: float,
            phi_exit: float, dphi) -> np.ndarray:
    """Position-coupling (process-stiffness) matrix, 2x2, in the CUT frame [N/m].

    The chain rule on the tooth-period average force above,

        dF/dq  =  dF/d[phi_en, phi_ex]  .  d[phi_en, phi_ex]/d[t, n]

                          c N a  [ -alpha'(phi_en)   alpha'(phi_ex) ]
        dF/d[phi] =      ------- [                                  ]
                           8 pi  [ -beta'(phi_en)    beta'(phi_ex)  ]

    `dphi` is the (2, 2) geometric block from
    `stabsim.engagement.engagement_gradients_along_path` — rows [phi_en, phi_ex],
    columns [d/dt, d/dn] in rad/mm.

    Coefficients in the usual N/mm^2 and every length in mm; the result is
    converted to N/m. The axial depth is ALREADY inside it, unlike `A_cut_0`
    which is per metre of depth — do not scale this by `a` again.
    """
    scale = feed_per_tooth_mm * n_teeth * axial_depth_mm / (8.0 * np.pi)
    dF_dphi = scale * np.array([
        [-_alpha_prime(phi_entry, Ktc_N_mm2, Krc_N_mm2),
         _alpha_prime(phi_exit, Ktc_N_mm2, Krc_N_mm2)],
        [-_beta_prime(phi_entry, Ktc_N_mm2, Krc_N_mm2),
         _beta_prime(phi_exit, Ktc_N_mm2, Krc_N_mm2)],
    ])                                          # N/rad
    return 1.0e3 * (dF_dphi @ np.asarray(dphi, dtype=float))     # N/mm -> N/m


def embed_2x2(A2: np.ndarray) -> np.ndarray:
    """Put an in-plane 2x2 cutting matrix into the 3-DOF (x, y, z) space."""
    A3 = np.zeros((3, 3))
    A3[:2, :2] = A2
    return A3


# ─────────────────────────────────────────────────────────────────────────────
# The three terms as 3x3 / 3-vectors, axial row included
# ─────────────────────────────────────────────────────────────────────────────

def F0_cut(fz_mm, a_mm: float, n_teeth: float, phi_en, phi_ex,
           Ktc: float, Krc: float, Kac: float = 0.0) -> np.ndarray:
    """Operating point (3,) [N] — the revolution-average force on the path."""
    return zero_order_force_analytical(c=fz_mm, a=a_mm, N=n_teeth,
                                       phi_st=phi_en, phi_ex=phi_ex,
                                       Ktc=Ktc, Krc=Krc, Kac=Kac)


def K_cut(fz_mm: float, a_mm: float, n_teeth: float, phi_en: float,
          phi_ex: float, dphi, Ktc: float, Krc: float,
          Kac: float = 0.0) -> np.ndarray:
    """dF/dx (3, 3) [N/m] — process stiffness. `dphi` is d[phi_en, phi_ex]/d[t, n].

    In-plane block from `K_cut_0`. The AXIAL ROW is the same chain rule applied
    to `Fz0 = N a c Kac (cos phi_en - cos phi_ex) / 2pi`: moving the tool
    in-plane shifts the engagement angles, and the axial force follows them. The
    axial COLUMN stays zero — nothing here moves along z, so there is no dFz/dz,
    which is the same constant-depth assumption the raster engine makes.
    """
    K = embed_2x2(K_cut_0(Ktc, Krc, fz_mm, n_teeth, a_mm, phi_en, phi_ex, dphi))
    dFz_dphi = (fz_mm * n_teeth * a_mm * Kac / (2.0 * np.pi)) * np.array(
        [-np.sin(phi_en), np.sin(phi_ex)])                       # N/rad
    K[2, :2] = 1e3 * (dFz_dphi @ np.asarray(dphi, dtype=float))  # N/mm -> N/m
    return K


def C_cut(a_mm: float, rpm: float, phi_en: float, phi_ex: float,
          Ktc: float, Krc: float, Kac: float = 0.0) -> np.ndarray:
    """dF/dxd (3, 3) [N s/m] — process damping. `A_cut_0` is per metre of depth.

    The axial row is the same velocity coupling with dFz/dh in place of the
    in-plane one:  A_z = +(Kac/Omega) [INT sin, INT cos] dphi.

    The PLUS sign is not a typo. `zero_order_force_analytical` carries the cut
    with x, y equal to MINUS the (Ktc t + Krc r) h integral but z equal to PLUS
    the Kac h integral — checked against a direct quadrature, ratio [-1, -1, +1].
    `A_cut_0` is written as -(1/Omega) INT (Ktc t + Krc r) r^T dphi, which in
    terms of that convention is +(1/Omega) INT (dF/dh) r^T dphi. Carrying the
    same form to the axial row therefore flips the sign relative to a naive
    derivation; getting it wrong shows up as a clean anticorrelation.
    """
    C = (a_mm * 1e-3) * embed_2x2(
        A_cut_0(Ktc * 1e6, Krc * 1e6, rpm, phi_en, phi_ex))
    S_s = np.cos(phi_en) - np.cos(phi_ex)          # INT sin(phi) dphi
    S_c = np.sin(phi_ex) - np.sin(phi_en)          # INT cos(phi) dphi
    C[2, :2] = (a_mm * 1e-3) * ((Kac * 1e6) * np.array([S_s, S_c])
                                / (2.0 * np.pi * (rpm / 60.0)))
    return C
