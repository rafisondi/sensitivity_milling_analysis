"""The RIGID-body dynamics, linearised — and carried to the tool.

    part  = tau_partials(robot, q, qd, qdd)          one operating point
    lin   = linearize_trajectory(robot, joints)      every sample of a command
    model = tcp_linear_model(robot, theta)           the 3x3 plant at the tool

The partial derivatives of the inverse dynamics

    tau = M(q) qdd + C(q, qd) qd + G(q)

taken about a commanded trajectory, so that a small deviation from the command
obeys the LINEAR TIME-VARYING relation

    dtau = M dqdd + dtau_dv dqd + dtau_dq dq          (at sample k)

with `tau0 = rnea(theta, thetaD, thetaDD)` the feedforward torque the deviation
is measured around. Every matrix is n x n in JOINT space, one per sample.

    M         dtau/dqdd   the joint-space inertia [kg m^2]
    dtau_dv   dtau/dqd    [N m s/rad]
    dtau_dq   dtau/dq     [N m/rad]

Scope
-----
This is the RIGID manipulator, and only that. It carries no joint springs: the
`K_m`, `D_m` of `robotsim/settings.py` are NOT in `dtau_dq`, `dtau_dv`, and
neither is the flexible/rigid joint split — all n joints are treated as driven.

COMPLIANCE IS ADDED SEPARATELY, on each matrix in turn, before the projection:

    K_total = dtau_dq[flex, flex] + diag(K_m[flex])
    D_total = dtau_dv[flex, flex] + diag(D_m[flex])

which is what `tcp_linear_model(..., springs=True)` does. Keeping the two apart
is the point: the rigid-body partials come from the URDF through pinocchio and
change with pose, velocity and gravity; the springs come from an identification
and do not. Either can be replaced without touching the other.

What each partial contains
--------------------------
`dtau_dq` is the full derivative and lumps three contributions that differ by
orders of magnitude:

    dG/dq         the gravitational stiffness — dominant, ~2.1e3 N m/rad on the
                  benchmark arm, and present whether or not the arm is moving
    d(M qdd)/dq   the inertial term. NOT negligible: ~1.8e2 N m/rad at a
                  machining feed, ~1.8e3 when the arm actually accelerates.
                  It vanishes only if thetaDD is zero.
    d(C qd)/dq    the Coriolis/centrifugal term — ~6.6 N m/rad at machining
                  speeds, i.e. the smallest of the three by far.

All three are three ORDERS OF MAGNITUDE below the joint springs (2.3e6 N m/rad),
so `dtau_dq` alone is not a structural stiffness and a plant built from it has
no business being integrated: it is the gravitational stiffening of a free arm,
asymmetric and generally indefinite, and it falls over. It is a CORRECTION to
the spring, which is why `springs=True` is the default at the tool.

`dtau_dv` is exactly `2 C` for the Christoffel factorisation that
`pin.computeCoriolisMatrix` returns, because `C qd` is a QUADRATIC form in qd
and the Christoffel symbols are symmetric in their velocity indices. It is NOT
`C`, and the factor 2 is real — see `_self_check`, which asserts it. The
derivative is taken from `pin.computeRNEADerivatives` rather than doubling `C`,
because the identity holds only for that particular factorisation and would
break silently under any other.

Where they come from
--------------------
One `pin.computeRNEADerivatives` call per sample — the analytical derivative of
the RNEA recursion (Carpentier & Mansard 2018), exact and a small multiple of
one RNEA, rather than the 2n+1 calls and sqrt(eps) accuracy of finite
differencing. The result is checked against central differences at one probe
sample before it is returned.

Gravity
-------
`dG/dq` is the largest entry of `dtau_dq`, so whether gravity is on changes the
answer completely. `gravity_on` overrides the model setting for the duration of
the call and restores it on the way out. `M` and `dtau_dv` do not depend on it.
Note the simulator runs gravity OFF by default (`Scene.gravity_on`), so match
the two deliberately rather than by accident.

Positions are MOTOR-side, as everywhere else in the package: under load the link
angles differ by the joint-spring deflection.
"""

from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from robotsim.linear import LinearModel

STANDARD_GRAVITY = np.array([0.0, 0.0, -9.81])

# Self-check tolerances. `2C` is an algebraic identity and holds to round-off;
# the finite-difference comparison is limited by the step, so it is checked
# relatively and loosely.
_TOL_2C = 1.0e-7
_TOL_FD_REL = 1.0e-5
_FD_STEP = 1.0e-6

# Above this the Jacobian is treated as singular: the arm has lost a Cartesian
# direction at this pose, and the projected stiffness in it would be reported as
# ~1e12 N/m rather than as the problem it is.
MAX_JACOBIAN_COND = 1.0e6


def _copy(x) -> np.ndarray:
    """Pinocchio returns VIEWS into `data`; the next call overwrites them."""
    return np.array(x, dtype=float, copy=True)


# ─────────────────────────────────────────────────────────────────────────────
# One sample
# ─────────────────────────────────────────────────────────────────────────────

def tau_partials(robot, q, qd=None, qdd=None, gravity_on: bool = True) -> dict:
    """The three partials of the inverse dynamics at one operating point.

    Returns `M`, `dtau_dv`, `dtau_dq` (n x n each) and `tau0` (n,), the torque
    the linearisation is taken about. `qd`, `qdd` default to zero — the standing
    arm, where `dtau_dq` is dG/dq alone.
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    qd = np.zeros_like(q) if qd is None else np.asarray(qd, dtype=float).reshape(-1)
    qdd = np.zeros_like(q) if qdd is None else np.asarray(qdd, dtype=float).reshape(-1)

    model, data = robot.model, robot.data
    saved_gravity = _copy(model.gravity.linear)
    model.gravity.linear[:] = STANDARD_GRAVITY if gravity_on else np.zeros(3)
    try:
        dtau_dq, dtau_dv, dtau_da = pin.computeRNEADerivatives(model, data, q, qd, qdd)
        out = {"dtau_dq": _copy(dtau_dq), "dtau_dv": _copy(dtau_dv),
               "M": _copy(dtau_da), "tau0": _copy(pin.rnea(model, data, q, qd, qdd))}
    finally:
        model.gravity.linear[:] = saved_gravity
    return out


def _self_check(robot, q, qd, qdd, part: dict, gravity_on: bool) -> dict:
    """Verify the partials at one probe sample. Returns the residuals.

    Two independent checks, because they fail in different ways:
      * `dtau_dv == 2C` catches a wrong factorisation or a stale view.
      * central differences on the full RNEA catch everything else, including a
        gravity setting that did not take.
    """
    model, data = robot.model, robot.data
    n = model.nv

    saved_gravity = _copy(model.gravity.linear)
    model.gravity.linear[:] = STANDARD_GRAVITY if gravity_on else np.zeros(3)
    try:
        C = _copy(pin.computeCoriolisMatrix(model, data, q, qd))
        res_2C = float(np.max(np.abs(part["dtau_dv"] - 2.0 * C)))

        fd_q, fd_v, fd_a = (np.zeros((n, n)) for _ in range(3))
        for j in range(n):
            e = np.zeros(n)
            e[j] = _FD_STEP
            fd_q[:, j] = (_copy(pin.rnea(model, data, q + e, qd, qdd))
                          - _copy(pin.rnea(model, data, q - e, qd, qdd))) / (2 * _FD_STEP)
            fd_v[:, j] = (_copy(pin.rnea(model, data, q, qd + e, qdd))
                          - _copy(pin.rnea(model, data, q, qd - e, qdd))) / (2 * _FD_STEP)
            fd_a[:, j] = (_copy(pin.rnea(model, data, q, qd, qdd + e))
                          - _copy(pin.rnea(model, data, q, qd, qdd - e))) / (2 * _FD_STEP)
    finally:
        model.gravity.linear[:] = saved_gravity

    def rel(analytic, fd):
        scale = max(float(np.max(np.abs(fd))), 1.0e-12)
        return float(np.max(np.abs(analytic - fd)) / scale)

    residuals = {
        "dtau_dv_vs_2C_Nm_s_per_rad": res_2C,
        "dtau_dq_vs_finite_diff_rel": rel(part["dtau_dq"], fd_q),
        "dtau_dv_vs_finite_diff_rel": rel(part["dtau_dv"], fd_v),
        "M_vs_finite_diff_rel": rel(part["M"], fd_a),
    }

    if res_2C > _TOL_2C:
        raise ValueError(
            f"dtau_dv disagrees with 2*C by {res_2C:.3e} N m s/rad — pinocchio's "
            "Coriolis factorisation is not the Christoffel one, or a returned "
            "view was overwritten before it was copied")
    worst = max(residuals["dtau_dq_vs_finite_diff_rel"],
                residuals["dtau_dv_vs_finite_diff_rel"],
                residuals["M_vs_finite_diff_rel"])
    if worst > _TOL_FD_REL:
        raise ValueError(
            f"the analytical partials disagree with central differences by "
            f"{worst:.3e} relative — the linearisation is not consistent with "
            "the inverse dynamics it claims to differentiate")
    return residuals


# ─────────────────────────────────────────────────────────────────────────────
# Carrying it to the tool
# ─────────────────────────────────────────────────────────────────────────────

def joint_subset(robot, subset):
    """Which joint indices the deflection lives on.

    "flex"  the flexible joints — the only ones that deviate from the command,
            so the only ones a compliance model may move. The rigid ones follow
            the motor exactly and are constrained out.
    "all"   every joint, i.e. the free rigid-body arm with no springs at all.
    """
    if subset == "all":
        return np.arange(robot.n)
    if subset == "flex":
        return np.asarray(robot.idx_flex)
    return np.asarray(subset, dtype=int)


def project_to_tcp(A_q, J, M_q=None):
    """Carry a joint-space matrix to the tool: 3x3, base-frame axes.

        square J    A_x = J^-T A_q J^-1                        exact
        wide J      A_x = Jbar^T A_q Jbar,  Jbar = M^-1 J^T (J M^-1 J^T)^-1

    `Jbar` is the dynamically consistent generalised inverse, so `M` projects to
    the operational-space inertia `(J M^-1 J^T)^-1` and the other matrices are
    carried on the same map. It reduces exactly to `J^-1` when J is square,
    which is the three-flexible-joint case of the benchmark arm.
    """
    J = np.asarray(J, dtype=float)
    if J.shape[1] < 3:
        raise ValueError(
            f"{J.shape[1]} joints cannot span the 3-D tool motion — the arm is "
            "rigid in at least one direction and its Cartesian stiffness is "
            "unbounded there. A 3x3 model needs at least three moving joints "
            "(see robotsim/settings.py: JOINT_STATE).")

    cond = np.linalg.cond(J)
    if not np.isfinite(cond) or cond > MAX_JACOBIAN_COND:
        raise ValueError(f"the Jacobian is singular at this pose (cond {cond:.2e}) "
                         "— there is no 3x3 model here")

    if J.shape[1] == 3:
        J_bar = np.linalg.inv(J)
    else:
        if M_q is None:
            raise ValueError("a wide Jacobian needs the joint inertia to build "
                             "the dynamically consistent inverse")
        M_inv_Jt = np.linalg.solve(M_q, J.T)
        J_bar = M_inv_Jt @ np.linalg.inv(J @ M_inv_Jt)
    A_x = J_bar.T @ np.asarray(A_q, dtype=float) @ J_bar
    return A_x


def tcp_linear_model(robot, theta, thetaD=None, thetaDD=None, *,
                     gravity_on: bool = True, springs: bool = True,
                     subset: str = None, symmetrise: bool = True,
                     ee_frame: str = "TCP", frame: str = "ee",
                     name: str = None) -> LinearModel:
    """The 3x3 `M dx'' + D dx' + K dx = F` at the tool, from the rigid-body
    partials — plus the joint springs, if you want them.

        M = P(dtau_dqdd)                                 always rigid-body
        D = P(dtau_dqd  + diag(D_m))                     springs optional
        K = P(dtau_dq   + diag(K_m))                     springs optional

    with `P` the Jacobian projection above, each matrix carried INDIVIDUALLY.

    springs   add the identified joint spring and damper onto `dtau_dq` and
              `dtau_dv`. The rigid-body terms are a ~0.1% correction to them, so
              with springs the model is a real structural one; WITHOUT, `K` is
              the gravitational stiffening of a free arm — asymmetric, indefinite
              and unstable to integrate. Read `springs=False` as "the plant a
              position controller has to stabilise", not as "a stiff machine".
    subset    which joints deflect: "flex" (default with springs) or "all"
              (default without). Rigid joints follow the motor exactly, so they
              carry no compliance and are constrained out of the spring model.
    symmetrise  `dtau_dq` is not symmetric and neither is its projection. On by
              default because `LinearModel` is read as a structural model and
              the asymmetry is ~0.1% of the spring; turn it off to keep the raw
              gyroscopic/gravitational asymmetry.
    """
    theta = np.asarray(theta, dtype=float).reshape(-1)
    theta = robot.expand_actuated(theta) if len(theta) != robot.n else theta
    subset = ("flex" if springs else "all") if subset is None else subset
    idx = joint_subset(robot, subset)

    part = tau_partials(robot, theta, thetaD, thetaDD, gravity_on=gravity_on)
    M_q = part["M"][np.ix_(idx, idx)]
    D_q = part["dtau_dv"][np.ix_(idx, idx)]
    K_q = part["dtau_dq"][np.ix_(idx, idx)]

    if springs:
        flex = set(np.asarray(robot.idx_flex).tolist())
        if not set(idx.tolist()) <= flex:
            raise ValueError(
                f"joints {sorted(set(idx.tolist()) - flex)} have no identified "
                "spring — a rigid joint is infinitely stiff, not zero-stiffness, "
                "so it cannot be part of the deflection subspace. Use "
                "subset='flex', or springs=False for the free rigid-body arm.")
        K_q = K_q + np.diag(np.asarray(robot.K_m_diag, dtype=float)[idx])
        D_q = D_q + np.diag(np.asarray(robot.D_m_diag, dtype=float)[idx])

    J = np.asarray(robot.jacobian(theta, ee_frame), dtype=float)[:3, idx]
    M_x, D_x, K_x = (project_to_tcp(A, J, M_q) for A in (M_q, D_q, K_q))
    if symmetrise:
        M_x, D_x, K_x = (0.5 * (A + A.T) for A in (M_x, D_x, K_x))

    model = LinearModel(
        M=M_x, D=D_x, K=K_x, frame="base", q_deg=tuple(np.rad2deg(theta)),
        ee_frame=ee_frame,
        name=name or (f"rigid-body @ pose"
                      + (" + joint springs" if springs else "")
                      + (", gravity on" if gravity_on else ", gravity off")))
    if frame == "base":
        return model
    if frame != "ee":
        raise ValueError("this writes the model in the 'base' or 'ee' frame; use "
                         "`LinearModel.in_workpiece(scene)` for the part frame")
    R_i_ee = np.asarray(robot.fkine(theta, ee_frame), dtype=float)[:3, :3]
    return model.rotated(R_i_ee.T, "ee")


# ─────────────────────────────────────────────────────────────────────────────
# Along a trajectory
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, eq=False)
class TrajectoryLinearization:
    """`dtau = M dqdd + dtau_dv dqd + dtau_dq dq` at every sample of a command.

    t, theta, thetaD, thetaDD   the operating point the partials were taken at
    tau0     (N, n)             feedforward torque there [N m]
    M        (N, n, n)          dtau/dqdd [kg m^2]
    dtau_dv  (N, n, n)          dtau/dqd  [N m s/rad]
    dtau_dq  (N, n, n)          dtau/dq   [N m/rad]
    """

    t: np.ndarray
    theta: np.ndarray
    thetaD: np.ndarray
    thetaDD: np.ndarray
    tau0: np.ndarray
    M: np.ndarray
    dtau_dv: np.ndarray
    dtau_dq: np.ndarray
    gravity_vector: np.ndarray = None
    residuals: dict = None

    def __len__(self) -> int:
        return len(self.theta)

    @property
    def n_joints(self) -> int:
        return self.theta.shape[1]

    def at(self, k: int) -> dict:
        """The four matrices at sample `k`, as `tau_partials` returns them."""
        return {"M": self.M[k], "dtau_dv": self.dtau_dv[k],
                "dtau_dq": self.dtau_dq[k], "tau0": self.tau0[k]}

    def state_space(self) -> tuple:
        """The LTV system `d/dt [dq; dqd] = A [dq; dqd] + B dtau`.

        A (N, 2n, 2n) = [[0, I], [-M^-1 dtau_dq, -M^-1 dtau_dv]]
        B (N, 2n, n)  = [[0], [M^-1]]

        Derived on demand rather than stored: the partials are the primitive,
        and A, B would otherwise be a second copy of the same information that
        could drift out of step with it.
        """
        n, N = self.n_joints, len(self)
        A = np.zeros((N, 2 * n, 2 * n))
        B = np.zeros((N, 2 * n, n))
        eye = np.eye(n)
        for k in range(N):
            A[k, :n, n:] = eye
            A[k, n:, :n] = -np.linalg.solve(self.M[k], self.dtau_dq[k])
            A[k, n:, n:] = -np.linalg.solve(self.M[k], self.dtau_dv[k])
            B[k, n:, :] = np.linalg.solve(self.M[k], eye)
        return A, B

    def summary(self) -> str:
        def rng(a):
            return f"{np.abs(a).min():.3g} .. {np.abs(a).max():.3g}"
        return (f"Rigid-body linearisation at {len(self)} samples "
                f"({self.n_joints} joints)\n"
                f"  |M|        {rng(self.M)} kg m^2\n"
                f"  |dtau_dv|  {rng(self.dtau_dv)} N m s/rad\n"
                f"  |dtau_dq|  {rng(self.dtau_dq)} N m/rad\n"
                f"  |tau0|     {rng(self.tau0)} N m\n"
                f"  gravity    {np.round(self.gravity_vector, 3)} m/s^2")


def linearize_trajectory(robot, joints, gravity_on: bool = True,
                         verbose: bool = False) -> TrajectoryLinearization:
    """Linearise the rigid-body dynamics at every sample of a joint command.

    joints  a `robotsim.kinematics.JointTrajectory` — anything with `t`, `theta`,
            `thetaD`, `thetaDD`.

    The partials are verified against central differences at one probe sample
    before returning; the residuals travel with the result.
    """
    theta = np.atleast_2d(np.asarray(joints.theta, dtype=float))
    thetaD = np.atleast_2d(np.asarray(joints.thetaD, dtype=float))
    thetaDD = np.atleast_2d(np.asarray(joints.thetaDD, dtype=float))
    t = np.asarray(joints.t, dtype=float).reshape(-1)

    if not (theta.shape == thetaD.shape == thetaDD.shape):
        raise ValueError(f"theta {theta.shape}, thetaD {thetaD.shape} and "
                         f"thetaDD {thetaDD.shape} must all match")
    if len(t) != len(theta):
        raise ValueError(f"t has {len(t)} samples but theta has {len(theta)}")

    n_samples, n = theta.shape
    M = np.zeros((n_samples, n, n))
    dtau_dv = np.zeros((n_samples, n, n))
    dtau_dq = np.zeros((n_samples, n, n))
    tau0 = np.zeros((n_samples, n))

    milestone = max(1, n_samples // 5)
    for k in range(n_samples):
        part = tau_partials(robot, theta[k], thetaD[k], thetaDD[k],
                            gravity_on=gravity_on)
        M[k], dtau_dv[k], dtau_dq[k], tau0[k] = (
            part["M"], part["dtau_dv"], part["dtau_dq"], part["tau0"])
        if verbose and (k + 1) % milestone == 0:
            print(f"  linearised {k + 1}/{n_samples} samples")

    probe = min(n_samples - 1, n_samples // 2)
    residuals = _self_check(
        robot, theta[probe], thetaD[probe], thetaDD[probe],
        {"M": M[probe], "dtau_dv": dtau_dv[probe], "dtau_dq": dtau_dq[probe]},
        gravity_on)
    residuals["probe_sample"] = int(probe)

    return TrajectoryLinearization(
        t=t, theta=theta, thetaD=thetaD, thetaDD=thetaDD, tau0=tau0,
        M=M, dtau_dv=dtau_dv, dtau_dq=dtau_dq,
        gravity_vector=STANDARD_GRAVITY if gravity_on else np.zeros(3),
        residuals=residuals)
