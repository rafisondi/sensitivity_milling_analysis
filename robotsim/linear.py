"""The arm as a 3x3 linear system at ONE pose.

    M dx'' + D dx' + K dx = F            at the tool tip, END-EFFECTOR frame

`dx` is the TCP DEVIATION from the commanded position [m] and `F` the force the
cut applies at the tool [N] — the same quantities `robotsim.dynamics` reports,
with the whole nonlinear flexible-joint arm collapsed into nine numbers per
matrix. Everything the arm does that is not this is gone: the pose is frozen, so
the model is only as good as the configuration it was linearised at.

    model  = default_model()                     # the base values below
    model  = LinearModel.load("mkd.json")        # yours, .json or .npz
    model  = LinearModel.from_robot(robot, theta)   # regenerate from a URDF arm

    plant  = model.rotated(scene.R_w_tcp).plant(dt)   # integrate in the part frame
    plant.reset()
    dx_w   = plant.step(force_w)                 # one dt, force held constant

FRAMES. The matrices are read in the EE frame by default — `frame="ee"`, axes of
`ee_frame` at the tool tip. `rotated(R)` writes the same system on other axes
(`A -> R A R.T`), which is how it reaches the workpiece frame the milling engine
works in: the tool holds a CONSTANT orientation over one of these paths, so
`R_w_ee = scene.R_w_tcp` exactly and one rotation covers the whole run.

INTEGRATION is exact for a force held constant across the step (zero-order hold,
matrix exponential) — the same assumption the coupled loop already makes when it
evaluates the cut once per `sim_dt`. There is no step-size stability limit to
respect, which matters here: these modes sit at 9-23 Hz but `K` is stiff enough
that an explicit scheme would need a far smaller step than the process does.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.linalg import expm

# ─────────────────────────────────────────────────────────────────────────────
# The base values: the Staeubli of `robot_model="joints123"`, reduced to the TCP
# at the start pose of the default `RunConfig` (theta = [-91.2, 26.2, 113.0,
# -2.1, -34.3, 1.4] deg), written on the EE axes. Regenerate with
#
#     from robotsim.linearize_robot_dynamics import tcp_linear_model
#     tcp_linear_model(cfg.robot(), theta, gravity_on=False)
#
# so these are not invented numbers: the rigid-body partials of that arm's own
# inverse dynamics, plus Huynh's identified joint springs, projected through its
# Jacobian. 9, 15 and 23 Hz, principal stiffness 0.44 / 0.55 / 5.81 N/um — soft
# across the tool axis, stiff along it. Standing still with gravity off the
# rigid-body terms vanish and these are the springs alone; at a machining feed
# with gravity on they move the fourth digit.
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_POSE_DEG = (-91.20, 26.24, 113.03, -2.06, -34.28, 1.39)

DEFAULT_M = np.array([[189.8683, 18.4152, -146.6844],           # [kg]
                      [18.4152, 143.7920, -21.7333],
                      [-146.6844, -21.7333, 206.2937]])

DEFAULT_D = np.array([[2671.7329, 18.7035, -646.8909],          # [N s/m]
                      [18.7035, 1071.2966, -9.9681],
                      [-646.8909, -9.9681, 436.4864]])

DEFAULT_K = np.array([[3828147.2007, 49827.2805, -2548157.5630],    # [N/m]
                      [49827.2805, 437526.5444, -39265.2354],
                      [-2548157.5630, -39265.2354, 2526565.2907]])

FRAMES = ("ee", "base", "workpiece")


def _dumps(d: dict) -> str:
    """JSON with each matrix ROW on one line — the file is meant to be edited."""
    rows = []
    for key, value in d.items():
        if isinstance(value, list) and value and isinstance(value[0], list):
            body = ",\n".join(f"    {json.dumps(r)}" for r in value)
            rows.append(f'  "{key}": [\n{body}\n  ]')
        else:
            rows.append(f'  "{key}": {json.dumps(value)}')
    return "{\n" + ",\n".join(rows) + "\n}\n"


def default_model() -> "LinearModel":
    """The base values above, as a model. What the runners use unless told otherwise."""
    return LinearModel(M=DEFAULT_M.copy(), D=DEFAULT_D.copy(), K=DEFAULT_K.copy(),
                       frame="ee", q_deg=DEFAULT_POSE_DEG,
                       name="staeubli joints123 @ default start pose")


# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LinearModel:
    """`M dx'' + D dx' + K dx = F`, 3x3, at the tool tip.

    M         (3, 3)  operational-space inertia [kg]
    D         (3, 3)  damping [N s/m]
    K         (3, 3)  stiffness [N/m]
    frame     which axes the three are written on: "ee" (default), "base" or
              "workpiece". A label — `rotated` is what actually changes them.
    q_deg     the configuration it was linearised at [deg], for the record
    ee_frame  the URDF frame it sits at
    """

    M: np.ndarray
    D: np.ndarray
    K: np.ndarray
    frame: str = "ee"
    q_deg: Optional[tuple] = None
    ee_frame: str = "TCP"
    name: str = "linear TCP model"

    def __post_init__(self):
        for field in ("M", "D", "K"):
            A = np.asarray(getattr(self, field), dtype=float)
            if A.shape != (3, 3):
                raise ValueError(f"{field} must be 3x3, got {A.shape}")
            object.__setattr__(self, field, A)
        if self.frame not in FRAMES:
            raise ValueError(f"frame {self.frame!r} — expected one of {FRAMES}")
        if np.linalg.matrix_rank(self.M) < 3:
            raise ValueError("M is singular — the tool would be massless in some "
                             "direction and the system has no dynamics there")
        if self.q_deg is not None:
            object.__setattr__(self, "q_deg", tuple(
                float(a) for a in np.asarray(self.q_deg, dtype=float).ravel()))

    # ── what the model says ──────────────────────────────────────────────────

    @property
    def natural_frequencies_hz(self) -> np.ndarray:
        """Undamped natural frequencies [Hz], ascending."""
        w2 = np.linalg.eigvals(np.linalg.solve(self.M, self.K))
        return np.sort(np.sqrt(np.abs(w2))) / (2.0 * np.pi)

    @property
    def damping_ratios(self) -> np.ndarray:
        """Modal damping ratios, in the order of `natural_frequencies_hz`.

        Read off the modes of the undamped problem — exact only when D is
        diagonalised by the same modes, which for a projected joint damper it is
        not quite. Indicative, and that is all it is used for.
        """
        w2, V = np.linalg.eig(np.linalg.solve(self.M, self.K))
        order = np.argsort(np.sqrt(np.abs(w2)))
        w = np.sqrt(np.abs(w2))[order]
        V = V[:, order]
        m = np.einsum("ij,jk,ki->i", V.T, self.M, V)
        d = np.einsum("ij,jk,ki->i", V.T, self.D, V)
        return np.real(d / (2.0 * m * np.where(w > 0, w, np.inf)))

    @property
    def compliance_um_per_n(self) -> np.ndarray:
        """`K^-1` in the units deflections are read in [um/N]."""
        return np.linalg.inv(self.K) * 1e6

    def static_deflection_um(self, force) -> np.ndarray:
        """TCP deflection [um] under a steady force [N], same frame as the model.

        The whole static content of the model — a coupled run whose force settles
        must settle here too, which is the cheapest check that a loaded file is
        the right way round.
        """
        f = np.asarray(force, dtype=float).reshape(-1)[:3]
        return np.linalg.solve(self.K, f) * 1e6

    @property
    def principal_stiffness(self) -> tuple:
        """(values [N/m], directions as columns) of K, weakest first."""
        return np.linalg.eigh(self.K)

    # ── moving it between frames ─────────────────────────────────────────────

    def rotated(self, R, frame: str = None) -> "LinearModel":
        """The same system on rotated axes: `A -> R A R.T`.

        `R` takes a vector's components in THIS model's frame to the new one, so
        an EE-frame model reaches the workpiece frame through `scene.R_w_tcp`
        and the base frame through `scene.R_i_tcp`. Give `frame` to relabel it.
        """
        R = np.asarray(R, dtype=float).reshape(3, 3)
        rot = lambda A: R @ A @ R.T
        return replace(self, M=rot(self.M), D=rot(self.D), K=rot(self.K),
                       frame=frame if frame is not None else self.frame)

    def in_workpiece(self, scene) -> "LinearModel":
        """Written on the WORKPIECE axes, for a scene the tool cuts at fixed attitude.

        `R_w_ee` is `scene.R_w_tcp`: the commanded TCP orientation is constant
        along one of these paths, so the EE axes hold still in the part frame and
        a single rotation is exact for the whole run. A model already labelled
        "workpiece" is returned untouched.
        """
        if self.frame == "workpiece":
            return self
        if self.frame == "base":
            return self.rotated(np.asarray(scene.R_iw, dtype=float).T, "workpiece")
        return self.rotated(np.asarray(scene.R_w_tcp, dtype=float), "workpiece")

    # ── integrating it ───────────────────────────────────────────────────────

    def plant(self, dt: float) -> "LinearPlant":
        """A stateful integrator for this model at a fixed step [s]."""
        return LinearPlant(self, dt)

    # ── persistence ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "frame": self.frame,
            "ee_frame": self.ee_frame,
            "q_deg": None if self.q_deg is None else list(self.q_deg),
            "M_kg": self.M.tolist(),
            "D_Ns_per_m": self.D.tolist(),
            "K_N_per_m": self.K.tolist(),
        }

    def save(self, path) -> Path:
        """Write to .json (readable, the one to hand-edit) or .npz."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".npz":
            np.savez(path, M=self.M, D=self.D, K=self.K, frame=self.frame,
                     ee_frame=self.ee_frame, name=self.name,
                     q_deg=(np.full(0, np.nan) if self.q_deg is None
                            else np.asarray(self.q_deg, dtype=float)))
        else:
            path.write_text(_dumps(self.to_dict()), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, d: dict) -> "LinearModel":
        def pick(*names):
            for n in names:
                if n in d:
                    return np.asarray(d[n], dtype=float)
            raise KeyError(f"none of {names} in the model file — it needs an M, "
                           "a D and a K, 3x3 each")
        q = d.get("q_deg")
        return cls(M=pick("M_kg", "M", "Lambda"), D=pick("D_Ns_per_m", "D", "C"),
                   K=pick("K_N_per_m", "K"),
                   frame=d.get("frame", "ee"), ee_frame=d.get("ee_frame", "TCP"),
                   q_deg=None if q is None else tuple(np.asarray(q, float).ravel()),
                   name=d.get("name", "linear TCP model"))

    @classmethod
    def load(cls, path) -> "LinearModel":
        """Read a model from .json or .npz."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"no linear model at {path}")
        if path.suffix == ".npz":
            z = np.load(path, allow_pickle=False)
            q = z["q_deg"] if "q_deg" in z.files else None
            return cls.from_dict({
                "M": z["M"], "D": z["D"], "K": z["K"],
                "frame": str(z["frame"]) if "frame" in z.files else "ee",
                "ee_frame": str(z["ee_frame"]) if "ee_frame" in z.files else "TCP",
                "name": str(z["name"]) if "name" in z.files else path.stem,
                "q_deg": None if q is None or q.size == 0 else q,
            })
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    # ── where a model comes from ─────────────────────────────────────────────

    @classmethod
    def from_robot(cls, robot, theta, **kwargs) -> "LinearModel":
        """Build the model from an arm at one configuration.

        A thin alias for `linearize_robot_dynamics.tcp_linear_model`, which is
        where the reduction lives: the RIGID-BODY partials of the inverse
        dynamics, plus the identified joint springs added onto `dtau_dq` and
        `dtau_dv` individually, each projected to the tool with the Jacobian.
        Takes that module's keywords — `springs`, `gravity_on`, `subset`,
        `thetaD`, `thetaDD`, `ee_frame`, `frame`.

        Imported here rather than at module scope: this module is the plant and
        knows nothing about pinocchio, so a model READ FROM A FILE never needs it.
        """
        from robotsim.linearize_robot_dynamics import tcp_linear_model
        return tcp_linear_model(robot, theta, **kwargs)

    # ── reporting ────────────────────────────────────────────────────────────

    def summary(self) -> str:
        k, _ = self.principal_stiffness
        pose = ("" if self.q_deg is None else
                f"\n         at q {np.round(np.asarray(self.q_deg), 2)} deg")
        return (f"model    '{self.name}' | {self.frame} frame, at '{self.ee_frame}'\n"
                f"         natural freq  {np.round(self.natural_frequencies_hz, 1)} Hz"
                f" | zeta {np.round(self.damping_ratios, 3)}\n"
                f"         principal K   {np.round(k * 1e-6, 3)} N/um (weakest "
                f"first) | 1 kN -> {np.linalg.norm(self.static_deflection_um([1e3, 0, 0])):.0f} "
                f"um along its own x{pose}")


# ─────────────────────────────────────────────────────────────────────────────

class LinearPlant:
    """`LinearModel` with a state, stepped one `dt` at a time.

    Zero-order hold on the force: `Ad`, `Bd` come from one matrix exponential of
    the 6-state system, so a step is exact for a force held constant across it
    and there is no step-size stability limit. You drive it — `step` is the whole
    interface, and the loop that calls it belongs in your script.
    """

    def __init__(self, model: LinearModel, dt: float, record: bool = True):
        self.model = model
        self.dt = float(dt)
        if self.dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        M_inv = np.linalg.inv(model.M)
        A = np.zeros((6, 6))
        A[:3, 3:] = np.eye(3)
        A[3:, :3] = -M_inv @ model.K
        A[3:, 3:] = -M_inv @ model.D
        B = np.vstack([np.zeros((3, 3)), M_inv])

        # expm([[A, B], [0, 0]] dt) = [[Ad, Bd], [0, I]] — valid whether or not A
        # is invertible, unlike the A^-1 (Ad - I) B form.
        block = np.zeros((9, 9))
        block[:6, :6], block[:6, 6:] = A * self.dt, B * self.dt
        E = expm(block)
        self.Ad, self.Bd = E[:6, :6], E[:6, 6:]

        self.record = bool(record)
        self.reset()

    # ── state ────────────────────────────────────────────────────────────────

    def reset(self, x0=None, v0=None, force=None) -> None:
        """Put the tool at rest. `force` starts it on the STATIC equilibrium of
        that load (`x = K^-1 F`) instead of at zero, which is what you want when
        a run opens already engaged — otherwise the step lands on the springs at
        t = 0 and rings for as long as the damping takes."""
        x = np.zeros(3) if x0 is None else np.asarray(x0, dtype=float).reshape(3)
        if force is not None:
            x = x + np.linalg.solve(self.model.K,
                                    np.asarray(force, dtype=float).reshape(-1)[:3])
        v = np.zeros(3) if v0 is None else np.asarray(v0, dtype=float).reshape(3)
        self.z = np.concatenate([x, v])
        self.time = 0.0
        self._t, self._x, self._f = [], [], []

    @property
    def x(self) -> np.ndarray:
        """Deflection [m] — the deviation of the tool from where it was told to be."""
        return self.z[:3]

    @property
    def v(self) -> np.ndarray:
        """Deflection rate [m/s]."""
        return self.z[3:]

    # ── the step ─────────────────────────────────────────────────────────────

    def step(self, force) -> np.ndarray:
        """Advance one `dt` under `force` [N] and return the new deflection [m]."""
        f = np.asarray(force, dtype=float).reshape(-1)[:3]
        if self.record:
            self._t.append(self.time)
            self._x.append(self.x.copy())
            self._f.append(f.copy())
        self.z = self.Ad @ self.z + self.Bd @ f
        self.time += self.dt
        return self.x

    # ── what it recorded ─────────────────────────────────────────────────────

    @property
    def history(self) -> tuple:
        """(t (N,), deflection (N, 3) [m], force (N, 3) [N]) as fed in."""
        if not self._t:
            return np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3))
        return (np.asarray(self._t), np.asarray(self._x), np.asarray(self._f))
