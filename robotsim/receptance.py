"""The arm as a receptance: `dx = G(s) F` at the tool tip, as a state space.

`robotsim.linear.LinearModel` carries the arm as `M dx'' + D dx' + K dx = F`.
`stabsim.stability` wants the same machine as a state space it can close a loop
around, and `stabsim.compensate` wants its DC gain. This module is that container,
plus the zero-order-hold integrator that steps it.

    model = tcp_linear_model(robot, theta)          # M/D/K at the tool tip
    r     = Receptance.from_mdk(model)              # the same, as a state space
    plant = r.in_workpiece(scene).plant(dt)         # stepped in the part frame
    plant.reset()
    dx_w  = plant.step(force_w)                     # one dt, force held constant

WHERE THIS ONE COMES FROM

Upstream this class held a MEASURED disturbance receptance — the closed-loop
`(I + PC)^-1 P` identified with a shaker, servo inside, valid over 1-20 Hz and
local to one pose. Nothing here reads that data. Every receptance in this
workspace is built by `Receptance.from_mdk` out of a pinocchio linearisation of
the arm, through the standard second-order form

    A = [[0, I], [-M^-1 K, -M^-1 D]]   B = [[0], [M^-1]]   C = [I, 0]   D = 0

so it is symmetric, passive and valid at DC, none of which was true of the
measured fit. That last point is what makes the DC compensation exact rather
than extrapolated: `dc_gain` of this realisation is exactly `K^-1`.

The one property inherited unchanged is the FROZEN POSE. `tcp_linear_model` is
evaluated at one configuration and the whole path is predicted with those
matrices, while the real arm's modes and compliance drift along it.

FRAMES. A model is built on the axes `tcp_linear_model` returned it on, this
repo's base frame (`frame="base"`). `in_workpiece(scene)` writes the same system
on the workpiece axes the milling engine works in: `A -> A`, `B -> B R^T`,
`C -> R C`, `D -> R D R^T`. One rotation is exact for a whole pass, because the
commanded tool orientation is constant along these paths.

INTEGRATION. `plant(dt)` discretises with one matrix exponential, zero-order
hold — exact for a force held constant across the step, which is the same
assumption the cutting engine already makes when it evaluates the cut once per
`sim_dt`, and with no step-size stability limit of its own.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.linalg import expm

FRAMES = ("ee", "base", "workpiece")


# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Receptance:
    """`dx = G(s) F` as a continuous state space, on one set of axes.

    A     (n, n)  continuous state matrix — the shared poles of all nine entries
    B     (n, 3)  force [N] in
    C     (3, n)  deflection [m] out
    D     (3, 3)  feedthrough; zero for anything built from M/D/K
    frame which axes B and C are written on: "base" (as linearised), "ee" or
          "workpiece". A label — `rotated` is what actually changes them.

    The rest is the model's own record, carried so a run can report what it was
    driven with: `modes_hz` / `damping` its modes, `q_deg` the pose it was
    linearised at, `source` where it came from.
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    frame: str = "base"
    name: str = "G"
    modes_hz: Optional[np.ndarray] = None
    damping: Optional[np.ndarray] = None
    q_deg: Optional[tuple] = None
    source: Optional[str] = None

    def __post_init__(self):
        for field in ("A", "B", "C", "D"):
            object.__setattr__(self, field,
                               np.asarray(getattr(self, field), dtype=float))
        n = self.A.shape[0]
        if (self.A.shape != (n, n) or self.B.shape != (n, 3)
                or self.C.shape != (3, n) or self.D.shape != (3, 3)):
            raise ValueError(
                f"shapes do not form a 3-in 3-out system: A {self.A.shape}, "
                f"B {self.B.shape}, C {self.C.shape}, D {self.D.shape}")
        if self.frame not in FRAMES:
            raise ValueError(f"frame {self.frame!r} — expected one of {FRAMES}")

    # ── where one comes from ─────────────────────────────────────────────────

    @classmethod
    def from_mdk(cls, model, *, name=None) -> "Receptance":
        """`LinearModel` (M/D/K at the tool tip) -> the second-order state space.

        Rejects a model written on the `ee` axes: `stabsim.stability` rotates the
        cut into the plant's frame via `scene.R_iw` and only knows how to do that
        for a plant in "base" or "workpiece". Build with `frame="base"`, or call
        `LinearModel.in_workpiece(scene)` first.
        """
        if model.frame not in ("base", "workpiece"):
            raise ValueError(
                f"the model is in the {model.frame!r} frame; stabsim reads a plant "
                "in 'base' or 'workpiece'. Build it with `frame='base'`, or rotate "
                "it with `in_workpiece(scene)` first.")

        M, D, K = (np.asarray(a, float) for a in (model.M, model.D, model.K))
        Mi = np.linalg.inv(M)
        Z, I = np.zeros((3, 3)), np.eye(3)
        damping = getattr(model, "damping_ratios", None)
        q_deg = getattr(model, "q_deg", None)
        return cls(
            A=np.block([[Z, I], [-Mi @ K, -Mi @ D]]),
            B=np.vstack([Z, Mi]),
            C=np.hstack([I, Z]),
            D=Z.copy(),
            frame=model.frame,
            name=name or f"M/D/K as a receptance ({model.name})",
            modes_hz=np.asarray(model.natural_frequencies_hz, float),
            damping=None if damping is None else np.asarray(damping, float),
            q_deg=None if q_deg is None else tuple(np.asarray(q_deg, float).ravel()),
            source=f"tcp_linear_model / {model.name}")

    # ── what it says ─────────────────────────────────────────────────────────

    @property
    def n_states(self) -> int:
        return self.A.shape[0]

    @property
    def dc_gain(self) -> np.ndarray:
        """`G(0)` [m/N] — the compliance a steady load sees. For a model built
        from M/D/K this is exactly `K^-1`, which is what makes the mean-force
        compensation an identity rather than an extrapolation."""
        return -self.C @ np.linalg.solve(self.A, self.B) + self.D

    @property
    def compliance_um_per_n(self) -> np.ndarray:
        """`G(0)` in the units deflections are read in [um/N]."""
        return self.dc_gain * 1e6

    def static_deflection_um(self, force) -> np.ndarray:
        """TCP deflection [um] under a steady force [N], same frame as the model."""
        f = np.asarray(force, dtype=float).reshape(-1)[:3]
        return self.dc_gain @ f * 1e6

    @property
    def max_real_pole(self) -> float:
        """Largest Re(pole) [1/s]. Negative is stable; near zero is barely damped."""
        return float(np.linalg.eigvals(self.A).real.max())

    def frf(self, f) -> np.ndarray:
        """`G(j 2 pi f)` as (K, 3, 3) complex [m/N], `f` in Hz."""
        f = np.atleast_1d(np.asarray(f, dtype=float))
        eye = np.eye(self.n_states)
        return np.stack([self.C @ np.linalg.solve(1j * w * eye - self.A, self.B) + self.D
                         for w in 2.0 * np.pi * f])

    # ── moving it between frames ─────────────────────────────────────────────

    def rotated(self, R, frame: str = None) -> "Receptance":
        """The same system on rotated axes: `B -> B R^T`, `C -> R C`, `D -> R D R^T`.

        `R` takes a vector's components in THIS frame to the new one — the force
        going in is rotated back, the deflection coming out is rotated forward,
        and the poles, which belong to the structure, do not move.
        """
        R = np.asarray(R, dtype=float).reshape(3, 3)
        return replace(self, B=self.B @ R.T, C=R @ self.C, D=R @ self.D @ R.T,
                       frame=frame if frame is not None else self.frame)

    def in_workpiece(self, scene) -> "Receptance":
        """Written on the WORKPIECE axes, for a scene the tool cuts at fixed attitude.

        Built in the BASE frame, so the rotation is `R_iw^T`. The commanded TCP
        orientation is constant along one of these paths, so a single rotation is
        exact for the whole run.
        """
        if self.frame == "workpiece":
            return self
        if self.frame == "base":
            return self.rotated(np.asarray(scene.R_iw, dtype=float).T, "workpiece")
        return self.rotated(np.asarray(scene.R_w_tcp, dtype=float), "workpiece")

    def scaled(self, factor: float) -> "Receptance":
        """Every entry of `G` multiplied by `factor` — the same modes on a softer
        or stiffer machine, for a sweep that asks how much the level matters."""
        return replace(self, C=self.C * float(factor), D=self.D * float(factor),
                       name=f"{self.name} x{factor:g}")

    # ── integrating it ───────────────────────────────────────────────────────

    def plant(self, dt: float) -> "ReceptancePlant":
        """A stateful integrator for this model at a fixed step [s]."""
        return ReceptancePlant(self, dt)

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path) -> Path:
        """Write the state space as it stands — rotation, scaling and all — so a
        saved run records the plant it was actually driven with."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, A=self.A, B=self.B, C=self.C, D=self.D, frame=self.frame,
                 name=self.name,
                 modes_hz=np.zeros(0) if self.modes_hz is None else self.modes_hz,
                 damping=np.zeros(0) if self.damping is None else self.damping,
                 q_deg=(np.zeros(0) if self.q_deg is None
                        else np.asarray(self.q_deg, float)),
                 source="" if self.source is None else self.source,
                 units="u [N] -> y [m], A/B/C/D continuous")
        return path

    # ── reporting ────────────────────────────────────────────────────────────

    def summary(self) -> str:
        modes = ""
        if self.modes_hz is not None and len(self.modes_hz):
            order = np.argsort(self.modes_hz)
            modes = ("         modes    "
                     + " ".join(f"{v:.2f}" for v in np.asarray(self.modes_hz)[order])
                     + " Hz\n")
            if self.damping is not None and len(self.damping) == len(self.modes_hz):
                modes += ("         zeta     "
                          + " ".join(f"{z:.3f}"
                                     for z in np.asarray(self.damping)[order]) + "\n")
        pose = ("" if self.q_deg is None else
                f"\n         at q {np.round(np.asarray(self.q_deg), 2)} deg "
                f"- the linearisation pose, FROZEN for the whole pass")
        return (f"plant    '{self.name}' | {self.frame} frame, "
                f"{self.n_states} states\n"
                f"{modes}"
                f"         max Re(pole) {self.max_real_pole:.2f} 1/s\n"
                f"         G(0) [um/N]   "
                + np.array2string(self.compliance_um_per_n, precision=2,
                                  prefix=" " * 23)
                + f"\n         1 kN along its own x -> "
                f"{np.linalg.norm(self.static_deflection_um([1e3, 0, 0])):.0f} um"
                f"{pose}")


# ─────────────────────────────────────────────────────────────────────────────

class ReceptancePlant:
    """`Receptance` with a state, stepped one `dt` at a time.

    Zero-order hold on the force: `Ad`, `Bd` come from one matrix exponential of
    the augmented system, so a step is exact for a force held constant across it.
    You drive it — `step` is the whole interface, and the loop that calls it
    belongs in your script.

    Interface-compatible with `robotsim.linear.LinearPlant`: `reset`, `x`,
    `step`, `history`. `x` is read BEFORE a step (where the tool is deflected to
    right now) and `step` returns the deflection AFTER it, which is the order the
    milling loop uses.
    """

    def __init__(self, model: Receptance, dt: float, record: bool = True):
        self.model = model
        self.dt = float(dt)
        if self.dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        n = model.n_states
        # expm([[A, B], [0, 0]] dt) = [[Ad, Bd], [0, I]] — valid whether or not A
        # is invertible, unlike the A^-1 (Ad - I) B form.
        block = np.zeros((n + 3, n + 3))
        block[:n, :n], block[:n, n:] = model.A * self.dt, model.B * self.dt
        E = expm(block)
        self.Ad, self.Bd = E[:n, :n], E[:n, n:]

        self.record = bool(record)
        self.reset()

    # ── state ────────────────────────────────────────────────────────────────

    def reset(self, state=None, force=None) -> None:
        """Put the tool at rest.

        `force` starts it on the STATIC state of that load instead of at zero,
        which is what you want when a run opens already engaged — otherwise the
        load lands on the plant at t = 0 and it rings for as long as the damping
        takes. `state` sets the n internal states directly; a deflection cannot,
        since three numbers do not fix n states.
        """
        n = self.model.n_states
        z = np.zeros(n) if state is None else np.asarray(state, float).reshape(n)
        f = (np.zeros(3) if force is None
             else np.asarray(force, dtype=float).reshape(-1)[:3])
        if force is not None:
            z = z - np.linalg.solve(self.model.A, self.model.B @ f)
        self.z, self.u = z, f
        self.time = 0.0
        self._t, self._x, self._f = [], [], []

    @property
    def x(self) -> np.ndarray:
        """Deflection [m] — the deviation of the tool from where it was told to be.

        `D u` uses the force of the LAST step. Anything built from M/D/K has
        `D = 0`, so this is `C z`; a biproper model would close an algebraic loop
        with the cut (the force depends on the position that depends on the
        force), and holding `u` one step back is how that loop is broken.
        """
        return self.model.C @ self.z + self.model.D @ self.u

    @property
    def v(self) -> np.ndarray:
        """Deflection rate [m/s], from the state derivative at the last force."""
        return self.model.C @ (self.model.A @ self.z + self.model.B @ self.u)

    # ── the step ─────────────────────────────────────────────────────────────

    def step(self, force) -> np.ndarray:
        """Advance one `dt` under `force` [N] and return the new deflection [m]."""
        f = np.asarray(force, dtype=float).reshape(-1)[:3]
        if self.record:
            self._t.append(self.time)
            self._x.append(self.x.copy())
            self._f.append(f.copy())
        self.z = self.Ad @ self.z + self.Bd @ f
        self.u = f
        self.time += self.dt
        return self.x

    # ── what it recorded ─────────────────────────────────────────────────────

    @property
    def history(self) -> tuple:
        """(t (N,), deflection (N, 3) [m], force (N, 3) [N]) as fed in."""
        if not self._t:
            return np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3))
        return (np.asarray(self._t), np.asarray(self._x), np.asarray(self._f))

    def sampling_note(self) -> str:
        """Whether `dt` resolves what the model has to say — the top mode wants
        several samples per cycle, and a step whose Nyquist sits under it aliases."""
        f_nyq = 0.5 / self.dt
        modes = np.asarray([] if self.model.modes_hz is None
                           else self.model.modes_hz, float)
        top = float(modes.max()) if modes.size else float("nan")
        warn = ("   ! under the top mode — it is aliased"
                if np.isfinite(top) and f_nyq < top else "")
        return (f"plant    {self.model.frame} frame | dt {self.dt:g} s "
                f"(Nyquist {f_nyq:.0f} Hz, top mode {top:.1f} Hz){warn}")
