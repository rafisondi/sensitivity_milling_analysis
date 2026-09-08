"""The cutting engine inside the flexible-joint robot.

The ONLY module where the two halves meet: `fastsim` imports `robotsim` here and
nowhere else. Nothing in `robotsim` knows this file exists.

`run_pass` is a script, not a framework — the whole coupled loop is the ten
lines at the bottom of this file. Copy them into your own script whenever you
want to change what happens per step; that is cheaper than an option.

The regenerative loop: the DEFLECTED TCP is what gets handed to `process.step`,
and the dexel surface record turns that displacement into chip thickness on the
next tooth pass.

The raster is a CONSTANT-DEPTH model — axial compliance is not fed back into the
geometry. Check `depth_check()` before trusting an absolute force level.
"""

from dataclasses import dataclass

import numpy as np

from robotsim import kinematics
from robotsim.dynamics import SimResult, Simulator
from robotsim.robot import Robot
from robotsim.scene import Scene
from robotsim.trajectory import OperationalPath
from robotsim.transforms import invert_transform, transform_points

from fastsim import metrics
from fastsim.config import MillConfig
from fastsim.geometry import Workpiece
from fastsim.prepare import build_process, prepare_steady_state_cut


class ProcessAdapter:
    """MillingProcess behind the 3-D-force interface the dynamics loop expects.

    The loop wants a force; the engine makes a 6-D wrench. The moments are
    recorded in `wrench_w` but NOT fed back — the loop applies a pure force.
    """

    def __init__(self, process, record: bool = True):
        self.process = process
        self.record = bool(record)
        self._wrench = []

    def step(self, tool_center_xy_mm, spindle_angle_signed: float) -> np.ndarray:
        self.process.step(tool_center_xy_mm, spindle_angle_signed)
        if self.record:
            self._wrench.append(self.process.wrench_wp.copy())
        return self.process.force_wp

    @property
    def wrench_w(self) -> np.ndarray:
        return np.asarray(self._wrench) if self._wrench else np.zeros((0, 6))

    @property
    def moment_w(self) -> np.ndarray:
        return self.wrench_w[:, 3:]

    def __getattr__(self, name):
        # Guard against recursion before __init__ has bound `process`.
        if name == "process":
            raise AttributeError(name)
        return getattr(self.process, name)


@dataclass
class MillingPass:
    """Everything one coupled pass produced."""

    cfg: MillConfig
    scene: Scene
    part: Workpiece
    path: OperationalPath
    robot: Robot
    joints: kinematics.JointTrajectory
    process: ProcessAdapter = None
    result: SimResult = None

    @property
    def tcp_w_mm(self) -> np.ndarray:
        """Simulated tool centre (N, 2), workpiece frame [mm]."""
        self._require_result()
        return transform_points(invert_transform(self.scene.T_iw),
                                self.result.tcp_i)[:, :2] * 1e3

    def commanded_w_mm(self) -> np.ndarray:
        """Commanded tool centre on the simulation grid [mm]."""
        self._require_result()
        return transform_points(invert_transform(self.scene.T_iw),
                                self.result.planned_i)[:, :2] * 1e3

    def engaged(self, force_threshold_n: float = 50.0) -> np.ndarray:
        self._require_result()
        return np.linalg.norm(self.result.force_w, axis=1) > force_threshold_n

    def deviation(self):
        """Across-path error against the NOMINAL path, perpendicular to it.

        The command IS the nominal unless it was aimed off it on purpose to
        cancel the mean deflection (`stabsim.compensate`), in which case
        `path.nom_w` is the geometry the part is supposed to end up with and the
        command is not. Error is measured against the former either way.
        """
        self._require_result()
        reference = (self.path.nom_w[:, :2] * 1e3 if self.path.has_nominal
                     else self.path.xy_mm)
        return metrics.deviation_from_path(
            self.tcp_w_mm, reference, workpiece=self.part,
            reference_at_time_xy=self.commanded_w_mm())

    def z_excursion_mm(self) -> np.ndarray:
        """Tool-axis travel of the simulated TCP [mm].

        The coupled loop hands the process only x and y, so the engine's own
        max_z_excursion_mm stays zero — recover it from the FK history instead.
        """
        self._require_result()
        p_w = transform_points(invert_transform(self.scene.T_iw), self.result.tcp_i)
        return p_w[:, 2] * 1e3 - p_w[0, 2] * 1e3

    def report(self) -> str:
        self._require_result()
        return "\n\n".join([
            self.result.report(),
            self.deviation().summary(self.engaged(), "PATH error vs the nominal"),
            self.depth_check()])

    def depth_check(self, force_threshold_n: float = 50.0) -> str:
        """Whether the constant-depth assumption survived, over ENGAGED samples."""
        dz = self.z_excursion_mm()
        eng = self.engaged(force_threshold_n)
        ap = self.part.height_mm
        span_all = float(np.ptp(dz))

        if not eng.any():
            return (f"depth    axial TCP travel {span_all:.4f} mm; never engaged "
                    f"above {force_threshold_n:g} N, nothing to check")

        dz_eng = dz[eng]
        span = float(np.ptp(dz_eng))
        frac = 100.0 * span / ap
        verdict = "ok" if frac < 2.0 else ("marginal" if frac < 10.0 else "VIOLATED")
        return (f"depth    axial TCP travel while cutting {span:.4f} mm of ap "
                f"{ap:g} mm = {frac:.2f}%  [{verdict}]\n"
                f"         drift {dz_eng[0]:+.4f} -> {dz_eng[-1]:+.4f} mm | "
                f"whole run {span_all:.4f} mm p-p (not fed back)")

    def save(self, npz_path) -> None:
        self._require_result()
        self.result.save_npz(npz_path)

    def _require_result(self) -> None:
        if self.result is None:
            raise RuntimeError("this pass was not simulated")


def solve_joints(scene: Scene, path: OperationalPath, robot: Robot = None,
                 seed_rad=None, verbose: bool = True):
    """(Robot, JointTrajectory) — the motor command along the path."""
    robot = kinematics.load_robot(scene) if robot is None else robot
    if seed_rad is None:
        seed_rad = np.deg2rad(scene.start_deg)      # the pose the scene was built on
    joints = kinematics.solve_ik(robot, path, seed_rad,
                                 ee_frame=scene.ee_frame, verbose=verbose)
    return robot, joints


def run_pass(cfg: MillConfig, part: Workpiece, path: OperationalPath,
             scene: Scene = None, robot: Robot = None, steady_state: bool = False,
             preload_i=None, carry_force_w=None, seed_rad=None,
             verbose: bool = True) -> MillingPass:
    """Solve the IK for `path`, then cut it with the robot in the loop.

    preload_i      constant external TCP wrench (6,), base frame [N, Nm]
    carry_force_w  mean cutting force [N, wp frame] carried by MOTOR torque
                   instead of by the joint springs. Without it a ~1 kN mean
                   force lands on the springs as a step at t = 0 and rings.
    """
    scene = Scene() if scene is None else scene
    robot, joints = solve_joints(scene, path, robot=robot, seed_rad=seed_rad,
                                 verbose=verbose)

    # The command sampled onto the process time grid.
    t, theta_cmd, thetaD_cmd = joints.resample(cfg.sim_dt)

    sim = Simulator(robot, scene, sim_dt=cfg.sim_dt)
    wrench_ff = (None if carry_force_w is None
                 else sim.wrench_from_force_w(carry_force_w))
    tau_ff = (lambda th: None) if wrench_ff is None else (
        lambda th: sim.tau_ff(th, wrench_ff))

    # Start ON the tracking equilibrium, or the run opens with its own transient.
    f_cut0_w = carry_force_w if steady_state else None
    sim.reset(theta_cmd[0], thetaD_cmd[0], joints.thetaDD[0],
              f_ext=sim.wrench_from_force_w(f_cut0_w, preload_i),
              tau_ff=tau_ff(theta_cmd[0]))

    if steady_state:
        # Carve along the line the arm actually starts on, not the ideal one:
        # the preset sits a few tens of um off it, and warming up on the wrong
        # line leaves a chip deficit that rings for revolutions.
        heading = path.xy_mm[min(len(path) - 1, 1)] - path.xy_mm[0]
        engine = prepare_steady_state_cut(cfg, part, sim.tcp_w_mm(theta_cmd[0]),
                                          heading, verbose=verbose)
    else:
        engine = build_process(cfg, part)
    process = ProcessAdapter(engine)

    if verbose:
        print(f"part     {part.name} | {part.height_mm:g} mm high = ap")
        print(f"engine   {process.n_slices} slices, raster {cfg.raster_mm:g} mm "
              f"({process.process._grids.nbytes / 1e6:.0f} MB)")
        print(f"sim      {len(t)} steps of {cfg.sim_dt:g} s")

    # ── THE COUPLED LOOP ─────────────────────────────────────────────────────
    mark = max(1, len(t) // 10)
    for i, ti in enumerate(t):
        xy_mm = sim.tcp_w_mm(theta_cmd[i])                  # deflected tool centre
        f_w = process.step(xy_mm, cfg.omega_rad_s * ti)     # -> cutting force [N]
        sim.step(theta_cmd[i], thetaD_cmd[i],               # -> deflects the joints
                 f_ext=sim.wrench_from_force_w(f_w, preload_i),
                 tau_ff=tau_ff(theta_cmd[i]))

        if verbose and (i + 1) % mark == 0:
            print(f"  sim {round(100 * (i + 1) / len(t)):3d}%  "
                  f"|F| = {np.linalg.norm(f_w):7.1f} N", flush=True)
    # ─────────────────────────────────────────────────────────────────────────

    return MillingPass(cfg=cfg, scene=scene, part=part, path=path, robot=robot,
                       joints=joints, process=process,
                       result=sim.result(path, nominal_force_w=carry_force_w))
