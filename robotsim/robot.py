import pinocchio as pin
import numpy as np
from pathlib import Path
from robotsim.settings import Settings


class Robot:
    def __init__(self, settings: Settings):
        self.urdf_path = str(Path(settings.URDF_PATH))   # kept for viz / reporting
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.model.gravity.linear[:] = np.array([0, 0, -9.81])
        self.data = self.model.createData()
        self.n = self.model.njoints - 1

        JOINT_STATE = settings.JOINT_STATE          # True = rigid, False = flexible
        self._check_joint_order(settings)
        if len(JOINT_STATE) != self.n:
            raise ValueError(
                f"JOINT_STATE has {len(JOINT_STATE)} entries but "
                f"{Path(settings.URDF_PATH).name} has {self.n} joints")
        self.idx_flex = np.where(~JOINT_STATE)[0]
        self.idx_rigid = np.where(JOINT_STATE)[0]
        self.K_m_diag = settings.K_m_diag
        self.D_m_diag = settings.D_m_diag
        self.sim_dt_max = getattr(settings, "SIM_DT_MAX", None)

        # Joints a motor can actually drive. Every joint, unless the settings
        # say otherwise — only the tri-joint model has passive DOFs (its virtual
        # bending springs), which the IK must leave alone.
        actuated = getattr(settings, "ACTUATED_JOINTS", None)
        if actuated is None:
            self.idx_actuated = np.arange(self.n)
        else:
            names = list(self.model.names)[1:]
            self.idx_actuated = np.array([names.index(a) for a in actuated])
        self.K_m_diag_flex = settings.K_m_diag[self.idx_flex].reshape(-1, 1)
        self.D_m_diag_flex = settings.D_m_diag[self.idx_flex].reshape(-1, 1)

    def _check_joint_order(self, settings: Settings) -> None:
        """The stiffness tables are positional — verify they line up with the model.

        `settings.JOINT_NAMES` lists the joints in the order its `K_m_diag` /
        `D_m_diag` / `JOINT_STATE` entries are written. pinocchio numbers joints
        by its own traversal of the URDF, so a joint inserted or reordered in
        the file would silently shift every stiffness onto the wrong axis. Cheap
        to check, expensive to debug. Settings without `JOINT_NAMES` are left
        alone.
        """
        names = getattr(settings, "JOINT_NAMES", None)
        if names is None:
            return
        model_names = list(self.model.names)[1:]     # names[0] is "universe"
        if list(names) != model_names:
            raise ValueError(
                f"{type(settings).__name__}.JOINT_NAMES does not match "
                f"{Path(settings.URDF_PATH).name}:\n"
                f"  settings: {list(names)}\n"
                f"  model:    {model_names}")

    def expand_actuated(self, q) -> np.ndarray:
        """Pad a MOTOR command out to the full joint vector of the URDF.

        Only `SettingsTriJoint` has more joints than motors: it inserts two
        passive bending springs after each of the six real axes. A six-value
        command then lands on `idx_actuated` and those springs start undeflected,
        which is what commanding a motor means there.

        ---> Padding used for tri-joint compliance, else not really important
        """
        q = np.asarray(q, dtype=float).reshape(-1)
        if len(q) == self.n:
            return q.copy()
        if len(q) != len(self.idx_actuated):
            raise ValueError(
                f"configuration of length {len(q)} is neither the full "
                f"{self.n} joints nor the {len(self.idx_actuated)} actuated ones")
        q_full = np.zeros(self.n)
        q_full[self.idx_actuated] = q
        return q_full

    def fkine(self, q, ee_name):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        ee_id = self.model.getFrameId(ee_name)
        T_ee = self.data.oMf[ee_id]

        T_matrix = np.vstack([
            np.hstack([T_ee.rotation, T_ee.translation.reshape(3, 1)]),
            np.array([0, 0, 0, 1])
        ])
        return T_matrix

    def clik(self, p: np.ndarray, R: np.ndarray, ee_name: str, q_init: np.ndarray = None,
             eps: float = 1e-4, it_max: int = 1000, update_rate: float = 1e-1, damp: float = 1e-12):
        """closed-loop inverse kinematics

        Source: https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/devel/doxygen-html/md_doc_b-examples_i-inverse-kinematics.html
        """
        ee_id = self.model.getFrameId(ee_name)

        oMdes = pin.SE3(R, p)
        if q_init is None:
            q = pin.neutral(self.model)
        else:
            q = self.expand_actuated(q_init)

        # Passive DOFs (the tri-joint bending springs) stay wherever the seed
        # put them: zeroing their Jacobian columns keeps the IK from steering
        # with joints no motor can drive.
        passive = np.setdiff1d(np.arange(self.n), self.idx_actuated)

        i = 0
        while True:
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            dMi = oMdes.actInv(self.data.oMf[ee_id])
            err = pin.log(dMi).vector
            if np.linalg.norm(err) < eps:
                success = True
                break
            if i >= it_max:
                success = False
                break
            J = pin.computeFrameJacobian(self.model, self.data, q, ee_id, pin.ReferenceFrame.LOCAL)
            J[:, passive] = 0.0
            v = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
            q = pin.integrate(self.model, q, v * update_rate)
            i += 1

        if not success:
            print("\nWarning: the iterative algorithm has not reached convergence to the desired precision")

        return q

    def jacobian(self, q, ee_name):
        ee_id = self.model.getFrameId(ee_name)
        pin.computeJointJacobians(self.model, self.data, q)
        return pin.computeFrameJacobian(self.model, self.data, q, ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

    def inertia_matrix(self, q):
        return pin.crba(self.model, self.data, q)  # Composite Rigid Body Algorithm

    def bias(self, q, v):
        bias = pin.rnea(self.model, self.data, q, v, np.zeros(self.model.nv))  # no acceleration
        return bias.reshape(-1, 1)

    def forward_dynamics(self, q_full, qD_full, tau_ext, tau_u):
        bias = self.bias(q_full, qD_full)[self.idx_flex, :]                     # Coriolis + gravity, flex rows
        inertia_matrix = self.inertia_matrix(q_full)[np.ix_(self.idx_flex, self.idx_flex)]

        rhs_q = -bias + tau_ext + tau_u
        qDD_flex = np.linalg.solve(inertia_matrix, rhs_q)

        return qDD_flex
