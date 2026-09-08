"""`Receptance.from_mdk` is the one module here that is not upstream's, so it is
the one that needs a check of its own.

Two claims are made about it and both are testable without a robot:

  * the state space it builds is `M x'' + D x' + K x = F` written as a first-order
    system — so its DC gain is `K^-1` and its poles are the model's modes;
  * closing the cut around it through `stabsim.stability.closed_loop_matrix` —
    the general output-feedback form, written for a plant that need not be second
    order — gives the same system as substituting `D - C_cut` and `K - K_cut`
    into the second-order form directly. That equivalence is what lets this
    workspace read `stability_along_path`'s verdict as a statement about the arm
    it actually simulates.

    python -m unittest robotsim.test_receptance
"""

import unittest

import numpy as np

from robotsim.linear import LinearModel
from robotsim.receptance import Receptance
from stabsim.stability import closed_loop_matrix


def _model():
    """A 3x3 M/D/K with nothing symmetric about it by accident."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 3))
    M = A @ A.T + 3.0 * np.eye(3)                       # SPD, O(1) kg
    B = rng.normal(size=(3, 3))
    K = (B @ B.T + 2.0 * np.eye(3)) * 1.0e6             # SPD, O(1) MN/m
    D = 5.0e-5 * K                                      # stiffness-proportional,
    #                                                     zeta ~ 1.5% at these modes
    return LinearModel(M=M, D=D, K=K, frame="base", name="test")


class ReceptanceFromMDK(unittest.TestCase):

    def setUp(self):
        self.m = _model()
        self.r = Receptance.from_mdk(self.m)

    def test_dc_gain_is_the_inverse_stiffness(self):
        """`G(0) = K^-1`. This is what makes the DC compensation exact rather
        than an extrapolation, so it is worth asserting rather than assuming."""
        np.testing.assert_allclose(self.r.dc_gain, np.linalg.inv(self.m.K),
                                   rtol=1e-10, atol=1e-18)

    def test_poles_are_the_model_modes(self):
        """The six poles are the three modes, each as a conjugate pair.

        `|lambda| = w_n` exactly for a proportionally damped mode, and this test
        model is stiffness-proportional, so the only slack needed is for the
        light damping being not quite proportional after the M-normalisation.
        """
        got = np.sort(np.abs(np.linalg.eigvals(self.r.A)) / (2.0 * np.pi))
        want = np.sort(np.repeat(self.m.natural_frequencies_hz, 2))
        np.testing.assert_allclose(got, want, rtol=1e-6)
        self.assertTrue(np.all(np.asarray(self.m.damping_ratios) < 0.1),
                        "the test model went overdamped; the pole check is then "
                        "about real poles, not modes")

    def test_strictly_proper_with_relative_degree_two(self):
        """`D = 0` and `C B = 0`, which is why the algebraic loop in
        `closed_loop_matrix` collapses to the identity for this plant."""
        np.testing.assert_allclose(self.r.D, 0.0, atol=0.0)
        np.testing.assert_allclose(self.r.C @ self.r.B, 0.0, atol=1e-18)

    def test_closed_loop_matches_the_classic_second_order_form(self):
        """The general output-feedback loop vs. substituting into M/D/K directly.

        `stabsim.cut` returns the cut as a FORCE PERTURBATION, `dF = K_cut x +
        C_cut x'`, so it moves to the left-hand side with a minus: the cut
        SOFTENS the machine and, where `C_cut` is positive, removes damping.
        """
        rng = np.random.default_rng(1)
        K_cut = rng.normal(size=(3, 3)) * 1.0e5
        C_cut = rng.normal(size=(3, 3)) * 1.0e2

        got = closed_loop_matrix(self.r, K_cut, C_cut)

        M, D, K = self.m.M, self.m.D, self.m.K
        Mi = np.linalg.inv(M)
        want = np.zeros((6, 6))
        want[:3, 3:] = np.eye(3)
        want[3:, :3] = -Mi @ (K - K_cut)
        want[3:, 3:] = -Mi @ (D - C_cut)

        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-6)
        # and therefore the same eigenvalues, which is the verdict itself
        np.testing.assert_allclose(np.sort_complex(np.linalg.eigvals(got)),
                                   np.sort_complex(np.linalg.eigvals(want)),
                                   rtol=1e-8, atol=1e-6)

    def test_rejects_a_model_on_the_ee_axes(self):
        """`stabsim` rotates the cut into the plant frame via `scene.R_iw`, which
        it can only do for a plant in 'base' or 'workpiece'."""
        from dataclasses import replace
        with self.assertRaises(ValueError):
            Receptance.from_mdk(replace(self.m, frame="ee"))


if __name__ == "__main__":
    unittest.main()
