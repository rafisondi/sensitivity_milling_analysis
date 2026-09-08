import unittest
from unittest.mock import patch

import numpy as np

from stabsim.engagement import Engagement, engagement_gradients_along_path
from stabsim.cut import K_cut


def _engagement(s, phi_en, phi_ex):
    n = len(s)
    return Engagement(
        s_mm=np.asarray(s, dtype=float),
        xy_mm=np.column_stack([s, np.zeros(n)]),
        phi_en=np.asarray(phi_en, dtype=float),
        phi_ex=np.asarray(phi_ex, dtype=float),
        contact=np.ones(n, dtype=bool),
        R_wc=np.repeat(np.eye(2)[None, :, :], n, axis=0),
        radius_mm=10.0,
    )


class EngagementGradientGapTests(unittest.TestCase):
    def setUp(self):
        self.s = np.arange(5.0)
        en = 0.1 * self.s
        ex = 1.0 + 0.2 * self.s
        en[2] = np.nan
        ex[2] = np.nan
        self.nominal = _engagement(self.s, en, ex)
        self.plus = _engagement(self.s, en + 0.01, ex + 0.02)
        self.minus = _engagement(self.s, en - 0.01, ex - 0.02)
        self.xy = np.column_stack([self.s, np.zeros(len(self.s))])

    def gradients(self, fill_gaps):
        with patch("stabsim.engagement.engagement_angles_along_path",
                   side_effect=[self.plus, self.minus]):
            return engagement_gradients_along_path(
                self.xy, object(), radius_mm=10.0, h_mm=0.01,
                s_mm=self.s, nominal=self.nominal, fill_gaps=fill_gaps)

    def test_default_leaves_nan_gradient_region_invalid(self):
        grad = self.gradients(False)
        np.testing.assert_array_equal(
            grad.valid, [True, False, False, False, True])
        np.testing.assert_allclose(grad.dphi[~grad.valid], 0.0)

    def test_opt_in_fills_before_differentiating(self):
        grad = self.gradients(True)
        self.assertTrue(grad.valid.all())
        np.testing.assert_array_equal(
            grad.interpolated, [False, True, True, True, False])
        np.testing.assert_allclose(grad.dphi[:, 0, 0], 0.1)
        np.testing.assert_allclose(grad.dphi[:, 1, 0], 0.2)
        np.testing.assert_allclose(grad.dphi[:, 0, 1], 1.0)
        np.testing.assert_allclose(grad.dphi[:, 1, 1], 2.0)
        stiffness = K_cut(
            0.1, 1.0, 4, 0.2, 1.4, grad.dphi[2], 1000.0, 300.0)
        self.assertGreater(np.linalg.norm(stiffness), 0.0)


if __name__ == "__main__":
    unittest.main()
