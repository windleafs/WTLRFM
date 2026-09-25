import unittest

import numpy as np

from analysis.sos_identifiability import (
    build_physical_basis,
    make_roi_mask,
    network_mode_diagnostics,
    real_gram,
    relative_response,
    solve_generalized_spectrum,
    state_gram,
    synthesize_modes,
)


class SoSIdentifiabilityTests(unittest.TestCase):
    def test_basis_shapes_bounds_and_metric(self):
        x = np.linspace(-.01, .01, 21)
        z = np.linspace(0., .04, 25)
        basis, names, roi = build_physical_basis(
            x, z, z_min=.004, z_max=.036,
            depth_slabs=4, lateral_modes=2, axial_modes=2,
            gaussian_x=2, gaussian_z=2, gaussian_sigma_mm=3.)
        self.assertEqual(basis.ndim, 3)
        self.assertEqual(basis.shape[1:], (21, 25))
        self.assertEqual(len(names), basis.shape[0])
        self.assertTrue(np.isfinite(basis).all())
        self.assertLessEqual(float(np.abs(basis).max()), 1.000001)
        self.assertTrue(roi.any())
        g = state_gram(basis, roi)
        self.assertEqual(g.shape, (len(names), len(names)))
        np.testing.assert_allclose(g, g.T, atol=1e-12)
        self.assertTrue(np.linalg.eigvalsh(g).max() > 0)

    def test_relative_response_and_complex_real_gram(self):
        base = np.ones((2, 3), np.complex64) * (2 + 1j)
        deriv_a = np.ones_like(base) * (1 + 2j)
        deriv_b = np.ones_like(base) * (2 - 1j)
        delta = 5.
        plus_a, minus_a = base + delta*deriv_a, base - delta*deriv_a
        plus_b, minus_b = base + delta*deriv_b, base - delta*deriv_b
        ra = relative_response(plus_a, minus_a, base, delta)
        rb = relative_response(plus_b, minus_b, base, delta)
        g = real_gram([ra, rb])
        self.assertTrue(np.isfinite(g).all())
        self.assertAlmostEqual(g[0, 1], g[1, 0])
        expected_a = np.mean(np.abs(deriv_a)**2) / np.mean(np.abs(base)**2)
        self.assertAlmostEqual(g[0, 0], expected_a, places=6)

    def test_generalized_spectrum_recovers_weak_direction(self):
        gc = np.eye(2)
        gy = np.diag([4.0, 0.01])
        sp = solve_generalized_spectrum(gy, gc)
        np.testing.assert_allclose(sp["singular_values"], [0.1, 2.0], atol=1e-12)
        weakest = sp["coefficients"][0]
        self.assertGreater(abs(weakest[1]), .999)

    def test_generalized_spectrum_handles_nonorthogonal_basis(self):
        basis = np.array([
            [[1., 0.], [0., 0.]],
            [[1., 1.], [0., 0.]],
        ])
        roi = np.ones((2, 2), bool)
        gc = state_gram(basis, roi)
        # Observation sees only the first physical pixel. In coefficient space
        # both columns look identical, leaving one physical combination null.
        y = np.array([[1., 1.]])
        gy = y.T @ y
        sp = solve_generalized_spectrum(gy, gc)
        self.assertLess(sp["singular_values"][0], 1e-8)
        modes = synthesize_modes(sp["coefficients"], basis)
        self.assertEqual(modes.shape, (2, 2, 2))

    def test_network_mode_diagnostics(self):
        coeff = np.eye(2)
        gc = np.eye(2)
        gn = np.diag([1., 4.])
        cross = np.diag([1., 1.])
        d = network_mode_diagnostics(coeff, gc, gn, cross)
        np.testing.assert_allclose(d[0], [1., 1., 0.], atol=1e-12)
        np.testing.assert_allclose(d[1], [1., 2., np.sqrt(3.)], atol=1e-12)

    def test_roi_validation(self):
        with self.assertRaises(ValueError):
            make_roi_mask(np.array([0., -1.]), np.array([0., 1.]))


if __name__ == "__main__":
    unittest.main()
